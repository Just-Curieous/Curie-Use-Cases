#!/usr/bin/env python
# Diabetic Retinopathy Detection using Multiple CNN Models
# APTOS 2019 Dataset - Experimental Group

import os
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def parse_args():
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Detection using Multiple CNN Models')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to train.csv')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test.csv')
    parser.add_argument('--train_images_dir', type=str, required=True, help='Path to train images directory')
    parser.add_argument('--test_images_dir', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    return parser.parse_args()

def preprocess_image(image_path, img_size):
    """
    Preprocess image for model input with circular crop
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image {image_path}")
            return np.zeros((img_size, img_size, 3), dtype=np.float32)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop the black borders (circular crop)
        h, w = img.shape[:2]
        radius = min(h, w) // 2
        center_x, center_y = w // 2, h // 2
        
        # Create a circular mask
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center <= radius
        
        # Apply mask
        masked_img = img.copy()
        masked_img[~mask] = 0
        
        # Resize image
        img_resized = cv2.resize(masked_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values
        img_normalized = img_resized / 255.0
        
        return img_normalized.astype(np.float32)
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return np.zeros((img_size, img_size, 3), dtype=np.float32)

class RetinaDataset(Dataset):
    def __init__(self, dataframe, img_dir, img_size=224, is_test=False, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_test = is_test
        
        # Data augmentation for training
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        
        # Load and preprocess image
        image = preprocess_image(img_path, self.img_size)
        image = self.transform(image)
        
        if not self.is_test and 'diagnosis' in self.dataframe.columns:
            label = self.dataframe.iloc[idx]['diagnosis']
            return image, label
        else:
            return image, img_id

def create_model(model_name, num_classes=5):
    """
    Create a model with pretrained weights based on model_name
    """
    if model_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights='DEFAULT')
        # Replace the classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes)
        )
        img_size = 224
    elif model_name == "densenet121":
        model = models.densenet121(weights='DEFAULT')
        # Replace the classifier
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )
        img_size = 224
    elif model_name == "inception_v3":
        model = models.inception_v3(weights='DEFAULT')
        # Replace the classifier
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )
        # Disable auxiliary outputs during inference
        model.aux_logits = False
        img_size = 299
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return model, img_size

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases"""
        logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_name, output_dir):
    """
    Train the model with early stopping
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'best_model_{model_name}.pt')
    early_stopping = EarlyStopping(patience=5, path=checkpoint_path)
    
    start_time = time.time()
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    logger.info(f'Training completed in {training_time:.2f} seconds')
    
    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model, history, training_time

def evaluate_model(model, test_loader, test_df, model_name, output_dir, device):
    """
    Evaluate the model on test data
    """
    start_time = time.time()
    
    model.eval()
    all_predictions = []
    all_img_ids = []
    
    with torch.no_grad():
        for inputs, img_ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_img_ids.extend(img_ids)
    
    inference_time = time.time() - start_time
    logger.info(f'Inference completed in {inference_time:.2f} seconds')
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id_code': all_img_ids,
        'diagnosis': all_predictions
    })
    
    # Save submission file
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, f'submission_{model_name}.csv')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'Submission saved to {submission_path}')
    
    # If test_df has ground truth labels, calculate metrics
    metrics = {'model': model_name, 'inference_time': inference_time}
    if 'diagnosis' in test_df.columns:
        true_classes = test_df['diagnosis'].values
        predicted_classes = all_predictions
        
        # Calculate metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        kappa = cohen_kappa_score(true_classes, predicted_classes, weights='quadratic')
        f1 = f1_score(true_classes, predicted_classes, average='weighted')
        
        # Print metrics
        logger.info(f"Model: {model_name}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Quadratic Weighted Kappa: {kappa:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        
        # Generate classification report
        class_report = classification_report(true_classes, predicted_classes)
        logger.info("Classification Report:")
        logger.info(class_report)
        
        # Save confusion matrix
        try:
            cm = confusion_matrix(true_classes, predicted_classes)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.colorbar()
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {e}")
        
        metrics.update({
            'accuracy': accuracy,
            'kappa': kappa,
            'f1_score': f1
        })
    
    return metrics, submission_df

def plot_training_history(history, model_name, output_dir):
    """
    Plot training history
    """
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'])
        plt.plot(history['val_acc'])
        plt.title(f'Model Accuracy - {model_name}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'])
        plt.plot(history['val_loss'])
        plt.title(f'Model Loss - {model_name}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_history_{model_name}.png'))
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting training history: {e}")

def compare_models(all_metrics, output_dir):
    """
    Create comparison plots for all models
    """
    try:
        # Create a DataFrame from all metrics
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save combined metrics
        metrics_df.to_csv(os.path.join(output_dir, 'all_models_metrics.csv'), index=False)
        
        # Plot comparison bar charts
        plt.figure(figsize=(15, 10))
        
        # Accuracy comparison
        plt.subplot(2, 2, 1)
        plt.bar(metrics_df['model'], metrics_df['accuracy'])
        plt.title('Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Kappa comparison
        plt.subplot(2, 2, 2)
        plt.bar(metrics_df['model'], metrics_df['kappa'])
        plt.title('Quadratic Weighted Kappa Comparison')
        plt.ylabel('Kappa Score')
        plt.ylim(0, 1)
        
        # F1 Score comparison
        plt.subplot(2, 2, 3)
        plt.bar(metrics_df['model'], metrics_df['f1_score'])
        plt.title('F1 Score Comparison')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1)
        
        # Training time comparison
        plt.subplot(2, 2, 4)
        plt.bar(metrics_df['model'], metrics_df['training_time'])
        plt.title('Training Time Comparison')
        plt.ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
        plt.close()
        
        # Inference time comparison
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_df['model'], metrics_df['inference_time'])
        plt.title('Inference Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'))
        plt.close()
        
        logger.info("Model comparison plots created successfully")
    except Exception as e:
        logger.error(f"Error creating model comparison: {e}")

def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load data
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
        
        logger.info(f"Training data: {train_df.shape[0]} samples")
        logger.info(f"Test data: {test_df.shape[0]} samples")
        
        # Split training data into train and validation sets
        train_subset, val_subset = train_test_split(
            train_df, 
            test_size=0.2,
            random_state=42,
            stratify=train_df['diagnosis'] if 'diagnosis' in train_df.columns else None
        )
        
        logger.info(f"Training subset: {train_subset.shape[0]} samples")
        logger.info(f"Validation subset: {val_subset.shape[0]} samples")
        
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Standard transform for validation and test
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # List of models to train and evaluate
        model_names = ["efficientnet_b4", "densenet121", "inception_v3"]
        all_metrics = []
        
        # Train and evaluate each model
        for model_name in model_names:
            logger.info(f"=== Starting training for {model_name} ===")
            
            # Create model
            model, model_img_size = create_model(model_name)
            model = model.to(device)
            
            logger.info(f"Using image size {model_img_size}x{model_img_size} for {model_name}")
            
            # Create datasets with appropriate image size
            train_dataset = RetinaDataset(train_subset, args.train_images_dir, model_img_size, transform=train_transform)
            val_dataset = RetinaDataset(val_subset, args.train_images_dir, model_img_size, transform=val_transform)
            test_dataset = RetinaDataset(test_df, args.test_images_dir, model_img_size, is_test=True, transform=val_transform)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
            
            # Train model
            model, history, training_time = train_model(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                args.epochs,
                device,
                model_name,
                args.output_dir
            )
            
            # Plot training history
            plot_training_history(history, model_name, args.output_dir)
            
            # Evaluate model
            metrics, submission_df = evaluate_model(model, test_loader, test_df, model_name, args.output_dir, device)
            metrics['training_time'] = training_time
            
            # Add metrics to the list
            all_metrics.append(metrics)
            
            # Save model
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_{model_name}.pth'))
            
            logger.info(f"=== Completed training and evaluation for {model_name} ===\n")
        
        # Compare models
        compare_models(all_metrics, args.output_dir)
        
        logger.info("All models trained and evaluated successfully!")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()