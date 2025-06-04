#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diabetic Retinopathy Detection using EfficientNetB4
===================================================

This script implements a diabetic retinopathy detection system using the APTOS 2019 dataset.
It includes:
- Data loading and preprocessing
- Model architecture (EfficientNetB4)
- Training and evaluation workflows
- Metrics calculation (quadratic weighted kappa, accuracy)
- Result logging and model saving

The preprocessing pipeline is modular to allow easy substitution of different methods.
"""

import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CustomTransforms:
    """Custom transforms using OpenCV and NumPy to replace torchvision transforms."""
    
    @staticmethod
    def resize(image, size):
        """Resize image to given size."""
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def center_crop(image, size):
        """Center crop image to given size."""
        h, w = image.shape[:2]
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        return image[start_y:start_y+size, start_x:start_x+size]
    
    @staticmethod
    def horizontal_flip(image, p=0.5):
        """Randomly flip image horizontally with probability p."""
        if np.random.random() < p:
            return cv2.flip(image, 1)
        return image
    
    @staticmethod
    def vertical_flip(image, p=0.5):
        """Randomly flip image vertically with probability p."""
        if np.random.random() < p:
            return cv2.flip(image, 0)
        return image
    
    @staticmethod
    def rotate(image, angle_range=10):
        """Randomly rotate image within angle_range."""
        angle = np.random.uniform(-angle_range, angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    @staticmethod
    def normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Normalize image with given mean and std."""
        # Convert to float32 and scale to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize with mean and std
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
        return image
    
    @staticmethod
    def to_tensor(image):
        """Convert image to PyTorch tensor."""
        # Convert HWC to CHW format
        image = image.transpose(2, 0, 1)
        return torch.tensor(image, dtype=torch.float32)


class DiabeticRetinopathyDataset(Dataset):
    """Dataset class for loading and preprocessing retina images."""
    
    def __init__(self, dataframe, img_dir, transform_type='val', preprocessing_method="basic"):
        """
        Args:
            dataframe: Pandas dataframe with image IDs and labels
            img_dir: Directory containing the images
            transform_type: Type of transforms to apply ('train' or 'val'/'test')
            preprocessing_method: Method to preprocess images ('basic', 'clahe', 'gaussian', 'ben_graham', 'vessel')
        """
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform_type = transform_type
        self.preprocessing_method = preprocessing_method
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        
        # Load image using OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing based on selected method
        image = self.preprocess_image(image)
        
        # Apply transforms
        image = self.apply_transforms(image)
        
        # Get label if available (for training data)
        if 'diagnosis' in self.dataframe.columns:
            label = torch.tensor(self.dataframe.iloc[idx]['diagnosis'], dtype=torch.long)
            return image, label
        else:
            return image, img_id
    
    def preprocess_image(self, image):
        """Apply preprocessing based on selected method."""
        if self.preprocessing_method == "basic":
            # Basic normalization - resize and center crop
            return image
        
        elif self.preprocessing_method == "clahe":
            # CLAHE enhancement - not implemented in this control group
            return image
        
        elif self.preprocessing_method == "gaussian":
            # Gaussian blur - not implemented in this control group
            return image
        
        elif self.preprocessing_method == "ben_graham":
            # Ben Graham preprocessing - not implemented in this control group
            return image
        
        elif self.preprocessing_method == "vessel":
            # Vessel extraction - not implemented in this control group
            return image
        
        else:
            return image
    
    def apply_transforms(self, image):
        """Apply transforms based on transform_type."""
        # Resize to 380x380
        image = CustomTransforms.resize(image, 380)
        
        # Center crop to 380x380
        image = CustomTransforms.center_crop(image, 380)
        
        # Apply data augmentation for training
        if self.transform_type == 'train':
            image = CustomTransforms.horizontal_flip(image)
            image = CustomTransforms.vertical_flip(image)
            image = CustomTransforms.rotate(image, 10)
        
        # Normalize and convert to tensor
        image = CustomTransforms.normalize(image)
        image = CustomTransforms.to_tensor(image)
        
        return image


class EfficientNetModel(nn.Module):
    """EfficientNetB4 model for diabetic retinopathy classification."""
    
    def __init__(self, num_classes=5):
        super(EfficientNetModel, self).__init__()
        # Load pre-trained EfficientNetB4
        self.model = models.efficientnet_b4(weights='DEFAULT')
        
        # Replace the classifier
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, patience, output_dir):
    """
    Train the model with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        output_dir: Directory to save model and results
    
    Returns:
        Trained model and training history
    """
    since = time.time()
    best_model_wts = model.state_dict()
    best_kappa = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_kappa': [], 'val_kappa': []}
    
    # Early stopping counter
    counter = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            all_labels = []
            all_preds = []
            
            # Disable tqdm progress bar output
            for inputs, labels in tqdm(dataloader, disable=True):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.detach().cpu().tolist())
                all_preds.extend(preds.detach().cpu().tolist())
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
            epoch_acc = accuracy_score(all_labels, all_preds)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Kappa: {epoch_kappa:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_kappa'].append(epoch_kappa)
                scheduler.step()
            else:
                history['val_loss'].append(epoch_loss)
                history['val_kappa'].append(epoch_kappa)
                
                # If best model, save it
                if epoch_kappa > best_kappa:
                    best_kappa = epoch_kappa
                    best_model_wts = model.state_dict()
                    best_epoch = epoch
                    counter = 0
                    
                    # Save the best model
                    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                else:
                    counter += 1
        
        print()
        
        # Early stopping
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Kappa: {best_kappa:.4f} at epoch {best_epoch+1}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    return model, history


def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
    
    Returns:
        Predictions, true labels, and metrics
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())
    
    # Calculate metrics
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    accuracy = accuracy_score(all_labels, all_preds)
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'kappa': kappa,
        'accuracy': accuracy,
        'confusion_matrix': conf_mat.tolist()
    }
    
    return all_preds, all_labels, all_probs, metrics


def predict(model, test_loader):
    """
    Generate predictions for test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
    
    Returns:
        Predictions and image IDs
    """
    model.eval()
    all_preds = []
    all_ids = []
    
    with torch.no_grad():
        for inputs, img_ids in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.detach().cpu().tolist())
            all_ids.extend(img_ids)
    
    return all_preds, all_ids


def plot_training_history(history, output_dir):
    """
    Plot training and validation loss and kappa.
    
    Args:
        history: Training history dictionary
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot kappa
    plt.subplot(1, 2, 2)
    plt.plot(history['train_kappa'], label='Train Kappa')
    plt.plot(history['val_kappa'], label='Val Kappa')
    plt.title('Quadratic Weighted Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('Kappa')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


def main(args):
    """Main function to run the experiment."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    
    # Split training data into train and validation sets
    train_df, val_df = train_test_split(
        train_df, test_size=args.val_size, random_state=42, stratify=train_df['diagnosis']
    )
    
    print(f"Train set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")
    
    # Create datasets
    train_dataset = DiabeticRetinopathyDataset(
        train_df, args.train_img_dir, transform_type='train',
        preprocessing_method=args.preprocessing_method
    )
    
    val_dataset = DiabeticRetinopathyDataset(
        val_df, args.train_img_dir, transform_type='val',
        preprocessing_method=args.preprocessing_method
    )
    
    test_dataset = DiabeticRetinopathyDataset(
        test_df, args.test_img_dir, transform_type='test',
        preprocessing_method=args.preprocessing_method
    )
    
    # Create data loaders with num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = EfficientNetModel(num_classes=5)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # Train model
    if args.mode in ['train', 'train_eval']:
        print("Training model...")
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            args.num_epochs, args.patience, args.output_dir
        )
        
        # Plot training history
        plot_training_history(history, args.output_dir)
    
    # Load best model for evaluation or prediction
    if args.mode in ['eval', 'predict', 'train_eval']:
        model_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"Model file {model_path} not found. Using the current model.")
    
    # Evaluate model
    if args.mode in ['eval', 'train_eval']:
        print("Evaluating model...")
        _, _, _, metrics = evaluate_model(model, val_loader)
        
        print(f"Validation Kappa: {metrics['kappa']:.4f}")
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        
        # Save metrics
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
    
    # Generate predictions for test set
    if args.mode in ['predict', 'train_eval']:
        print("Generating predictions...")
        predictions, img_ids = predict(model, test_loader)
        
        # Create submission file
        submission_df = pd.DataFrame({
            'id_code': img_ids,
            'diagnosis': predictions
        })
        
        submission_path = os.path.join(args.output_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        print(f"Predictions saved to {submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diabetic Retinopathy Detection")
    
    # Data paths
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--train_img_dir", type=str, required=True, help="Path to training images directory")
    parser.add_argument("--test_img_dir", type=str, required=True, help="Path to test images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    
    # Model parameters
    parser.add_argument("--preprocessing_method", type=str, default="basic", 
                        choices=["basic", "clahe", "gaussian", "ben_graham", "vessel"],
                        help="Preprocessing method")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr_step_size", type=int, default=5, help="LR scheduler step size")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="LR scheduler gamma")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation set size")
    
    # Mode
    parser.add_argument("--mode", type=str, default="train_eval", 
                        choices=["train", "eval", "predict", "train_eval"],
                        help="Mode: train, eval, predict, or train_eval")
    
    args = parser.parse_args()
    main(args)