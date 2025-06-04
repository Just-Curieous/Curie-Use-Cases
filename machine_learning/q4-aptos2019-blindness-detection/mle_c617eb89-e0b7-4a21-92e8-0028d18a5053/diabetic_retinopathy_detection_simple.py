#!/usr/bin/env python
# Diabetic Retinopathy Detection - Simplified PyTorch Version
# APTOS 2019 Dataset

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
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Detection - Simplified')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to train.csv')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test.csv')
    parser.add_argument('--train_images_dir', type=str, required=True, help='Path to train images directory')
    parser.add_argument('--test_images_dir', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--max_train_samples', type=int, default=200, help='Maximum number of training samples to use')
    return parser.parse_args()

def preprocess_image(image_path, img_size):
    """
    Simple preprocessing function for retina images
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image {image_path}")
            return np.zeros((img_size, img_size, 3), dtype=np.float32)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Simple resize (no circular crop for simplicity)
        img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values
        img_normalized = img_resized / 255.0
        
        return img_normalized.astype(np.float32)
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return np.zeros((img_size, img_size, 3), dtype=np.float32)

class RetinaDataset(Dataset):
    def __init__(self, dataframe, img_dir, img_size=128, is_test=False):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_test = is_test
        self.transform = transforms.Compose([
            transforms.ToTensor(),
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

def create_model(num_classes=5):
    """
    Create a simple model using pretrained MobileNetV2
    """
    # Use a pretrained model for feature extraction
    model = models.mobilenet_v2(weights='DEFAULT')
    
    # Replace the classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Train the model
    """
    start_time = time.time()
    
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
        logger.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
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
        logger.info(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    training_time = time.time() - start_time
    logger.info(f'Training completed in {training_time:.2f} seconds')
    
    return model, training_time

def evaluate_model(model, test_loader, test_df, output_dir, device):
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
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'Submission saved to {submission_path}')
    
    metrics = {
        'inference_time': inference_time
    }
    
    return metrics, submission_df

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
        
        # Limit training data
        if args.max_train_samples is not None and args.max_train_samples < train_df.shape[0]:
            logger.info(f"Limiting training data to {args.max_train_samples} samples")
            train_df = train_df.sample(n=args.max_train_samples, random_state=42)
        
        # Split training data into train and validation sets
        from sklearn.model_selection import train_test_split
        train_subset, val_subset = train_test_split(
            train_df, 
            test_size=0.2,
            random_state=42,
            stratify=train_df['diagnosis'] if 'diagnosis' in train_df.columns else None
        )
        
        logger.info(f"Training subset: {train_subset.shape[0]} samples")
        logger.info(f"Validation subset: {val_subset.shape[0]} samples")
        
        # Create datasets
        train_dataset = RetinaDataset(train_subset, args.train_images_dir, args.img_size)
        val_dataset = RetinaDataset(val_subset, args.train_images_dir, args.img_size)
        test_dataset = RetinaDataset(test_df, args.test_images_dir, args.img_size, is_test=True)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        # Create model
        model = create_model()
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        model, training_time = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            args.epochs,
            device
        )
        
        # Save model
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))
        
        # Evaluate model
        metrics, submission_df = evaluate_model(model, test_loader, test_df, args.output_dir, device)
        metrics['training_time'] = training_time
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
        
        logger.info("Done!")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()