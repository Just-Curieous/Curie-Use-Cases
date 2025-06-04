#!/usr/bin/env python
# PCam Cancer Detection Experiment
# Model: EfficientNetB0 with ImageNet pretrained weights
# 5-fold cross-validation

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import models, transforms
from torchvision.io import read_image
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

# PCam Dataset class
class PCamDataset(Dataset):
    def __init__(self, img_dir, labels_file=None, transform=None, is_test=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
        # Get all image files
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
        
        # Load labels if provided
        if labels_file and not is_test:
            self.labels_df = pd.read_csv(labels_file)
            # Create a dictionary for faster lookup
            self.labels = {row['id']: row['label'] for _, row in self.labels_df.iterrows()}
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Read image using PIL for better compatibility
        image = Image.open(img_path)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # For test set, return image and ID
        if self.is_test:
            img_id = os.path.splitext(img_name)[0]
            return image, img_id
        
        # For training set, return image and label
        img_id = os.path.splitext(img_name)[0]
        label = self.labels.get(img_id, 0)  # Default to 0 if not found
        return image, torch.tensor(label, dtype=torch.float32)

# Data transformations
def get_transforms():
    # Training transformations with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test transformations (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Model definition
def get_model(device):
    # Load pretrained EfficientNetB0
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    # Modify the classifier for binary classification
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    
    # Move model to device
    model = model.to(device)
    logger.info(f"Model loaded: EfficientNetB0 (pretrained)")
    
    return model

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    logger.info(f"Starting training for {num_epochs} epochs")
    
    # Lists to track metrics
    train_losses = []
    val_losses = []
    val_aucs = []
    
    best_val_auc = 0.0
    best_model_weights = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1, 1)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                # Store predictions and labels for AUC calculation
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Calculate AUC
        val_auc = roc_auc_score(all_labels, all_preds)
        val_aucs.append(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_weights = model.state_dict().copy()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val AUC: {val_auc:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    return model, train_losses, val_losses, val_aucs

# Evaluation function
def evaluate_model(model, test_loader, device):
    logger.info("Evaluating model")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)
            
            # Forward pass
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            # Store predictions and labels
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Calculate metrics
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    
    auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    
    metrics = {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    logger.info(f"Evaluation metrics: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    return metrics, all_preds, all_labels

# Main experiment function
def run_experiment(data_dir, output_dir, num_epochs=10, batch_size=32, n_folds=5):
    start_time = time.time()
    logger.info("Starting PCam cancer detection experiment")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Paths
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    labels_file = os.path.join(data_dir, 'train_labels.csv')
    
    # Get data transformations
    train_transform, val_transform = get_transforms()
    
    # Create dataset
    full_dataset = PCamDataset(train_dir, labels_file, transform=train_transform)
    
    # K-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Lists to store results
    fold_metrics = []
    
    # Run K-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        logger.info(f"Starting fold {fold+1}/{n_folds}")
        
        # Create data samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # Create data loaders
        train_loader = DataLoader(
            full_dataset, batch_size=batch_size, sampler=train_sampler, 
            num_workers=4, pin_memory=True
        )
        
        # For validation, we use the validation transform
        val_dataset = PCamDataset(train_dir, labels_file, transform=val_transform)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler, 
            num_workers=4, pin_memory=True
        )
        
        # Initialize model
        model = get_model(device)
        
        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        model, train_losses, val_losses, val_aucs = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, num_epochs
        )
        
        # Evaluate model
        metrics, _, _ = evaluate_model(model, val_loader, device)
        fold_metrics.append(metrics)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_fold_{fold+1}.pth'))
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Fold {fold+1} - Loss Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(val_aucs, label='Val AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.title(f'Fold {fold+1} - AUC Curve')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_curves_fold_{fold+1}.png'))
        plt.close()
    
    # Calculate average metrics across folds
    avg_metrics = {
        'auc': np.mean([m['auc'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1': np.mean([m['f1'] for m in fold_metrics]),
        'auc_std': np.std([m['auc'] for m in fold_metrics]),
        'precision_std': np.std([m['precision'] for m in fold_metrics]),
        'recall_std': np.std([m['recall'] for m in fold_metrics]),
        'f1_std': np.std([m['f1'] for m in fold_metrics])
    }
    
    # Log average metrics
    logger.info("Cross-validation completed")
    logger.info(f"Average AUC: {avg_metrics['auc']:.4f} ± {avg_metrics['auc_std']:.4f}")
    logger.info(f"Average Precision: {avg_metrics['precision']:.4f} ± {avg_metrics['precision_std']:.4f}")
    logger.info(f"Average Recall: {avg_metrics['recall']:.4f} ± {avg_metrics['recall_std']:.4f}")
    logger.info(f"Average F1: {avg_metrics['f1']:.4f} ± {avg_metrics['f1_std']:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'fold_metrics.csv'), index=False)
    
    # Save average metrics
    avg_metrics_df = pd.DataFrame([avg_metrics])
    avg_metrics_df.to_csv(os.path.join(output_dir, 'average_metrics.csv'), index=False)
    
    # Train final model on all data
    logger.info("Training final model on all data")
    
    # Create data loader for all training data
    full_loader = DataLoader(
        full_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    # Initialize final model
    final_model = get_model(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    
    # Train final model (no validation)
    final_model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, labels in full_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = final_model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(full_loader.dataset)
        logger.info(f"Final model - Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
    
    # Save final model
    torch.save(final_model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # Calculate total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Experiment completed in {execution_time:.2f} seconds")
    
    return avg_metrics

if __name__ == "__main__":
    # Parse command line arguments
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/mle_dataset"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/mle_b3f788d8-5097-4fb4-a60b-5e1198e43a7b/output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiment
    metrics = run_experiment(
        data_dir=data_dir,
        output_dir=output_dir,
        num_epochs=10,
        batch_size=32,
        n_folds=5
    )
    
    # Print final results
    print("\n=== FINAL RESULTS ===")
    print(f"AUC-ROC: {metrics['auc']:.4f} ± {metrics['auc_std']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} ± {metrics['precision_std']:.4f}")
    print(f"Recall: {metrics['recall']:.4f} ± {metrics['recall_std']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f} ± {metrics['f1_std']:.4f}")