#!/usr/bin/env python
# Diabetic Retinopathy Detection using EfficientNetB4 - IMPROVED VERSION
# Control Group: Best Single Model Approach - MEMORY OPTIMIZED
#
# This version includes:
# 1. Reduced batch size (4 instead of 16)
# 2. Gradient accumulation to simulate larger batch sizes
# 3. Optimized data loading pipeline
# 4. Mixed precision training using torch.cuda.amp
# 5. Memory-efficient training techniques
# 6. Consistent training across all 5 cross-validation folds
# 7. Minimum 10 epochs per fold

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import traceback
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from tqdm import tqdm
import gc
import json
from torch.cuda.amp import autocast, GradScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Configuration class
class Config:
    def __init__(self):
        self.seed = 42
        self.img_size = 456  # EfficientNetB4 recommended input size
        self.num_classes = 5  # 5 severity levels (0-4)
        self.batch_size = 4   # Reduced batch size for memory efficiency
        self.grad_accum_steps = 4  # Gradient accumulation steps (simulates batch size of 16)
        self.num_workers = 0  # No worker processes for stable data loading
        self.num_folds = 5    # 5-fold cross-validation
        self.epochs = 20      # Train for 20 epochs with early stopping
        self.lr = 3e-4        # Learning rate
        self.min_epochs = 10  # Minimum number of epochs to train each fold
        self.patience = 7     # Early stopping patience (only after min_epochs)
        self.mixed_precision = True  # Use mixed precision training

# Apply CLAHE to improve contrast in images
def apply_clahe(image):
    # Convert tuple to list for item assignment
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image

# Custom dataset class
class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, preprocess=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.preprocess = preprocess
        self.img_paths = [os.path.join(img_dir, f"{i}.png") for i in df['id_code']]
        
        if 'diagnosis' in df.columns:
            self.labels = df['diagnosis'].values
            self.has_labels = True
        else:
            self.has_labels = False

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Read and preprocess image with timeout handling
        try:
            image = cv2.imread(img_path)
            if image is None:
                logging.error(f"Failed to read image: {img_path}")
                # Return a blank image as fallback
                image = np.zeros((Config().img_size, Config().img_size, 3), dtype=np.uint8)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing (CLAHE)
            if self.preprocess:
                image = self.preprocess(image)
            
            # Apply augmentations
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
                
            # If no labels are available (test data)
            if not self.has_labels:
                return image, self.df['id_code'].iloc[idx]
                
            label = self.labels[idx]
            return image, label
            
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {str(e)}")
            # Return a blank image and either 0 as label or the id
            image = torch.zeros((3, Config().img_size, Config().img_size))
            if self.has_labels:
                return image, 0
            else:
                return image, self.df['id_code'].iloc[idx]

# Create EfficientNetB4 model
def create_model(config):
    model = EfficientNet.from_pretrained('efficientnet-b4')
    
    # Replace final layer with custom classifier for 5 classes
    num_ftrs = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, config.num_classes)
    )
    
    return model

# Train model function with memory optimizations
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, fold, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Setup gradient scaler for mixed precision training
    scaler = GradScaler() if config.mixed_precision else None
    
    best_kappa = -1
    best_epoch = -1
    train_losses = []
    val_losses = []
    
    # For early stopping
    patience_counter = 0
    
    # Clean up memory before training
    torch.cuda.empty_cache()
    gc.collect()
    
    # Track model checkpoints for each fold
    checkpoint_path = os.path.join(output_dir, f'best_model_fold{fold}.pth')
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the beginning
        
        batch_iterator = tqdm(train_loader, desc=f"Fold {fold+1}/{config.num_folds} Epoch {epoch+1}/{config.epochs} [Train]")
        
        # Track steps for gradient accumulation
        steps = 0
        
        for i, (inputs, labels) in enumerate(batch_iterator):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast(enabled=config.mixed_precision):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / config.grad_accum_steps  # Scale loss
            
            # Mixed precision backward pass
            if config.mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            train_loss += loss.item() * config.grad_accum_steps
            steps += 1
            
            # Update weights after accumulating gradients
            if steps % config.grad_accum_steps == 0 or i == len(train_loader) - 1:
                if config.mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Update progress bar
            batch_iterator.set_postfix({"train_loss": f"{train_loss/(i+1):.4f}"})
            
            # Clean up to save memory
            del inputs, labels, outputs
            torch.cuda.empty_cache()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        batch_iterator = tqdm(val_loader, desc=f"Fold {fold+1}/{config.num_folds} Epoch {epoch+1}/{config.epochs} [Val]")
        
        with torch.no_grad():
            for inputs, labels in batch_iterator:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Mixed precision inference
                with autocast(enabled=config.mixed_precision):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                
                # Store predictions and labels for metrics
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())
                
                # Clean up to save memory
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate metrics
        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Update learning rate
        if scheduler:
            scheduler.step(val_loss)
        
        # Log progress
        logging.info(f"Fold {fold+1}/{config.num_folds} Epoch {epoch+1}/{config.epochs}: "
                     f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                     f"Kappa: {kappa:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save best model
        if kappa > best_kappa:
            best_kappa = kappa
            best_epoch = epoch
            patience_counter = 0
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'kappa': kappa,
                'accuracy': accuracy,
                'epoch': epoch
            }, checkpoint_path)
            
            logging.info(f"Saved best model for fold {fold+1} with kappa: {kappa:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping check (only after minimum epochs)
        if epoch >= config.min_epochs - 1 and patience_counter >= config.patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
        
        # Force garbage collection between epochs
        gc.collect()
    
    # Plot training curve
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title(f'Fold {fold+1} Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'training_curve_fold{fold}.png'))
    plt.close()
    
    return best_kappa, best_epoch

# Prediction function with memory optimizations
def predict(model, test_loader, device):
    model.eval()
    all_ids = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, img_ids in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device, non_blocking=True)
            
            # Use mixed precision for inference
            with autocast(enabled=True):
                outputs = model(inputs)
            
            preds = torch.argmax(outputs, dim=1)
            
            all_ids.extend(img_ids)
            all_preds.extend(preds.detach().cpu().tolist())
            
            # Clean up to save memory
            del inputs, outputs, preds
            torch.cuda.empty_cache()
    
    return all_ids, all_preds

def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) != 6:
        logging.error("Usage: python script.py <train_csv> <test_csv> <train_img_dir> <test_img_dir> <output_dir>")
        sys.exit(1)

    # Parse arguments
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    train_img_dir = sys.argv[3]
    test_img_dir = sys.argv[4]
    output_dir = sys.argv[5]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize config
    config = Config()
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Load data
    try:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        logging.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples.")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        sys.exit(1)
    
    # Define transformations and augmentations
    preprocess = apply_clahe
    
    # Training augmentations
    train_transforms = A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # Validation/Test transformations (no augmentation)
    val_transforms = A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # Start cross-validation training
    kf = KFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    cv_results = []
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # For each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        logging.info(f"Training fold {fold+1}/{config.num_folds}")
        
        # Split data
        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        val_data = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Create datasets
        train_dataset = DRDataset(train_data, train_img_dir, transform=train_transforms, preprocess=preprocess)
        val_dataset = DRDataset(val_data, train_img_dir, transform=val_transforms, preprocess=preprocess)
        
        # Calculate class weights to handle imbalance
        if 'diagnosis' in train_df.columns:
            class_counts = train_data['diagnosis'].value_counts().sort_index().values
            class_weights = 1.0 / class_counts
            class_weights = torch.tensor(class_weights / class_weights.sum() * len(class_counts), dtype=torch.float32)
            class_weights = class_weights.to(device)
            logging.info(f"Class weights: {class_weights}")
        else:
            class_weights = None
        
        # Create data loaders with prefetch and memory optimizations
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=config.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Create model
        model = create_model(config)
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        # Train model
        try:
            fold_kappa, best_epoch = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                scheduler, config, fold, output_dir
            )
            
            cv_results.append({
                'fold': fold,
                'kappa': fold_kappa,
                'best_epoch': best_epoch
            })
            
            # Clean up memory after each fold
            del model, train_dataset, val_dataset, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logging.error(f"Error during training fold {fold+1}: {str(e)}")
            logging.error(traceback.format_exc())
            # Continue to next fold
            continue
    
    # Save cross-validation results
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)
    
    avg_kappa = cv_df['kappa'].mean()
    logging.info(f"Average validation kappa: {avg_kappa:.4f}")
    
    # Generate ensemble prediction for test data
    logging.info("Generating test predictions...")
    
    # Create test dataset
    test_dataset = DRDataset(test_df, test_img_dir, transform=val_transforms, preprocess=preprocess)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers, 
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Ensemble prediction from all folds
    all_fold_preds = []
    
    for fold in range(config.num_folds):
        checkpoint_path = os.path.join(output_dir, f'best_model_fold{fold}.pth')
        
        if os.path.exists(checkpoint_path):
            # Load model
            model = create_model(config)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Make predictions
            ids, preds = predict(model, test_loader, device)
            all_fold_preds.append(preds)
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            gc.collect()
    
    # Average predictions from all folds
    final_preds = []
    if all_fold_preds:
        # Convert to numpy arrays for easier handling
        all_fold_preds_np = np.array(all_fold_preds)
        
        # Take mode (most common prediction) across folds
        for i in range(len(test_df)):
            # Get predictions from all folds for this sample
            fold_preds = all_fold_preds_np[:, i]
            
            # Take the most common prediction (mode)
            unique_vals, counts = np.unique(fold_preds, return_counts=True)
            most_common_pred = unique_vals[np.argmax(counts)]
            final_preds.append(most_common_pred)
    else:
        logging.error("No fold predictions available. Using zeros as fallback.")
        final_preds = [0] * len(test_df)
    
    # Create submission file
    submission = pd.DataFrame({
        'id_code': test_df['id_code'],
        'diagnosis': final_preds
    })
    submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
    
    # Save metrics
    metrics = {
        'avg_kappa': float(avg_kappa),
        'fold_results': cv_results,
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Also save as text file for easier reading
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Average Validation Kappa: {avg_kappa:.4f}\n\n")
        f.write("Fold Results:\n")
        for result in cv_results:
            f.write(f"Fold {result['fold']+1} - Kappa: {result['kappa']:.4f}, Best Epoch: {result['best_epoch']+1}\n")
    
    logging.info("Done!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        logging.error("Traceback:", exc_info=True)
        sys.exit(1)
