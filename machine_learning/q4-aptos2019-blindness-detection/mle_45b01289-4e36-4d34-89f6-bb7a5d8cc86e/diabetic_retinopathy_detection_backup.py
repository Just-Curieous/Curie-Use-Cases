#!/usr/bin/env python
# Diabetic Retinopathy Detection using EfficientNetB4
# Control Group: Best Single Model Approach

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
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import traceback
from tqdm import tqdm
import signal
import json
import gc
from functools import partial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Timeout handler for operations that might hang
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def with_timeout(func, timeout_duration=30, default_return=None):
    """Run a function with a timeout"""
    def wrapper(*args, **kwargs):
        # Set the timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_duration)
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Disable the alarm
            return result
        except TimeoutError:
            logging.warning(f"Function {func.__name__} timed out after {timeout_duration} seconds")
            return default_return
        except Exception as e:
            signal.alarm(0)  # Disable the alarm
            logging.error(f"Error in {func.__name__}: {str(e)}")
            return default_return
    return wrapper

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
class Config:
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 380  # EfficientNet recommended size
    num_classes = 5  # 0-4 DR levels
    batch_size = 8  # Reduced batch size to prevent memory issues
    num_epochs = 2  # Reduced for testing
    learning_rate = 0.0001
    num_folds = 2  # Reduced for testing
    model_name = 'efficientnet-b4'
    patience = 3  # Early stopping patience
    early_stopping = True  # Enable early stopping
    mixed_precision = True  # Enable mixed precision training
    checkpoint_dir = None  # Will be set to output_dir
    resume_training = True  # Resume training from checkpoint if available
    image_load_timeout = 10  # Timeout for image loading in seconds
    max_retries = 3  # Maximum number of retries for operations
    memory_cleanup_freq = 50  # Clean up memory every N batches
    
# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Convert tuple to list to allow item assignment
        lab_planes = list(lab_planes)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image
    except Exception as e:
        logging.warning(f"CLAHE application failed: {str(e)}. Using original image.")
        return image

# Safe image loading with timeout
def safe_load_image(img_path, timeout=10):
    def _load_image(path):
        image = cv2.imread(path)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return None
    
    # Apply timeout to image loading
    load_with_timeout = with_timeout(_load_image, timeout_duration=timeout)
    image = load_with_timeout(img_path)
    
    if image is None:
        logging.warning(f"Failed to load image with timeout: {img_path}")
        # Create a blank image as fallback
        image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    return image

# Dataset class for retinal images
class RetinalDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False, config=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.config = config
        self.image_cache = {}  # Cache for frequently accessed images
        self.max_cache_size = 100  # Maximum number of images to cache
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        for attempt in range(self.config.max_retries if self.config else 3):
            try:
                img_id = self.df.iloc[idx]['id_code']
                img_path = os.path.join(self.img_dir, img_id + '.png')
                
                # Try to get from cache first
                if img_id in self.image_cache:
                    image = self.image_cache[img_id]
                else:
                    # Check if image exists
                    if not os.path.exists(img_path):
                        logging.warning(f"Image not found: {img_path}")
                        # Create a blank image as fallback
                        image = np.zeros((512, 512, 3), dtype=np.uint8)
                    else:
                        # Load image with timeout
                        timeout = self.config.image_load_timeout if self.config else 10
                        image = safe_load_image(img_path, timeout=timeout)
                        
                        # Apply CLAHE for contrast enhancement
                        image = apply_clahe(image)
                        
                        # Cache the image if cache isn't full
                        if len(self.image_cache) < self.max_cache_size:
                            self.image_cache[img_id] = image
                
                # Apply transformations
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                
                if self.is_test:
                    return image
                else:
                    label = self.df.iloc[idx]['diagnosis']
                    return image, label
                    
            except Exception as e:
                if attempt < (self.config.max_retries if self.config else 3) - 1:
                    logging.warning(f"Error processing image at index {idx}, attempt {attempt+1}: {str(e)}. Retrying...")
                    time.sleep(0.5)  # Short delay before retry
                else:
                    logging.error(f"Failed to process image at index {idx} after {attempt+1} attempts: {str(e)}")
                    # Return a fallback image and label
                    image = torch.zeros((3, 380, 380))
                    if self.is_test:
                        return image
                    else:
                        label = 0 if self.is_test else self.df.iloc[idx].get('diagnosis', 0)
                        return image, label

# Create train and validation transforms
def get_transforms(config):
    train_transform = A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Rotate(limit=20, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    valid_transform = A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, valid_transform

# Create model
def create_model(config):
    model = EfficientNet.from_pretrained(config.model_name, num_classes=config.num_classes)
    model = model.to(config.device)
    return model

# Memory cleanup function
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Train function with memory management
def train_epoch(model, dataloader, criterion, optimizer, config):
    model.train()
    train_loss = 0.0
    batch_count = 0
    
    # Set up progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    # Set up mixed precision training if enabled
    scaler = torch.amp.GradScaler() if config.mixed_precision else None
    
    for images, labels in pbar:
        try:
            images = images.to(config.device)
            labels = labels.to(config.device)
            
            optimizer.zero_grad()
            
            # Use mixed precision if enabled
            if config.mixed_precision:
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({"batch_loss": loss.item()})
            
            # Periodic memory cleanup
            batch_count += 1
            if batch_count % config.memory_cleanup_freq == 0:
                cleanup_memory()
                
        except Exception as e:
            logging.error(f"Error in training batch: {str(e)}")
            logging.error(traceback.format_exc())
            cleanup_memory()  # Clean up memory after error
            continue
    
    train_loss = train_loss / len(dataloader.dataset)
    return train_loss

# Validation function
def validate_epoch(model, dataloader, criterion, config):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Set up progress bar
    pbar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for images, labels in pbar:
            try:
                images = images.to(config.device)
                labels = labels.to(config.device)
                
                # Use mixed precision if enabled
                if config.mixed_precision:
                    with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({"batch_loss": loss.item()})
                
            except Exception as e:
                logging.error(f"Error in validation batch: {str(e)}")
                logging.error(traceback.format_exc())
                continue
    
    val_loss = val_loss / len(dataloader.dataset)
    val_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    val_acc = accuracy_score(all_labels, all_preds)
    
    return val_loss, val_kappa, val_acc, all_preds, all_labels

# Predict function
def predict(model, dataloader, config):
    model.eval()
    predictions = []
    
    # Set up progress bar
    pbar = tqdm(dataloader, desc="Predicting")
    
    with torch.no_grad():
        for images in pbar:
            try:
                images = images.to(config.device)
                
                # Use mixed precision if enabled
                if config.mixed_precision:
                    with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
            except Exception as e:
                logging.error(f"Error in prediction batch: {str(e)}")
                logging.error(traceback.format_exc())
                # Add fallback predictions
                predictions.extend([0] * images.size(0))
    
    return predictions

# Main training function with k-fold cross-validation
def train_and_validate(train_df, test_df, train_img_dir, test_img_dir, config, output_dir):
    set_seed(config.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set checkpoint directory
    config.checkpoint_dir = output_dir
    
    # Initialize KFold
    kfold = KFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    
    # Initialize lists to store metrics
    fold_kappas = []
    fold_accuracies = []
    test_predictions = np.zeros((len(test_df), config.num_folds))
    
    # Get transforms
    train_transform, valid_transform = get_transforms(config)
    
    # Check for checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoint.json')
    start_fold = 0
    if os.path.exists(checkpoint_path) and config.resume_training:
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                start_fold = checkpoint.get('next_fold', 0)
                fold_kappas = checkpoint.get('fold_kappas', [])
                fold_accuracies = checkpoint.get('fold_accuracies', [])
                if 'test_predictions' in checkpoint:
                    test_predictions = np.array(checkpoint['test_predictions'])
                logging.info(f"Resuming from fold {start_fold} with {len(fold_kappas)} completed folds")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            start_fold = 0
    
    # Start k-fold cross-validation
    folds = list(kfold.split(train_df))
    for fold, (train_idx, val_idx) in enumerate(folds[start_fold:], start=start_fold):
        logging.info(f"Training fold {fold + 1}/{config.num_folds}")
        
        try:
            # Split data
            train_data = train_df.iloc[train_idx].reset_index(drop=True)
            val_data = train_df.iloc[val_idx].reset_index(drop=True)
            
            # Create datasets
            train_dataset = RetinalDataset(train_data, train_img_dir, transform=train_transform, config=config)
            val_dataset = RetinalDataset(val_data, train_img_dir, transform=valid_transform, config=config)
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
            
            # Create model, criterion, and optimizer
            model = create_model(config)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            
            # Training loop
            best_kappa = -1.0
            best_model_path = os.path.join(output_dir, f"best_model_fold{fold}.pth")
            patience_counter = 0
            
            for epoch in range(config.num_epochs):
                # Train
                train_loss = train_epoch(model, train_loader, criterion, optimizer, config)
                
                # Validate
                val_loss, val_kappa, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, config)
                
                logging.info(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Kappa: {val_kappa:.4f}, Val Acc: {val_acc:.4f}")
                
                # Save best model
                if val_kappa > best_kappa:
                    best_kappa = val_kappa
                    # Save model with error handling
                    try:
                        torch.save(model.state_dict(), best_model_path)
                        logging.info(f"Saved best model with kappa: {best_kappa:.4f}")
                    except Exception as e:
                        logging.error(f"Error saving model: {str(e)}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logging.info(f"No improvement for {patience_counter} epochs")
                
                # Early stopping
                if config.early_stopping and patience_counter >= config.patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                
                # Clean up memory after each epoch
                cleanup_memory()
            
            # Load best model with error handling
            try:
                if os.path.exists(best_model_path):
                    model.load_state_dict(torch.load(best_model_path))
                    logging.info(f"Loaded best model from {best_model_path}")
                else:
                    logging.warning(f"Best model file not found at {best_model_path}, using current model")
            except Exception as e:
                logging.error(f"Error loading best model: {str(e)}")
                logging.warning("Continuing with current model state")
            
            # Validate best model
            val_loss, val_kappa, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, config)
            fold_kappas.append(val_kappa)
            fold_accuracies.append(val_acc)
            
            logging.info(f"Fold {fold+1} - Best Validation Kappa: {val_kappa:.4f}, Accuracy: {val_acc:.4f}")
            
            # Create confusion matrix
            cm = confusion_matrix(val_labels, val_preds)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - Fold {fold+1}')
            plt.colorbar()
            classes = ['0', '1', '2', '3', '4']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_fold{fold}.png'))
            plt.close()
            
            # Predict on test data
            test_dataset = RetinalDataset(test_df, test_img_dir, transform=valid_transform, is_test=True, config=config)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
            test_preds = predict(model, test_loader, config)
            test_predictions[:, fold] = test_preds
            
            # Save checkpoint after each fold
            checkpoint = {
                'next_fold': fold + 1,
                'fold_kappas': fold_kappas,
                'fold_accuracies': fold_accuracies,
                'test_predictions': test_predictions.tolist()
            }
            
            try:
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint, f)
                logging.info(f"Checkpoint saved after fold {fold+1}")
            except Exception as e:
                logging.error(f"Error saving checkpoint: {str(e)}")
            
            # Free up memory
            del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
            cleanup_memory()
            
        except Exception as e:
            logging.error(f"Error in fold {fold+1}: {str(e)}")
            logging.error(traceback.format_exc())
            
            # Save checkpoint even if fold fails
            checkpoint = {
                'next_fold': fold + 1,
                'fold_kappas': fold_kappas,
                'fold_accuracies': fold_accuracies,
                'test_predictions': test_predictions.tolist() if len(test_predictions) > 0 else []
            }
            
            try:
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint, f)
                logging.info(f"Checkpoint saved after failed fold {fold+1}")
            except Exception as e:
                logging.error(f"Error saving checkpoint: {str(e)}")
            
            # Continue with next fold
            cleanup_memory()
            continue
    
    # Calculate average metrics
    avg_kappa = np.mean(fold_kappas) if fold_kappas else 0.0
    avg_accuracy = np.mean(fold_accuracies) if fold_accuracies else 0.0
    
    logging.info(f"Average Validation Kappa: {avg_kappa:.4f}")
    logging.info(f"Average Validation Accuracy: {avg_accuracy:.4f}")
    
    # Save cross-validation results
    cv_results = pd.DataFrame({
        'Fold': range(1, len(fold_kappas) + 1),
        'Kappa': fold_kappas,
        'Accuracy': fold_accuracies
    })
    cv_results.to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)
    
    # Create final test predictions (mean of all folds)
    final_predictions = np.zeros(len(test_df))
    for i in range(len(test_df)):
        # Use only the folds that were completed
        completed_folds = min(len(fold_kappas), config.num_folds)
        if completed_folds > 0:
            final_predictions[i] = np.round(np.mean(test_predictions[i, :completed_folds]))
        else:
            final_predictions[i] = 0  # Default prediction if no folds completed
    
    # Create submission file
    submission = pd.DataFrame({
        'id_code': test_df['id_code'],
        'diagnosis': final_predictions.astype(int)
    })
    submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
    
    # Save model metrics
    metrics = {
        'avg_kappa': avg_kappa,
        'avg_accuracy': avg_accuracy,
        'fold_kappas': fold_kappas,
        'fold_accuracies': fold_accuracies
    }
    
    # Clean up checkpoint file if all folds completed successfully
    if len(fold_kappas) == config.num_folds and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logging.info("Removed checkpoint file as all folds completed successfully")
        except Exception as e:
            logging.error(f"Error removing checkpoint file: {str(e)}")
    
    return metrics, submission

def main(train_csv, test_csv, train_img_dir, test_img_dir, output_dir):
    try:
        # Load data
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        
        # Print dataset info
        logging.info(f"Train set: {train_df.shape[0]} images")
        logging.info(f"Test set: {test_df.shape[0]} images")
        logging.info(f"Class distribution in train set:")
        logging.info(train_df['diagnosis'].value_counts().sort_index())
        
        # Initialize config
        config = Config()
        logging.info(f"Using device: {config.device}")
        
        # Train and validate
        metrics, submission = train_and_validate(train_df, test_df, train_img_dir, test_img_dir, config, output_dir)
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Average Validation Kappa: {metrics['avg_kappa']:.4f}\n")
            f.write(f"Average Validation Accuracy: {metrics['avg_accuracy']:.4f}\n")
            f.write("\nFold Results:\n")
            for i, (kappa, acc) in enumerate(zip(metrics['fold_kappas'], metrics['fold_accuracies'])):
                f.write(f"Fold {i+1} - Kappa: {kappa:.4f}, Accuracy: {acc:.4f}\n")
        
        logging.info(f"Results saved to {output_dir}")
        return metrics, submission
    
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user. Saving intermediate results...")
        # Create minimal output files to ensure the workflow completes
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metrics file with warning
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Process interrupted by user at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Partial results may be available in the checkpoint file.\n")
        
        # Create empty CV results if not exists
        if not os.path.exists(os.path.join(output_dir, 'cv_results.csv')):
            pd.DataFrame({
                'Fold': range(1, 6),
                'Kappa': [0.0] * 5,
                'Accuracy': [0.0] * 5
            }).to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)
        
        # Create empty submission file if not exists
        if not os.path.exists(os.path.join(output_dir, 'submission.csv')) and os.path.exists(test_csv):
            test_df = pd.read_csv(test_csv)
            submission = pd.DataFrame({
                'id_code': test_df['id_code'],
                'diagnosis': [0] * len(test_df)
            })
            submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
        
        sys.exit(1)
    
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        logging.error(traceback.format_exc())
        
        # Create minimal output files to ensure the workflow completes
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metrics file with error
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Error occurred: {str(e)}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Check logs for more details.\n")
        
        # Create empty CV results
        pd.DataFrame({
            'Fold': range(1, 6),
            'Kappa': [0.0] * 5,
            'Accuracy': [0.0] * 5
        }).to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)
        
        # Create empty submission file
        if os.path.exists(test_csv):
            test_df = pd.read_csv(test_csv)
            submission = pd.DataFrame({
                'id_code': test_df['id_code'],
                'diagnosis': [0] * len(test_df)
            })
            submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
        
        raise

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python diabetic_retinopathy_detection.py <train_csv> <test_csv> <train_img_dir> <test_img_dir> <output_dir>")
        sys.exit(1)
    
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    train_img_dir = sys.argv[3]
    test_img_dir = sys.argv[4]
    output_dir = sys.argv[5]
    
    main(train_csv, test_csv, train_img_dir, test_img_dir, output_dir)