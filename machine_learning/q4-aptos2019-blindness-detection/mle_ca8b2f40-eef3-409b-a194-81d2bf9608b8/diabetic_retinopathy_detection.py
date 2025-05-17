#!/usr/bin/env python
# Diabetic Retinopathy Detection Workflow
# Control Group (Partition 1)

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("/workspace/mle_ca8b2f40-eef3-409b-a194-81d2bf9608b8", "experiment.log"))
    ]
)
logger = logging.getLogger(__name__)

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration
DATASET_DIR = "/workspace/mle_dataset"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train_images")
RESULTS_DIR = "/workspace/mle_ca8b2f40-eef3-409b-a194-81d2bf9608b8"
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "efficientnet_b3_model.pth")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
BATCH_SIZE = 8  # Reduced from 16 to 8 to address memory issues
NUM_EPOCHS = 3  # Reduced from 10 to 3 for faster experimentation
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
NUM_CLASSES = 5  # 0 to 4 severity levels
PATIENCE = 2  # Number of epochs to wait for improvement before early stopping

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Custom Dataset for APTOS 2019 with robust error handling
class APTOSDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.valid_indices = self._validate_images()
        logger.info(f"Found {len(self.valid_indices)} valid images out of {len(dataframe)}")
        
    def _validate_images(self):
        valid_indices = []
        for idx in range(len(self.dataframe)):
            img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0] + '.png')
            if os.path.exists(img_name):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        img = Image.open(img_name)
                        img.verify()  # Verify the image
                        valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Invalid image at index {idx}: {img_name}, Error: {str(e)}")
        return valid_indices
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_id = self.dataframe.iloc[real_idx, 0]
        img_name = os.path.join(self.img_dir, img_id + '.png')
        label = self.dataframe.iloc[real_idx, 1]
        
        try:
            image = Image.open(img_name).convert('RGB')  # Ensure RGB format
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_name}: {str(e)}")
            # Return a placeholder image in case of error
            placeholder = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
            return placeholder, label

# Standard preprocessing for control group
def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

# Load and prepare data
def load_data():
    logger.info("Loading data...")
    df = pd.read_csv(TRAIN_CSV)
    
    # Display class distribution
    logger.info("Class distribution:")
    class_dist = df['diagnosis'].value_counts().sort_index()
    logger.info(f"\n{class_dist}")
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diagnosis'])
    
    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Create datasets
    transform = get_transforms()
    train_dataset = APTOSDataset(train_df, TRAIN_IMAGES_DIR, transform=transform)
    val_dataset = APTOSDataset(val_df, TRAIN_IMAGES_DIR, transform=transform)
    
    # Create data loaders with num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, val_df

# Build EfficientNet-B3 model
def build_model():
    logger.info("Building EfficientNet-B3 model...")
    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=NUM_CLASSES)
    model = model.to(device)
    return model

# Training function with early stopping and checkpointing
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    logger.info("Starting training...")
    best_val_kappa = -1.0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Calculate quadratic weighted kappa
        val_kappa = quadratic_weighted_kappa(all_labels, all_preds)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Val Kappa: {val_kappa:.4f}, Time: {epoch_time:.1f}s")
        
        # Save checkpoint for each epoch
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'val_kappa': val_kappa,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"Best model saved with Kappa: {val_kappa:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation Kappa. Patience: {patience_counter}/{PATIENCE}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f} seconds")
    logger.info(f"Best validation Kappa: {best_val_kappa:.4f}")
    return model

# Quadratic weighted kappa calculation
def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate the quadratic weighted kappa
    """
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# Evaluate model and generate metrics
def evaluate_model(model, val_loader, val_df):
    logger.info("Evaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    kappa = quadratic_weighted_kappa(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print metrics
    logger.info(f"Quadratic Weighted Kappa: {kappa:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    # Per-class metrics
    logger.info("\nPer-class metrics:")
    for i in range(NUM_CLASSES):
        logger.info(f"Class {i} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(NUM_CLASSES), 
                yticklabels=range(NUM_CLASSES))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    
    # Return metrics as a dictionary
    metrics = {
        'kappa': kappa,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return metrics

# Main workflow
def main():
    logger.info("Starting diabetic retinopathy detection workflow...")
    
    try:
        # Load data
        train_loader, val_loader, val_df = load_data()
        
        # Build model
        model = build_model()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train model
        model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        
        # Evaluate model
        metrics = evaluate_model(model, val_loader, val_df)
        
        logger.info("Workflow completed successfully!")
        return metrics
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()