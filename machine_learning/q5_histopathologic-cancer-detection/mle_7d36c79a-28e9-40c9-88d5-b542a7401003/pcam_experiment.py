#!/usr/bin/env python
# PatchCamelyon (PCam) Cancer Detection Experiment
# Control Group Configuration

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.io import read_image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import glob
from PIL import Image
import sys

# Set up logging
def log(message):
    print(message)
    sys.stdout.flush()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}")

# Define dataset paths
DATASET_PATH = "/workspace/mle_dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")
TRAIN_LABELS_PATH = os.path.join(DATASET_PATH, "train_labels.csv")

# Define hyperparameters (control group configuration)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Custom Dataset for PCam
class PCamDataset(Dataset):
    def __init__(self, image_dir, labels_df=None, transform=None, is_test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        
        if not is_test:
            self.labels_df = labels_df
            self.image_ids = self.labels_df['id'].tolist()
        else:
            # For test set, just get the file names without extension
            self.image_ids = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(image_dir, "*.tif"))]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")
        
        # Read image using PIL (more compatible with .tif files)
        image = Image.open(img_path)
        
        # Convert to tensor
        image = transforms.ToTensor()(image)
        
        if self.transform:
            image = self.transform(image)
        
        if not self.is_test:
            # Get label for training/validation
            label = self.labels_df.loc[self.labels_df['id'] == img_id, 'label'].values[0]
            label = torch.tensor(label, dtype=torch.float32)
            return image, label
        else:
            # For test set, return image and ID
            return image, img_id

def main():
    start_time = time.time()
    log("Starting PCam cancer detection experiment (Control Group)")
    
    # Load labels
    log("Loading labels...")
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    
    # Define transformations (standard normalization for control group)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    log("Creating dataset...")
    full_dataset = PCamDataset(TRAIN_PATH, labels_df, transform=transform)
    
    # Split dataset into train, validation, and test sets
    log("Splitting dataset...")
    dataset_size = len(full_dataset)
    test_size = int(TEST_SPLIT * dataset_size)
    val_size = int(VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    log(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # Create model (ResNet18)
    log("Creating ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify the final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Calculate model size
    model_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)
    log(f"Model size: {model_size_mb:.2f} MB")
    
    # Training loop
    log("Starting training...")
    training_start_time = time.time()
    
    best_val_auc = 0.0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        # Save the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        log(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    
    training_time = time.time() - training_start_time
    log(f"Training completed in {training_time:.2f} seconds")
    
    # Load the best model for evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    log("Evaluating on test set...")
    inference_start_time = time.time()
    
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze()
            
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    inference_time = time.time() - inference_start_time
    test_auc = roc_auc_score(test_labels, test_preds)
    
    log(f"Test AUC-ROC: {test_auc:.4f}")
    log(f"Inference time: {inference_time:.2f} seconds for {len(test_dataset)} samples")
    log(f"Average inference time per sample: {inference_time/len(test_dataset)*1000:.2f} ms")
    
    # Record results
    results = {
        "AUC-ROC": test_auc,
        "Training Time (s)": training_time,
        "Inference Time (s)": inference_time,
        "Inference Time per Sample (ms)": inference_time/len(test_dataset)*1000,
        "Model Size (MB)": model_size_mb
    }
    
    log("\nExperiment Results:")
    for key, value in results.items():
        log(f"{key}: {value:.4f}")
    
    total_time = time.time() - start_time
    log(f"\nTotal experiment time: {total_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    main()