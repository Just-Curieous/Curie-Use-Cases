#!/usr/bin/env python3
"""
PatchCamelyon Cancer Detection Experiment
This script implements a complete workflow for training and evaluating
multiple model architectures on the PatchCamelyon dataset.
"""

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchvision import transforms, models
import timm
from sklearn.metrics import roc_auc_score
from PIL import Image, ImageEnhance
import gc

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths
DATASET_DIR = Path("/workspace/mle_dataset")
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
LABELS_PATH = DATASET_DIR / "train_labels.csv"

class PCamDataset(Dataset):
    """PatchCamelyon Dataset"""
    
    def __init__(self, image_dir, labels_df=None, transform=None, is_test=False):
        """
        Args:
            image_dir: Directory with images
            labels_df: DataFrame with image IDs and labels
            transform: Optional transform to be applied on images
            is_test: Whether this is the test set (no labels)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        
        if not is_test:
            self.labels_df = labels_df
            self.image_ids = self.labels_df['id'].values
        else:
            # For test set, just get the first 1000 images to save time
            self.image_ids = [f.stem for f in list(Path(image_dir).glob('*.tif'))[:1000]]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        if not self.is_test:
            label = self.labels_df.loc[self.labels_df['id'] == img_id, 'label'].values[0]
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            return image, img_id

def create_data_loaders(batch_size=32, model_type="standard"):
    """Create train, validation, and test data loaders"""
    
    # Load labels
    labels_df = pd.read_csv(LABELS_PATH)
    
    # Use a smaller subset of the data for faster training (10% of the data)
    # This is just for demonstration purposes
    labels_df = labels_df.sample(frac=0.1, random_state=42)
    print(f"Using {len(labels_df)} samples for training and validation")
    
    # Define transformations
    if model_type == "attention":
        # Enhanced augmentation for attention model
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Standard augmentation for other models
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Validation and test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = PCamDataset(TRAIN_DIR, labels_df, transform=None)
    
    # Split dataset: 70% train, 20% validation, 10% test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y
            
        def __len__(self):
            return len(self.subset)
    
    train_dataset = TransformedSubset(train_dataset, train_transform)
    val_dataset = TransformedSubset(val_dataset, val_transform)
    test_dataset = TransformedSubset(test_dataset, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

class AttentionBlock(nn.Module):
    """Simple attention mechanism"""
    def __init__(self, in_features):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, in_features // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class CustomAttentionModel(nn.Module):
    """Custom model with attention mechanism based on ResNet50"""
    def __init__(self):
        super(CustomAttentionModel, self).__init__()
        # Load pretrained ResNet50 without classification head
        resnet = models.resnet50(weights='DEFAULT')
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add attention mechanism
        self.attention = AttentionBlock(2048)
        
        # Global average pooling and classification head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_model(model_name):
    """Create and return a model based on the model name"""
    if model_name == "resnet50":
        model = models.resnet50(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == "densenet121":
        model = models.densenet121(weights='DEFAULT')
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif model_name == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif model_name == "seresnext50":
        model = timm.create_model('seresnext50_32x4d', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == "custom_attention":
        model = CustomAttentionModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

def train_model(model, train_loader, val_loader, model_name, epochs, learning_rate):
    """Train the model and return training history"""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss instead of BCELoss for better numerical stability
    
    # Configure optimizer and scheduler based on model type
    if model_name == "custom_attention":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=epochs, 
                              steps_per_epoch=len(train_loader))
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # For early stopping
    best_val_auc = 0
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # Make sure outputs and labels have the same shape
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0 and labels.dim() == 1:
                outputs = outputs.unsqueeze(0)  # Convert scalar to [1] tensor if needed
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if model_name == "custom_attention":
                scheduler.step()
                
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                if outputs.dim() == 0 and labels.dim() == 1:
                    outputs = outputs.unsqueeze(0)  # Convert scalar to [1] tensor if needed
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # Apply sigmoid since we're using BCEWithLogitsLoss
                sigmoid_outputs = torch.sigmoid(outputs)
                all_preds.extend(sigmoid_outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(all_labels, all_preds)
        
        if model_name != "custom_attention":
            scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate_model(model, test_loader):
    """Evaluate the model on the test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            
            # Apply sigmoid since we're using BCEWithLogitsLoss
            sigmoid_outputs = torch.sigmoid(outputs)
            all_preds.extend(sigmoid_outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    inference_time = (time.time() - start_time) / len(test_loader.dataset)
    
    # Calculate metrics
    test_auc = roc_auc_score(all_labels, all_preds)
    
    return {
        'auc': test_auc,
        'inference_time': inference_time
    }

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def run_experiment():
    """Run the complete experiment workflow"""
    # Model configurations
    model_configs = [
        {"name": "resnet50", "type": "standard", "epochs": 3, "lr": 0.0005, "batch_size": 32},
        {"name": "densenet121", "type": "standard", "epochs": 3, "lr": 0.0005, "batch_size": 32},
        {"name": "efficientnet_b0", "type": "standard", "epochs": 3, "lr": 0.0005, "batch_size": 32},
        {"name": "seresnext50", "type": "standard", "epochs": 3, "lr": 0.0005, "batch_size": 32},
        {"name": "custom_attention", "type": "attention", "epochs": 5, "lr": 0.0003, "batch_size": 32}
    ]
    
    results = []
    
    for config in model_configs:
        print(f"\n{'='*50}")
        print(f"Training {config['name']} model")
        print(f"{'='*50}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=config['batch_size'],
            model_type=config['type']
        )
        
        # Create model
        model = get_model(config['name'])
        
        # Measure model size
        model_size = get_model_size(model)
        
        # Measure training time
        start_time = time.time()
        
        # Train model
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=config['name'],
            epochs=config['epochs'],
            learning_rate=config['lr']
        )
        
        training_time = time.time() - start_time
        
        # Evaluate model
        eval_metrics = evaluate_model(model, test_loader)
        
        # Collect results
        result = {
            'model_name': config['name'],
            'auc': eval_metrics['auc'],
            'training_time': training_time,
            'inference_time': eval_metrics['inference_time'],
            'model_size_mb': model_size,
            'best_val_auc': max(history['val_auc'])
        }
        
        results.append(result)
        
        print(f"\nResults for {config['name']}:")
        print(f"AUC-ROC: {result['auc']:.4f}")
        print(f"Training time: {result['training_time']:.2f} seconds")
        print(f"Inference time: {result['inference_time']*1000:.2f} ms per sample")
        print(f"Model size: {result['model_size_mb']:.2f} MB")
        
        # Clear GPU memory
        model = model.cpu()
        del model, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    # Print summary of results
    print("\n\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'AUC-ROC':<10} {'Train Time (s)':<15} {'Inference (ms)':<15} {'Size (MB)':<10}")
    print("-"*80)
    
    for result in results:
        print(f"{result['model_name']:<20} {result['auc']:.4f}     {result['training_time']:.2f}           {result['inference_time']*1000:.2f}           {result['model_size_mb']:.2f}")
    
    # Find best model
    best_model = max(results, key=lambda x: x['auc'])
    print("\nBest model by AUC-ROC:")
    print(f"{best_model['model_name']} with AUC-ROC of {best_model['auc']:.4f}")

if __name__ == "__main__":
    run_experiment()