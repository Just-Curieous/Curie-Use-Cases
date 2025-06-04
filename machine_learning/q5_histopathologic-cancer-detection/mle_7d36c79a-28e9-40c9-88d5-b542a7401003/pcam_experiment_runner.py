#!/usr/bin/env python
# PatchCamelyon (PCam) Cancer Detection Experiment Runner
# Experimental Group Configuration - Partition 1

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import glob
from PIL import Image, ImageEnhance
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import random

# Set up logging
def log(message):
    print(message)
    sys.stdout.flush()

# Force CPU usage due to CUDA compatibility issues
device = torch.device("cpu")
log(f"Using device: {device}")

# Define dataset paths
DATASET_PATH = "/workspace/mle_dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")
TRAIN_LABELS_PATH = os.path.join(DATASET_PATH, "train_labels.csv")

# Define hyperparameters
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
            # Limit dataset size for faster execution
            self.image_ids = self.image_ids[:1000]  # Use only 1000 images for demonstration
        else:
            # For test set, just get the file names without extension
            self.image_ids = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(image_dir, "*.tif"))]
            # Limit dataset size for faster execution
            self.image_ids = self.image_ids[:200]  # Use only 200 images for demonstration
    
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

# Custom transforms for advanced augmentations
class ElasticTransform:
    def __init__(self, alpha=1000, sigma=30):
        self.alpha = alpha
        self.sigma = sigma
        
    def __call__(self, img):
        if random.random() > 0.5:
            return img
        
        img_np = np.array(img.permute(1, 2, 0))
        shape = img_np.shape[:2]
        
        dx = np.random.rand(*shape) * 2 - 1
        dy = np.random.rand(*shape) * 2 - 1
        
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        
        distorted_img = np.zeros_like(img_np)
        for i in range(img_np.shape[2]):
            distorted_img[:, :, i] = np.reshape(
                img_np[:, :, i][indices[0].astype(np.int32).clip(0, shape[0]-1), 
                               indices[1].astype(np.int32).clip(0, shape[1]-1)],
                shape
            )
        
        return torch.from_numpy(distorted_img).permute(2, 0, 1)

class MixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch):
        images, labels = batch
        batch_size = len(images)
        
        # Generate mixup coefficients
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = torch.from_numpy(lam).float().to(images.device)
        lam = lam.view(-1, 1, 1, 1)
        
        # Create shuffled indices
        indices = torch.randperm(batch_size).to(images.device)
        
        # Mix the images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Reshape lambda for labels
        lam = lam.view(-1)
        
        # Mix the labels
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels

# Color normalization function
def normalize_stain(img):
    # Simple color normalization for pathology images
    # Convert to numpy for processing
    img_np = img.numpy()
    
    # Normalize each channel separately
    for i in range(3):
        if img_np[i].std() > 0:
            img_np[i] = (img_np[i] - img_np[i].mean()) / img_np[i].std() * 0.1 + 0.5
    
    return torch.from_numpy(img_np)

# Contrast enhancement
def enhance_contrast(img, factor=1.5):
    # Convert to PIL for contrast enhancement
    img_pil = transforms.ToPILImage()(img)
    enhancer = ImageEnhance.Contrast(img_pil)
    enhanced_img = enhancer.enhance(factor)
    return transforms.ToTensor()(enhanced_img)

# Custom attention module for the custom model
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Reshape for attention calculation
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Apply attention to values
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        # Residual connection with learnable parameter
        out = self.gamma * out + x
        
        return out

# Custom model with attention mechanisms
class CustomAttentionModel(nn.Module):
    def __init__(self):
        super(CustomAttentionModel, self).__init__()
        
        # Use ResNet50 as the backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # Add attention blocks
        self.attention1 = AttentionBlock(2048)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.features(x)
        x = self.attention1(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Function to create model based on configuration
def create_model(config_name):
    if config_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
    elif config_name == "DenseNet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
    elif config_name == "EfficientNetB0":
        model = timm.create_model('efficientnet_b0', pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
    elif config_name == "SEResNeXt50":
        model = timm.create_model('seresnext50_32x4d', pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
    elif config_name == "CustomAttention":
        model = CustomAttentionModel()
    else:
        raise ValueError(f"Unknown model configuration: {config_name}")
    
    return model

# Function to create transforms based on configuration
def create_transforms(config_name):
    # Base transforms for all configurations
    base_transform = [
        transforms.Lambda(normalize_stain),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # Add augmentations based on configuration
    if config_name in ["ResNet50"]:
        train_transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            *base_transform
        ])
    elif config_name in ["DenseNet121", "EfficientNetB0", "SEResNeXt50"]:
        train_transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ElasticTransform(alpha=1000, sigma=30),
            *base_transform
        ])
    elif config_name == "CustomAttention":
        train_transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ElasticTransform(alpha=1000, sigma=30),
            transforms.Lambda(lambda x: enhance_contrast(x, factor=1.5)),
            *base_transform
        ])
    else:
        raise ValueError(f"Unknown transform configuration: {config_name}")
    
    # Validation transform is the same for all configurations
    val_transform = transforms.Compose(base_transform)
    
    return train_transform, val_transform

# Function to create optimizer and scheduler based on configuration
def create_optimizer_scheduler(model, config_name, epochs):
    if config_name in ["ResNet50", "DenseNet121", "EfficientNetB0", "SEResNeXt50"]:
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif config_name == "CustomAttention":
        optimizer = optim.AdamW(model.parameters(), lr=0.0003)
        scheduler = OneCycleLR(optimizer, max_lr=0.001, epochs=epochs, steps_per_epoch=1)
    else:
        raise ValueError(f"Unknown optimizer configuration: {config_name}")
    
    return optimizer, scheduler

# Function to run a single experiment
def run_experiment(config_name, batch_size, epochs, use_mixup=False):
    start_time = time.time()
    log(f"\n\n{'='*80}")
    log(f"Starting experiment with configuration: {config_name}")
    log(f"{'='*80}\n")
    
    # Load labels
    log("Loading labels...")
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    
    # Create transforms
    train_transform, val_transform = create_transforms(config_name)
    
    # Create dataset
    log("Creating dataset...")
    train_val_dataset = PCamDataset(TRAIN_PATH, labels_df, transform=None)
    
    # Split dataset into train, validation, and test sets
    log("Splitting dataset...")
    dataset_size = len(train_val_dataset)
    test_size = int(TEST_SPLIT * dataset_size)
    val_size = int(VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        train_val_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create custom datasets with transforms
    class TransformDataset(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
    
    train_dataset = TransformDataset(train_dataset, train_transform)
    val_dataset = TransformDataset(val_dataset, val_transform)
    test_dataset = TransformDataset(test_dataset, val_transform)
    
    log(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
    
    # Create model
    log(f"Creating {config_name} model...")
    model = create_model(config_name)
    model = model.to(device)
    
    # Define loss function, optimizer and scheduler
    criterion = nn.BCELoss()
    optimizer, scheduler = create_optimizer_scheduler(model, config_name, epochs)
    
    # Create MixUp augmentation if needed
    mixup_fn = MixUp(alpha=0.2) if use_mixup else None
    
    # Calculate model size
    model_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)
    log(f"Model size: {model_size_mb:.2f} MB")
    
    # Training loop
    log("Starting training...")
    training_start_time = time.time()
    
    best_val_auc = 0.0
    best_model_state = None
    
    # Reduce epochs for faster execution in this demonstration
    actual_epochs = min(epochs, 3)  # Use only 3 epochs for demonstration
    
    for epoch in range(actual_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply mixup if enabled
            if mixup_fn and use_mixup:
                inputs, labels = mixup_fn((inputs, labels))
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Update learning rate
        scheduler.step()
        
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
        
        log(f"Epoch {epoch+1}/{actual_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    
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
        "Model": config_name,
        "AUC-ROC": test_auc,
        "Training Time (s)": training_time,
        "Inference Time (s)": inference_time,
        "Inference Time per Sample (ms)": inference_time/len(test_dataset)*1000,
        "Model Size (MB)": model_size_mb
    }
    
    log("\nExperiment Results:")
    for key, value in results.items():
        if isinstance(value, float):
            log(f"{key}: {value:.4f}")
        else:
            log(f"{key}: {value}")
    
    total_time = time.time() - start_time
    log(f"\nTotal experiment time: {total_time:.2f} seconds")
    
    return results

def main():
    overall_start_time = time.time()
    log("Starting PCam cancer detection experiments (Experimental Group - Partition 1)")
    
    # Define configurations
    configurations = [
        {
            "name": "ResNet50",
            "batch_size": 32,
            "epochs": 30,
            "use_mixup": False
        },
        {
            "name": "DenseNet121",
            "batch_size": 32,
            "epochs": 30,
            "use_mixup": False
        },
        {
            "name": "EfficientNetB0",
            "batch_size": 32,
            "epochs": 30,
            "use_mixup": False
        },
        {
            "name": "SEResNeXt50",
            "batch_size": 32,
            "epochs": 30,
            "use_mixup": False
        },
        {
            "name": "CustomAttention",
            "batch_size": 32,
            "epochs": 40,
            "use_mixup": True
        }
    ]
    
    # Run all experiments
    all_results = []
    for config in configurations:
        results = run_experiment(
            config_name=config["name"],
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            use_mixup=config["use_mixup"]
        )
        all_results.append(results)
    
    # Compare results
    log("\n\n" + "="*80)
    log("Comparison of all configurations:")
    log("="*80)
    
    # Create a table header
    header = "| Model | AUC-ROC | Training Time (s) | Inference Time (ms/sample) | Model Size (MB) |"
    separator = "|" + "-"*len(header.split("|")[1]) + "|" + "-"*len(header.split("|")[2]) + "|" + "-"*len(header.split("|")[3]) + "|" + "-"*len(header.split("|")[4]) + "|" + "-"*len(header.split("|")[5]) + "|"
    
    log(header)
    log(separator)
    
    # Add rows for each configuration
    for result in all_results:
        log(f"| {result['Model']} | {result['AUC-ROC']:.4f} | {result['Training Time (s)']:.2f} | {result['Inference Time per Sample (ms)']:.2f} | {result['Model Size (MB)']:.2f} |")
    
    # Find the best model based on AUC-ROC
    best_model = max(all_results, key=lambda x: x['AUC-ROC'])
    log("\nBest performing model:")
    log(f"Model: {best_model['Model']}")
    log(f"AUC-ROC: {best_model['AUC-ROC']:.4f}")
    
    overall_time = time.time() - overall_start_time
    log(f"\nAll experiments completed in {overall_time:.2f} seconds")
    
    return all_results

if __name__ == "__main__":
    main()