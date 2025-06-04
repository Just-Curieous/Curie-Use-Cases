#!/usr/bin/env python
# PatchCamelyon (PCam) Cancer Detection Experiment
# Experimental Group Configuration

import os
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
from PIL import Image
import sys
import timm
import logging
import gc
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import random
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define dataset paths
DATASET_PATH = "/workspace/mle_dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")
TRAIN_LABELS_PATH = os.path.join(DATASET_PATH, "train_labels.csv")
RESULTS_DIR = "/workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003/model_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
            self.image_ids = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(image_dir, "*.tif"))]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")
        
        # Load image using PIL
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        if not self.is_test:
            label = self.labels_df.loc[self.labels_df['id'] == img_id, 'label'].values[0]
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            return image, img_id

# Define transforms for each model
def get_transforms(model_name, is_train=True):
    # Common normalization values for medical images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Base transforms for all models
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    # Training augmentations
    if is_train:
        if model_name == "resnet50":
            train_transforms = [
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ] + base_transforms
        
        elif model_name in ["densenet121", "efficientnet_b0", "seresnext50"]:
            train_transforms = [
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ElasticTransform(alpha=50.0, sigma=5.0),
            ] + base_transforms
        
        elif model_name == "custom_attention":
            # Enhanced augmentations for custom model
            train_transforms = [
                transforms.RandomRotation(45),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
                transforms.ElasticTransform(alpha=75.0, sigma=5.0),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.RandomAutocontrast(),
            ] + base_transforms
        
        return transforms.Compose(train_transforms)
    
    # Validation/Test transforms (no augmentation)
    return transforms.Compose(base_transforms)

# Mixup augmentation
def mixup_data(x, y, alpha=0.2):
    """Applies mixup augmentation to the batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates the loss with mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Convolutional Block Attention Module (CBAM)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# Custom ResNet50 with CBAM
class ResNet50WithCBAM(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet50WithCBAM, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Add CBAM after each residual block
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)
        
        # Replace the original fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x):
        # First conv and maxpool
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # Layer 1 with CBAM
        x = self.resnet.layer1(x)
        x = self.cbam1(x)
        
        # Layer 2 with CBAM
        x = self.resnet.layer2(x)
        x = self.cbam2(x)
        
        # Layer 3 with CBAM
        x = self.resnet.layer3(x)
        x = self.cbam3(x)
        
        # Layer 4 with CBAM
        x = self.resnet.layer4(x)
        x = self.cbam4(x)
        
        # Final layers
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        
        return x

# Function to get model
def get_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)
    
    elif model_name == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)
    
    elif model_name == "seresnext50":
        model = timm.create_model('seresnext50_32x4d', pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    
    elif model_name == "custom_attention":
        model = ResNet50WithCBAM(num_classes=1)
    
    return model

# Function to get optimizer and scheduler
def get_optimizer_scheduler(model, model_name, epochs):
    if model_name == "custom_attention":
        optimizer = optim.AdamW(model.parameters(), lr=0.0003)
        scheduler = OneCycleLR(optimizer, max_lr=0.0003, epochs=epochs, steps_per_epoch=len(train_loader))
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    return optimizer, scheduler

# Training function
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience=5):
    model.to(device)
    best_val_auc = 0.0
    best_model_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pth")
    early_stop_counter = 0
    train_time_start = time.time()
    
    # For storing metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Apply mixup for custom_attention model
            if model_name == "custom_attention":
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
                
            optimizer.zero_grad()
            
            outputs = model(inputs)
            outputs = outputs.squeeze()
            
            # Calculate loss (with mixup if applicable)
            if model_name == "custom_attention":
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            # Update learning rate for OneCycleLR
            if model_name == "custom_attention":
                scheduler.step()
                
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # Validation phase
        val_loss, val_auc = evaluate_model(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        # Update learning rate for CosineAnnealingLR
        if model_name != "custom_attention":
            scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0
            logger.info(f"Saved best model with Val AUC: {val_auc:.4f}")
        else:
            early_stop_counter += 1
            
        # Early stopping
        if early_stop_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    train_time = time.time() - train_time_start
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    return model, history, train_time, best_val_auc

# Evaluation function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            outputs = outputs.squeeze()
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # Store predictions and labels for AUC calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
    
    # Calculate metrics
    val_loss = running_loss / len(dataloader.dataset)
    val_auc = roc_auc_score(all_labels, all_preds)
    
    return val_loss, val_auc

# Function to measure inference time and model size
def measure_model_metrics(model, val_loader):
    # Measure model size
    model_size_bytes = 0
    for param in model.parameters():
        model_size_bytes += param.nelement() * param.element_size()
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    # Measure inference time
    model.eval()
    batch = next(iter(val_loader))[0].to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch)
    
    # Measure inference time
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model(batch)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    inference_time = (end_time - start_time) / 100
    
    return model_size_mb, inference_time

# Main function
def main():
    # Load and prepare data
    logger.info("Loading dataset...")
    train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    
    # Create full dataset
    full_dataset = PCamDataset(
        image_dir=TRAIN_PATH,
        labels_df=train_labels_df,
        transform=get_transforms("resnet50", is_train=False),  # Basic transform for splitting
        is_test=False
    )
    
    # Split dataset into train, validation, and test sets
    dataset_size = len(full_dataset)
    test_size = int(0.1 * dataset_size)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    logger.info(f"Dataset split - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
    
    # Model configurations
    model_configs = [
        {
            "name": "resnet50",
            "batch_size": 32,
            "epochs": 30,
        },
        {
            "name": "densenet121",
            "batch_size": 32,
            "epochs": 30,
        },
        {
            "name": "efficientnet_b0",
            "batch_size": 32,
            "epochs": 30,
        },
        {
            "name": "seresnext50",
            "batch_size": 32,
            "epochs": 30,
        },
        {
            "name": "custom_attention",
            "batch_size": 32,
            "epochs": 40,
        }
    ]
    
    # Results storage
    results = []
    
    # Train and evaluate each model
    for config in model_configs:
        model_name = config["name"]
        batch_size = config["batch_size"]
        epochs = config["epochs"]
        
        logger.info(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
        
        # Apply specific transforms for this model
        train_dataset.dataset.transform = get_transforms(model_name, is_train=True)
        val_dataset.dataset.transform = get_transforms(model_name, is_train=False)
        test_dataset.dataset.transform = get_transforms(model_name, is_train=False)
        
        # Create data loaders
        global train_loader, val_loader, test_loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Initialize model, loss function, optimizer, and scheduler
        model = get_model(model_name)
        criterion = nn.BCEWithLogitsLoss()
        optimizer, scheduler = get_optimizer_scheduler(model, model_name, epochs)
        
        try:
            # Train model
            model, history, train_time, best_val_auc = train_model(
                model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, epochs
            )
            
            # Evaluate on test set
            test_loss, test_auc = evaluate_model(model, test_loader, criterion)
            
            # Measure model metrics
            model_size_mb, inference_time = measure_model_metrics(model, val_loader)
            
            # Store results
            model_result = {
                "name": model_name,
                "train_time": train_time,
                "inference_time": inference_time,
                "model_size_mb": model_size_mb,
                "val_auc": best_val_auc,
                "test_auc": test_auc,
                "test_loss": test_loss
            }
            
            results.append(model_result)
            
            logger.info(f"\nResults for {model_name}:")
            logger.info(f"Training time: {train_time:.2f} seconds")
            logger.info(f"Inference time: {inference_time*1000:.2f} ms per batch")
            logger.info(f"Model size: {model_size_mb:.2f} MB")
            logger.info(f"Best validation AUC: {best_val_auc:.4f}")
            logger.info(f"Test AUC: {test_auc:.4f}")
            
            # Clean up GPU memory
            del model, optimizer, scheduler, history
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # Print final comparison
    logger.info("\n\n" + "="*50)
    logger.info("Final Results Comparison")
    logger.info("="*50)
    
    # Sort models by test AUC
    results.sort(key=lambda x: x["test_auc"], reverse=True)
    
    for i, result in enumerate(results):
        logger.info(f"{i+1}. {result['name']}:")
        logger.info(f"   Test AUC: {result['test_auc']:.4f}")
        logger.info(f"   Training time: {result['train_time']:.2f} seconds")
        logger.info(f"   Inference time: {result['inference_time']*1000:.2f} ms per batch")
        logger.info(f"   Model size: {result['model_size_mb']:.2f} MB")
    
    # Identify best model
    best_model = results[0]
    logger.info(f"\nBest model: {best_model['name']} with Test AUC: {best_model['test_auc']:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison_results.csv"), index=False)
    logger.info(f"Results saved to {os.path.join(RESULTS_DIR, 'model_comparison_results.csv')}")

if __name__ == "__main__":
    main()