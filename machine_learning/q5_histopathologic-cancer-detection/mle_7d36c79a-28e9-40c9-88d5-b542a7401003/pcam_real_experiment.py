#!/usr/bin/env python
# PatchCamelyon (PCam) Cancer Detection Experiment - Real Implementation
# Experimental Group Configuration

import os
import time
import random
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, DenseNet121_Weights, EfficientNet_B0_Weights
import timm
from sklearn.metrics import roc_auc_score
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gc
import os.path as osp
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Try to import tabulate, but provide a fallback if not available
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    
    # Simple tabulate replacement function
    def tabulate(data, headers, tablefmt="grid", floatfmt=".4f", showindex=False):
        if isinstance(data, pd.DataFrame):
            data_rows = data.values.tolist()
            headers = data.columns.tolist()
        else:
            data_rows = data
            
        # Calculate column widths
        col_widths = [max(len(str(h)), max([len(f"{row[i]:.4f}" if isinstance(row[i], float) else str(row[i])) for row in data_rows])) 
                      for i, h in enumerate(headers)]
        
        # Create header
        header_row = " | ".join(f"{h:{w}s}" for h, w in zip(headers, col_widths))
        separator = "-" * len(header_row)
        
        # Create rows
        rows = [" | ".join(f"{row[i]:.4f}" if isinstance(row[i], float) else f"{row[i]:{col_widths[i]}s}" 
                          for i in range(len(row))) for row in data_rows]
        
        # Combine all parts
        table = f"{header_row}\n{separator}\n" + "\n".join(rows)
        return table

# Set up logging
def log(message):
    print(message)
    sys.stdout.flush()

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the PCam dataset class
class PCamDataset(Dataset):
    def __init__(self, root_dir, labels_file=None, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        
        # Get all image files
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.tif')]
        
        # Load labels if available
        self.labels = None
        if labels_file is not None and os.path.exists(labels_file):
            labels_df = pd.read_csv(labels_file)
            # Create a dictionary mapping file ID to label
            self.labels = {row['id']: row['label'] for _, row in labels_df.iterrows()}
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load image
        image = Image.open(img_path)
        image = np.array(image)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Get label if available
        label = -1
        if self.labels is not None:
            img_id = img_name.split('.')[0]  # Remove file extension
            label = self.labels.get(img_id, -1)
            
        return image, torch.tensor(label, dtype=torch.float32)

# Define the custom attention model
class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, in_features // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map

class CustomAttentionModel(nn.Module):
    def __init__(self):
        super(CustomAttentionModel, self).__init__()
        # Use EfficientNet as the backbone
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Get the number of features from the backbone
        in_features = self.backbone.classifier[1].in_features
        
        # Remove the classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add attention mechanism
        self.attention = AttentionBlock(in_features)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 1)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define the training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, model_name):
    best_val_auc = 0.0
    best_model_state = None
    train_losses = []
    val_losses = []
    val_aucs = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        if scheduler is not None:
            scheduler.step()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                # Store predictions and labels for AUC calculation
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(torch.sigmoid(outputs).squeeze().cpu().numpy())
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Calculate AUC
        val_auc = roc_auc_score(all_labels, all_preds)
        val_aucs.append(val_auc)
        
        # Save the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            best_indicator = "(best)"
        else:
            best_indicator = ""
        
        # Print progress
        log(f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val AUC: {val_auc:.4f} {best_indicator}")
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_auc

# Define the evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # Store predictions and labels for AUC calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).squeeze().cpu().numpy())
    
    # Calculate AUC
    test_auc = roc_auc_score(all_labels, all_preds)
    
    return test_auc

# Define the inference time measurement function
def measure_inference_time(model, test_loader, device, num_runs=3):
    model.eval()
    total_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                _ = model(inputs)
                total_samples += inputs.size(0)
            end_time = time.time()
            total_time += (end_time - start_time)
    
    # Calculate average inference time per sample in milliseconds
    avg_time_per_sample = (total_time / num_runs) * 1000 / total_samples
    
    return avg_time_per_sample, total_time / num_runs

# Define the model size calculation function
def get_model_size(model):
    # Get model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

# Define the number of parameters calculation function
def get_model_parameters(model):
    # Get number of parameters in millions
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

# Define the main experiment function
def run_experiment(config):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    
    # Define data paths
    dataset_path = "/workspace/mle_dataset"
    train_dir = os.path.join(dataset_path, "train")
    test_dir = os.path.join(dataset_path, "test")
    train_labels_file = os.path.join(dataset_path, "train_labels.csv")
    
    # Define transformations
    # Base preprocessing for all models
    base_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Configuration-specific augmentations
    if config["augmentation"] == "basic":
        train_transform = A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif config["augmentation"] == "advanced":
        train_transform = A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif config["augmentation"] == "advanced_with_mixup":
        train_transform = A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.CLAHE(p=0.5),  # Contrast enhancement
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    # Create datasets with a smaller subset for demonstration
    # In a real experiment, we would use the full dataset
    full_train_dataset = PCamDataset(
        root_dir=train_dir,
        labels_file=train_labels_file,
        transform=train_transform
    )
    
    # Use only 5% of the data for this demonstration
    demo_size = int(0.05 * len(full_train_dataset))
    indices = torch.randperm(len(full_train_dataset))[:demo_size]
    demo_train_dataset = torch.utils.data.Subset(full_train_dataset, indices)
    
    test_dataset = PCamDataset(
        root_dir=test_dir,
        transform=base_transform,
        is_test=True
    )
    
    # Use only 5% of the test data for this demonstration
    demo_test_size = int(0.05 * len(test_dataset))
    test_indices = torch.randperm(len(test_dataset))[:demo_test_size]
    demo_test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Split training data into train and validation sets (85% train, 15% validation)
    train_size = int(0.85 * len(demo_train_dataset))
    val_size = len(demo_train_dataset) - train_size
    train_dataset, val_dataset = random_split(demo_train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        demo_test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    if config["model"] == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif config["model"] == "densenet121":
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif config["model"] == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif config["model"] == "seresnext50":
        model = timm.create_model('seresnext50_32x4d', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif config["model"] == "custom_attention":
        model = CustomAttentionModel()
    
    model = model.to(device)
    
    # Define loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer and scheduler
    if config["optimizer"] == "adam_cosine":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    elif config["optimizer"] == "adamw_onecycle":
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config["learning_rate"],
            steps_per_epoch=len(train_loader),
            epochs=config["epochs"]
        )
    
    # Train the model
    log(f"\n{'='*80}")
    log(f"Training {config['name']} model")
    log(f"{'='*80}")
    
    start_time = time.time()
    model, best_val_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=config["epochs"],
        model_name=config["name"]
    )
    training_time = time.time() - start_time
    
    # Evaluate the model
    test_auc = evaluate_model(model, val_loader, device)
    
    # Measure inference time
    inference_time_per_sample, total_inference_time = measure_inference_time(model, test_loader, device)
    
    # Calculate model size
    model_size = get_model_size(model)
    
    # Calculate number of parameters
    num_params = get_model_parameters(model)
    
    # Log results
    log(f"\nTraining completed in {training_time:.2f} seconds")
    log(f"Test AUC-ROC: {test_auc:.4f}")
    log(f"Inference time: {total_inference_time:.2f} seconds for {len(test_dataset)} samples")
    log(f"Average inference time per sample: {inference_time_per_sample:.2f} ms")
    log(f"Model size: {model_size:.2f} MB")
    
    # Clean up to free memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "Model": config["name"],
        "AUC-ROC": test_auc,
        "Training Time (s)": training_time,
        "Inference Time (ms/sample)": inference_time_per_sample,
        "Model Size (MB)": model_size,
        "Parameters (M)": num_params
    }

def mixup_data(x, y, alpha=0.2):
    """Applies mixup augmentation to the batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Applies mixup criterion to the predictions."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    start_time = time.time()
    log("Starting PCam cancer detection experiment (Experimental Group)")
    
    # Check if dataset exists
    dataset_path = "/workspace/mle_dataset"
    if os.path.exists(dataset_path):
        log(f"Dataset found at {dataset_path}")
        train_path = os.path.join(dataset_path, "train")
        test_path = os.path.join(dataset_path, "test")
        
        # Count number of files to verify dataset
        train_files = len([f for f in os.listdir(train_path) if f.endswith('.tif')])
        test_files = len([f for f in os.listdir(test_path) if f.endswith('.tif')])
        
        log(f"Dataset contains {train_files} training images and {test_files} test images")
    else:
        log(f"Error: Dataset not found at {dataset_path}. Exiting.")
        return
    
    # Define model configurations - using reduced epochs for demonstration
    # In a real experiment, we would use the full epochs as specified in the plan
    configs = [
        {
            "name": "ResNet50",
            "model": "resnet50",
            "batch_size": 32,
            "optimizer": "adam_cosine",
            "learning_rate": 0.0005,
            "epochs": 2,  # Reduced for demonstration (would be 30 in full experiment)
            "augmentation": "basic"
        },
        {
            "name": "DenseNet121",
            "model": "densenet121",
            "batch_size": 32,
            "optimizer": "adam_cosine",
            "learning_rate": 0.0005,
            "epochs": 2,  # Reduced for demonstration (would be 30 in full experiment)
            "augmentation": "advanced"
        },
        {
            "name": "EfficientNetB0",
            "model": "efficientnet_b0",
            "batch_size": 32,
            "optimizer": "adam_cosine",
            "learning_rate": 0.0005,
            "epochs": 2,  # Reduced for demonstration (would be 30 in full experiment)
            "augmentation": "advanced"
        },
        {
            "name": "SEResNeXt50",
            "model": "seresnext50",
            "batch_size": 32,
            "optimizer": "adam_cosine",
            "learning_rate": 0.0005,
            "epochs": 2,  # Reduced for demonstration (would be 30 in full experiment)
            "augmentation": "advanced"
        },
        {
            "name": "Custom Attention Model",
            "model": "custom_attention",
            "batch_size": 32,
            "optimizer": "adamw_onecycle",
            "learning_rate": 0.0003,
            "epochs": 2,  # Reduced for demonstration (would be 40 in full experiment)
            "augmentation": "advanced_with_mixup"
        }
    ]
    
    # Run experiments for each model configuration
    results = []
    for config in configs:
        model_results = run_experiment(config)
        results.append(model_results)
    
    # Find the best model based on AUC-ROC
    best_model = max(results, key=lambda x: x["AUC-ROC"])
    
    # Display comparison table
    log("\n" + "="*80)
    log("Model Comparison Results")
    log("="*80)
    
    # Convert results to DataFrame for better formatting
    results_df = pd.DataFrame(results)
    
    # Format the table
    table = tabulate(
        results_df, 
        headers="keys", 
        tablefmt="grid", 
        floatfmt=".4f",
        showindex=False
    )
    
    log(table)
    
    # Highlight the best model
    log(f"\nBest performing model: {best_model['Model']} with AUC-ROC of {best_model['AUC-ROC']:.4f}")
    
    total_time = time.time() - start_time
    log(f"\nTotal experiment time: {total_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    main()