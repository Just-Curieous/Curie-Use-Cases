import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Utility functions
def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_img

def load_and_preprocess_image(image_path, size=(380, 380), apply_clahe_enhancement=True):
    """Load and preprocess an image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if apply_clahe_enhancement:
            image = apply_clahe(image)
        image = cv2.resize(image, size)
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)

def calculate_metrics(y_true, y_pred):
    """Calculate metrics for model evaluation."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['quadratic_weighted_kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return metrics

def save_results(metrics, predictions, true_labels, output_dir, filename_prefix="results"):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, f"{filename_prefix}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save predictions
    preds_df = pd.DataFrame({
        'true_label': true_labels,
        'prediction': predictions
    })
    preds_path = os.path.join(output_dir, f"{filename_prefix}_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    
    # Create and save confusion matrix
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(true_labels, predictions):
        cm[t][p] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    class_names = [str(i) for i in range(5)]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{filename_prefix}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    return metrics_path, preds_path, cm_path

# Dataset class
class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, apply_clahe=True):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.apply_clahe = apply_clahe
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        
        # Load and preprocess image
        image = load_and_preprocess_image(img_path, apply_clahe_enhancement=self.apply_clahe)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Get label if available (for training data)
        if 'diagnosis' in self.df.columns:
            label = self.df.iloc[idx]['diagnosis']
            return image, label
        else:
            return image, img_id

# Model class
class DiabeticRetinopathyModel(nn.Module):
    def __init__(self, num_classes=5, model_name='efficientnet-b4'):
        super().__init__()
        
        # Load pre-trained EfficientNet
        self.model = EfficientNet.from_pretrained(model_name)
        
        # Replace classifier
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define transformations
    train_transform = A.Compose([
        A.Resize(height=380, width=380),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(height=380, width=380),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Load data
    train_df = pd.read_csv(args.train_csv)
    
    # Handle class imbalance by stratifying the split
    train_df, val_df = train_test_split(
        train_df, 
        test_size=args.val_split, 
        random_state=args.seed,
        stratify=train_df['diagnosis']
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Create datasets
    train_dataset = APTOSDataset(
        train_df, 
        args.train_img_dir, 
        transform=train_transform,
        apply_clahe=args.apply_clahe
    )
    
    val_dataset = APTOSDataset(
        val_df, 
        args.train_img_dir, 
        transform=val_transform,
        apply_clahe=args.apply_clahe
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Calculate class weights for handling imbalance
    if args.use_class_weights:
        class_counts = train_df['diagnosis'].value_counts().sort_index()
        total_samples = len(train_df)
        weights = total_samples / (len(class_counts) * class_counts)
        weights = weights / weights.sum() * len(class_counts)
        class_weights = torch.tensor(weights.values, dtype=torch.float32)
    else:
        class_weights = None
    
    # Initialize model
    model = DiabeticRetinopathyModel(
        num_classes=5,  # 5 classes for diabetic retinopathy
        model_name=args.model_name
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function with class weights if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    best_val_kappa = 0.0
    best_epoch = 0
    
    print(f"Starting training for {args.max_epochs} epochs...")
    
    for epoch in range(args.max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Disable progress bar to reduce output
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                
                # Store predictions and targets for metrics calculation
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        val_metrics = calculate_metrics(val_targets, val_preds)
        val_acc = val_metrics['accuracy'] * 100
        val_kappa = val_metrics['quadratic_weighted_kappa']
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.max_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Kappa: {val_kappa:.4f}")
        
        # Save model after each epoch
        epoch_model_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model saved to {epoch_model_path}")
        
        # Track best model
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch + 1
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
    
    print(f"Training completed. Best model from epoch {best_epoch} with kappa: {best_val_kappa:.4f}")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
    model.eval()
    
    # Final evaluation on validation set
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
    
    # Calculate and save final metrics
    final_metrics = calculate_metrics(val_targets, val_preds)
    print(f"Final validation metrics:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Quadratic Weighted Kappa: {final_metrics['quadratic_weighted_kappa']:.4f}")
    
    # Save results
    metrics_path, preds_path, cm_path = save_results(
        final_metrics,
        val_preds,
        val_targets,
        args.output_dir,
        filename_prefix="validation_results"
    )
    
    print(f"Results saved to {args.output_dir}")
    
    # Generate predictions for test set if available
    if os.path.exists(args.test_csv):
        test_df = pd.read_csv(args.test_csv)
        test_dataset = APTOSDataset(
            test_df, 
            args.test_img_dir, 
            transform=val_transform,
            apply_clahe=args.apply_clahe
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        test_preds = []
        test_img_ids = []
        
        with torch.no_grad():
            for images, img_ids in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_preds.extend(predicted.cpu().numpy())
                test_img_ids.extend(img_ids)
        
        # Create submission file
        submission = pd.DataFrame({
            'id_code': test_img_ids,
            'diagnosis': test_preds
        })
        submission_path = os.path.join(args.output_dir, 'submission.csv')
        submission.to_csv(submission_path, index=False)
        print(f"Test predictions saved to {submission_path}")
    
    return final_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for diabetic retinopathy detection')
    
    # Data paths
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--train_img_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--test_img_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=2, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='efficientnet-b4', help='EfficientNet model variant')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for handling imbalance')
    
    # Preprocessing parameters
    parser.add_argument('--apply_clahe', action='store_true', help='Apply CLAHE preprocessing')
    
    args = parser.parse_args()
    main(args)
