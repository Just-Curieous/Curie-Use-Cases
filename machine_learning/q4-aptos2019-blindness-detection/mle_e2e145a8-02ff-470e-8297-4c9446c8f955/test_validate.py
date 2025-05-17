import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import get_data_loaders
from models.model import get_model
from utils.train_utils import validate

def test_validate_function():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define paths
    dataset_dir = "/workspace/mle_dataset"
    train_csv = os.path.join(dataset_dir, "train.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")
    train_img_dir = os.path.join(dataset_dir, "train_images")
    test_img_dir = os.path.join(dataset_dir, "test_images")
    
    # Get data loaders for fold 0
    train_loader, valid_loader, test_loader = get_data_loaders(
        train_csv, test_csv, train_img_dir, test_img_dir,
        fold=0, num_folds=5, batch_size=16, num_workers=4, seed=42
    )
    
    # Create model
    model = get_model(num_classes=5, pretrained=True, device=device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Test validate function
    print("Testing validate function...")
    val_loss, val_preds, val_labels = validate(model, valid_loader, criterion, device)
    
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation predictions shape: {val_preds.shape}")
    print(f"Validation labels shape: {val_labels.shape}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_validate_function()