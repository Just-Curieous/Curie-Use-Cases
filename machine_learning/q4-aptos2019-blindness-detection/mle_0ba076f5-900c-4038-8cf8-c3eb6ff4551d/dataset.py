import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class APTOSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether this is a test dataset (no labels)
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0] + '.png')
        
        # Read image
        image = cv2.imread(img_name)
        
        # Apply CLAHE enhancement
        image = self.apply_clahe(image)
        
        # Convert to RGB (from BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, self.data_frame.iloc[idx, 0]  # Return image and id_code
        else:
            label = self.data_frame.iloc[idx, 1]
            return image, label
    
    def apply_clahe(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split the LAB channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L channel with the original A and B channels
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convert back to BGR color space
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr

def get_transforms(is_train=True):
    """
    Get transforms for training and validation/test
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(train_csv, test_csv, train_img_dir, test_img_dir, batch_size=8, val_split=0.2):
    """
    Create train, validation, and test dataloaders
    
    Args:
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file
        train_img_dir: Path to training images directory
        test_img_dir: Path to test images directory
        batch_size: Batch size for dataloaders
        val_split: Fraction of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load training data
    train_df = pd.read_csv(train_csv)
    
    # Split into train and validation
    val_size = int(len(train_df) * val_split)
    train_size = len(train_df) - val_size
    
    # Use random_split to create train and validation datasets
    train_dataset = APTOSDataset(
        csv_file=train_csv,
        img_dir=train_img_dir,
        transform=get_transforms(is_train=True)
    )
    
    # Create train and validation datasets
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Override the transform for validation subset
    val_subset.dataset.transform = get_transforms(is_train=False)
    
    # Create test dataset
    test_dataset = APTOSDataset(
        csv_file=test_csv,
        img_dir=test_img_dir,
        transform=get_transforms(is_train=False),
        is_test=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader