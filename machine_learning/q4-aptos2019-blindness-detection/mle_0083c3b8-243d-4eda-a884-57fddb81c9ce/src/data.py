import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.utils import circular_crop
from src.config import (
    TRAIN_CSV, TEST_CSV, TRAIN_IMAGES_DIR, TEST_IMAGES_DIR,
    BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE, SEED
)

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether this is the test dataset.
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply circular crop
        image = circular_crop(image)
        
        # Convert to PIL Image for torchvision transforms
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image
        else:
            diagnosis = self.data_frame.iloc[idx, 1]
            return image, diagnosis

def get_transforms(augmentation_level="standard"):
    """
    Get train and validation transforms.
    
    Args:
        augmentation_level (str): Level of augmentation to apply.
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Training transform with augmentation
    if augmentation_level == "standard":
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif augmentation_level == "strong":
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    return train_transform, val_transform

def create_data_loaders(augmentation_level="standard", validation_split=0.2):
    """
    Create train, validation, and test data loaders.
    
    Args:
        augmentation_level (str): Level of augmentation to apply.
        validation_split (float): Fraction of training data to use for validation.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform, val_transform = get_transforms(augmentation_level)
    
    # Load training data
    train_df = pd.read_csv(TRAIN_CSV)
    
    # Split into train and validation
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_size = int(len(train_df) * validation_split)
    train_df_split = train_df.iloc[val_size:]
    val_df_split = train_df.iloc[:val_size]
    
    # Save the split dataframes temporarily
    train_split_path = os.path.join(os.path.dirname(TRAIN_CSV), "train_split.csv")
    val_split_path = os.path.join(os.path.dirname(TRAIN_CSV), "val_split.csv")
    train_df_split.to_csv(train_split_path, index=False)
    val_df_split.to_csv(val_split_path, index=False)
    
    # Create datasets
    train_dataset = RetinopathyDataset(
        csv_file=train_split_path,
        img_dir=TRAIN_IMAGES_DIR,
        transform=train_transform
    )
    
    val_dataset = RetinopathyDataset(
        csv_file=val_split_path,
        img_dir=TRAIN_IMAGES_DIR,
        transform=val_transform
    )
    
    test_dataset = RetinopathyDataset(
        csv_file=TEST_CSV,
        img_dir=TEST_IMAGES_DIR,
        transform=val_transform,
        is_test=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Clean up temporary files
    os.remove(train_split_path)
    os.remove(val_split_path)
    
    return train_loader, val_loader, test_loader