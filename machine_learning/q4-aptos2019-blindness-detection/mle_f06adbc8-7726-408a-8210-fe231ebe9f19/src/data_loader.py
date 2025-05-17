import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class APTOSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0] + '.png')
        image = Image.open(img_name)
        
        # Some images might be grayscale, convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if 'diagnosis' in self.data_frame.columns:
            diagnosis = self.data_frame.iloc[idx, 1]
            diagnosis = torch.tensor(diagnosis, dtype=torch.long)
        else:
            diagnosis = torch.tensor(-1, dtype=torch.long)  # For test set

        if self.transform:
            image = self.transform(image)

        return image, diagnosis

def get_data_loaders(data_dir, batch_size=32, val_split=0.2, random_state=42):
    """
    Create train and validation data loaders
    
    Args:
        data_dir (string): Path to the data directory
        batch_size (int): Batch size for the data loaders
        val_split (float): Validation split ratio
        random_state (int): Random state for reproducibility
        
    Returns:
        train_loader, val_loader: PyTorch data loaders for training and validation
    """
    # Define transformations
    # Basic augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Only resize and normalize for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the CSV file
    csv_path = os.path.join(data_dir, 'train.csv')
    df = pd.read_csv(csv_path)
    
    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=random_state, stratify=df['diagnosis'])
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Save the split datasets for reproducibility
    train_csv_path = os.path.join(data_dir, 'train_split.csv')
    val_csv_path = os.path.join(data_dir, 'val_split.csv')
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    # Create datasets
    train_dataset = APTOSDataset(
        csv_file=train_csv_path,
        img_dir=os.path.join(data_dir, 'train_images'),
        transform=train_transform
    )
    
    val_dataset = APTOSDataset(
        csv_file=val_csv_path,
        img_dir=os.path.join(data_dir, 'train_images'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    return train_loader, val_loader

def get_test_loader(data_dir, batch_size=32):
    """
    Create test data loader
    
    Args:
        data_dir (string): Path to the data directory
        batch_size (int): Batch size for the data loader
        
    Returns:
        test_loader: PyTorch data loader for testing
    """
    # Define transformations for test data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the CSV file
    csv_path = os.path.join(data_dir, 'test.csv')
    
    # Create dataset
    test_dataset = APTOSDataset(
        csv_file=csv_path,
        img_dir=os.path.join(data_dir, 'test_images'),
        transform=test_transform
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    return test_loader