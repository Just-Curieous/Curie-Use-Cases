import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class PCamDataset(Dataset):
    def __init__(self, image_dir, labels_df=None, transform=None, is_test=False):
        """
        Args:
            image_dir (string): Directory with all the images.
            labels_df (pandas.DataFrame): DataFrame containing image IDs and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether this is the test dataset.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        
        if not is_test:
            self.labels_df = labels_df
            self.image_ids = self.labels_df['id'].values
        else:
            # For test set, just get all image files
            self.image_ids = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.tif')]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")
        
        # Load image
        image = Image.open(img_path)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if not self.is_test:
            # Get label for training/validation
            label = self.labels_df.loc[self.labels_df['id'] == img_id, 'label'].values[0]
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            # For test set, return image ID as well
            return image, img_id

def get_data_loaders(data_dir, batch_size=32, num_workers=4, val_size=0.15, test_size=0.15, seed=42):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir (string): Root directory of the dataset.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of workers for the data loaders.
        val_size (float): Proportion of the training data to use for validation.
        test_size (float): Proportion of the training data to use for testing.
        seed (int): Random seed for reproducibility.
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for training, validation, and testing.
    """
    # Define data transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load labels
    labels_path = os.path.join(data_dir, 'train_labels.csv')
    labels_df = pd.read_csv(labels_path)
    
    # Split data into train, validation, and test sets
    train_val_ids, test_ids = train_test_split(
        labels_df['id'].values, 
        test_size=test_size, 
        random_state=seed, 
        stratify=labels_df['label']
    )
    
    train_ids, val_ids = train_test_split(
        train_val_ids, 
        test_size=val_size/(1-test_size),  # Adjust validation size
        random_state=seed, 
        stratify=labels_df.loc[labels_df['id'].isin(train_val_ids), 'label']
    )
    
    # Create DataFrames for each split
    train_df = labels_df[labels_df['id'].isin(train_ids)]
    val_df = labels_df[labels_df['id'].isin(val_ids)]
    test_df = labels_df[labels_df['id'].isin(test_ids)]
    
    # Create datasets
    train_dataset = PCamDataset(
        image_dir=os.path.join(data_dir, 'train'),
        labels_df=train_df,
        transform=train_transform
    )
    
    val_dataset = PCamDataset(
        image_dir=os.path.join(data_dir, 'train'),
        labels_df=val_df,
        transform=val_test_transform
    )
    
    test_dataset = PCamDataset(
        image_dir=os.path.join(data_dir, 'train'),
        labels_df=test_df,
        transform=val_test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_df, val_df, test_df

def get_external_test_loader(data_dir, batch_size=32, num_workers=4):
    """
    Create a data loader for the external test set.
    
    Args:
        data_dir (string): Root directory of the dataset.
        batch_size (int): Batch size for the data loader.
        num_workers (int): Number of workers for the data loader.
        
    Returns:
        test_loader: DataLoader object for the external test set.
    """
    # Define data transformations
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    test_dataset = PCamDataset(
        image_dir=os.path.join(data_dir, 'test'),
        transform=test_transform,
        is_test=True
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader