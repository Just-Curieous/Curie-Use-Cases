import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import random

class PCamDataset(Dataset):
    def __init__(self, image_dir, labels_df=None, transform=None, test_mode=False):
        """
        Args:
            image_dir (string): Directory with all the images.
            labels_df (pandas.DataFrame): DataFrame containing image IDs and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            test_mode (bool): If True, no labels are expected.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.test_mode = test_mode
        
        if test_mode:
            self.image_ids = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.tif')]
            self.labels = None
        else:
            self.image_ids = labels_df['id'].values
            self.labels = labels_df['label'].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        if self.test_mode:
            return image, img_id
        else:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.float32)

def get_transforms(augment=True):
    """
    Get transforms for training and validation/test sets
    
    Args:
        augment (bool): Whether to apply data augmentation
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    # Standard normalization values for ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Basic transformation for validation/test
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    if augment:
        # Augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = val_transform
        
    return train_transform, val_transform

def load_data(data_dir, batch_size=64, val_split=0.15, test_split=0.15, num_workers=4, seed=42):
    """
    Load and prepare PCam dataset
    
    Args:
        data_dir (str): Path to the dataset directory
        batch_size (int): Batch size for dataloaders
        val_split (float): Validation split ratio
        test_split (float): Test split ratio
        num_workers (int): Number of workers for dataloaders
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Load labels
    train_labels_path = os.path.join(data_dir, 'train_labels.csv')
    train_labels_df = pd.read_csv(train_labels_path)
    
    # Get transforms
    train_transform, val_transform = get_transforms(augment=True)
    
    # Create full dataset
    full_dataset = PCamDataset(
        image_dir=os.path.join(data_dir, 'train'),
        labels_df=train_labels_df,
        transform=None  # We'll apply transforms after splitting
    )
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size - test_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create custom datasets with appropriate transforms
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, idx):
            image, label = self.subset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
            
        def __len__(self):
            return len(self.subset)
    
    # Apply transforms
    train_dataset = TransformedSubset(train_dataset, train_transform)
    val_dataset = TransformedSubset(val_dataset, val_transform)
    test_dataset = TransformedSubset(test_dataset, val_transform)
    
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
    
    print(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")
    
    return train_loader, val_loader, test_loader

def get_test_loader(data_dir, batch_size=64, num_workers=4):
    """
    Load test dataset for inference
    
    Args:
        data_dir (str): Path to the dataset directory
        batch_size (int): Batch size for dataloader
        num_workers (int): Number of workers for dataloader
    
    Returns:
        DataLoader: Test data loader
    """
    _, val_transform = get_transforms(augment=False)
    
    test_dataset = PCamDataset(
        image_dir=os.path.join(data_dir, 'test'),
        transform=val_transform,
        test_mode=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader