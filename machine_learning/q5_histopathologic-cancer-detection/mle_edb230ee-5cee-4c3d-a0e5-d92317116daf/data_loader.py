import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PatchCamelyonDataset(Dataset):
    def __init__(self, root_dir, csv_file=None, transform=None, is_test=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether this is the test dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        
        if not is_test:
            self.labels_df = pd.read_csv(csv_file)
            self.image_ids = self.labels_df['id'].values
            self.labels = self.labels_df['label'].values
        else:
            # For test set, just get the image IDs from the directory
            self.image_ids = [f.split('.')[0] for f in os.listdir(root_dir) if f.endswith('.tif')]
            self.labels = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.image_ids[idx] + '.tif')
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        if not self.is_test:
            label = self.labels[idx]
            return image, label
        else:
            return image, self.image_ids[idx]

def get_transforms(mode='train'):
    """
    Get the transforms for the specified mode.
    
    Args:
        mode (str): 'train' or 'val' or 'test'
    
    Returns:
        transforms.Compose: The composed transforms
    """
    # Standard RGB normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(96),  # Resize to slightly larger than needed
            transforms.CenterCrop(96),  # Center crop to get consistent size
            transforms.RandomRotation(15),  # Medium augmentation: rotation ±15°
            transforms.RandomHorizontalFlip(),  # Medium augmentation: horizontal flip
            transforms.RandomVerticalFlip(),  # Medium augmentation: vertical flip
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Medium augmentation: slight color jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir (str): Path to the dataset directory
        batch_size (int): Batch size for the data loaders
        num_workers (int): Number of workers for the data loaders
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    train_csv = os.path.join(data_dir, 'train_labels.csv')
    
    # Load the full training dataset
    full_dataset = PatchCamelyonDataset(
        root_dir=train_dir,
        csv_file=train_csv,
        transform=get_transforms('train')
    )
    
    # Split into train and validation sets (90% train, 10% validation)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Override the transform for validation dataset
    val_dataset.dataset.transform = get_transforms('val')
    
    # Create test dataset
    test_dataset = PatchCamelyonDataset(
        root_dir=test_dir,
        transform=get_transforms('test'),
        is_test=True
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
    
    return train_loader, val_loader, test_loader