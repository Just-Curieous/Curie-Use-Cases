import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CLAHEPreprocessor:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to images.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, img):
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        
        # Check if image is grayscale or RGB
        if len(img_np.shape) == 2 or img_np.shape[2] == 1:
            # Apply CLAHE to grayscale image
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img_np = clahe.apply(img_np)
        else:
            # Convert to LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            lab = cv2.merge((l, a, b))
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
        # Convert back to PIL Image
        return Image.fromarray(img_np)


class RetinopathyDataset(Dataset):
    """
    Dataset class for loading and preprocessing retinopathy images.
    """
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether this is a test dataset (no labels).
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
            
        img_name = os.path.join(self.img_dir, f"{self.data_frame.iloc[idx, 0]}.png")
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return {'image': image, 'id_code': self.data_frame.iloc[idx, 0]}
        else:
            label = self.data_frame.iloc[idx, 1]
            return {'image': image, 'label': torch.tensor(label, dtype=torch.long), 'id_code': self.data_frame.iloc[idx, 0]}


def get_transforms(phase):
    """
    Get transforms for training or validation/testing phases.
    
    Args:
        phase (str): 'train', 'valid', or 'test'
        
    Returns:
        transforms.Compose: Composed transforms
    """
    if phase == 'train':
        return transforms.Compose([
            CLAHEPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8)),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # 'valid' or 'test'
        return transforms.Compose([
            CLAHEPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_data_loaders(train_csv, test_csv, train_img_dir, test_img_dir, batch_size=8, valid_size=0.2, random_state=42):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_csv (str): Path to training CSV file
        test_csv (str): Path to test CSV file
        train_img_dir (str): Path to training images directory
        test_img_dir (str): Path to test images directory
        batch_size (int): Batch size
        valid_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # Load training data
    train_df = pd.read_csv(train_csv)
    
    # Split into train and validation
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    valid_df = train_df.sample(frac=valid_size, random_state=random_state)
    train_df = train_df.drop(valid_df.index)
    
    # Save the split dataframes to temporary CSV files
    train_temp_csv = '/workspace/mle_cb2a24fa-7167-4bd7-a9fb-616fea266e41/train_temp.csv'
    valid_temp_csv = '/workspace/mle_cb2a24fa-7167-4bd7-a9fb-616fea266e41/valid_temp.csv'
    train_df.to_csv(train_temp_csv, index=False)
    valid_df.to_csv(valid_temp_csv, index=False)
    
    # Create datasets
    train_dataset = RetinopathyDataset(
        csv_file=train_temp_csv,
        img_dir=train_img_dir,
        transform=get_transforms('train')
    )
    
    valid_dataset = RetinopathyDataset(
        csv_file=valid_temp_csv,
        img_dir=train_img_dir,
        transform=get_transforms('valid')
    )
    
    test_dataset = RetinopathyDataset(
        csv_file=test_csv,
        img_dir=test_img_dir,
        transform=get_transforms('test'),
        is_test=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # As specified in requirements
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # As specified in requirements
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # As specified in requirements
    )
    
    return train_loader, valid_loader, test_loader