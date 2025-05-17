import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from sklearn.model_selection import StratifiedKFold

class RetinopathyDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(height=300, width=300),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=300, width=300),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

def prepare_dataloader(df, image_dir, phase, fold=None, num_folds=5, batch_size=16, num_workers=4, seed=42):
    """
    Prepare dataloader for training, validation, or testing
    
    Args:
        df: DataFrame containing image IDs and labels
        image_dir: Directory containing images
        phase: 'train', 'valid', or 'test'
        fold: Current fold number (only used if phase is 'train' or 'valid')
        num_folds: Total number of folds for cross-validation
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        seed: Random seed for reproducibility
        
    Returns:
        DataLoader object
    """
    if phase == 'test':
        # For test set, we don't have labels
        image_paths = [os.path.join(image_dir, f"{img_id}.png") for img_id in df['id_code'].values]
        dataset = RetinopathyDataset(
            image_paths=image_paths,
            transform=get_transforms(phase)
        )
    else:
        # For train/valid, we need to split the data based on fold
        assert fold is not None, "Fold must be specified for train/valid phase"
        
        # Create folds
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        df['fold'] = -1
        
        for i, (train_idx, val_idx) in enumerate(skf.split(df, df['diagnosis'])):
            df.loc[val_idx, 'fold'] = i
        
        if phase == 'train':
            df_phase = df[df['fold'] != fold].reset_index(drop=True)
        else:  # valid
            df_phase = df[df['fold'] == fold].reset_index(drop=True)
        
        image_paths = [os.path.join(image_dir, f"{img_id}.png") for img_id in df_phase['id_code'].values]
        dataset = RetinopathyDataset(
            image_paths=image_paths,
            labels=df_phase['diagnosis'].values,
            transform=get_transforms(phase)
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(phase == 'train')
    )
    
    return dataloader

def get_data_loaders(train_csv_path, test_csv_path, train_img_dir, test_img_dir, 
                    fold, num_folds=5, batch_size=16, num_workers=4, seed=42):
    """
    Get train, validation, and test dataloaders for a specific fold
    
    Args:
        train_csv_path: Path to train CSV file
        test_csv_path: Path to test CSV file
        train_img_dir: Directory containing training images
        test_img_dir: Directory containing test images
        fold: Current fold number
        num_folds: Total number of folds for cross-validation
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, valid_loader, test_loader
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load CSVs
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Prepare dataloaders
    train_loader = prepare_dataloader(
        train_df, train_img_dir, 'train', fold, num_folds, batch_size, num_workers, seed
    )
    
    valid_loader = prepare_dataloader(
        train_df, train_img_dir, 'valid', fold, num_folds, batch_size, num_workers, seed
    )
    
    test_loader = prepare_dataloader(
        test_df, test_img_dir, 'test', batch_size=batch_size, num_workers=num_workers
    )
    
    return train_loader, valid_loader, test_loader