import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from utils import load_and_preprocess_image

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

class APTOSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        test_csv_path,
        train_img_dir,
        test_img_dir,
        batch_size=16,
        val_split=0.2,
        num_workers=4,
        apply_clahe=True,
        seed=42
    ):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.apply_clahe = apply_clahe
        self.seed = seed
        
        # Define transformations
        self.train_transform = A.Compose([
            A.Resize(height=380, width=380),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.val_transform = A.Compose([
            A.Resize(height=380, width=380),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
    def setup(self, stage=None):
        # Load data
        train_df = pd.read_csv(self.train_csv_path)
        
        # Handle class imbalance by stratifying the split
        train_df, val_df = train_test_split(
            train_df, 
            test_size=self.val_split, 
            random_state=self.seed,
            stratify=train_df['diagnosis']
        )
        
        # Reset indices
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        
        # Create datasets
        self.train_dataset = APTOSDataset(
            train_df, 
            self.train_img_dir, 
            transform=self.train_transform,
            apply_clahe=self.apply_clahe
        )
        
        self.val_dataset = APTOSDataset(
            val_df, 
            self.train_img_dir, 
            transform=self.val_transform,
            apply_clahe=self.apply_clahe
        )
        
        # Load test data if available
        if os.path.exists(self.test_csv_path):
            test_df = pd.read_csv(self.test_csv_path)
            self.test_dataset = APTOSDataset(
                test_df, 
                self.test_img_dir, 
                transform=self.val_transform,
                apply_clahe=self.apply_clahe
            )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        return None
    
    def get_class_weights(self):
        """Calculate class weights to handle class imbalance."""
        train_df = pd.read_csv(self.train_csv_path)
        class_counts = train_df['diagnosis'].value_counts().sort_index()
        total_samples = len(train_df)
        
        # Calculate weights as inverse of frequency
        weights = total_samples / (len(class_counts) * class_counts)
        
        # Normalize weights
        weights = weights / weights.sum() * len(class_counts)
        
        return torch.tensor(weights.values, dtype=torch.float32)