import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import (
    TRAIN_CSV, TEST_CSV, TRAIN_IMAGES_DIR, TEST_IMAGES_DIR,
    IMAGE_SIZE, BATCH_SIZE, SEED, NUM_FOLDS
)

def get_train_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

class RetinopathyDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['id_code']
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        # Get label if available
        if 'diagnosis' in self.df.columns:
            label = torch.tensor(self.df.iloc[idx]['diagnosis'], dtype=torch.long)
            return img, label
        else:
            return img, img_id

def prepare_dataloader(df, image_dir, transform, batch_size, shuffle=True):
    ds = RetinopathyDataset(df, image_dir, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )

def create_folds(df, n_splits=NUM_FOLDS):
    df['fold'] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    for fold, (_, val_idx) in enumerate(skf.split(df, df['diagnosis'])):
        df.loc[val_idx, 'fold'] = fold
    
    return df

def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Create folds for cross-validation
    train_df = create_folds(train_df)
    
    return train_df, test_df

def get_fold_dataloader(train_df, fold):
    train_idx = train_df[train_df['fold'] != fold].index
    valid_idx = train_df[train_df['fold'] == fold].index
    
    train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
    valid_fold_df = train_df.iloc[valid_idx].reset_index(drop=True)
    
    train_loader = prepare_dataloader(
        train_fold_df,
        TRAIN_IMAGES_DIR,
        get_train_transforms(),
        BATCH_SIZE,
        shuffle=True
    )
    
    valid_loader = prepare_dataloader(
        valid_fold_df,
        TRAIN_IMAGES_DIR,
        get_valid_transforms(),
        BATCH_SIZE,
        shuffle=False
    )
    
    return train_loader, valid_loader, train_fold_df, valid_fold_df