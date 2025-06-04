import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class RetinopathyDataset(Dataset):
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
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split the LAB channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L channel with the original A and B channels
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR color space
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return enhanced_img

def get_data_loaders(train_csv, test_csv, train_img_dir, test_img_dir, batch_size=32):
    """
    Create data loaders for training and testing
    
    Args:
        train_csv (string): Path to the training csv file
        test_csv (string): Path to the test csv file
        train_img_dir (string): Path to the training images directory
        test_img_dir (string): Path to the test images directory
        batch_size (int): Batch size for the data loaders
        
    Returns:
        train_loader, val_loader, test_loader: Data loaders for training, validation, and testing
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load training data
    train_data = pd.read_csv(train_csv)
    
    # Split training data into train and validation sets (85% train, 15% validation)
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    split_idx = int(len(train_data) * 0.85)
    train_df = train_data[:split_idx]
    val_df = train_data[split_idx:]
    
    # Save the split dataframes to CSV
    train_df.to_csv('/workspace/mle_d4bfab9d-b8a0-41ef-9c6c-a28b75e69bc7/train_split.csv', index=False)
    val_df.to_csv('/workspace/mle_d4bfab9d-b8a0-41ef-9c6c-a28b75e69bc7/val_split.csv', index=False)
    
    # Create datasets
    train_dataset = RetinopathyDataset(
        csv_file='/workspace/mle_d4bfab9d-b8a0-41ef-9c6c-a28b75e69bc7/train_split.csv',
        img_dir=train_img_dir,
        transform=train_transform
    )
    
    val_dataset = RetinopathyDataset(
        csv_file='/workspace/mle_d4bfab9d-b8a0-41ef-9c6c-a28b75e69bc7/val_split.csv',
        img_dir=train_img_dir,
        transform=val_test_transform
    )
    
    test_dataset = RetinopathyDataset(
        csv_file=test_csv,
        img_dir=test_img_dir,
        transform=val_test_transform,
        is_test=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def get_class_weights(train_csv):
    """
    Calculate class weights for weighted loss function
    
    Args:
        train_csv (string): Path to the training csv file
        
    Returns:
        class_weights: Tensor of class weights
    """
    train_data = pd.read_csv(train_csv)
    class_counts = train_data['diagnosis'].value_counts().sort_index()
    total_samples = len(train_data)
    
    # Calculate weights as inverse of class frequency
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    return torch.FloatTensor(class_weights.values)