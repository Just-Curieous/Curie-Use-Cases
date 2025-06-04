import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def train_model(model, train_loader, val_loader, device, num_epochs=50, 
                lr=0.001, patience=10, model_save_path=None, log_file=None):
    """
    Train the model with early stopping based on validation AUC-ROC.
    
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to use for training.
        num_epochs (int): Maximum number of epochs to train for.
        lr (float): Learning rate.
        patience (int): Number of epochs to wait for improvement before early stopping.
        model_save_path (str): Path to save the best model.
        log_file (str): Path to log file.
        
    Returns:
        tuple: (trained model, training history)
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize variables for early stopping
    best_val_auc = 0.0
    patience_counter = 0
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * images.size(0)
            train_preds.extend(outputs.squeeze().detach().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_auc = roc_auc_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                
                # Track metrics
                val_loss += loss.item() * images.size(0)
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate epoch metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Train AUC: {train_auc:.4f} | '
              f'Val AUC: {val_auc:.4f}')
        
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f'Epoch {epoch+1}/{num_epochs} | '
                      f'Train Loss: {train_loss:.4f} | '
                      f'Val Loss: {val_loss:.4f} | '
                      f'Train AUC: {train_auc:.4f} | '
                      f'Val AUC: {val_auc:.4f}\n')
        
        # Check if this is the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            
            # Save the best model
            if model_save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_auc': train_auc,
                    'val_auc': val_auc,
                }, model_save_path)
                print(f'Model saved to {model_save_path}')
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(f'Model saved to {model_save_path}\n')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f'Early stopping after {epoch+1} epochs\n')
            break
    
    # Load the best model
    if model_save_path and os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded best model from epoch {checkpoint["epoch"]+1} with validation AUC: {checkpoint["val_auc"]:.4f}')
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f'Loaded best model from epoch {checkpoint["epoch"]+1} with validation AUC: {checkpoint["val_auc"]:.4f}\n')
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_auc': train_aucs,
        'val_auc': val_aucs
    }
    
    return model, history