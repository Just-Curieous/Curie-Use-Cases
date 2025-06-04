import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

def train_model(model, train_loader, val_loader, device, num_epochs=20, learning_rate=0.001, 
                save_dir='/workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf'):
    """
    Train the model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (str): Device to use for training
        num_epochs (int): Number of epochs to train for
        learning_rate (float): Learning rate
        save_dir (str): Directory to save model checkpoints
    
    Returns:
        dict: Training history
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }
    
    best_val_auc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc='Training', disable=True):
            inputs = inputs.to(device)
            labels = labels.float().to(device).view(-1, 1)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        val_loss, val_auc = evaluate_model(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}')
        
        # Update learning rate
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(save_dir, 'models', 'best_model.pth'))
            print(f'Saved new best model with AUC: {val_auc:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': best_val_auc,
                'history': history
            }, os.path.join(save_dir, 'models', f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'models', 'final_model.pth'))
    
    return history

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model.
    
    Args:
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): Data loader
        criterion (nn.Module): Loss function
        device (str): Device to use for evaluation
    
    Returns:
        tuple: (loss, auc)
    """
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Validation', disable=True):
            inputs = inputs.to(device)
            labels = labels.float().to(device).view(-1, 1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            
            # Store predictions and labels for AUC calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
    
    val_loss = val_loss / len(data_loader.dataset)
    
    # Calculate AUC
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    
    return val_loss, auc