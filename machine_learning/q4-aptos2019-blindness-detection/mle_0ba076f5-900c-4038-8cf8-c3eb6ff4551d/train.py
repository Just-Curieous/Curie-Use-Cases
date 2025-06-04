import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import calculate_class_weights, calculate_metrics, quadratic_weighted_kappa

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        epoch_loss: Average loss for the epoch
        epoch_kappa: Kappa score for the epoch
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        
        # Store targets and predictions for metric calculation
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_kappa = quadratic_weighted_kappa(np.array(all_targets), np.array(all_predictions))
    
    return epoch_loss, epoch_kappa

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    
    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        val_loss: Validation loss
        val_kappa: Validation kappa score
        all_targets: True labels
        all_predictions: Predicted labels
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store targets and predictions for metric calculation
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
    
    # Calculate validation metrics
    val_loss = running_loss / len(dataloader.dataset)
    val_kappa = quadratic_weighted_kappa(np.array(all_targets), np.array(all_predictions))
    
    return val_loss, val_kappa, all_targets, all_predictions

def train_model(model, train_loader, val_loader, num_epochs=50, patience=10, 
                lr=0.0001, device='cuda', output_dir='./'):
    """
    Train the model
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        lr: Learning rate
        device: Device to use
        output_dir: Directory to save model and results
        
    Returns:
        model: Trained model
        history: Training history
        best_val_targets: Validation targets at best epoch
        best_val_preds: Validation predictions at best epoch
    """
    # Get class weights for imbalanced dataset
    train_labels = []
    for _, labels in train_loader.dataset:
        train_labels.append(labels)
    
    class_weights = calculate_class_weights(train_labels)
    class_weights = class_weights.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Initialize variables for early stopping
    best_val_kappa = -1.0
    best_epoch = 0
    best_model_state = None
    best_val_targets = None
    best_val_preds = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_kappa': [],
        'val_kappa': [],
        'lr': []
    }
    
    # Start training
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_kappa = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_kappa, val_targets, val_preds = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_kappa)
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f}, Train Kappa: {train_kappa:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Kappa: {val_kappa:.4f}')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_kappa'].append(train_kappa)
        history['val_kappa'].append(val_kappa)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Check if this is the best model
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            best_val_targets = val_targets
            best_val_preds = val_preds
            
            # Save the best model
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f'New best model saved with validation kappa: {best_val_kappa:.4f}')
        
        # Early stopping
        if epoch - best_epoch >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Calculate total training time
    total_time = time.time() - start_time
    print(f'Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
    print(f'Best validation Kappa: {best_val_kappa:.4f} at epoch {best_epoch+1}')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, history, best_val_targets, best_val_preds

def predict(model, dataloader, device):
    """
    Make predictions with the model
    
    Args:
        model: The model to use
        dataloader: Test data loader
        device: Device to use
        
    Returns:
        ids: Image IDs
        predictions: Predicted labels
    """
    model.eval()
    all_ids = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, ids in dataloader:
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store IDs and predictions
            all_ids.extend(ids)
            all_predictions.extend(preds.cpu().numpy())
    
    return all_ids, all_predictions