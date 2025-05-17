import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        epoch_loss: Average loss for the epoch
    """
    model.train()
    epoch_loss = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= len(dataloader)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    
    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        val_loss: Average validation loss
        val_preds: Predictions
        val_labels: Ground truth labels
    """
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(dataloader)
    return val_loss, np.array(val_preds), np.array(val_labels)

def calculate_metrics(preds, labels):
    """
    Calculate evaluation metrics
    
    Args:
        preds: Predictions
        labels: Ground truth labels
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Calculate quadratic weighted kappa
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, preds)
    
    metrics = {
        'accuracy': accuracy,
        'quadratic_weighted_kappa': kappa,
        'confusion_matrix': cm
    }
    
    return metrics

def train_and_validate(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=10, scheduler=None):
    """
    Train and validate the model
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        valid_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train for
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        history: Dictionary of training history
        best_model_state: State dict of the best model
        best_metrics: Best validation metrics
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_kappa': []
    }
    
    best_kappa = -1
    best_model_state = None
    best_metrics = None
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_preds, val_labels = validate(model, valid_loader, criterion, device)
        
        # Calculate metrics
        metrics = calculate_metrics(val_preds, val_labels)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(metrics['accuracy'])
        history['val_kappa'].append(metrics['quadratic_weighted_kappa'])
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(metrics['quadratic_weighted_kappa'])
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save best model
        if metrics['quadratic_weighted_kappa'] > best_kappa:
            best_kappa = metrics['quadratic_weighted_kappa']
            best_model_state = model.state_dict().copy()
            best_metrics = metrics.copy()
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {metrics['accuracy']:.4f}, "
              f"Val Kappa: {metrics['quadratic_weighted_kappa']:.4f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return history, best_model_state, best_metrics, training_time

def predict(model, dataloader, device):
    """
    Make predictions with the model
    
    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device to predict on
        
    Returns:
        predictions: Model predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images in dataloader:
            if isinstance(images, list):
                images = images[0]  # In case the dataloader returns a list
            
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions)