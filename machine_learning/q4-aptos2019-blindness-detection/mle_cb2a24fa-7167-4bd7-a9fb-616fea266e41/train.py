import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import logging

from utils import calculate_metrics, save_predictions


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to use
        
    Returns:
        tuple: (epoch_loss, epoch_kappa)
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    for batch in dataloader:
        images = batch['image'].to(device)
        targets = batch['label'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_predictions))
    epoch_kappa = metrics['kappa']
    
    return epoch_loss, epoch_kappa


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model (nn.Module): The model
        dataloader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        
    Returns:
        tuple: (val_loss, val_kappa, all_targets, all_predictions, all_probabilities)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    # Calculate metrics
    val_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_predictions))
    val_kappa = metrics['kappa']
    
    return val_loss, val_kappa, np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)


def predict(model, dataloader, device):
    """
    Make predictions with the model.
    
    Args:
        model (nn.Module): The model
        dataloader (DataLoader): Test data loader
        device (torch.device): Device to use
        
    Returns:
        tuple: (ids, predictions, probabilities)
    """
    model.eval()
    all_ids = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            ids = batch['id_code']
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_ids.extend(ids)
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    return all_ids, np.array(all_predictions), np.array(all_probabilities)


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, 
                device, num_epochs=25, patience=5, model_save_path=None):
    """
    Train the model.
    
    Args:
        model (nn.Module): The model
        train_loader (DataLoader): Training data loader
        valid_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        device (torch.device): Device to use
        num_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        model_save_path (str): Path to save the best model
        
    Returns:
        tuple: (best_model, history)
    """
    # Initialize variables
    best_val_kappa = -1.0
    best_model_wts = None
    counter = 0
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_kappa': [],
        'val_kappa': []
    }
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_kappa = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_kappa, _, _, _ = validate(model, valid_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_kappa'].append(train_kappa)
        history['val_kappa'].append(val_kappa)
        
        # Print epoch results
        logging.info(f'Train Loss: {train_loss:.4f} | Train Kappa: {train_kappa:.4f}')
        logging.info(f'Val Loss: {val_loss:.4f} | Val Kappa: {val_kappa:.4f}')
        
        # Check if this is the best model
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_model_wts = model.state_dict().copy()
            counter = 0
            
            # Save the best model
            if model_save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_kappa': best_val_kappa,
                    'history': history
                }, model_save_path)
                logging.info(f'Saved best model with kappa: {best_val_kappa:.4f}')
        else:
            counter += 1
            
        # Early stopping
        if counter >= patience:
            logging.info(f'Early stopping at epoch {epoch+1}')
            break
    
    # Calculate total training time
    time_elapsed = time.time() - start_time
    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best validation kappa: {best_val_kappa:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


def evaluate_model(model, valid_loader, criterion, device, output_dir):
    """
    Evaluate the model on the validation set.
    
    Args:
        model (nn.Module): The model
        valid_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        output_dir (str): Directory to save results
        
    Returns:
        dict: Metrics
    """
    # Validate
    val_loss, val_kappa, targets, predictions, probabilities = validate(model, valid_loader, criterion, device)
    
    # Calculate metrics
    metrics = calculate_metrics(targets, predictions, probabilities)
    
    # Log metrics
    logging.info(f'Validation Loss: {val_loss:.4f}')
    logging.info(f'Validation Kappa: {val_kappa:.4f}')
    logging.info(f'Validation Accuracy: {metrics["accuracy"]:.4f}')
    
    # Log per-class metrics
    for i in range(len(metrics['precision_per_class'])):
        logging.info(f'Class {i} - Precision: {metrics["precision_per_class"][i]:.4f}, '
                    f'Recall: {metrics["recall_per_class"][i]:.4f}, '
                    f'F1: {metrics["f1_per_class"][i]:.4f}')
    
    # Log severe and proliferative metrics
    logging.info(f'Severe (Class 3) - Precision: {metrics["severe_precision"]:.4f}, '
                f'Recall: {metrics["severe_recall"]:.4f}, '
                f'F1: {metrics["severe_f1"]:.4f}')
    
    logging.info(f'Proliferative (Class 4) - Precision: {metrics["proliferative_precision"]:.4f}, '
                f'Recall: {metrics["proliferative_recall"]:.4f}, '
                f'F1: {metrics["proliferative_f1"]:.4f}')
    
    # Save validation predictions
    ids = [batch['id_code'] for batch in valid_loader.dataset]
    save_predictions(ids, predictions, probabilities, os.path.join(output_dir, 'validation_predictions.csv'))
    
    return metrics