import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
import json

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            path (str): Path to save the checkpoint
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        """Save model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(outputs.squeeze().detach().cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Calculate AUC-ROC
    try:
        epoch_auc = roc_auc_score(all_targets, all_predictions)
    except:
        epoch_auc = 0.5  # Default value if calculation fails
    
    return epoch_loss, epoch_auc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.squeeze().cpu().numpy())
    
    val_loss = running_loss / len(val_loader.dataset)
    
    # Calculate AUC-ROC
    try:
        val_auc = roc_auc_score(all_targets, all_predictions)
    except:
        val_auc = 0.5  # Default value if calculation fails
    
    return val_loss, val_auc

def train_model(model, train_loader, val_loader, output_dir, 
                lr=0.001, num_epochs=20, patience=5):
    """
    Train the model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Directory to save model and results
        lr: Learning rate
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
    
    Returns:
        dict: Training history and metrics
    """
    device = next(model.parameters()).device
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize early stopping
    checkpoint_path = os.path.join(output_dir, 'best_model.pt')
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path)
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_auc': [],
        'val_loss': [],
        'val_auc': [],
        'epochs': [],
        'best_epoch': 0,
        'best_val_auc': 0,
        'training_time': 0
    }
    
    # Start timer
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        
        # Train and validate
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_auc'].append(float(train_auc))
        history['val_loss'].append(val_loss)
        history['val_auc'].append(float(val_auc))
        history['epochs'].append(epoch)
        
        # Update best metrics
        if val_auc > history['best_val_auc']:
            history['best_val_auc'] = float(val_auc)
            history['best_epoch'] = epoch
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch}')
            break
    
    # Calculate training time
    history['training_time'] = time.time() - start_time
    print(f'Training completed in {history["training_time"]:.2f} seconds')
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Save history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    return history

def evaluate_model(model, test_loader, output_dir):
    """
    Evaluate the model on test data
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        output_dir: Directory to save results
    
    Returns:
        dict: Evaluation metrics
    """
    device = next(model.parameters()).device
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics
    metrics = {
        'test_loss': 0.0,
        'test_auc': 0.0,
        'inference_time': 0.0
    }
    
    # Define loss function
    criterion = nn.BCELoss()
    
    # Start timer
    start_time = time.time()
    
    # Evaluate
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.squeeze().cpu().numpy())
    
    # Calculate metrics
    metrics['test_loss'] = running_loss / len(test_loader.dataset)
    metrics['test_auc'] = float(roc_auc_score(all_targets, all_predictions))
    metrics['inference_time'] = time.time() - start_time
    
    # Print metrics
    print(f'Test Loss: {metrics["test_loss"]:.4f}, Test AUC: {metrics["test_auc"]:.4f}')
    print(f'Inference completed in {metrics["inference_time"]:.2f} seconds')
    
    # Save metrics
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics