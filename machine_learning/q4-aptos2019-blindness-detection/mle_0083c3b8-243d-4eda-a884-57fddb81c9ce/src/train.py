import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_PATH
)
from src.utils import quadratic_weighted_kappa

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=NUM_EPOCHS):
    """
    Train the model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of epochs to train for
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Initialize history dictionary to store metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_kappa': []
    }
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Initialize best validation metrics
    best_val_loss = float('inf')
    best_val_kappa = -float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Store predictions and labels for metrics calculation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # Calculate validation accuracy
        val_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        history['val_accuracy'].append(val_accuracy)
        
        # Calculate quadratic weighted kappa
        val_kappa = quadratic_weighted_kappa(all_labels, all_preds)
        history['val_kappa'].append(val_kappa)
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step(val_kappa)  # Pass the validation kappa score to the scheduler
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, "
              f"Val Kappa: {val_kappa:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model based on validation kappa score
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved new best model with Kappa: {val_kappa:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH))
    
    return model, history

def create_trainer(model):
    """
    Create trainer components.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (criterion, optimizer, scheduler)
    """
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    return criterion, optimizer, scheduler