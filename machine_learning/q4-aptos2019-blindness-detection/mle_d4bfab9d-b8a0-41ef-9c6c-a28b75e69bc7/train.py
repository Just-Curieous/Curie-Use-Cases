import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_data_loaders, get_class_weights
from model import get_model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs=10, early_stopping_patience=3, output_dir=None):
    """
    Train the model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use ('cuda' or 'cpu')
        num_epochs: Number of epochs to train for
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        output_dir: Directory to save model checkpoints
        
    Returns:
        model: The trained model
        history: Dictionary containing training history
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_kappa': []
    }
    
    best_val_kappa = -1
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar but disable it to avoid excessive output
        for inputs, labels in tqdm(train_loader, desc="Training", disable=True):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", disable=True):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        val_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_acc)
        history['val_kappa'].append(val_kappa)
        
        print(f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        print(f'Val Acc: {val_acc:.4f}, Val Kappa: {val_kappa:.4f}')
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Save best model
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            early_stopping_counter = 0
            
            if output_dir is not None:
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f'Saved best model with kappa: {val_kappa:.4f}')
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
        
        print()
    
    # Load best model
    if output_dir is not None:
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    
    return model, history

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", disable=True):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = accuracy_score(all_labels, all_preds)
    test_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Calculate per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    metrics = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_kappa': test_kappa,
        'per_class_acc': per_class_acc,
        'confusion_matrix': cm
    }
    
    return metrics

def predict(model, test_loader, device):
    """
    Make predictions on the test set
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        predictions: Dictionary mapping image IDs to predicted classes
    """
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for inputs, img_ids in tqdm(test_loader, desc="Predicting", disable=True):
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store predictions
            for img_id, pred in zip(img_ids, preds.cpu().numpy()):
                predictions[img_id] = int(pred)
    
    return predictions

def save_confusion_matrix(cm, output_dir):
    """
    Save confusion matrix as an image
    
    Args:
        cm: Confusion matrix
        output_dir: Directory to save the image
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_metrics(metrics, history, output_dir):
    """
    Save metrics and training history
    
    Args:
        metrics: Dictionary containing evaluation metrics
        history: Dictionary containing training history
        output_dir: Directory to save the metrics
    """
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Test Loss: {metrics['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {metrics['test_acc']:.4f}\n")
        f.write(f"Test Quadratic Weighted Kappa: {metrics['test_kappa']:.4f}\n")
        f.write("\nPer-Class Accuracy:\n")
        for i, acc in enumerate(metrics['per_class_acc']):
            f.write(f"Class {i}: {acc:.4f}\n")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Accuracy')
    plt.plot(history['val_kappa'], label='Kappa')
    plt.title('Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def save_submission(predictions, output_dir):
    """
    Save predictions as a submission file
    
    Args:
        predictions: Dictionary mapping image IDs to predicted classes
        output_dir: Directory to save the submission file
    """
    submission = pd.DataFrame({
        'id_code': list(predictions.keys()),
        'diagnosis': list(predictions.values())
    })
    
    submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)

def main(train_csv, test_csv, train_img_dir, test_img_dir, output_dir, batch_size=32, num_epochs=10):
    """
    Main function to run the training and evaluation pipeline
    
    Args:
        train_csv: Path to the training CSV file
        test_csv: Path to the test CSV file
        train_img_dir: Path to the training images directory
        test_img_dir: Path to the test images directory
        output_dir: Directory to save outputs
        batch_size: Batch size for training
        num_epochs: Number of epochs to train for
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_csv, test_csv, train_img_dir, test_img_dir, batch_size
    )
    
    # Get class weights for weighted loss
    class_weights = get_class_weights(train_csv).to(device)
    print(f"Class weights: {class_weights}")
    
    # Get model
    model = get_model(num_classes=5, device=device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, num_epochs=num_epochs, early_stopping_patience=3, output_dir=output_dir
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model on validation set
    print("Evaluating model...")
    metrics = evaluate_model(model, val_loader, criterion, device)
    
    # Save confusion matrix
    save_confusion_matrix(metrics['confusion_matrix'], output_dir)
    
    # Save metrics and history
    save_metrics(metrics, history, output_dir)
    
    # Make predictions on test set
    print("Making predictions on test set...")
    predictions = predict(model, test_loader, device)
    
    # Save submission file
    save_submission(predictions, output_dir)
    
    print("Done!")