import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        labels: List or array of class labels
        
    Returns:
        weights: Tensor of class weights
    """
    # Count samples per class
    class_counts = np.bincount(labels)
    
    # Calculate weights as inverse of frequency
    weights = 1.0 / class_counts
    
    # Normalize weights
    weights = weights / weights.sum() * len(class_counts)
    
    return torch.FloatTensor(weights)

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate quadratic weighted kappa score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        kappa: Quadratic weighted kappa score
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def calculate_metrics(y_true, y_pred):
    """
    Calculate various metrics for evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Calculate metrics
    kappa = quadratic_weighted_kappa(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    metrics = {
        'kappa': kappa,
        'accuracy': acc,
        'per_class_accuracy': per_class_acc
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
                yticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot kappa
    plt.subplot(1, 2, 2)
    plt.plot(history['train_kappa'], label='Train Kappa')
    plt.plot(history['val_kappa'], label='Validation Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('Quadratic Weighted Kappa')
    plt.title('Training and Validation Kappa')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_predictions(ids, preds, output_path):
    """
    Save predictions to CSV file
    
    Args:
        ids: List of image IDs
        preds: List of predictions
        output_path: Path to save the CSV file
    """
    # Create DataFrame
    df = pd.DataFrame({
        'id_code': ids,
        'diagnosis': preds
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df