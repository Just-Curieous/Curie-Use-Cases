import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import seaborn as sns
import psutil
import pandas as pd

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    Get the device to use for training.
    
    Returns:
        torch.device: The device to use.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def measure_inference_time(model, dataloader, device, num_runs=100):
    """
    Measure the inference time of a model.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The data loader to use.
        device (torch.device): The device to use.
        num_runs (int): Number of runs to average over.
        
    Returns:
        float: Average inference time per batch in milliseconds.
    """
    model.eval()
    model.to(device)
    
    # Warm-up
    for images, _ in dataloader:
        images = images.to(device)
        with torch.no_grad():
            _ = model(images)
        break
    
    # Measure inference time
    start_time = time.time()
    batch_count = 0
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_runs:
                break
                
            images = images.to(device)
            _ = model(images)
            batch_count += 1
    
    end_time = time.time()
    avg_time_per_batch = (end_time - start_time) * 1000 / batch_count  # in milliseconds
    
    return avg_time_per_batch

def measure_memory_usage(model, dataloader, device):
    """
    Measure the memory usage of a model during inference.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The data loader to use.
        device (torch.device): The device to use.
        
    Returns:
        float: Peak memory usage in MB.
    """
    model.eval()
    model.to(device)
    
    # Record initial memory usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        
    # Run inference
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            break
    
    # Measure peak memory usage
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB
    else:
        peak_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # in MB
    
    return peak_memory

def plot_training_curves(train_losses, val_losses, train_aucs, val_aucs, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        train_losses (list): Training losses.
        val_losses (list): Validation losses.
        train_aucs (list): Training AUC-ROC scores.
        val_aucs (list): Validation AUC-ROC scores.
        save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot AUC-ROC scores
    plt.subplot(1, 2, 2)
    plt.plot(train_aucs, label='Train AUC-ROC')
    plt.plot(val_aucs, label='Validation AUC-ROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC-ROC')
    plt.title('Training and Validation AUC-ROC')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_roc_curve(y_true, y_pred, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities.
        save_path (str, optional): Path to save the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()
    
    return auc

def plot_confusion_matrix(y_true, y_pred_class, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array-like): True labels.
        y_pred_class (array-like): Predicted classes.
        save_path (str, optional): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred_class)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def save_metrics_to_csv(metrics, save_path):
    """
    Save metrics to a CSV file.
    
    Args:
        metrics (dict): Dictionary of metrics.
        save_path (str): Path to save the CSV file.
    """
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)

def log_results(metrics, log_file):
    """
    Log results to a file.
    
    Args:
        metrics (dict): Dictionary of metrics.
        log_file (str): Path to the log file.
    """
    with open(log_file, 'a') as f:
        f.write('\n' + '=' * 50 + '\n')
        f.write('EXPERIMENT RESULTS\n')
        f.write('=' * 50 + '\n')
        
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f'{key}: {value:.4f}\n')
            else:
                f.write(f'{key}: {value}\n')
        
        f.write('=' * 50 + '\n')