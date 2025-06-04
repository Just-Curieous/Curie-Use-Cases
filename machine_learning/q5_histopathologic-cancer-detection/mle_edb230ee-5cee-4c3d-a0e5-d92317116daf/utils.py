import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_training_history(history, output_dir):
    """
    Plot training history.
    
    Args:
        history (dict): Training history
        output_dir (str): Directory to save plots
    """
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot validation AUC
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'training_history.png'))
    plt.close()

def save_experiment_config(config, output_dir):
    """
    Save experiment configuration.
    
    Args:
        config (dict): Experiment configuration
        output_dir (str): Directory to save configuration
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp
    config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save as JSON
    with open(os.path.join(output_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def visualize_predictions(model, data_loader, device, num_samples=5, output_dir=None):
    """
    Visualize model predictions.
    
    Args:
        model (nn.Module): The trained model
        data_loader (DataLoader): Data loader
        device (str): Device to use
        num_samples (int): Number of samples to visualize
        output_dir (str): Directory to save visualizations
    """
    if output_dir:
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    model.eval()
    
    # Get a batch of data
    images, labels = next(iter(data_loader))
    
    # Select a subset of images
    images = images[:num_samples]
    labels = labels[:num_samples] if isinstance(labels, torch.Tensor) else None
    
    # Make predictions
    with torch.no_grad():
        images_tensor = images.to(device)
        outputs = model(images_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    
    # Plot images with predictions
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        
        if labels is not None:
            title = f"True: {labels[i].item()}\nPred: {probs[i]:.4f}"
        else:
            title = f"Pred: {probs[i]:.4f}"
        
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'plots', 'predictions.png'))
        plt.close()
    else:
        plt.show()

def log_gpu_info():
    """
    Log GPU information.
    
    Returns:
        dict: GPU information
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    gpu_info = {
        "gpu_available": True,
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda
    }
    
    return gpu_info