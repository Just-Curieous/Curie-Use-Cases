import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_training_history(history, save_dir=None):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        save_dir: Directory to save the plots
    """
    # Plot training & validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        plt.close()
    else:
        plt.show()
    
    # Plot validation kappa
    plt.figure(figsize=(8, 4))
    plt.plot(history['val_kappa'])
    plt.title('Validation Quadratic Weighted Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('Kappa')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'validation_kappa.png'))
        plt.close()
    else:
        plt.show()

def plot_sample_images(data_loader, num_samples=5, save_dir=None):
    """
    Plot sample images from the data loader
    
    Args:
        data_loader: Data loader
        num_samples: Number of samples to plot
        save_dir: Directory to save the plot
    """
    # Get a batch of images
    images, labels = next(iter(data_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Convert from tensor to numpy array
    images = images.cpu().numpy()
    
    # Denormalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std.reshape(1, 3, 1, 1) * images + mean.reshape(1, 3, 1, 1)
    
    # Move color channel from index 1 to index 3
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Clip values to [0, 1]
    images = np.clip(images, 0, 1)
    
    # Plot images
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(images[i])
        axes[i].set_title(f'Class: {labels[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'sample_images.png'))
        plt.close()
    else:
        plt.show()