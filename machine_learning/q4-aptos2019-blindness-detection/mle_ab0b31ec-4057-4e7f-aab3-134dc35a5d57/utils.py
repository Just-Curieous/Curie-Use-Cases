import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score
import pandas as pd

def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image."""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the original A and B channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to RGB color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return enhanced_img

def load_and_preprocess_image(image_path, size=(380, 380), apply_clahe_enhancement=True):
    """Load and preprocess an image."""
    try:
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if apply_clahe_enhancement:
            image = apply_clahe(image)
        
        # Resize
        image = cv2.resize(image, size)
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        # Return a blank image as fallback
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)

def visualize_batch(images, labels=None, predictions=None, class_names=None, num_images=5, figsize=(15, 10)):
    """Visualize a batch of images with optional labels and predictions."""
    num_images = min(num_images, len(images))
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    for i in range(num_images):
        img = images[i]
        
        # If image is a tensor, convert to numpy
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
            
            # Denormalize if needed
            if img.max() <= 1.0:
                img = img * 255
                
        img = img.astype(np.uint8)
        
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
            
        ax.imshow(img)
        
        title = ""
        if labels is not None and i < len(labels):
            label = labels[i]
            if class_names is not None:
                label_name = class_names[label]
                title += f"Label: {label_name}"
            else:
                title += f"Label: {label}"
                
        if predictions is not None and i < len(predictions):
            pred = predictions[i]
            if class_names is not None:
                pred_name = class_names[pred]
                title += f"\nPred: {pred_name}"
            else:
                title += f"\nPred: {pred}"
                
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    return fig

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate various metrics for model evaluation."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Quadratic weighted kappa (primary metric for this task)
    metrics['quadratic_weighted_kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Extract metrics from the report
    for label, values in report.items():
        if label.isdigit() or label in ['macro avg', 'weighted avg']:
            for metric, value in values.items():
                if metric != 'support':
                    metrics[f"{label}_{metric}"] = value
    
    return metrics

def save_results(metrics, predictions, true_labels, output_dir, filename_prefix="results"):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, f"{filename_prefix}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save predictions
    preds_df = pd.DataFrame({
        'true_label': true_labels,
        'prediction': predictions
    })
    preds_path = os.path.join(output_dir, f"{filename_prefix}_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    
    # Create and save confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    class_names = [str(i) for i in range(len(np.unique(true_labels)))]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{filename_prefix}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    return metrics_path, preds_path, cm_path