import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import seaborn as sns


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate various metrics for model evaluation.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels
        y_prob (numpy.ndarray, optional): Predicted probabilities
        
    Returns:
        dict: Dictionary containing various metrics
    """
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Specific focus on severe (class 3) and proliferative (class 4) cases
    metrics['severe_precision'] = metrics['precision_per_class'][3] if 3 < len(metrics['precision_per_class']) else 0
    metrics['severe_recall'] = metrics['recall_per_class'][3] if 3 < len(metrics['recall_per_class']) else 0
    metrics['severe_f1'] = metrics['f1_per_class'][3] if 3 < len(metrics['f1_per_class']) else 0
    
    metrics['proliferative_precision'] = metrics['precision_per_class'][4] if 4 < len(metrics['precision_per_class']) else 0
    metrics['proliferative_recall'] = metrics['recall_per_class'][4] if 4 < len(metrics['recall_per_class']) else 0
    metrics['proliferative_f1'] = metrics['f1_per_class'][4] if 4 < len(metrics['f1_per_class']) else 0
    
    # Quadratic weighted kappa
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels
        save_path (str, optional): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
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
    Plot training history.
    
    Args:
        history (dict): Dictionary containing training history
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
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


def get_class_weights(train_csv):
    """
    Calculate class weights to handle class imbalance.
    
    Args:
        train_csv (str): Path to training CSV file
        
    Returns:
        torch.Tensor: Class weights
    """
    df = pd.read_csv(train_csv)
    class_counts = df['diagnosis'].value_counts().sort_index()
    total_samples = len(df)
    
    # Calculate weights as inverse of frequency
    weights = total_samples / (len(class_counts) * class_counts)
    
    # Convert to tensor
    weights = torch.tensor(weights.values, dtype=torch.float)
    
    return weights


def save_predictions(ids, preds, probs, output_file):
    """
    Save predictions to a CSV file.
    
    Args:
        ids (list): List of image IDs
        preds (numpy.ndarray): Predicted labels
        probs (numpy.ndarray): Predicted probabilities
        output_file (str): Path to output file
    """
    df = pd.DataFrame({
        'id_code': ids,
        'diagnosis': preds
    })
    
    # Add probability columns
    for i in range(probs.shape[1]):
        df[f'prob_class_{i}'] = probs[:, i]
    
    df.to_csv(output_file, index=False)
    
    # Also create a submission file in the required format
    submission = pd.DataFrame({
        'id_code': ids,
        'diagnosis': preds
    })
    submission.to_csv(output_file.replace('.csv', '_submission.csv'), index=False)