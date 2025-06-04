import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from utils import measure_inference_time, measure_memory_usage, plot_roc_curve, plot_confusion_matrix
from model import get_model_complexity, get_model_size_mb

def evaluate_model(model, test_loader, device, threshold=0.5, log_file=None):
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to use for evaluation.
        threshold (float): Classification threshold.
        log_file (str): Path to log file.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Store predictions and targets
            if isinstance(labels, torch.Tensor):
                all_preds.extend(outputs.squeeze().cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    auc_roc = roc_auc_score(all_targets, all_preds)
    
    # Convert probabilities to binary predictions
    all_preds_binary = (all_preds >= threshold).astype(int)
    
    accuracy = accuracy_score(all_targets, all_preds_binary)
    precision = precision_score(all_targets, all_preds_binary)
    recall = recall_score(all_targets, all_preds_binary)
    f1 = f1_score(all_targets, all_preds_binary)
    
    # Measure inference time
    inference_time = measure_inference_time(model, test_loader, device)
    
    # Measure memory usage
    memory_usage = measure_memory_usage(model, test_loader, device)
    
    # Calculate model complexity
    flops, params = get_model_complexity(model)
    model_size = get_model_size_mb(model)
    
    # Create metrics dictionary
    metrics = {
        'auc_roc': auc_roc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'inference_time_ms': inference_time,
        'memory_usage_mb': memory_usage,
        'flops': flops,
        'params': params,
        'model_size_mb': model_size
    }
    
    # Print metrics
    print('\nTest Metrics:')
    print(f'AUC-ROC: {auc_roc:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Inference Time: {inference_time:.2f} ms per batch')
    print(f'Memory Usage: {memory_usage:.2f} MB')
    print(f'FLOPs: {flops/1e9:.2f} G')
    print(f'Parameters: {params/1e6:.2f} M')
    print(f'Model Size: {model_size:.2f} MB')
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write('\nTest Metrics:\n')
            f.write(f'AUC-ROC: {auc_roc:.4f}\n')
            f.write(f'Accuracy: {accuracy:.4f}\n')
            f.write(f'Precision: {precision:.4f}\n')
            f.write(f'Recall: {recall:.4f}\n')
            f.write(f'F1 Score: {f1:.4f}\n')
            f.write(f'Inference Time: {inference_time:.2f} ms per batch\n')
            f.write(f'Memory Usage: {memory_usage:.2f} MB\n')
            f.write(f'FLOPs: {flops/1e9:.2f} G\n')
            f.write(f'Parameters: {params/1e6:.2f} M\n')
            f.write(f'Model Size: {model_size:.2f} MB\n')
    
    return metrics, all_targets, all_preds, all_preds_binary

def generate_predictions_for_submission(model, test_loader, device, submission_file):
    """
    Generate predictions for submission.
    
    Args:
        model (nn.Module): The model to use.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to use for inference.
        submission_file (str): Path to save the submission file.
    """
    model.eval()
    model = model.to(device)
    
    predictions = {}
    
    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Store predictions
            for i, img_id in enumerate(img_ids):
                predictions[img_id] = outputs[i].item()
    
    # Write predictions to file
    with open(submission_file, 'w') as f:
        f.write('id,label\n')
        for img_id, pred in predictions.items():
            f.write(f'{img_id},{pred}\n')
    
    print(f'Predictions saved to {submission_file}')