import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model import measure_inference_time

def evaluate_and_predict(model, data_loader, device, threshold=0.5):
    """
    Evaluate the model and make predictions.
    
    Args:
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): Data loader
        device (str): Device to use for evaluation
        threshold (float): Threshold for binary classification
    
    Returns:
        tuple: (predictions, true_labels, probabilities)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating', disable=True):
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.extend(probs)
            
            if isinstance(labels, torch.Tensor):  # For validation data
                labels = labels.numpy()
                all_labels.extend(labels)
    
    all_probs = np.array(all_probs).flatten()
    
    if all_labels:  # If we have labels (validation set)
        all_labels = np.array(all_labels)
        all_preds = (all_probs >= threshold).astype(int)
        return all_preds, all_labels, all_probs
    else:  # For test set
        all_preds = (all_probs >= threshold).astype(int)
        return all_preds, None, all_probs

def generate_submission(image_ids, probabilities, output_file):
    """
    Generate a submission file.
    
    Args:
        image_ids (list): List of image IDs
        probabilities (np.array): Predicted probabilities
        output_file (str): Path to save the submission file
    """
    submission = pd.DataFrame({
        'id': image_ids,
        'label': probabilities.flatten()
    })
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")

def evaluate_model_performance(model, val_loader, test_loader, device, output_dir):
    """
    Comprehensive evaluation of model performance.
    
    Args:
        model (nn.Module): The trained model
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        device (str): Device to use for evaluation
        output_dir (str): Directory to save results
    
    Returns:
        dict: Performance metrics
    """
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Evaluate on validation set
    val_preds, val_labels, val_probs = evaluate_and_predict(model, val_loader, device)
    
    # Calculate metrics
    val_auc = roc_auc_score(val_labels, val_probs)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(val_labels, val_preds)
    
    # Classification report
    class_report = classification_report(val_labels, val_preds, output_dict=True)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(val_labels, val_probs)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'AUC = {val_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'plots', 'roc_curve.png'))
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'plots', 'confusion_matrix.png'))
    plt.close()
    
    # Measure inference time
    inference_time = measure_inference_time(model, device=device)
    
    # Generate predictions for test set
    test_preds, _, test_probs = evaluate_and_predict(model, test_loader, device)
    
    # Get test image IDs
    test_image_ids = []
    for _, ids in test_loader:
        if isinstance(ids, torch.Tensor):
            ids = [str(i.item()) for i in ids]
        test_image_ids.extend(ids)
    
    # Generate submission file
    submission_file = os.path.join(output_dir, 'submission.csv')
    generate_submission(test_image_ids, test_probs, submission_file)
    
    # Compile metrics
    metrics = {
        'val_auc': val_auc,
        'accuracy': class_report['accuracy'],
        'precision': class_report['1']['precision'],
        'recall': class_report['1']['recall'],
        'f1_score': class_report['1']['f1-score'],
        'inference_time_ms': inference_time
    }
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    return metrics