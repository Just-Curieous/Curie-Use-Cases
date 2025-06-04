import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.config import DEVICE, TEST_CSV, SUBMISSION_PATH
from src.utils import quadratic_weighted_kappa

def evaluate_model(model, val_loader):
    """
    Evaluate the model on the validation set.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    kappa = quadratic_weighted_kappa(all_labels, all_preds)
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    metrics = {
        'accuracy': accuracy,
        'kappa': kappa,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_acc
    }
    
    return metrics

def generate_predictions(model, test_loader):
    """
    Generate predictions for the test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
    
    Returns:
        numpy.ndarray: Predicted classes
    """
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Generating predictions"):
            images = images.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store predictions
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_preds)

def create_submission(predictions):
    """
    Create submission file.
    
    Args:
        predictions: Numpy array of predictions
    """
    # Read test CSV to get image IDs
    test_df = pd.read_csv(TEST_CSV)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id_code': test_df['id_code'],
        'diagnosis': predictions
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    
    # Save submission file
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")