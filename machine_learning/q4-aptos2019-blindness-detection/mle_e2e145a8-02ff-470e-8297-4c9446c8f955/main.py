import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import get_data_loaders
from models.model import get_model, get_model_size, measure_inference_time
from utils.train_utils import train_and_validate, validate, predict, calculate_metrics

def set_seed(seed=42):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiment(args):
    """
    Run the diabetic retinopathy detection experiment
    
    Args:
        args: Command line arguments
        
    Returns:
        results: Dictionary of experiment results
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'fold_results': [],
        'overall_metrics': {},
        'training_time': 0,
        'inference_time': 0,
        'model_size_mb': 0
    }
    
    # Cross-validation
    all_val_preds = []
    all_val_labels = []
    total_training_time = 0
    
    for fold in range(args.num_folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{args.num_folds}")
        print(f"{'='*50}")
        
        # Get data loaders for this fold
        train_loader, valid_loader, test_loader = get_data_loaders(
            args.train_csv, args.test_csv, args.train_img_dir, args.test_img_dir,
            fold=fold, num_folds=args.num_folds, batch_size=args.batch_size,
            num_workers=args.num_workers, seed=args.seed
        )
        
        # Create model
        model = get_model(num_classes=args.num_classes, pretrained=True, device=device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        
        # Train and validate
        history, best_model_state, best_metrics, training_time = train_and_validate(
            model, train_loader, valid_loader, criterion, optimizer, device,
            num_epochs=args.num_epochs, scheduler=scheduler
        )
        
        # Add to total training time
        total_training_time += training_time
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Get validation predictions and labels directly using validate function
        val_loss, val_preds, val_labels = validate(model, valid_loader, criterion, device)
        
        # Check if validate function returned valid results
        if val_preds is not None and val_labels is not None:
            # Extend overall validation predictions and labels
            all_val_preds.extend(val_preds)
            all_val_labels.extend(val_labels)
        else:
            print(f"Warning: validate function returned None for predictions or labels in fold {fold}")
            # Use the best metrics from training instead
            val_preds = []
            val_labels = []
            for batch in valid_loader:
                images, labels = batch
                images = images.to(device)
                with torch.no_grad():
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())
            all_val_preds.extend(val_preds)
            all_val_labels.extend(val_labels)
        
        # Save fold results
        fold_result = {
            'fold': fold,
            'best_val_accuracy': best_metrics['accuracy'],
            'best_val_kappa': best_metrics['quadratic_weighted_kappa'],
            'confusion_matrix': best_metrics['confusion_matrix'].tolist(),
            'training_time': training_time
        }
        results['fold_results'].append(fold_result)
        
        # Save model for this fold
        torch.save(model.state_dict(), os.path.join(args.results_dir, f"model_fold_{fold}.pt"))
        
        print(f"Fold {fold+1} completed. Best validation kappa: {best_metrics['quadratic_weighted_kappa']:.4f}")
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(np.array(all_val_preds), np.array(all_val_labels))
    results['overall_metrics'] = {
        'accuracy': float(overall_metrics['accuracy']),
        'quadratic_weighted_kappa': float(overall_metrics['quadratic_weighted_kappa']),
        'confusion_matrix': overall_metrics['confusion_matrix'].tolist()
    }
    
    # Calculate average training time
    results['training_time'] = total_training_time / args.num_folds
    
    # Create a final model for inference time and size measurement
    final_model = get_model(num_classes=args.num_classes, pretrained=True, device=device)
    
    # Measure inference time
    inference_time = measure_inference_time(final_model, device=device)
    results['inference_time'] = float(inference_time)
    
    # Measure model size
    model_size = get_model_size(final_model)
    results['model_size_mb'] = float(model_size)
    
    # Save results
    with open(os.path.join(args.results_dir, 'experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Detection')
    
    # Data paths
    parser.add_argument('--train_csv', type=str, required=True, help='Path to train CSV file')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--train_img_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--test_img_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save results')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_experiment(args)
    
    # Print summary
    print("\nExperiment completed!")
    print(f"Overall accuracy: {results['overall_metrics']['accuracy']:.4f}")
    print(f"Overall quadratic weighted kappa: {results['overall_metrics']['quadratic_weighted_kappa']:.4f}")
    print(f"Average training time: {results['training_time']:.2f} seconds")
    print(f"Inference time: {results['inference_time']:.2f} ms")
    print(f"Model size: {results['model_size_mb']:.2f} MB")

if __name__ == '__main__':
    main()