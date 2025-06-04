import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from data_loader import get_data_loaders, get_external_test_loader
from model import EfficientNetB0Model
from train import train_model
from evaluate import evaluate_model, generate_predictions_for_submission
from utils import set_seed, get_device, plot_training_curves, plot_roc_curve, plot_confusion_matrix, save_metrics_to_csv, log_results

def main(args):
    # Create output directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"experiment_log_{timestamp}.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"Experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Using device: {device}\n")
        f.write("\nExperiment Configuration:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, train_df, val_df, test_df = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed
    )
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    with open(log_file, 'a') as f:
        f.write(f"Train set size: {len(train_df)}\n")
        f.write(f"Validation set size: {len(val_df)}\n")
        f.write(f"Test set size: {len(test_df)}\n\n")
    
    # Create model
    print("Creating model...")
    model = EfficientNetB0Model(pretrained=True)
    
    # Define model save path
    model_save_path = os.path.join(args.model_dir, f"efficientnet_b0_{timestamp}.pt")
    
    # Train model
    print("Training model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        patience=args.patience,
        model_save_path=model_save_path,
        log_file=log_file
    )
    
    # Plot training curves
    plot_save_path = os.path.join(args.log_dir, f"training_curves_{timestamp}.png")
    plot_training_curves(
        train_losses=history['train_loss'],
        val_losses=history['val_loss'],
        train_aucs=history['train_auc'],
        val_aucs=history['val_auc'],
        save_path=plot_save_path
    )
    
    # Evaluate model on test set
    print("Evaluating model...")
    metrics, y_true, y_pred, y_pred_binary = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=args.threshold,
        log_file=log_file
    )
    
    # Plot ROC curve
    roc_save_path = os.path.join(args.log_dir, f"roc_curve_{timestamp}.png")
    plot_roc_curve(y_true, y_pred, save_path=roc_save_path)
    
    # Plot confusion matrix
    cm_save_path = os.path.join(args.log_dir, f"confusion_matrix_{timestamp}.png")
    plot_confusion_matrix(y_true, y_pred_binary, save_path=cm_save_path)
    
    # Save metrics to CSV
    metrics_save_path = os.path.join(args.log_dir, f"metrics_{timestamp}.csv")
    save_metrics_to_csv(metrics, metrics_save_path)
    
    # Log results
    log_results(metrics, log_file)
    
    # Generate predictions for external test set if requested
    if args.generate_submission:
        print("Generating predictions for external test set...")
        external_test_loader = get_external_test_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        submission_file = os.path.join(args.log_dir, f"submission_{timestamp}.csv")
        generate_predictions_for_submission(
            model=model,
            test_loader=external_test_loader,
            device=device,
            submission_file=submission_file
        )
    
    print(f"Experiment completed. Results saved to {args.log_dir}")
    with open(log_file, 'a') as f:
        f.write(f"\nExperiment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate EfficientNetB0 model on PCam dataset")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="/workspace/mle_dataset", 
                        help="Path to the dataset directory")
    parser.add_argument("--val_size", type=float, default=0.15, 
                        help="Proportion of training data to use for validation")
    parser.add_argument("--test_size", type=float, default=0.15, 
                        help="Proportion of training data to use for testing")
    
    # Model parameters
    parser.add_argument("--model_dir", type=str, default="/workspace/mle_a6604f22-7b59-46c5-9a9c-4953fbd2e0f0/models", 
                        help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="/workspace/mle_a6604f22-7b59-46c5-9a9c-4953fbd2e0f0/logs", 
                        help="Directory to save logs and results")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=50, 
                        help="Maximum number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate for optimizer")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Classification threshold")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of workers for data loading")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--generate_submission", action="store_true", 
                        help="Generate predictions for external test set")
    
    args = parser.parse_args()
    main(args)