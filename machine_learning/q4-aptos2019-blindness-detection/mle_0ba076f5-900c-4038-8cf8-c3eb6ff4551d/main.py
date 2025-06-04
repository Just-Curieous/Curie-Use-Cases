import os
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import get_model
from train import train_model, predict
from utils import plot_confusion_matrix, plot_training_history, save_predictions, calculate_metrics

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get dataloaders
    print('Creating dataloaders...')
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        train_img_dir=args.train_img_dir,
        test_img_dir=args.test_img_dir,
        batch_size=args.batch_size,
        val_split=args.val_split
    )
    print(f'Train size: {len(train_loader.dataset)}, '
          f'Validation size: {len(val_loader.dataset)}, '
          f'Test size: {len(test_loader.dataset)}')
    
    # Get model
    print('Creating model...')
    model = get_model(num_classes=5, device=device)
    
    # Train model
    print('Training model...')
    model, history, val_targets, val_preds = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        patience=args.patience,
        lr=args.learning_rate,
        device=device,
        output_dir=args.output_dir
    )
    
    # Calculate and print validation metrics
    print('Calculating validation metrics...')
    val_metrics = calculate_metrics(val_targets, val_preds)
    print(f'Validation Kappa: {val_metrics["kappa"]:.4f}')
    print(f'Validation Accuracy: {val_metrics["accuracy"]:.4f}')
    print('Per-class Accuracy:')
    for i, acc in enumerate(val_metrics["per_class_accuracy"]):
        print(f'  Class {i}: {acc:.4f}')
    
    # Plot confusion matrix
    print('Plotting confusion matrix...')
    confusion_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(val_targets, val_preds, save_path=confusion_matrix_path)
    
    # Plot training history
    print('Plotting training history...')
    history_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=history_path)
    
    # Make predictions on test set
    print('Making predictions on test set...')
    test_ids, test_preds = predict(model, test_loader, device)
    
    # Save predictions
    print('Saving predictions...')
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    save_predictions(test_ids, test_preds, predictions_path)
    
    # Save metrics to file
    print('Saving metrics...')
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Validation Kappa: {val_metrics["kappa"]:.4f}\n')
        f.write(f'Validation Accuracy: {val_metrics["accuracy"]:.4f}\n')
        f.write('Per-class Accuracy:\n')
        for i, acc in enumerate(val_metrics["per_class_accuracy"]):
            f.write(f'  Class {i}: {acc:.4f}\n')
    
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Detection')
    
    # Data paths
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--train_img_dir', type=str, required=True, help='Path to training images directory')
    parser.add_argument('--test_img_dir', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    main(args)