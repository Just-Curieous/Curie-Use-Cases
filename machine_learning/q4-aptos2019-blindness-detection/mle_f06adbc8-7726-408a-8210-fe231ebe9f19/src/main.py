import os
import argparse
import torch
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from data_loader import get_data_loaders, get_test_loader
from model import get_model
from trainer import Trainer
from evaluation import evaluate_model, plot_confusion_matrix, calculate_metrics, save_results
from visualization import plot_training_history, plot_sample_images

def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    """
    Main function to run the experiment
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device to CPU only
    device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Get data loaders
    print('Loading data...')
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        random_state=args.seed
    )
    print(f'Train set size: {len(train_loader.dataset)}, Val set size: {len(val_loader.dataset)}')
    
    # Plot sample images
    if args.plot_samples:
        print('Plotting sample images...')
        plot_sample_images(train_loader, num_samples=5, save_dir=args.output_dir)
    
    # Create model
    print('Creating model...')
    model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    # Train model
    print('Training model...')
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate
    )
    
    history, best_model_path, conf_matrix, training_time = trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.output_dir
    )
    
    # Plot training history
    print('Plotting training history...')
    plot_training_history(history, save_dir=args.output_dir)
    
    # Load best model
    if best_model_path and os.path.exists(best_model_path):
        print(f'Loading best model from {best_model_path}...')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # Evaluate model
    print('Evaluating model...')
    y_pred, y_true = evaluate_model(model, val_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Plot confusion matrix
    print('Plotting confusion matrix...')
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Save results
    print('Saving results...')
    save_results(
        metrics=metrics,
        conf_matrix=conf_matrix,
        training_time=training_time,
        history=history,
        output_file=args.results_file
    )
    
    print(f'Results saved to {args.results_file}')
    
    # Make predictions on test set if required
    if args.predict_test:
        print('Making predictions on test set...')
        test_loader = get_test_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
        
        from evaluation import predict_test_set
        predict_test_set(
            model=model,
            test_loader=test_loader,
            device=device,
            output_file=os.path.join(args.output_dir, 'test_predictions.csv')
        )
        
        print(f'Test predictions saved to {os.path.join(args.output_dir, "test_predictions.csv")}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Detection')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='/workspace/mle_dataset',
                        help='Path to the data directory')
    parser.add_argument('--output_dir', type=str, default='/workspace/mle_f06adbc8-7726-408a-8210-fe231ebe9f19/output',
                        help='Path to the output directory')
    parser.add_argument('--results_file', type=str, 
                        default='/workspace/mle_f06adbc8-7726-408a-8210-fe231ebe9f19/results_f06adbc8-7726-408a-8210-fe231ebe9f19_control_group_partition_1.txt',
                        help='Path to the results file')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    # Other parameters
    parser.add_argument('--plot_samples', action='store_true',
                        help='Plot sample images')
    parser.add_argument('--predict_test', action='store_true',
                        help='Make predictions on test set')
    
    args = parser.parse_args()
    
    main(args)