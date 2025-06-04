import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random

from dataset import get_data_loaders
from model import get_model
from train import train_model, evaluate_model, predict
from utils import get_class_weights, plot_confusion_matrix, plot_training_history, save_predictions


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """
    Main function.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Log arguments
    logging.info(f'Arguments: {args}')
    
    # Set random seed
    set_seed(args.seed)
    logging.info(f'Set random seed to {args.seed}')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Get data loaders
    logging.info('Creating data loaders...')
    train_loader, valid_loader, test_loader = get_data_loaders(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        train_img_dir=args.train_img_dir,
        test_img_dir=args.test_img_dir,
        batch_size=args.batch_size,
        valid_size=args.valid_size,
        random_state=args.seed
    )
    logging.info(f'Train size: {len(train_loader.dataset)}, '
                f'Valid size: {len(valid_loader.dataset)}, '
                f'Test size: {len(test_loader.dataset)}')
    
    # Get model
    logging.info('Creating model...')
    model = get_model(num_classes=args.num_classes, device=device)
    
    # Get class weights for handling imbalance
    class_weights = get_class_weights(args.train_csv).to(device)
    logging.info(f'Class weights: {class_weights}')
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6)
    
    # Train model
    if args.train:
        logging.info('Training model...')
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.num_epochs,
            patience=args.patience,
            model_save_path=os.path.join(args.output_dir, 'best_model.pth')
        )
        
        # Plot training history
        logging.info('Plotting training history...')
        plot_training_history(
            history=history,
            save_path=os.path.join(args.output_dir, 'training_history.png')
        )
    else:
        # Load model
        logging.info('Loading model...')
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    logging.info('Evaluating model...')
    metrics = evaluate_model(
        model=model,
        valid_loader=valid_loader,
        criterion=criterion,
        device=device,
        output_dir=args.output_dir
    )
    
    # Plot confusion matrix
    logging.info('Plotting confusion matrix...')
    from train import validate
    val_loss, val_kappa, targets, predictions, _ = validate(model, valid_loader, criterion, device)
    plot_confusion_matrix(
        y_true=targets,
        y_pred=predictions,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Make predictions on test set
    logging.info('Making predictions on test set...')
    ids, preds, probs = predict(
        model=model,
        dataloader=test_loader,
        device=device
    )
    
    # Save predictions
    logging.info('Saving predictions...')
    save_predictions(
        ids=ids,
        preds=preds,
        probs=probs,
        output_file=os.path.join(args.output_dir, 'test_predictions.csv')
    )
    
    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Detection')
    
    # Data paths
    parser.add_argument('--train_csv', type=str, default='/workspace/mle_dataset/train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str, default='/workspace/mle_dataset/test.csv',
                        help='Path to test CSV file')
    parser.add_argument('--train_img_dir', type=str, default='/workspace/mle_dataset/train_images',
                        help='Path to training images directory')
    parser.add_argument('--test_img_dir', type=str, default='/workspace/mle_dataset/test_images',
                        help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, default='/workspace/mle_cb2a24fa-7167-4bd7-a9fb-616fea266e41/output',
                        help='Path to output directory')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--valid_size', type=float, default=0.2,
                        help='Validation set size')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    
    args = parser.parse_args()
    
    main(args)