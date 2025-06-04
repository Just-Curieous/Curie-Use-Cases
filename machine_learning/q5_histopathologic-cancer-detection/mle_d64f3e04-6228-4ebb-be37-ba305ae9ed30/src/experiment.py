import os
import argparse
import json
import torch
from datetime import datetime

from data import load_data
from model import get_resnet18_model, get_device
from train import train_model, evaluate_model

def run_experiment(config):
    """
    Run the experiment with the given configuration
    
    Args:
        config (dict): Experiment configuration
    
    Returns:
        dict: Experiment results
    """
    print("Starting experiment with configuration:")
    print(json.dumps(config, indent=4))
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Get device
    device = get_device()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        test_split=config['test_split'],
        num_workers=config['num_workers'],
        seed=config['seed']
    )
    
    # Create model
    print("Creating model...")
    model = get_resnet18_model(pretrained=config['pretrained'])
    model = model.to(device)
    
    # Train model
    print("Training model...")
    training_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=config['output_dir'],
        lr=config['learning_rate'],
        num_epochs=config['num_epochs'],
        patience=config['patience']
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        output_dir=config['output_dir']
    )
    
    # Combine results
    results = {
        'training_history': training_history,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results
    with open(os.path.join(config['output_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Experiment completed successfully!")
    return results

def main():
    parser = argparse.ArgumentParser(description='Run PCam experiment')
    parser.add_argument('--data_dir', type=str, default='/workspace/mle_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='/workspace/mle_d64f3e04-6228-4ebb-be37-ba305ae9ed30/output',
                        help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'seed': args.seed,
        'val_split': 0.15,
        'test_split': 0.15,
        'num_workers': 4,
        'pretrained': True
    }
    
    # Run experiment
    run_experiment(config)

if __name__ == '__main__':
    main()