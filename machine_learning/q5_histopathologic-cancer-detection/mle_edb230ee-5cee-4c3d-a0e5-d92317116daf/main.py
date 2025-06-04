import os
import argparse
import torch
import json
from datetime import datetime

from data_loader import get_data_loaders
from model import CancerDetectionModel, measure_inference_time
from train import train_model
from evaluate import evaluate_model_performance
from utils import set_seed, plot_training_history, save_experiment_config, visualize_predictions, log_gpu_info

def main(args):
    """
    Main function to run the experiment.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log GPU information
    gpu_info = log_gpu_info()
    print(f"GPU Information: {json.dumps(gpu_info, indent=2)}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {args.model_name} model...")
    model = CancerDetectionModel(model_name=args.model_name, pretrained=args.pretrained)
    model = model.to(device)
    
    # Save experiment configuration
    config = vars(args)
    config.update(gpu_info)
    config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_experiment_config(config, args.output_dir)
    
    # Train model
    if not args.skip_training:
        print("Training model...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            save_dir=args.output_dir
        )
        
        # Plot training history
        plot_training_history(history, args.output_dir)
    else:
        # Load pre-trained model
        print(f"Loading pre-trained model from {os.path.join(args.output_dir, 'models', 'best_model.pth')}")
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'models', 'best_model.pth')))
    
    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(model, val_loader, device, num_samples=5, output_dir=args.output_dir)
    
    # Evaluate model performance
    print("Evaluating model performance...")
    metrics = evaluate_model_performance(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir
    )
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    print(f"\nExperiment completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancer Detection Experiment")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="/workspace/mle_dataset",
                        help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="/workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf/results",
                        help="Directory to save results")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0"],
                        help="Model architecture to use")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and load pre-trained model")
    
    args = parser.parse_args()
    
    main(args)