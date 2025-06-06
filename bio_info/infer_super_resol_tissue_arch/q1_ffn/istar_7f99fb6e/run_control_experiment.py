import os
import sys
import time
import torch
import numpy as np
from flexible_nn import FlexibleNeuralNetwork
from impute import get_data, normalize, SpotDataset, predict
from utils import read_string

def main():
    # Record start time
    start_time = time.time()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set parameters for control configuration
    prefix = sys.argv[1]
    epochs = 400
    batch_size = 27
    hidden_layers = "256,256,256,256"
    learning_rate = 0.0001
    
    # Get data
    print("Loading data...")
    embs, cnts, locs = get_data(prefix)
    
    # Get radius
    factor = 16
    radius = int(read_string(f'{prefix}radius.txt'))
    radius = radius / factor
    
    # Normalize data
    print("Normalizing data...")
    embs, cnts, embs_norm, cnts_norm = normalize(embs, cnts)
    
    # Create dataset
    print("Creating dataset...")
    try:
        dataset = SpotDataset(embs, cnts, locs, radius)
        print(f"Dataset created with {len(dataset)} samples")
    except KeyError as e:
        print(f"Error creating dataset: {e}")
        print("Attempting to fix the issue...")
        # Handle potential KeyError by ensuring data is properly formatted
        if isinstance(cnts, np.ndarray):
            print("cnts is already a numpy array")
        else:
            print("Converting cnts to numpy array")
            cnts = cnts.to_numpy()
        
        # Try creating the dataset again
        dataset = SpotDataset(embs, cnts, locs, radius)
        print(f"Dataset created with {len(dataset)} samples")
    
    # Create model
    print("Creating model with baseline configuration:")
    print(f"  - Architecture: 4 layers of 256 units each")
    print(f"  - Activation: LeakyReLU(0.1)")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Optimizer: Adam")
    print(f"  - No regularization")
    
    model_kwargs = {
        'n_inp': embs.shape[-1],
        'n_out': cnts.shape[-1] if isinstance(cnts, np.ndarray) else cnts.shape[1],
        'hidden_layers': hidden_layers,
        'activation': 'leaky_relu',
        'activation_params': {'negative_slope': 0.1},
        'output_activation': 'elu',
        'output_activation_params': {'alpha': 0.01, 'beta': 0.01},
        'learning_rate': learning_rate,
        'optimizer': 'adam'
    }
    
    # Create model directory
    os.makedirs(f'{prefix}states/00/', exist_ok=True)
    
    # Import train module
    from train import train_model
    
    # Train model
    print(f"Training model for {epochs} epochs with batch size {batch_size}...")
    model = FlexibleNeuralNetwork(**model_kwargs)
    model, history, trainer = train_model(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        device=device
    )
    
    # Save model
    checkpoint_file = f'{prefix}states/00/model.pt'
    trainer.save_checkpoint(checkpoint_file)
    print(f'Model saved to {checkpoint_file}')
    
    # Prepare for prediction
    names = cnts.columns if hasattr(cnts, 'columns') else np.arange(cnts.shape[-1])
    cnts_range = np.stack([cnts_norm[0], cnts_norm[1]], -1)
    mask_size = dataset.mask.sum()
    cnts_range /= mask_size
    
    # Prepare batches for prediction
    batch_size_row = 50
    n_batches_row = embs.shape[0] // batch_size_row + 1
    embs_batches = np.array_split(embs, n_batches_row)
    
    # Make predictions
    print("Making predictions...")
    predict(
        model_states=[model],
        x_batches=embs_batches,
        name_list=names,
        y_range=cnts_range,
        prefix=prefix,
        device=device
    )
    
    # Calculate and report execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Experiment completed successfully in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    # Write final summary
    print("\n===== FINAL RESULTS =====")
    print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"Model saved to: {checkpoint_file}")

if __name__ == "__main__":
    main()
