import os
import time
import torch
from src.config import DEVICE, NUM_EPOCHS, SEED, AUGMENTATION_LEVEL
from src.utils import set_seed
from src.data import create_data_loaders
from src.model import create_model
from src.train import create_trainer, train_model
from src.evaluate import evaluate_model, generate_predictions, create_submission

def run_experiment():
    """Run the complete experiment workflow."""
    start_time = time.time()
    
    # Print experiment configuration
    print(f"Running experiment with the following configuration:")
    print(f"- Device: {DEVICE}")
    print(f"- Seed: {SEED}")
    print(f"- Epochs: {NUM_EPOCHS}")
    print(f"- Augmentation Level: {AUGMENTATION_LEVEL}")
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        augmentation_level=AUGMENTATION_LEVEL
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    model = model.to(DEVICE)
    
    # Create trainer components
    criterion, optimizer, scheduler = create_trainer(model)
    
    # Train model
    print("\nTraining model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS
    )
    
    # Evaluate model on validation set
    print("\nEvaluating model on validation set...")
    metrics = evaluate_model(model, val_loader)
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation Kappa: {metrics['kappa']:.4f}")
    print("Per-class Accuracy:")
    for i, acc in enumerate(metrics['per_class_accuracy']):
        print(f"  Class {i}: {acc:.4f}")
    
    # Generate predictions for test set
    print("\nGenerating predictions for test set...")
    predictions = generate_predictions(model, test_loader)
    
    # Create submission file
    print("\nCreating submission file...")
    create_submission(predictions)
    
    # Print total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    run_experiment()