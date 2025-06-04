import os
import argparse
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt

from data_module import APTOSDataModule
from model import DiabeticRetinopathyModel
from utils import visualize_batch, save_results

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for diabetic retinopathy detection')
    
    # Data paths
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--train_img_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--test_img_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Number of batches to accumulate gradients')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='efficientnet-b4', help='EfficientNet model variant')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for handling imbalance')
    
    # Preprocessing parameters
    parser.add_argument('--apply_clahe', action='store_true', help='Apply CLAHE preprocessing')
    
    return parser.parse_args()

def visualize_sample_images(data_module, output_dir, num_samples=5):
    """Visualize sample images from the training set."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a batch of training data
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Get a batch from each loader
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    # Visualize training samples
    train_images, train_labels = train_batch
    fig = visualize_batch(
        images=[img.permute(1, 2, 0).cpu().numpy() * 255 for img in train_images[:num_samples]],
        labels=train_labels[:num_samples].cpu().numpy(),
        class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'],
        num_images=num_samples
    )
    fig.savefig(os.path.join(output_dir, 'train_samples.png'))
    plt.close(fig)
    
    # Visualize validation samples
    val_images, val_labels = val_batch
    fig = visualize_batch(
        images=[img.permute(1, 2, 0).cpu().numpy() * 255 for img in val_images[:num_samples]],
        labels=val_labels[:num_samples].cpu().numpy(),
        class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'],
        num_images=num_samples
    )
    fig.savefig(os.path.join(output_dir, 'val_samples.png'))
    plt.close(fig)

def main(args):
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data module
    data_module = APTOSDataModule(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        train_img_dir=args.train_img_dir,
        test_img_dir=args.test_img_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        apply_clahe=args.apply_clahe,
        seed=args.seed
    )
    
    # Setup data module
    data_module.setup()
    
    # Visualize sample images
    visualize_sample_images(data_module, args.output_dir)
    
    # Get class weights if needed
    class_weights = data_module.get_class_weights() if args.use_class_weights else None
    
    # Initialize model
    model = DiabeticRetinopathyModel(
        num_classes=5,  # 5 classes for diabetic retinopathy
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        model_name=args.model_name
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='dr-{epoch:02d}-{val_kappa:.4f}',
        monitor='val_kappa',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_kappa',
        patience=5,
        mode='max',
        verbose=True
    )
    
    # Setup logger
    logger = CSVLogger(
        save_dir=args.output_dir,
        name='logs'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision='16-mixed' if torch.cuda.is_available() else '32'
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    test_results = trainer.test(model, data_module, ckpt_path='best')
    
    # Save best model
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    torch.save(model.state_dict(), best_model_path)
    
    # Generate predictions for test set
    trainer.predict(model, data_module.test_dataloader(), ckpt_path='best')
    
    # Save test results
    test_loader = data_module.test_dataloader()
    all_preds = []
    all_img_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, img_ids = batch
            x = x.to(model.device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_img_ids.extend(img_ids)
    
    # Create submission file
    submission = pd.DataFrame({
        'id_code': all_img_ids,
        'diagnosis': all_preds
    })
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"Training completed. Best model saved to {best_model_path}")
    print(f"Test predictions saved to {submission_path}")
    
    return best_model_path, submission_path

if __name__ == '__main__':
    args = parse_args()
    main(args)