import os
import argparse
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from data_module import APTOSDataModule
from model import DiabeticRetinopathyModel
from utils import save_results, visualize_batch

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained model for diabetic retinopathy detection')
    
    # Data paths
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--train_img_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--test_img_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--apply_clahe', action='store_true', help='Apply CLAHE preprocessing')
    
    return parser.parse_args()

def visualize_predictions(data_loader, model, output_dir, num_samples=10):
    """Visualize model predictions on sample images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a batch of data
    batch = next(iter(data_loader))
    images, labels = batch
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        logits = model(images.to(model.device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    # Visualize
    fig = visualize_batch(
        images=[img.permute(1, 2, 0).cpu().numpy() * 255 for img in images[:num_samples]],
        labels=labels[:num_samples].cpu().numpy(),
        predictions=preds[:num_samples],
        class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'],
        num_images=num_samples,
        figsize=(20, 10)
    )
    fig.savefig(os.path.join(output_dir, 'prediction_samples.png'))
    plt.close(fig)

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data module
    data_module = APTOSDataModule(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        train_img_dir=args.train_img_dir,
        test_img_dir=args.test_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        apply_clahe=args.apply_clahe
    )
    
    # Setup data module
    data_module.setup()
    
    # Load model
    model = DiabeticRetinopathyModel.load_from_checkpoint(args.model_path)
    model.eval()
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Evaluate on validation set
    val_loader = data_module.val_dataloader()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # Create metrics dictionary
    metrics = {
        'quadratic_weighted_kappa': kappa,
        'accuracy': accuracy
    }
    
    # Add detailed metrics from classification report
    for label, values in report.items():
        if label.isdigit() or label in ['macro avg', 'weighted avg']:
            for metric, value in values.items():
                if metric != 'support':
                    metrics[f"{label}_{metric}"] = value
    
    # Save results
    metrics_path, preds_path, cm_path = save_results(
        metrics,
        all_preds,
        all_labels,
        args.output_dir,
        filename_prefix="validation_results"
    )
    
    # Visualize predictions
    visualize_predictions(val_loader, model, args.output_dir)
    
    # Generate predictions for test set
    test_loader = data_module.test_dataloader()
    test_preds = []
    test_img_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            images, img_ids = batch
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_img_ids.extend(img_ids)
    
    # Create submission file
    submission = pd.DataFrame({
        'id_code': test_img_ids,
        'diagnosis': test_preds
    })
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"Evaluation completed.")
    print(f"Validation metrics:")
    print(f"  Quadratic Weighted Kappa: {kappa:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"Results saved to {args.output_dir}")
    
    return metrics, submission_path

if __name__ == '__main__':
    args = parse_args()
    main(args)