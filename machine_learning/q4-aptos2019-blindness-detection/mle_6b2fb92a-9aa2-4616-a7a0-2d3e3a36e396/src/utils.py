import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def quadratic_weighted_kappa(y_true, y_pred):
    """Calculate quadratic weighted kappa."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def save_summary(summary, file_path):
    """Save summary results to a text file."""
    with open(file_path, 'w') as f:
        f.write("Diabetic Retinopathy Detection - Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Average Validation Kappa: {summary['avg_val_kappa']:.4f}\n")
        f.write(f"Average Generalization Gap: {summary['avg_gen_gap']:.4f}\n\n")
        
        f.write("Fold Results:\n")
        f.write("-" * 50 + "\n")
        
        for i, fold_result in enumerate(summary['fold_results']):
            f.write(f"Fold {i}:\n")
            f.write(f"  Best Epoch: {fold_result['best_epoch']}\n")
            f.write(f"  Best Validation Kappa: {fold_result['best_val_kappa']:.4f}\n")
            f.write(f"  Final Train Kappa: {fold_result['final_train_kappa']:.4f}\n")
            f.write(f"  Final Validation Kappa: {fold_result['final_val_kappa']:.4f}\n")
            f.write(f"  Generalization Gap: {fold_result['generalization_gap']:.4f}\n")
            f.write("-" * 50 + "\n")