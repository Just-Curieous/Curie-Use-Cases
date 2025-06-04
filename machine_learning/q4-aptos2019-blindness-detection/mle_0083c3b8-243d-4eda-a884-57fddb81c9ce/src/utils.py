import os
import random
import numpy as np
import torch
import cv2
from sklearn.metrics import cohen_kappa_score

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def circular_crop(image):
    """Apply circular crop to retinal images."""
    height, width = image.shape[:2]
    
    # Create a circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(width, height) // 2
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply the mask
    if len(image.shape) == 3:  # Color image
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)
        result = cv2.bitwise_and(image, mask)
    else:  # Grayscale image
        result = cv2.bitwise_and(image, mask)
    
    return result

def quadratic_weighted_kappa(y_true, y_pred):
    """Calculate quadratic weighted kappa score."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')