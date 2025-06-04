import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from src.config import MODEL_NAME, NUM_CLASSES, DROPOUT_RATE

class RetinopathyModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        """
        Initialize the model.
        
        Args:
            model_name (str): Name of the EfficientNet model.
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(RetinopathyModel, self).__init__()
        
        # Load pre-trained EfficientNet
        self.efficientnet = EfficientNet.from_pretrained(model_name)
        
        # Get the number of features in the last layer
        in_features = self.efficientnet._fc.in_features
        
        # Replace the final fully connected layer
        self.efficientnet._fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.efficientnet(x)

def create_model():
    """Create and initialize the model."""
    model = RetinopathyModel()
    return model