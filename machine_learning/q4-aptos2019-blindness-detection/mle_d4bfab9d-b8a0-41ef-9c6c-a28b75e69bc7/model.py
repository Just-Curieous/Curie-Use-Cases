import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class RetinopathyModel(nn.Module):
    def __init__(self, num_classes=5):
        """
        Initialize the model with EfficientNetB4 backbone
        
        Args:
            num_classes (int): Number of output classes
        """
        super(RetinopathyModel, self).__init__()
        
        # Load pre-trained EfficientNetB4
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        
        # Get the number of features from the last layer
        in_features = self.backbone._fc.in_features
        
        # Replace the final fully connected layer
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)

def get_model(num_classes=5, device='cuda'):
    """
    Create and return the model
    
    Args:
        num_classes (int): Number of output classes
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        model: The initialized model
    """
    model = RetinopathyModel(num_classes=num_classes)
    model = model.to(device)
    return model