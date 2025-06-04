import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class DiabeticRetinopathyModel(nn.Module):
    def __init__(self, num_classes=5):
        """
        Initialize the model with EfficientNetB4 backbone
        
        Args:
            num_classes: Number of output classes (default: 5 for DR stages 0-4)
        """
        super(DiabeticRetinopathyModel, self).__init__()
        
        # Load pre-trained EfficientNetB4
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        
        # Get the number of features from the last layer
        n_features = self.backbone._fc.in_features
        
        # Replace the final fully connected layer
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)

def get_model(num_classes=5, device='cuda'):
    """
    Create and return the model
    
    Args:
        num_classes: Number of output classes
        device: Device to put the model on
        
    Returns:
        model: The initialized model
    """
    model = DiabeticRetinopathyModel(num_classes=num_classes)
    model = model.to(device)
    return model