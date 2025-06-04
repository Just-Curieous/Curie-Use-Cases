import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class RetinopathyModel(nn.Module):
    """
    EfficientNetB4 model for diabetic retinopathy classification.
    """
    def __init__(self, num_classes=5):
        super(RetinopathyModel, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        
        # Get the number of features in the last layer
        n_features = self.model._fc.in_features
        
        # Replace the final fully connected layer
        self.model._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)


def get_model(num_classes=5, device='cuda'):
    """
    Create and return the model.
    
    Args:
        num_classes (int): Number of output classes
        device (str): Device to move the model to
        
    Returns:
        RetinopathyModel: The model
    """
    model = RetinopathyModel(num_classes=num_classes)
    model = model.to(device)
    return model