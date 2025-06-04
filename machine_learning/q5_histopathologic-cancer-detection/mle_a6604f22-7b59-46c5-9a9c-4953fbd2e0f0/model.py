import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from thop import profile

class EfficientNetB0Model(nn.Module):
    def __init__(self, pretrained=True):
        """
        EfficientNetB0 model for binary classification.
        
        Args:
            pretrained (bool): Whether to use pretrained weights.
        """
        super(EfficientNetB0Model, self).__init__()
        
        # Load the EfficientNetB0 model
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=pretrained)
        
        # Get the number of features in the last layer
        num_features = self.efficientnet.classifier.in_features
        
        # Replace the classifier with a custom one
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(num_features, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.efficientnet(x)
        return torch.sigmoid(x)

def get_model_complexity(model, input_size=(3, 96, 96)):
    """
    Calculate the model complexity in terms of FLOPs and parameters.
    
    Args:
        model (nn.Module): The model to analyze.
        input_size (tuple): The input size (C, H, W).
        
    Returns:
        tuple: (FLOPs, parameters)
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    return flops, params

def get_model_size_mb(model):
    """
    Calculate the model size in MB.
    
    Args:
        model (nn.Module): The model to analyze.
        
    Returns:
        float: Model size in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb