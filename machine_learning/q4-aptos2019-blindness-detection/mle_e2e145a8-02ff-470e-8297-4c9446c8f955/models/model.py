import torch
import torch.nn as nn
import torchvision.models as models
import time

class RetinopathyModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(RetinopathyModel, self).__init__()
        # Load pretrained EfficientNet-B3
        self.model = models.efficientnet_b3(weights='DEFAULT' if pretrained else None)
        
        # Modify the classifier for our task
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def get_model(num_classes=5, pretrained=True, device='cuda'):
    """
    Create and return the model
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to put the model on
        
    Returns:
        model: The model
    """
    model = RetinopathyModel(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model

def get_model_size(model):
    """
    Calculate the size of the model in MB
    
    Args:
        model: PyTorch model
        
    Returns:
        size_mb: Size of the model in MB
    """
    torch.save(model.state_dict(), "temp.pt")
    size_mb = os.path.getsize("temp.pt") / (1024 * 1024)
    os.remove("temp.pt")
    return size_mb

def measure_inference_time(model, input_size=(1, 3, 300, 300), device='cuda', num_iterations=100):
    """
    Measure the inference time of the model
    
    Args:
        model: PyTorch model
        input_size: Input size for the model
        device: Device to run inference on
        num_iterations: Number of iterations to average over
        
    Returns:
        avg_time: Average inference time in milliseconds
    """
    model.eval()
    dummy_input = torch.randn(input_size, device=device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) * 1000 / num_iterations  # Convert to milliseconds
    return avg_time

import os