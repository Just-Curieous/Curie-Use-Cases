import torch
import torch.nn as nn
import torchvision.models as models
import time

class CancerDetectionModel(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        """
        Initialize the model.
        
        Args:
            model_name (str): Name of the model architecture to use
            pretrained (bool): Whether to use pretrained weights
        """
        super(CancerDetectionModel, self).__init__()
        
        if model_name == 'resnet50':
            self.model = models.resnet50(weights='DEFAULT' if pretrained else None)
            # Modify the final layer for binary classification
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, 1)
            )
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
            # Modify the final layer for binary classification
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, 1)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.model(x)

def measure_inference_time(model, input_size=(1, 3, 96, 96), device='cuda', num_iterations=100):
    """
    Measure the inference time of the model.
    
    Args:
        model (nn.Module): The model to measure
        input_size (tuple): The input size
        device (str): The device to use
        num_iterations (int): Number of iterations to average over
    
    Returns:
        float: Average inference time in milliseconds
    """
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) * 1000 / num_iterations  # Convert to milliseconds
    return avg_time