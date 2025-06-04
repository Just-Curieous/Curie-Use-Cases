import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18_model(pretrained=True):
    """
    Create a ResNet18 model for binary classification
    
    Args:
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: ResNet18 model
    """
    # Load pretrained ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Modify the final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    
    return model

def get_device():
    """
    Get the device to use for training/inference
    
    Returns:
        torch.device: Device to use
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    
    return device