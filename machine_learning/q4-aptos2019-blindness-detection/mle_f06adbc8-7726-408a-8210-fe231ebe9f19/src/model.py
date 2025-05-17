import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name='resnet50', num_classes=5, pretrained=True):
    """
    Create a model for diabetic retinopathy detection
    
    Args:
        model_name (string): Name of the model architecture
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        model: PyTorch model
    """
    if model_name == 'resnet50':
        # Load a pretrained ResNet50 model
        if pretrained:
            model = models.resnet50(weights='IMAGENET1K_V2')
        else:
            model = models.resnet50(weights=None)
        
        # Modify the final fully connected layer for our classification task
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model