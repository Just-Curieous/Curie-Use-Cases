import torch
import torch.nn as nn
import torchvision.models as models
from src.config import NUM_CLASSES, DROPOUT_RATE

class EfficientNetModel(nn.Module):
    def __init__(self, model_name='efficientnet_b5', pretrained=True, dropout_rate=DROPOUT_RATE):
        super(EfficientNetModel, self).__init__()
        
        # Load pre-trained EfficientNet model
        if model_name == 'efficientnet_b5':
            self.backbone = models.efficientnet_b5(weights='DEFAULT' if pretrained else None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Get the number of features in the last layer
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=NUM_CLASSES)
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_model(model_name='efficientnet_b5', pretrained=True):
    return EfficientNetModel(model_name=model_name, pretrained=pretrained)