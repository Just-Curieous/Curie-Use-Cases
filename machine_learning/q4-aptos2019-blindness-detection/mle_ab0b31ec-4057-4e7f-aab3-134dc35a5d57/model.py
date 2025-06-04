import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
from torchmetrics.classification import Accuracy, F1Score, CohenKappa
import numpy as np
from utils import calculate_metrics

class DiabeticRetinopathyModel(pl.LightningModule):
    def __init__(
        self,
        num_classes=5,
        learning_rate=1e-4,
        weight_decay=1e-5,
        class_weights=None,
        model_name='efficientnet-b4'
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained EfficientNet
        self.model = EfficientNet.from_pretrained(model_name)
        
        # Replace classifier
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)
        
        # Loss function with class weights if provided
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_kappa = CohenKappa(task="multiclass", num_classes=num_classes, weights='quadratic')
        
        # For tracking best metrics
        self.best_val_kappa = 0.0
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_kappa",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        f1 = self.val_f1(preds, y)
        kappa = self.val_kappa(preds, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_kappa', kappa, prog_bar=True)
        
        # Track best kappa score
        if kappa > self.best_val_kappa:
            self.best_val_kappa = kappa
            self.log('best_val_kappa', self.best_val_kappa)
        
        return {'val_loss': loss, 'val_preds': preds, 'val_targets': y}
    
    def validation_epoch_end(self, outputs):
        # Aggregate predictions and targets
        all_preds = torch.cat([x['val_preds'] for x in outputs])
        all_targets = torch.cat([x['val_targets'] for x in outputs])
        
        # Calculate metrics
        metrics = calculate_metrics(
            all_targets.cpu().numpy(),
            all_preds.cpu().numpy()
        )
        
        # Log detailed metrics
        for name, value in metrics.items():
            self.log(f'val_detailed_{name}', value)
    
    def test_step(self, batch, batch_idx):
        # For test data with labels
        if len(batch) == 2 and isinstance(batch[1], torch.Tensor) and batch[1].dim() == 1:
            x, y = batch
            logits = self(x)
            preds = torch.argmax(logits, dim=1)
            return {'test_preds': preds, 'test_targets': y}
        # For test data without labels (just IDs)
        else:
            x, img_ids = batch
            logits = self(x)
            preds = torch.argmax(logits, dim=1)
            return {'test_preds': preds, 'test_img_ids': img_ids}
    
    def test_epoch_end(self, outputs):
        # Check if we have targets (labeled test data)
        if 'test_targets' in outputs[0]:
            all_preds = torch.cat([x['test_preds'] for x in outputs])
            all_targets = torch.cat([x['test_targets'] for x in outputs])
            
            # Calculate metrics
            metrics = calculate_metrics(
                all_targets.cpu().numpy(),
                all_preds.cpu().numpy()
            )
            
            # Log metrics
            for name, value in metrics.items():
                self.log(f'test_{name}', value)
                
            return {
                'test_preds': all_preds.cpu().numpy(),
                'test_targets': all_targets.cpu().numpy(),
                'metrics': metrics
            }
        else:
            # For unlabeled test data, just return predictions and IDs
            all_preds = torch.cat([x['test_preds'] for x in outputs])
            all_img_ids = np.concatenate([x['test_img_ids'] for x in outputs])
            
            return {
                'test_preds': all_preds.cpu().numpy(),
                'test_img_ids': all_img_ids
            }