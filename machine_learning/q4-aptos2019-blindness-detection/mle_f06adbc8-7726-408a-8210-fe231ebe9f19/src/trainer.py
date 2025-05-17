import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.0001):
        """
        Initialize the trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to run the training on
            learning_rate: Learning rate for the optimizer
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self):
        """
        Train for one epoch
        
        Returns:
            epoch_loss: Average loss for the epoch
            epoch_acc: Accuracy for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Disable tqdm progress bar
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        Validate the model
        
        Returns:
            val_loss: Validation loss
            val_acc: Validation accuracy
            all_preds: All predictions
            all_labels: All true labels
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / total
        val_acc = correct / total
        
        return val_loss, val_acc, np.array(all_preds), np.array(all_labels)
    
    def train(self, num_epochs=10, save_dir=None):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train for
            save_dir: Directory to save the model
            
        Returns:
            history: Dictionary containing training history
            best_model_path: Path to the best model
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_kappa': []
        }
        
        best_val_kappa = -1.0
        best_model_path = None
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, all_preds, all_labels = self.validate()
            
            # Calculate quadratic weighted kappa
            val_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_kappa'].append(val_kappa)
            
            # Print statistics
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Kappa: {val_kappa:.4f}')
            
            # Save the best model
            if val_kappa > best_val_kappa and save_dir is not None:
                best_val_kappa = val_kappa
                os.makedirs(save_dir, exist_ok=True)
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f'Saved best model with kappa: {val_kappa:.4f}')
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f'Training completed in {training_time:.2f} seconds')
        
        # Calculate final confusion matrix
        _, _, all_preds, all_labels = self.validate()
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return history, best_model_path, conf_matrix, training_time