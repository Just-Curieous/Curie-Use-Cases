import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.config import (
    MODELS_DIR, RESULTS_DIR, NUM_EPOCHS, LEARNING_RATE, 
    NUM_FOLDS, MODEL_NAME, SEED
)
from src.data import load_data, get_fold_dataloader
from src.model import get_model
from src.utils import set_seed, quadratic_weighted_kappa

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_outputs = []
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        all_targets.extend(targets.cpu().numpy())
        all_outputs.extend(outputs.softmax(dim=1).detach().cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = train_loss / len(loader)
    
    # Calculate quadratic weighted kappa
    all_preds = np.argmax(np.array(all_outputs), axis=1)
    kappa = quadratic_weighted_kappa(all_targets, all_preds)
    
    return avg_loss, accuracy, kappa, all_targets, all_preds, all_outputs

def validate_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.softmax(dim=1).detach().cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = val_loss / len(loader)
    
    # Calculate quadratic weighted kappa
    all_preds = np.argmax(np.array(all_outputs), axis=1)
    kappa = quadratic_weighted_kappa(all_targets, all_preds)
    
    return avg_loss, accuracy, kappa, all_targets, all_preds, all_outputs

def train_and_validate(fold):
    set_seed(SEED + fold)
    
    # Load data
    train_df, _ = load_data()
    train_loader, valid_loader, train_fold_df, valid_fold_df = get_fold_dataloader(train_df, fold)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = get_model(model_name=MODEL_NAME)
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3
    )
    
    # Store initial learning rate for manual logging
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_kappa': [],
        'val_loss': [], 'val_acc': [], 'val_kappa': []
    }
    
    best_kappa = -1
    best_epoch = -1
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f'Fold {fold} | Epoch {epoch+1}/{NUM_EPOCHS}')
        
        # Train
        train_loss, train_acc, train_kappa, train_targets, train_preds, train_outputs = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_kappa, val_targets, val_preds, val_outputs = validate_epoch(
            model, valid_loader, criterion, device
        )
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_kappa)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Manually log learning rate changes
        if new_lr != current_lr:
            print(f'Learning rate changed from {current_lr:.6f} to {new_lr:.6f}')
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_kappa'].append(train_kappa)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_kappa'].append(val_kappa)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Kappa: {train_kappa:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Kappa: {val_kappa:.4f}')
        
        # Save best model
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            best_epoch = epoch
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_fold{fold}.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path}')
            
            # Save confusion matrix
            cm = confusion_matrix(val_targets, val_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - Fold {fold}')
            cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_fold{fold}.png')
            plt.savefig(cm_path)
            plt.close()
            
            # Save prediction confidence distribution
            plt.figure(figsize=(10, 6))
            for i in range(5):
                class_probs = [output[i] for output in val_outputs]
                sns.kdeplot(class_probs, label=f'Class {i}')
            plt.xlabel('Prediction Confidence')
            plt.ylabel('Density')
            plt.title(f'Prediction Confidence Distribution - Fold {fold}')
            plt.legend()
            conf_path = os.path.join(RESULTS_DIR, f'confidence_dist_fold{fold}.png')
            plt.savefig(conf_path)
            plt.close()
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves - Fold {fold}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_kappa'], label='Train Kappa')
    plt.plot(history['val_kappa'], label='Validation Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('Quadratic Weighted Kappa')
    plt.title(f'Kappa Curves - Fold {fold}')
    plt.legend()
    
    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, f'learning_curves_fold{fold}.png')
    plt.savefig(curves_path)
    plt.close()
    
    # Calculate generalization gap
    gen_gap = history['train_kappa'][-1] - history['val_kappa'][-1]
    
    # Load best model for final evaluation
    model_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_fold{fold}.pth')
    model.load_state_dict(torch.load(model_path))
    
    # Final validation
    val_loss, val_acc, val_kappa, val_targets, val_preds, val_outputs = validate_epoch(
        model, valid_loader, criterion, device
    )
    
    # Return metrics
    fold_results = {
        'fold': fold,
        'best_epoch': best_epoch,
        'best_val_kappa': best_kappa,
        'final_train_kappa': history['train_kappa'][-1],
        'final_val_kappa': history['val_kappa'][-1],
        'generalization_gap': gen_gap,
        'val_targets': val_targets,
        'val_preds': val_preds,
        'val_outputs': val_outputs
    }
    
    return fold_results

def run_cross_validation():
    results = []
    
    for fold in range(NUM_FOLDS):
        print(f'Training fold {fold+1}/{NUM_FOLDS}')
        fold_results = train_and_validate(fold)
        results.append(fold_results)
    
    # Calculate average metrics
    avg_val_kappa = np.mean([r['best_val_kappa'] for r in results])
    avg_gen_gap = np.mean([r['generalization_gap'] for r in results])
    
    print(f'Cross-validation completed.')
    print(f'Average Validation Kappa: {avg_val_kappa:.4f}')
    print(f'Average Generalization Gap: {avg_gen_gap:.4f}')
    
    # Save summary results
    summary = {
        'avg_val_kappa': avg_val_kappa,
        'avg_gen_gap': avg_gen_gap,
        'fold_results': results
    }
    
    return summary