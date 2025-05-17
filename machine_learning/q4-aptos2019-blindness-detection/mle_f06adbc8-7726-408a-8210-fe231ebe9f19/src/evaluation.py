import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import torch
import pandas as pd

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the given data loader
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device to run the evaluation on
        
    Returns:
        all_preds: All predictions
        all_labels: All true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    metrics = {}
    
    # Calculate quadratic weighted kappa
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['accuracy'] = report['accuracy']
    metrics['macro_avg_precision'] = report['macro avg']['precision']
    metrics['macro_avg_recall'] = report['macro avg']['recall']
    metrics['macro_avg_f1'] = report['macro avg']['f1-score']
    
    return metrics

def save_results(metrics, conf_matrix, training_time, history, output_file):
    """
    Save results to a file
    
    Args:
        metrics: Dictionary containing evaluation metrics
        conf_matrix: Confusion matrix
        training_time: Training time in seconds
        history: Training history
        output_file: Path to the output file
    """
    with open(output_file, 'w') as f:
        f.write("Diabetic Retinopathy Detection Results\n")
        f.write("=====================================\n\n")
        
        f.write("Model: ResNet50\n")
        f.write("Batch Size: 32\n")
        f.write("Augmentation: Basic (rotation, flip, shift)\n")
        f.write("Preprocessing: Standard resize to 224x224 + normalization\n")
        f.write("Learning Rate: 0.0001\n\n")
        
        f.write("Training Time: {:.2f} seconds\n\n".format(training_time))
        
        f.write("Evaluation Metrics:\n")
        f.write("------------------\n")
        f.write("Accuracy: {:.4f}\n".format(metrics['accuracy']))
        f.write("Quadratic Weighted Kappa: {:.4f}\n".format(metrics['kappa']))
        f.write("Macro Avg Precision: {:.4f}\n".format(metrics['macro_avg_precision']))
        f.write("Macro Avg Recall: {:.4f}\n".format(metrics['macro_avg_recall']))
        f.write("Macro Avg F1-score: {:.4f}\n\n".format(metrics['macro_avg_f1']))
        
        f.write("Confusion Matrix:\n")
        f.write("----------------\n")
        f.write(np.array2string(conf_matrix, separator=', '))
        f.write("\n\n")
        
        f.write("Training History:\n")
        f.write("----------------\n")
        for epoch in range(len(history['train_loss'])):
            f.write("Epoch {}: Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}, Val Kappa: {:.4f}\n".format(
                epoch+1,
                history['train_loss'][epoch],
                history['train_acc'][epoch],
                history['val_loss'][epoch],
                history['val_acc'][epoch],
                history['val_kappa'][epoch]
            ))

def predict_test_set(model, test_loader, device, output_file):
    """
    Make predictions on the test set and save to a CSV file
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run the predictions on
        output_file: Path to the output CSV file
    """
    model.eval()
    all_preds = []
    all_ids = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            
            # Get the image IDs from the test loader
            batch_ids = [test_loader.dataset.data_frame.iloc[i, 0] for i in range(len(predicted))]
            all_ids.extend(batch_ids)
    
    # Create a DataFrame with the predictions
    df = pd.DataFrame({
        'id_code': all_ids,
        'diagnosis': all_preds
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)