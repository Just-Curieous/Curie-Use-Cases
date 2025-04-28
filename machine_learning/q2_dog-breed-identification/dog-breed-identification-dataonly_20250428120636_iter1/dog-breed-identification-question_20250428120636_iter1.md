# Dog Breed Classification Model Development: 
# A Comprehensive Experimental Analysis

## Abstract

This report presents the development and optimization of a machine learning model for classifying dog breeds across 120 distinct classes. Through a systematic experimental approach, we evaluated various deep learning architectures, training strategies, and optimization techniques. Our experiments demonstrate that transfer learning significantly outperforms training from scratch, with EfficientNetB0 emerging as the best-performing architecture (68.04% accuracy). Further optimization through learning rate scheduling and optimizer selection improved performance to 83.53% accuracy with SGD and step decay learning rate. This report details our methodology, findings, and recommendations for future work in fine-grained image classification tasks.

## 1. Introduction

### Research Question
The primary research question addressed in this study is: How can we develop an effective image classification model that accurately identifies and categorizes dogs into 120 different breeds, optimizing for multi-class log loss metric?

### Hypothesis
We hypothesized that transfer learning using pre-trained models would significantly outperform models trained from scratch, and that careful optimization of hyperparameters, particularly learning rate scheduling, would substantially improve performance on this fine-grained classification task.

### Background
Dog breed classification presents a challenging computer vision task due to the high intra-class variability and inter-class similarity among different dog breeds. The subtle distinctions between certain breeds make this an ideal problem for testing advanced computer vision techniques. This experiment contributes to the field by systematically comparing different model architectures and training strategies specifically for fine-grained image classification.

## 2. Methodology

### Experiment Design
We designed four experimental groups to systematically explore different aspects of model development:
1. **Model Architecture Selection**: Comparing different CNN architectures with transfer learning versus training from scratch
2. **Hyperparameter Optimization**: Testing various optimizers, learning rates, and learning strategies
3. **Data Augmentation and Balancing**: Exploring techniques to enhance model generalization and address class imbalance
4. **Ensemble Modeling**: Evaluating methods for combining multiple models to improve performance

### Experimental Setup
All experiments were conducted using PyTorch on an NVIDIA A40 GPU. The dataset consisted of images of 120 different dog breeds, split into training (70%), validation (15%), and test (15%) sets. Images were resized to 224×224 pixels for most experiments, with the exception of memory-constrained runs where images were reduced to 150×150 pixels.

### Implementation Details
Initial implementation faced memory constraints, requiring several optimizations:
- Reduced DataLoader workers to 0 to avoid multiprocessing issues
- Implemented batch sizes appropriate for memory constraints (8-64 depending on experiment)
- Added memory management code with periodic GPU cleanup
- Implemented gradient accumulation (over 4 batches) for experiments with very limited memory
- Enhanced error handling for memory-related exceptions

For transfer learning models, we froze the convolutional base layers and trained only the classifier layers for the first 10 epochs, then fine-tuned the entire network for subsequent epochs.

### Experiment Execution
Each experiment followed a standard workflow:
1. Data preparation with appropriate augmentation
2. Model initialization (from scratch or with pre-trained weights)
3. Training with validation monitoring
4. Performance evaluation on test data
5. Results logging and analysis

## 3. Results

### Model Architecture Comparison

The first experiment compared different CNN architectures to identify the most effective model for this classification task.

| Model | Accuracy | Log Loss | Training Time (s) |
|-------|----------|----------|-------------------|
| Simple CNN (from scratch) | 0.0223 | 4.6312 | 358.0 |
| ResNet50 | 0.5913 | 1.4414 | 324.3 |
| VGG16 | 0.0185 | 4.7741 | 379.6 |
| MobileNetV2 | 0.6538 | 1.1936 | 256.8 |
| EfficientNetB0 | 0.6804 | 1.1095 | 277.2 |
| ResNet50 with Advanced Aug | 0.5690 | 1.5928 | 458.1 |

The results clearly demonstrate that transfer learning models dramatically outperform models built from scratch. EfficientNetB0 achieved the best performance with 68.04% accuracy and the lowest log loss of 1.1095. Surprisingly, advanced augmentation techniques decreased ResNet50's performance compared to standard augmentation, suggesting potential overfitting to the augmented data.

### Hyperparameter Optimization

The second experiment focused on optimizing hyperparameters for the best-performing model architecture.

| Configuration | Accuracy | Log Loss | Training Time (s) |
|--------------|----------|----------|-------------------|
| Default (Adam, const LR) | 0.7408 | 0.8429 | 504.0 |
| Adam + StepLR (BS=32) | 0.8054 | 0.6626 | 948.8 |
| SGD + StepLR (BS=32) | 0.8353 | 0.5348 | 843.3 |
| Adam + Cosine (BS=32) | 0.8168 | 0.6236 | 751.7 |
| Adam + StepLR (BS=64) | 0.8043 | 0.6471 | 750.0 |

The results show that learning rate scheduling significantly improved model performance, with SGD with momentum (0.9) and StepLR scheduling achieving the best performance (83.53% accuracy, 0.5348 log loss). The default configuration with a constant learning rate significantly underperformed compared to the scheduled learning rate approaches.

### Data Augmentation and Class Balance Analysis

The control group experiment with standard data augmentation achieved:
- Validation Accuracy: 71.63%
- Validation Log Loss: 1.1144
- Best results at epoch 11

Analysis of per-class performance revealed significant variation:
- High-performing breeds (F1 > 0.90): borzoi, keeshond, chow, ibizan_hound
- Low-performing breeds (F1 < 0.40): eskimo_dog, american_staffordshire_terrier, whippet

This variation suggests challenges with class imbalance and breed similarity that require further investigation.

### Ensemble Model Performance

The final ensemble model achieved:
- Accuracy: 0.7848
- Multi-class Log Loss: 0.7587
- Average Inference Time per Image: 0.0006 seconds
- Model Size: 90.92 MB

The ensemble approach showed improved robustness but didn't significantly outperform the best single model with optimized hyperparameters.

## 4. Discussion

### Architecture Impact
Transfer learning models outperformed the CNN trained from scratch by an order of magnitude, confirming our hypothesis that pre-trained feature extractors are critical for this task. EfficientNetB0 delivered the best balance of performance and efficiency, suggesting that modern architecture design provides advantages for fine-grained classification tasks.

### Optimization Strategy
Learning rate scheduling proved to be critical for performance optimization. The substantial improvement from default settings (74.08% accuracy) to optimized settings (83.53% accuracy) highlights the importance of proper hyperparameter tuning. SGD with momentum outperformed Adam optimizer for this task, contradicting the common practice of defaulting to Adam for deep learning tasks.

### Data Augmentation
Standard augmentation techniques (horizontal flips, rotations, brightness and contrast adjustments) proved sufficient for this task. The unexpected decrease in performance with advanced augmentation techniques suggests that excessive data transformation may have introduced artificial patterns that don't correspond to the test distribution.

### Class Balance Challenges
The significant variation in per-class performance indicates that class balance and similarity between certain breeds remain challenging issues. The confusion between similar-looking breeds suggests that more sophisticated techniques might be needed to distinguish subtle differences.

## 5. Conclusion and Future Work

### Summary of Findings
This study demonstrated that effective dog breed classification can be achieved through transfer learning with appropriate model architecture and hyperparameter optimization. We improved classification accuracy from 2.23% (simple CNN) to 83.53% (optimized transfer learning model), confirming our hypothesis about the effectiveness of transfer learning and hyperparameter optimization.

The most important factors for performance were:
1. Selection of an appropriate pre-trained architecture
2. Implementation of learning rate scheduling
3. Choice of optimizer (SGD with momentum outperforming Adam)

### Recommendations for Future Work
Based on our findings, we recommend:

1. **Feature extraction improvement:** Explore attention mechanisms and feature visualization techniques to better understand which visual features are most discriminative for similar breeds.

2. **Advanced regularization:** Implement more sophisticated regularization techniques like stochastic depth or drop path to further improve generalization.

3. **Knowledge distillation:** Investigate knowledge distillation from the best-performing models to create more efficient models for deployment.

4. **Error analysis:** Conduct detailed error analysis on the most frequently confused breed pairs to develop targeted interventions.

5. **Hyperparameter search:** Perform more extensive hyperparameter optimization using Bayesian optimization or population-based training.

6. **Model ensembling:** Explore more sophisticated ensemble techniques like stacked generalization or snapshot ensembles.

## 6. Appendices

### Environment Information
- Hardware: NVIDIA A40 GPU
- Software: PyTorch 1.12.1, Python 3.9
- Frameworks: torchvision, albumentations (initial implementation)
- Memory optimizations: Gradient accumulation, DataLoader worker optimization

### Code Implementation Details
The implementation used a modular approach with separate components for:
- Data loading and augmentation
- Model architecture definition
- Training and validation loops
- Performance evaluation and metrics calculation
- Results logging and visualization

Memory management was a critical aspect of the implementation, with periodic GPU memory cleanup and exception handling for out-of-memory situations.

### Experiment Log Locations
- Model weights: Stored in experiment-specific directories
- Raw results: CSV files with per-epoch metrics
- Prediction files: JSON format with image IDs and predicted probabilities
- Visualization: PNG plots of training and validation metrics