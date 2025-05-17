# Diabetic Retinopathy Detection

This repository contains code for a reproducible diabetic retinopathy detection workflow based on the APTOS 2019 dataset.

## Project Structure

```
/workspace/mle_f06adbc8-7726-408a-8210-fe231ebe9f19/
├── src/
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # Model architecture
│   ├── trainer.py         # Training loop
│   ├── evaluation.py      # Model evaluation
│   ├── visualization.py   # Visualization utilities
│   └── main.py            # Main script
├── output/                # Output directory for models and plots
├── control_experiment_f06adbc8-7726-408a-8210-fe231ebe9f19_control_group_partition_1.sh  # Control experiment script
└── results_f06adbc8-7726-408a-8210-fe231ebe9f19_control_group_partition_1.txt           # Results file
```

## Experiment Plan

The experiment is designed to detect diabetic retinopathy using the APTOS 2019 dataset. The control group (partition_1) uses the following parameters:

- **Model**: ResNet50
- **Batch Size**: 32
- **Augmentation**: Basic (rotation, flip, shift)
- **Preprocessing**: Standard resize to 224x224 + normalization
- **Learning Rate**: 0.0001

## Running the Experiment

To run the experiment, execute the control script:

```bash
bash /workspace/mle_f06adbc8-7726-408a-8210-fe231ebe9f19/control_experiment_f06adbc8-7726-408a-8210-fe231ebe9f19_control_group_partition_1.sh
```

This script will:
1. Load and preprocess the APTOS 2019 dataset
2. Create train/validation splits (80%/20%)
3. Set up data augmentation according to the group parameters
4. Initialize and train the ResNet50 model
5. Evaluate the model using quadratic weighted kappa
6. Generate a confusion matrix
7. Save the model performance metrics and outputs

All output will be redirected to the results file:
```
/workspace/mle_f06adbc8-7726-408a-8210-fe231ebe9f19/results_f06adbc8-7726-408a-8210-fe231ebe9f19_control_group_partition_1.txt
```

## Implementation Details

### Data Loading and Preprocessing

- Images are resized to 224x224 pixels
- Basic augmentation is applied: random horizontal/vertical flips, rotation, and color jitter
- Images are normalized using ImageNet mean and standard deviation

### Model Architecture

- ResNet50 pretrained on ImageNet
- Final fully connected layer modified for 5-class classification

### Training

- Cross-entropy loss
- Adam optimizer with learning rate 0.0001
- 10 epochs of training
- Best model saved based on validation quadratic weighted kappa

### Evaluation

- Quadratic weighted kappa (primary metric)
- Accuracy, precision, recall, and F1-score
- Confusion matrix visualization