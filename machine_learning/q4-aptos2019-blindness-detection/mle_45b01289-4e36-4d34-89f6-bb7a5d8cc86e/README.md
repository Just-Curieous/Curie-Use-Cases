# Diabetic Retinopathy Detection Experiment

## Overview
This repository contains a reproducible experimental workflow for diabetic retinopathy detection using the APTOS 2019 dataset. The implementation follows a controlled experiment approach with a "best single model" strategy for the control group.

## Dataset
The APTOS 2019 Blindness Detection dataset consists of retinal images that are rated for the severity of diabetic retinopathy on a scale of 0 to 4:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

## Implementation Details

### Model Architecture
- EfficientNetB4 pre-trained model fine-tuned for the diabetic retinopathy classification task
- Output layer modified to predict 5 classes (0-4 severity levels)

### Preprocessing Pipeline
- Resize images to 380x380 pixels (recommended size for EfficientNetB4)
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
- Normalize pixel values using ImageNet mean and standard deviation

### Data Augmentation
- Random rotations (±20 degrees)
- Horizontal and vertical flips
- Random brightness and contrast adjustments

### Training Strategy
- 5-fold cross-validation for reliable performance estimation
- Adam optimizer with learning rate of 0.0001
- Cross-entropy loss function
- Early stopping based on validation quadratic weighted kappa score
- Batch size of 16

### Evaluation Metrics
- Primary: Quadratic Weighted Kappa (the competition's official metric)
- Secondary: Accuracy, Confusion Matrix

## Files
- `diabetic_retinopathy_detection.py`: Main Python script implementing the model and training pipeline
- `control_experiment_45b01289-4e36-4d34-89f6-bb7a5d8cc86e_control_group_partition_1.sh`: Shell script to run the complete experimental workflow
- `results_45b01289-4e36-4d34-89f6-bb7a5d8cc86e_control_group_partition_1.txt`: Output log file with experiment results

## Running the Experiment
To run the experiment, execute the control script:
```bash
./control_experiment_45b01289-4e36-4d34-89f6-bb7a5d8cc86e_control_group_partition_1.sh
```

This script will:
1. Set up the necessary environment
2. Check for GPU availability
3. Verify the dataset
4. Run the training and evaluation pipeline
5. Generate and save results

## Output
The experiment produces the following outputs in the `/workspace/mle_45b01289-4e36-4d34-89f6-bb7a5d8cc86e/output` directory:
1. Trained model files for each fold (`best_model_fold{0-4}.pth`)
2. Cross-validation results (`cv_results.csv`)
3. Evaluation metrics (`metrics.txt`)
4. Confusion matrix visualizations for each fold
5. Test set predictions (`submission.csv`)

All console output and logs are redirected to `results_45b01289-4e36-4d34-89f6-bb7a5d8cc86e_control_group_partition_1.txt`.