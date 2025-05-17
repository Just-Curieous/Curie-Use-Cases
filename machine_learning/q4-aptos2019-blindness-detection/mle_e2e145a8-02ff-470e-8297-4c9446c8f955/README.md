# Diabetic Retinopathy Detection

This project implements a reproducible workflow for diabetic retinopathy detection using the APTOS 2019 dataset.

## Project Structure

```
/workspace/mle_e2e145a8-02ff-470e-8297-4c9446c8f955/
├── data/
│   └── data_loader.py       # Data loading and preprocessing utilities
├── models/
│   └── model.py             # Model definition and utilities
├── utils/
│   └── train_utils.py       # Training and evaluation utilities
├── results/                 # Directory to store experiment results
├── main.py                  # Main script to run the experiment
├── control_experiment_e2e145a8-02ff-470e-8297-4c9446c8f955_control_group_partition_1.sh  # Control experiment script
└── README.md                # Project documentation
```

## Dataset

The APTOS 2019 Blindness Detection dataset consists of retina images with the following diagnosis labels:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

## Experiment Plan

For the control group (partition_1), the workflow implements:
- Standard fine-tuning of all layers of an EfficientNet-B3 model
- No explainability methods
- No post-processing
- No threshold optimization

## Implementation Details

1. **Data Processing**:
   - Images are resized to 300x300 pixels
   - Standard normalization using ImageNet mean and std
   - Basic augmentations (horizontal and vertical flips) for training

2. **Model**:
   - EfficientNet-B3 pretrained on ImageNet
   - Fine-tuning of all layers

3. **Training**:
   - 5-fold cross-validation
   - Categorical cross-entropy loss
   - Adam optimizer
   - Learning rate scheduler (ReduceLROnPlateau)

4. **Evaluation Metrics**:
   - Quadratic weighted kappa (primary metric)
   - Accuracy
   - Confusion matrix

5. **Performance Metrics**:
   - Training time
   - Inference time
   - Model size

## Running the Experiment

To run the full experiment, execute the control experiment script:

```bash
./control_experiment_e2e145a8-02ff-470e-8297-4c9446c8f955_control_group_partition_1.sh
```

This script will:
1. Set up the environment
2. Run the experiment with 5-fold cross-validation
3. Save the results to the specified output file
4. Print a summary of the results

## Results

The experiment results are saved in:
- `results/experiment_results.json`: Detailed metrics for each fold and overall
- `results_e2e145a8-02ff-470e-8297-4c9446c8f955_control_group_partition_1.txt`: Full experiment log