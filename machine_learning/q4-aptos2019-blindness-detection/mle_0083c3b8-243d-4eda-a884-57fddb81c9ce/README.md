# Diabetic Retinopathy Detection

This repository contains a PyTorch-based solution for diabetic retinopathy detection using the APTOS 2019 dataset.

## Project Structure

- `src/`: Source code directory
  - `config.py`: Configuration parameters
  - `utils.py`: Utility functions
  - `data.py`: Data loading and preprocessing
  - `model.py`: Model architecture
  - `train.py`: Training functions
  - `evaluate.py`: Evaluation and prediction functions
  - `main.py`: Main script to run the experiment
- `control_experiment_0083c3b8-243d-4eda-a884-57fddb81c9ce_control_group_partition_1.sh`: Control experiment script
- `results_0083c3b8-243d-4eda-a884-57fddb81c9ce_control_group_partition_1.txt`: Results file (generated after running the experiment)

## Experiment Details

### Control Group (Partition 1)

- **Model Architecture**: EfficientNetB4
- **Regularization**: Standard (weight decay, dropout)
- **Training Approach**: End-to-end (train all layers together)

### Implementation Features

1. **Data Preprocessing**:
   - Circular crop for retinal images
   - Standard normalization and augmentation

2. **Model**:
   - EfficientNetB4 with standard regularization
   - End-to-end training approach

3. **Evaluation**:
   - Quadratic weighted kappa (main metric)
   - Accuracy and per-class accuracy

## Running the Experiment

To run the experiment, execute the control script:

```bash
bash control_experiment_0083c3b8-243d-4eda-a884-57fddb81c9ce_control_group_partition_1.sh
```

The script will:
1. Set up the environment
2. Run the experiment
3. Save all output to the results file

## Future Extensions

The implementation is modular to support future experimental groups with:
1. Different regularization techniques (strong_dropout or mixup)
2. Different training strategies (gradual_unfreezing or progressive_resizing)