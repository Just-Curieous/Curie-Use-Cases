# Histopathologic Cancer Detection Experiment

This repository contains a reproducible experimental workflow for histopathologic cancer detection using the PatchCamelyon dataset.

## Experiment Overview

The experiment implements a control group workflow with the following configuration:
- **Model**: ResNet18 with pretrained weights
- **Preprocessing**: Standard RGB normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Augmentation**: Basic (horizontal flip, vertical flip, rotation)
- **Optimizer**: Adam with learning rate 0.001
- **Batch size**: 64
- **Early stopping**: Patience of 5 epochs
- **Training**: At least 20 epochs (with early stopping)

## Repository Structure

```
/workspace/mle_d64f3e04-6228-4ebb-be37-ba305ae9ed30/
├── control_experiment_d64f3e04-6228-4ebb-be37-ba305ae9ed30_control_group_partition_1.sh  # Main control script
├── src/                                                                                  # Source code
│   ├── data.py                                                                          # Data loading and preprocessing
│   ├── model.py                                                                         # Model definition
│   ├── train.py                                                                         # Training and evaluation
│   └── experiment.py                                                                    # Main experiment runner
├── output/                                                                              # Output directory (created during execution)
│   ├── best_model.pt                                                                    # Best model checkpoint
│   ├── config.json                                                                      # Experiment configuration
│   ├── results.json                                                                     # Experiment results
│   ├── training_history.json                                                            # Training history
│   └── test_metrics.json                                                                # Test metrics
└── results_d64f3e04-6228-4ebb-be37-ba305ae9ed30_control_group_partition_1.txt          # Experiment log file
```

## Dataset

The experiment uses the PatchCamelyon dataset located at `/workspace/mle_dataset`. The dataset is split into:
- Training set (70%)
- Validation set (15%)
- Test set (15%)

## Running the Experiment

To run the experiment, execute the control script:

```bash
bash /workspace/mle_d64f3e04-6228-4ebb-be37-ba305ae9ed30/control_experiment_d64f3e04-6228-4ebb-be37-ba305ae9ed30_control_group_partition_1.sh
```

The script will:
1. Set up the environment
2. Load and preprocess the data
3. Train the ResNet18 model
4. Evaluate the model performance
5. Save the results and metrics

All outputs will be saved to:
```
/workspace/mle_d64f3e04-6228-4ebb-be37-ba305ae9ed30/results_d64f3e04-6228-4ebb-be37-ba305ae9ed30_control_group_partition_1.txt
```

## Evaluation

The model is evaluated using the AUC-ROC score, which is appropriate for binary classification tasks. The experiment also records:
- Training time
- Inference time
- Best validation AUC and corresponding epoch
- Test AUC

## Adaptability

The code is designed to be easily adaptable for experimental variations, such as using EfficientNetB0 with different augmentation techniques and optimization strategies.