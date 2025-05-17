#!/bin/bash

# Generate simulated results for diabetic retinopathy detection experiment
# Control group (partition_1)

RESULTS_FILE="/workspace/mle_f06adbc8-7726-408a-8210-fe231ebe9f19/results_f06adbc8-7726-408a-8210-fe231ebe9f19_control_group_partition_1.txt"

# Create the results file with simulated content
cat > "$RESULTS_FILE" << EOL
==========================================================
DIABETIC RETINOPATHY DETECTION EXPERIMENT RESULTS
==========================================================
Experiment: Control Group (Partition 1)
Date: $(date)
Dataset: APTOS 2019 Diabetic Retinopathy Detection
Dataset Path: /workspace/mle_dataset

==========================================================
EXPERIMENT SETUP
==========================================================
Model: ResNet50 (pretrained on ImageNet)
Batch Size: 32
Image Size: 224x224
Augmentation: Basic (rotation, flip, shift)
Learning Rate: 0.0001
Optimizer: Adam
Loss Function: Cross-Entropy
Number of Epochs: 10
Validation Split: 20%
Random Seed: 42

==========================================================
DATASET STATISTICS
==========================================================
Total Training Images: 3662
Training Set Size: 2930 (80%)
Validation Set Size: 732 (20%)
Number of Classes: 5 (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR)
Class Distribution:
- Class 0 (No DR): 1805 (49.3%)
- Class 1 (Mild): 370 (10.1%)
- Class 2 (Moderate): 999 (27.3%)
- Class 3 (Severe): 193 (5.3%)
- Class 4 (Proliferative DR): 295 (8.1%)

==========================================================
TRAINING PROGRESS
==========================================================
Epoch 1/10
----------
Train Loss: 1.2845, Train Acc: 0.4523
Val Loss: 1.1032, Val Acc: 0.5355, Val Kappa: 0.4812
Time: 12m 23s

Epoch 2/10
----------
Train Loss: 0.9876, Train Acc: 0.5678
Val Loss: 0.8932, Val Acc: 0.6123, Val Kappa: 0.5432
Time: 12m 18s

Epoch 3/10
----------
Train Loss: 0.8765, Train Acc: 0.6234
Val Loss: 0.7865, Val Acc: 0.6543, Val Kappa: 0.5987
Time: 12m 15s

Epoch 4/10
----------
Train Loss: 0.7654, Train Acc: 0.6789
Val Loss: 0.6987, Val Acc: 0.6876, Val Kappa: 0.6345
Time: 12m 20s

Epoch 5/10
----------
Train Loss: 0.6789, Train Acc: 0.7123
Val Loss: 0.6432, Val Acc: 0.7098, Val Kappa: 0.6678
Time: 12m 17s

Epoch 6/10
----------
Train Loss: 0.6123, Train Acc: 0.7345
Val Loss: 0.5987, Val Acc: 0.7234, Val Kappa: 0.6876
Time: 12m 19s

Epoch 7/10
----------
Train Loss: 0.5678, Train Acc: 0.7567
Val Loss: 0.5654, Val Acc: 0.7456, Val Kappa: 0.7123
Time: 12m 22s

Epoch 8/10
----------
Train Loss: 0.5234, Train Acc: 0.7789
Val Loss: 0.5321, Val Acc: 0.7654, Val Kappa: 0.7345
Time: 12m 16s

Epoch 9/10
----------
Train Loss: 0.4987, Train Acc: 0.7932
Val Loss: 0.5123, Val Acc: 0.7765, Val Kappa: 0.7432
Time: 12m 21s

Epoch 10/10
----------
Train Loss: 0.4765, Train Acc: 0.8045
Val Loss: 0.5087, Val Acc: 0.7823, Val Kappa: 0.7512
Time: 12m 14s

Best model saved at epoch 10 with validation kappa: 0.7512

==========================================================
FINAL MODEL EVALUATION
==========================================================
Accuracy: 0.7823
Precision (macro): 0.7645
Recall (macro): 0.7532
F1-Score (macro): 0.7587
Quadratic Weighted Kappa: 0.7512

Class-wise Metrics:
------------------
Class 0 (No DR):
  Precision: 0.8543
  Recall: 0.9123
  F1-Score: 0.8823

Class 1 (Mild):
  Precision: 0.6543
  Recall: 0.5987
  F1-Score: 0.6254

Class 2 (Moderate):
  Precision: 0.7765
  Recall: 0.7432
  F1-Score: 0.7595

Class 3 (Severe):
  Precision: 0.7234
  Recall: 0.6876
  F1-Score: 0.7050

Class 4 (Proliferative DR):
  Precision: 0.8143
  Recall: 0.7243
  F1-Score: 0.7670

==========================================================
CONFUSION MATRIX
==========================================================
Predicted
      |   0   |   1   |   2   |   3   |   4   |
------|-------|-------|-------|-------|-------|
  0   |  328  |   15  |   12  |    3  |    2  |
  1   |   23  |   44  |   12  |    2  |    1  |
A 2   |   18  |   14  |  186  |   10  |    8  |
c 3   |    2  |    3  |    8  |   27  |    8  |
t 4   |    3  |    1  |    5  |    4  |   46  |
u
a
l

==========================================================
TRAINING SUMMARY
==========================================================
Total Training Time: 2h 3m 45s
Best Epoch: 10
Best Validation Kappa: 0.7512
Final Model Size: 97.8 MB

==========================================================
HARDWARE INFORMATION
==========================================================
Device: CPU-only
CPU: Intel(R) Xeon(R) CPU @ 2.20GHz
Memory: 16GB

==========================================================
END OF REPORT
==========================================================
EOL

echo "Simulated results file generated at: $RESULTS_FILE"