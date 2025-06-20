# Feed-Forward Network for Super-Resolution Tissue Architecture Inference

This directory contains Curie's experiments with Feed-Forward Neural Networks (FFN) for super-resolution tissue architecture inference.
 
💥 Curie is able to find models that **converge fast and have smaller loss** (fig. 1) on the  super-resolution tissue architecture inference task!


## Running Experiment with Curie
To run resolution enhancement experiments with Curie:
```
key_dict = {
    "MODEL": "claude-3-7-sonnet-20250219",
    "ANTHROPIC_API_KEY": "your-anthropic-key"
}

import curie
result = curie.experiment(api_keys=key_dict, 
                          question="Find an optimal model architecture, hyperparameters and training algorithms to train the feedforward neural network \
                          to minimize the loss or RMSE of the model.\
                          The baseline solution code is provided at 'impute.py', you only need to work on top of it.", 
                          workspace_name='/home/ubuntu/hest_analyze',
                          dataset_dir='/home/ubuntu/hest_data')
```


## Curie Experiment Results
- Full experiment report: [istar_1749179423_20250605231023_iter1.md](./istar_1749179423_20250605231023_iter1.md)
- Raw results: [istar_1749179423_20250605231023_iter1_all_results.txt](./istar_1749179423_20250605231023_iter1_all_results.txt)
- Raw **codebase** generated by Curie: `istar_*`.




## Report Snippet

### 3.1 Overall Performance Comparison

The experimental results showed significant differences in performance across the tested architectures. The AdamW optimizer variation with [256, 128] architecture demonstrated the best performance, achieving an RMSE of 0.0735, compared to the baseline model's RMSE of 0.0827.

![convergence_curves](convergence_comparison.png)
*<small>Fig 1: Comparison of training convergence (RMSE vs. epochs) for different neural network architectures in gene expression prediction. The AdamW optimizer with [256, 128] architecture achieves the lowest RMSE.</small>*

### 3.2 Final Performance Metrics

The best RMSE values achieved by each configuration are summarized in the figure below:

![best_rmse](best_rmse_comparison.png)
*<small>Fig 2: Comparison of best RMSE values achieved by different neural network architectures. The AdamW optimizer with [256, 128] architecture shows the best performance with an RMSE of 0.0735.</small>*

### 3.3 Training Efficiency and Model Complexity

We analyzed the relationship between model complexity (number of parameters) and training efficiency (iterations per second) to evaluate the computational cost of each approach:

![training_efficiency](training_efficiency_comparison.png)
*<small>Fig 3: Comparison of training efficiency (iterations per second) and model complexity (thousands of parameters) across different architectures. The AdamW [256, 128] architecture offers an optimal balance between training speed and model complexity.</small>*

### 3.4 Convergence Speed

The convergence speed was measured as the number of epochs required to reach an RMSE threshold of 0.08. The AdamW [256, 128] architecture demonstrated significantly faster convergence:

![convergence_speed](convergence_speed_comparison.png)
*<small>Fig 4: Comparison of convergence speed measured as epochs needed to reach an RMSE threshold of 0.08. The AdamW [256, 128] architecture converges significantly faster, requiring only 103 epochs versus 175 for the baseline model.</small>*

### 3.5 Training Dynamics

The baseline model showed a typical learning pattern:
- Rapid initial improvement in the first 50 epochs
- Slower but steady progress in middle epochs
- Minor fluctuations between RMSE 0.07-0.09 in later stages
- Final RMSE of 0.0827 after 390 epochs

In comparison, the AdamW optimizer with [256, 128] architecture:
- Demonstrated faster initial convergence
- Reached optimal performance around epoch 130 with RMSE of 0.0735
- Maintained more stable RMSE values during later training stages
- Achieved better final performance with fewer parameters
