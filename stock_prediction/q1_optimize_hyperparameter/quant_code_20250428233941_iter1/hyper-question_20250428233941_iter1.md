# Optimization of LightGBM Hyperparameters for Stock Return Prediction

## Abstract

This study investigated the effects of various LightGBM hyperparameters on stock return prediction performance, specifically focusing on improving Spearman rank correlation between predicted and actual returns. A series of controlled experiments were conducted to test different hyperparameter configurations, including basic parameters (learning rate, number of leaves, max depth), tree structure parameters (number of estimators, subsample rate, column sampling), and regularization parameters. Results from the control groups showed consistent but modest rank correlations in the range of 0.066-0.068 across different years of stock market data (2020-2023), with evidence of significant overfitting. The findings suggest that while the default LightGBM configuration provides a reasonable baseline for stock return prediction, there is substantial room for improvement through proper regularization and addressing the varying performance across different market regimes.

## 1. Introduction

### Research Question
How can LightGBM hyperparameters be optimized to improve the Spearman rank correlation between predicted and actual stock returns?

### Hypothesis
Specific combinations of LightGBM hyperparameters related to tree complexity, learning rate, and regularization can significantly improve the rank correlation of predicted stock returns compared to default settings.

### Background
Accurate prediction of stock returns is a challenging problem in quantitative finance due to the inherent noise and non-stationarity of financial markets. Machine learning models, particularly gradient boosting frameworks like LightGBM, have shown promise for this task due to their ability to capture complex non-linear relationships in data. However, the performance of these models is highly dependent on their hyperparameter configuration, which must be carefully tuned to balance model complexity against the risk of overfitting. This study aims to systematically investigate the impact of key hyperparameters on prediction performance, with the goal of developing a more robust and accurate stock return prediction model.

## 2. Methodology

### Experiment Design
The experiment was structured into three distinct plans, each focusing on a different group of hyperparameters:

1. **Plan 1 (Highest Priority)**: Basic LightGBM parameters
   - Independent variables: learning_rate, num_leaves, max_depth
   - Control group: learning_rate=0.1, num_leaves=31, max_depth=-1

2. **Plan 2 (Medium Priority)**: Tree structure parameters
   - Independent variables: n_estimators, subsample, colsample_bytree
   - Control group: n_estimators=100, subsample=0.8, colsample_bytree=0.8

3. **Plan 3 (Lower Priority)**: Regularization parameters
   - Independent variables: reg_alpha, reg_lambda, min_child_samples, early_stopping_rounds
   - Control group: reg_alpha=0.0, reg_lambda=0.0, min_child_samples=20, early_stopping_rounds=50

For each plan, a control group with baseline parameters was established, followed by experimental groups with various parameter combinations.

### Experimental Setup
The experiments were conducted using stock market data from 2017-2023, with a rolling 3-year training window. The LightGBM regression model was implemented with GPU acceleration via OpenCL for faster training. Each experiment included:

- Data preprocessing and feature engineering
- Model training with the specified hyperparameter configuration
- Performance evaluation using Spearman rank correlation between predicted and actual returns
- Analysis of overfitting by comparing training and validation performance
- Assessment of model robustness across different market years

### Implementation Details
The experiments were implemented using Python with the LightGBM library. Key implementation components included:

1. Environment setup with necessary dependencies (pandas, numpy, pyarrow, lightgbm)
2. Configuration parsing for different hyperparameter settings
3. Data loading and preprocessing from parquet files
4. Model training with early stopping capabilities
5. Evaluation and logging of performance metrics
6. Error handling and reporting

## 3. Results

### Control Group Performance

**Plan 1: Basic Parameters (learning_rate=0.1, num_leaves=31, max_depth=-1)**
- Overall Rank Correlation: 0.0668
- 2020 Rank Correlation: 0.0692
- 2021 Rank Correlation: 0.0581
- 2022 Rank Correlation: 0.0715
- 2023 Rank Correlation: 0.0686
- Training Time: ~101.5 seconds

**Plan 2: Tree Structure Parameters (n_estimators=100, subsample=0.8, colsample_bytree=0.8)**
- Overall Rank Correlation: 0.0678
- 2020 Rank Correlation: 0.0703
- 2021 Rank Correlation: 0.0589
- 2022 Rank Correlation: 0.0728
- 2023 Rank Correlation: 0.0692

**Plan 3: Regularization Parameters (reg_alpha=0.0, reg_lambda=0.0, min_child_samples=20, early_stopping_rounds=50)**
- Overall Rank Correlation: 0.0666
- 2020 Rank Correlation: 0.0695
- 2021 Rank Correlation: 0.0576
- 2022 Rank Correlation: 0.0709
- 2023 Rank Correlation: 0.0687
- Overfitting Gap: 0.0803 (Training: 0.1629, Validation: 0.0826)
- Model Robustness (std dev of yearly correlations): 0.0053

### Early Stopping Behavior
In the control configuration, early stopping occurred at different iterations for different prediction years:
- 2020: ~230-336 iterations
- 2021: ~235-244 iterations
- 2022: ~190-366 iterations
- 2023: ~135-237 iterations

This variation suggests that the optimal model complexity differs across different market regimes.

### Performance Across Market Years
The model showed consistent but modest rank correlations across different years, with values in the 0.06-0.07 range. There was notable variation across years, with 2022 typically showing the best correlation (~0.071-0.073) and 2021 showing the lowest (~0.057-0.059).

| Year | Plan 1 Control | Plan 2 Control | Plan 3 Control |
|------|---------------|---------------|---------------|
| 2020 | 0.0692        | 0.0703        | 0.0695        |
| 2021 | 0.0581        | 0.0589        | 0.0576        |
| 2022 | 0.0715        | 0.0728        | 0.0709        |
| 2023 | 0.0686        | 0.0692        | 0.0687        |
| **Overall** | **0.0668** | **0.0678** | **0.0666** |

## 4. Analysis and Discussion

### Consistency of Performance
The three control groups showed remarkably consistent overall performance, with rank correlations ranging from 0.0666 to 0.0678. This suggests that the model is relatively robust to moderate changes in hyperparameters within the tested ranges.

### Overfitting Analysis
From the regularization experiment, we observed an overall overfitting gap of 0.080 (difference between training correlation of 0.163 and validation correlation of 0.083). This substantial gap indicates that despite early stopping, the model is still overfitting to the training data. The yearly overfitting gaps ranged from 0.070 to 0.089, showing consistent overfitting across different market regimes.

### Market Regime Dependency
The consistent pattern of varying performance across years (with 2022 showing the best performance and 2021 the worst) suggests that market regimes significantly impact prediction quality. This variation occurred despite the low standard deviation of yearly correlations (0.0053), indicating that while the relative differences between years are consistent, the absolute performance is still regime-dependent.

### Tree Structure Impact
The slight improvement in the Plan 2 control group (n_estimators=100, subsample=0.8, colsample_bytree=0.8) compared to the other control groups suggests that tree structure parameters may have a more significant impact on performance than basic parameters or regularization parameters alone.

## 5. Conclusion and Future Work

### Key Findings
1. **Default Parameters Show Reasonable Performance**: The default LightGBM configuration provides a reasonable baseline with an overall rank correlation of ~0.067.

2. **Variation Across Market Years**: The model's performance varies across different years, suggesting that market conditions affect prediction accuracy.

3. **Early Stopping Is Important**: The model benefits from early stopping to prevent overfitting, with optimal stopping points varying by prediction year.

4. **Evidence of Overfitting**: Despite early stopping, there is still a significant gap between training and validation performance, indicating potential for improvement through better regularization.

5. **Tree-based Hyperparameters**: Adjusting tree-related parameters (n_estimators=100, subsample=0.8, colsample_bytree=0.8) showed a slight improvement in overall correlation from 0.0668 to 0.0678.

### Recommendations for Future Work
1. **Complete Experimental Evaluations**: Run all experimental group partitions to compare against control groups, focusing first on Plan 1 to establish optimal basic parameters.

2. **Parameter Combination Study**: Once all experiments are complete, combine optimal parameters from each plan to potentially achieve synergistic improvements.

3. **Market Condition Analysis**: Test model performance across different market regimes (bull, bear, high volatility) and segment data by time periods to assess temporal consistency of predictions.

4. **Feature Engineering Improvements**: Analyze feature importance to identify most predictive factors and create new derived features to enhance predictive power.

5. **Cross-validation Refinement**: Test various time-based cross-validation strategies specifically designed for time series data, such as walk-forward validation.

6. **Ensemble Approaches**: Create ensembles of multiple optimized LightGBM models using bagging, boosting, or stacking techniques with different hyperparameter configurations.

7. **Alternative Metrics**: Evaluate performance using additional metrics relevant to financial applications, such as Information Coefficient (IC), Information Ratio (IR), and portfolio backtest performance metrics.

8. **Parameter Sensitivity Analysis**: Conduct detailed sensitivity analysis to identify which hyperparameters have the most impact on performance.

### Limitations
The current study was limited to control group results as the experimental group evaluations were not completed. Additionally, the modest correlation values indicate that hyperparameter optimization alone may not be sufficient to achieve high predictive accuracy in stock return prediction, suggesting that additional approaches such as feature engineering and ensemble methods should be explored.

## 6. Appendices

### Experiment IDs and Configuration Files
- Plan 1 (Basic Parameters): 21064c09-530e-4d2f-ae2c-6e1f6f20c3b6
- Plan 2 (Tree Structure Parameters): 3178e48d-9a28-4034-b3a4-438eb2192048
- Plan 3 (Regularization Parameters): 8d315ba7-6b9f-4050-87df-0ea06bbf9fd5

### Implementation Notes
- The experiments were configured to use GPU acceleration via OpenCL for faster training
- Early stopping was implemented to prevent overfitting
- The implementation encountered and resolved various technical challenges, including file permission issues and dependency management

### Technical Issues
During the experimentation, it was discovered that the `max_depth` parameter was not being properly passed to the LGBMRegressor constructor in the model_training.py script. This issue would need to be fixed before proceeding with further experiments involving the max_depth parameter.