# Hyperparameter Optimization for Stock Prediction Model

This directory contains experiment results and logs for hyperparameter optimization of the stock prediction model. The implementation is based on the starter code (`../../starter_code`).

## Running Experiments

To run hyperparameter optimization experiments:

```bash
cd Curie/
python3 -m curie.main -f ~/Curie-Use-Cases/stock_prediction/starter_code/questions/hyper-question.txt --task_config curie/configs/quant_config.json --report
```

Requirements:
- A40 GPU for model training
- Configure your dataset path in `quant_config.json`

## Configuration

Key hyperparameters to optimize:
```json
{
    "num_leaves": 511,
    "learning_rate": 0.02, 
    "min_child_samples": 30,
    "n_estimators": 10000,
    "subsample": 0.7,
    "colsample_bytree": 0.7, 
}
```

## Experiment Results

The following files contain experiment outputs:

- **Report: [hyper-question_20250428233941_iter1.md](./hyper-question_20250428233941_iter1.md)**
- Results Summary: [hyper-question_20250428233941_iter1_all_results.txt](./hyper-question_20250428233941_iter1_all_results.txt)
- Execution Log: [hyper-question_20250428233941_iter1.log](./hyper-question_20250428233941_iter1.log)
- Configuration: [quant_code_config_hyper-question_20250428233941_iter1.json](./quant_code_config_hyper-question_20250428233941_iter1.json)

## Workspace References

Additional workspace files related to this experiment:
- Plan 1: [quant_code_21064c09-530e-4d2f-ae2c-6e1f6f20c3b6](./quant_code_21064c09-530e-4d2f-ae2c-6e1f6f20c3b6)
- Plan 2: [quant_code_3178e48d-9a28-4034-b3a4-438eb2192048](./quant_code_3178e48d-9a28-4034-b3a4-438eb2192048)
- Plan 3: [quant_code_8d315ba7-6b9f-4050-87df-0ea06bbf9fd5](./quant_code_8d315ba7-6b9f-4050-87df-0ea06bbf9fd5)
