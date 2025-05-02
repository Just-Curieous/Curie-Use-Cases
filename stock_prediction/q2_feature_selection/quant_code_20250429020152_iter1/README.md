## Feature Selection for Stock Prediction Model


This directory contains logs and results for feature selection experiments on stock prediction models. The implementation is based on the starter code (`../../starter_code`) with optimizations for feature selection. (The raw code is missing QAQ.)

### Running Experiments
To run feature selection experiments:
```bash
cd Curie/
python3 -m curie.main -f ~/Curie-Use-Cases/stock_prediction/starter_code/questions/feature-question.txt  --task_config curie/configs/quant_config.json  --report
```
- We use an A40 GPU as the local machine.
- You need to modify the dataset and starter code path specified in `quant_config.json`.

### Configuration
Key parameters to tune from sample_config.json:
```json
{
    "feature_threshold": 0.75,
    "min_samples": 1650,
    "num_workers": 40,
    "num_simulations": 3
}
```

### Curie Results

Experiment results and analysis can be found in the following files:
- **Experiment Report: [feature-question_20250429020152_iter1.md](./feature-question_20250429020152_iter1.md)**
- Experiment results (summarized from the raw results): [feature-question_20250429020152_iter1_all_results.txt](./feature-question_20250429020152_iter1_all_results.txt)
- Raw Curie execution Log: [feature-question_20250429020152_iter1.log](./feature-question_20250429020152_iter1.log)
- Raw Curie configuration Files: [quant_code_config_feature-question_20250429020152_iter1.json](./quant_code_config_feature-question_20250429020152_iter1.json)  
