#!/bin/bash
set -e

WORKSPACE_DIR="/workspace/quant_code_21064c09-530e-4d2f-ae2c-6e1f6f20c3b6"
RESULTS_FILE="${WORKSPACE_DIR}/results_21064c09-530e-4d2f-ae2c-6e1f6f20c3b6_control_group_partition_1.txt"
METRICS_FILE="${WORKSPACE_DIR}/metrics_21064c09-530e-4d2f-ae2c-6e1f6f20c3b6_control_group_partition_1.json"
PYTHON_PATH="${WORKSPACE_DIR}/venv/bin/python3"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${WORKSPACE_DIR}/results_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}"

# Create config file directly
cat > "${WORKSPACE_DIR}/control_config.json" << EOC
{
    "data_path": "/workspace/quant_code_dataset", 
    "num_years_train": 3,
    "start_year": 2017,
    "end_year": 2023,
    "min_samples": 1650,
    "min_trading_volume": 5000000,
    "feature_threshold": 0.75,
    "min_price": 2,
    "lgbm_params": {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "max_depth": -1,
        "verbose": -1,
        "min_child_samples": 30,
        "n_estimators": 10000,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "early_stopping_rounds": 100,
        "log_evaluation_freq": 500
    },
    "num_workers": 40,
    "num_simulations": 3,
    "device_type": "gpu",
    "results_path": "${RESULTS_DIR}"
}
EOC

# Log experiment start
echo "==================================================" | tee "${RESULTS_FILE}"
echo "EXPERIMENT: 21064c09-530e-4d2f-ae2c-6e1f6f20c3b6" | tee -a "${RESULTS_FILE}"
echo "CONTROL GROUP: Partition 1" | tee -a "${RESULTS_FILE}"
echo "TIMESTAMP: $(date)" | tee -a "${RESULTS_FILE}"
echo "==================================================" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"

# Log configuration
echo "Configuration:" | tee -a "${RESULTS_FILE}"
echo "- num_leaves: 31" | tee -a "${RESULTS_FILE}"
echo "- learning_rate: 0.1" | tee -a "${RESULTS_FILE}"
echo "- max_depth: -1" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"

# Run the model training script
echo "Starting model training..." | tee -a "${RESULTS_FILE}"
START_TIME=$(date +%s)

# Execute the model training script and capture output
"${PYTHON_PATH}" "${WORKSPACE_DIR}/model_training.py" --config "${WORKSPACE_DIR}/control_config.json" 2>&1 | tee -a "${RESULTS_FILE}"

# Calculate execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))
echo "" | tee -a "${RESULTS_FILE}"
echo "Total execution time: ${EXECUTION_TIME} seconds" | tee -a "${RESULTS_FILE}"

# Find latest metrics file
LATEST_METRICS=$(find "${RESULTS_DIR}" -name "metrics_*.json" -type f | sort | tail -1)

if [ -f "${LATEST_METRICS}" ]; then
    echo "Metrics file found: ${LATEST_METRICS}" | tee -a "${RESULTS_FILE}"
    echo "Copying to: ${METRICS_FILE}" | tee -a "${RESULTS_FILE}"
    cp "${LATEST_METRICS}" "${METRICS_FILE}"
    echo "Metrics Summary:" | tee -a "${RESULTS_FILE}"
    cat "${METRICS_FILE}" >> "${RESULTS_FILE}"
else
    echo "WARNING: No metrics file found!" | tee -a "${RESULTS_FILE}"
fi

echo "" | tee -a "${RESULTS_FILE}"
echo "==================================================" | tee -a "${RESULTS_FILE}"
echo "EXPERIMENT COMPLETED" | tee -a "${RESULTS_FILE}"
echo "Results saved to: ${RESULTS_FILE}" | tee -a "${RESULTS_FILE}"
echo "==================================================" | tee -a "${RESULTS_FILE}"
