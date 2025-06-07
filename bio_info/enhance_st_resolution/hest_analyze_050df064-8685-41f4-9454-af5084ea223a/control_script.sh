#!/bin/bash
# Install necessary dependencies
echo "Installing additional required packages..."
pip install python-igraph leidenalg

# Set up the directories
WORKSPACE_DIR="/workspace/hest_analyze_050df064-8685-41f4-9454-af5084ea223a"
DATASET_DIR="/workspace/hest_analyze_dataset"
RESULTS_DIR="\${WORKSPACE_DIR}/results"
OUTPUT_LOG="\${WORKSPACE_DIR}/results_050df064-8685-41f4-9454-af5084ea223a_control_group_partition_1.txt"

# Ensure results directory exists
mkdir -p "\${RESULTS_DIR}"

# Run the analysis and capture the output
echo "Running ST data analysis for control group..."
python3 "\${WORKSPACE_DIR}/run_analysis.py" \
  --dataset_dir "\${DATASET_DIR}" \
  --output_dir "\${RESULTS_DIR}" 2>&1 | tee "\${OUTPUT_LOG}"

# Verify results were generated
echo -e "\nChecking for results..."
if [ -f "\${RESULTS_DIR}/control_group_metrics.txt" ]; then
  echo "Results were successfully generated."
  echo "Contents of metrics file:"
  cat "\${RESULTS_DIR}/control_group_metrics.txt"
  
  # Create a symlink with the correct filename format
  cp "\${WORKSPACE_DIR}/control_script.sh" "\${WORKSPACE_DIR}/control_experiment_050df064-8685-41f4-9454-af5084ea223a_control_group_partition_1.sh"
  echo "Experiment completed successfully."
  exit 0
else
  echo "ERROR: Results file was not generated!"
  exit 1
fi
