#!/bin/bash
WORKSPACE_DIR="/workspace/hest_analyze_050df064-8685-41f4-9454-af5084ea223a"
DATASET_DIR="/workspace/hest_analyze_dataset"
RESULTS_DIR="/workspace/hest_analyze_050df064-8685-41f4-9454-af5084ea223a/results"
OUTPUT_LOG="/workspace/hest_analyze_050df064-8685-41f4-9454-af5084ea223a/results_050df064-8685-41f4-9454-af5084ea223a_control_group_partition_1.txt"

mkdir -p "\$RESULTS_DIR"

# Run analysis
python3 /workspace/hest_analyze_050df064-8685-41f4-9454-af5084ea223a/run_analysis.py \
  --dataset_dir "\$DATASET_DIR" \
  --output_dir "\$RESULTS_DIR" | tee -a "\$OUTPUT_LOG"

exit 0
