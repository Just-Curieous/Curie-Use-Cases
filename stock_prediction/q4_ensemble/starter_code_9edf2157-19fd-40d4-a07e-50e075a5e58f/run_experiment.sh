#!/bin/bash
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
export PATH="/openhands/micromamba/bin:\$PATH"
eval "\$(micromamba shell hook --shell bash)"
export VENV_PATH="/workspace/starter_code_9edf2157-19fd-40d4-a07e-50e075a5e58f/venv"
micromamba activate \$VENV_PATH/
cd /workspace/starter_code_9edf2157-19fd-40d4-a07e-50e075a5e58f/
python model_training.py --config config_mse_mae_huber_averaging.json > results_9edf2157-19fd-40d4-a07e-50e075a5e58f_experimental_group_partition_1.txt 2>&1

