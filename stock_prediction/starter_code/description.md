# Running the Model Training Script with Multiple Configurations
Your main job is to copy `sample_config.json` and edit the configuratoins. 
The main python file is `model_training.py`, which is correct executable. 

Here's how to configure and run the model training script with different parameter sets:

0. **Setup**
   **Be sure to include this in your workflow script!!!** to support efficient model training.
   ```bash
   mkdir -p /etc/OpenCL/vendors
   echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
   ```

1. **Basic usage** of the training code.
   ```bash
   python model_training.py --config sample_config.json
   ```

2. **Read `sample_config.json` configuration file** with different parameter values:

- Understand the training configurations within `sample_config.json`. 
- Copy `sample_config.json` and edit the new configuration file to tune the variables, such as `data_path`.
- Note that if you configure the hyperparameters out of the range, it might cause bugs.

3. **Run each configuration**: 

Each run will create its own timestamped output file `predictions_*.parquet` and result file `metrics_*.json` in the results directory, making it easy to compare performance across different parameter settings.
You just need to focus on the performance reported in the result file `metrics_*.json`.