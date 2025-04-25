import json
import sys

def extract_metrics(metrics_file, config_file):
    # Load metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract metrics
    metrics = data.get('metrics', {})
    
    # Print overall correlation
    print(f"Overall Rank Correlation: {metrics.get('overall', 'N/A')}")
    
    # Print yearly metrics
    for year in sorted([k for k in metrics.keys() if k != 'overall']):
        print(f"{year} Rank Correlation: {metrics.get(year, 'N/A')}")
    
    # Print configuration details
    print("\nMODEL CONFIGURATION:")
    print(f"- Model: LightGBM Regressor")
    print(f"- Training Years: {config.get('num_years_train', 'N/A')}")
    print(f"- Start Year: {config.get('start_year', 'N/A')}")
    print(f"- End Year: {config.get('end_year', 'N/A')}")
    print(f"- Number of Leaves: {config.get('lgbm_params', {}).get('num_leaves', 'N/A')}")
    print(f"- Learning Rate: {config.get('lgbm_params', {}).get('learning_rate', 'N/A')}")
    print(f"- Number of Simulations: {config.get('num_simulations', 'N/A')}")
    print(f"- Device Type: {config.get('device_type', 'N/A')}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_metrics.py <metrics_file> <config_file>")
        sys.exit(1)
    
    extract_metrics(sys.argv[1], sys.argv[2])