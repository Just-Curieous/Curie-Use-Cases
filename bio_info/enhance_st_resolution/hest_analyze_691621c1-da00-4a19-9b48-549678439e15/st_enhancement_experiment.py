#!/usr/bin/env python3
"""
Spatial Transcriptomics Enhancement Experiment

This script implements various methods for enhancing the resolution of spatial transcriptomics data:
1. Bicubic interpolation
2. Deep learning super-resolution (SRCNN)
3. Histology-guided deep learning enhancement
4. Gene expression aware variational autoencoder (VAE)

For each method, it evaluates:
- Spatial resolution improvement
- Gene expression accuracy preservation
- Computational efficiency
- Detail retention in spatial patterns
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import psutil
import gc
from sample_data_loader import sample_one_pair
from experimental_st_enhancement import enhance_st_data
from experimental_st_analysis import run_experimental_analysis, compare_methods

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def run_enhancement_experiment(dataset_path, output_dir, sample_id=None):
    """
    Run the enhancement experiment with all methods.
    
    Args:
        dataset_path (str): Path to the HEST dataset
        output_dir (str): Directory to save output files
        sample_id (str, optional): ID of the specific sample to analyze
        
    Returns:
        dict: Dictionary of results for each method
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Methods to evaluate
    methods = ["bicubic", "srcnn", "histology_guided", "gene_vae"]
    all_results = {}
    
    # Run analysis with each method
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running analysis with method: {method}")
        print(f"{'='*50}")
        
        results = run_experimental_analysis(dataset_path, output_dir, method, sample_id)
        all_results[method] = results
    
    # Compare all methods
    compare_methods(all_results, output_dir)
    
    return all_results

def generate_summary_report(results_dict, output_dir):
    """
    Generate a comprehensive summary report of all methods.
    
    Args:
        results_dict (dict): Dictionary of results for each method
        output_dir (str): Directory to save the report
    """
    methods = list(results_dict.keys())
    sample_id = results_dict[methods[0]]['sample_id']
    
    report_file = os.path.join(output_dir, f"{sample_id}_enhancement_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("SPATIAL TRANSCRIPTOMICS ENHANCEMENT EXPERIMENT REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Sample ID: {sample_id}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY OF ENHANCEMENT METHODS\n")
        f.write("-"*60 + "\n\n")
        
        # Method descriptions
        method_descriptions = {
            "bicubic": "Classical image processing technique that uses bicubic interpolation to increase resolution.",
            "srcnn": "Deep learning super-resolution using convolutional neural networks.",
            "histology_guided": "Enhancement guided by histology images to improve spatial resolution.",
            "gene_vae": "Variational autoencoder that preserves gene expression patterns during enhancement."
        }
        
        for method in methods:
            f.write(f"{method.upper()}:\n")
            f.write(f"  Description: {method_descriptions.get(method, 'No description available')}\n")
            f.write(f"  Enhancement Time: {results_dict[method]['timings']['enhancement_time']:.4f} seconds\n")
            if 'resolution_factor' in results_dict[method]['metrics']['spatial_resolution']:
                f.write(f"  Resolution Factor: {results_dict[method]['metrics']['spatial_resolution']['resolution_factor']:.2f}x\n")
            f.write("\n")
        
        # Best method for each metric
        f.write("BEST METHODS BY METRIC\n")
        f.write("-"*60 + "\n\n")
        
        metrics_to_compare = {
            "Spatial Resolution (SSIM)": ("spatial_resolution", "ssim", max),
            "Gene Expression Preservation": ("gene_expression", "correlation", max),
            "Computational Efficiency": ("computational_efficiency", "memory_usage_mb", min),
            "Detail Retention": ("detail_retention", "edge_preservation", max),
            "Processing Speed": ("timings", "enhancement_time", min)
        }
        
        for metric_name, (category, metric, best_func) in metrics_to_compare.items():
            f.write(f"{metric_name}:\n")
            
            try:
                if category == "timings":
                    values = {method: results_dict[method]['timings'][metric] for method in methods}
                else:
                    values = {method: results_dict[method]['metrics'][category][metric] for method in methods if metric in results_dict[method]['metrics'][category]}
                
                if values:
                    best_method = best_func(values.items(), key=lambda x: x[1])[0]
                    f.write(f"  Best Method: {best_method}\n")
                    f.write(f"  Value: {values[best_method]:.4f}\n")
                else:
                    f.write(f"  No valid data available for this metric\n")
            except Exception as e:
                f.write(f"  Error computing best method: {str(e)}\n")
            
            f.write("\n")
        
        # Overall recommendation
        f.write("OVERALL RECOMMENDATION\n")
        f.write("-"*60 + "\n\n")
        
        try:
            # Simple scoring system (lower is better)
            scores = {method: 0 for method in methods}
            
            for category, metric, best_func in metrics_to_compare.values():
                try:
                    if category == "timings":
                        values = {method: results_dict[method]['timings'][metric] for method in methods}
                    else:
                        values = {method: results_dict[method]['metrics'][category][metric] for method in methods if metric in results_dict[method]['metrics'][category]}
                    
                    if not values:
                        continue
                        
                    # Normalize values to 0-1 range
                    min_val = min(values.values())
                    max_val = max(values.values())
                    
                    if min_val == max_val:
                        continue  # Skip metrics where all methods have the same value
                        
                    for method in methods:
                        if method in values:
                            if best_func == min:
                                scores[method] += (values[method] - min_val) / (max_val - min_val + 1e-10)
                            else:  # max
                                scores[method] += 1 - (values[method] - min_val) / (max_val - min_val + 1e-10)
                except Exception as e:
                    f.write(f"Warning: Error processing metric {metric}: {str(e)}\n")
            
            if scores:
                best_method = min(scores.items(), key=lambda x: x[1])[0]
                
                f.write(f"Based on the overall performance across all metrics, the recommended method is: {best_method.upper()}\n\n")
                f.write("Method rankings (lower score is better):\n")
                for method, score in sorted(scores.items(), key=lambda x: x[1]):
                    f.write(f"  {method}: {score:.4f}\n")
            else:
                f.write("Unable to determine overall recommendation due to lack of comparable metrics.\n")
        except Exception as e:
            f.write(f"Error generating recommendation: {str(e)}\n")
    
    print(f"Summary report saved to {report_file}")
    return report_file

if __name__ == "__main__":
    # Parameters
    dataset_path = "/workspace/hest_analyze_dataset"
    output_dir = "/workspace/hest_analyze_691621c1-da00-4a19-9b48-549678439e15/results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the enhancement experiment
    results = run_enhancement_experiment(dataset_path, output_dir)
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    print("\nExperiment completed successfully!")
