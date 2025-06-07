#!/usr/bin/env python3
"""
ST Data Enhancement Experiment Analysis

This module implements the analysis of different enhancement methods for spatial transcriptomics data.
It evaluates:
1. Spatial resolution improvement
2. Gene expression accuracy preservation
3. Computational efficiency
4. Detail retention in spatial patterns
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import psutil
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.ndimage import sobel
from sample_data_loader import sample_one_pair
from experimental_st_enhancement import enhance_st_data, visualize_enhancement

class ExperimentalSTAnalyzer:
    """
    Class for analyzing ST data enhancement methods.
    """
    def __init__(self, dataset_path, output_dir):
        """
        Initialize the analyzer.
        
        Args:
            dataset_path (str): Path to the HEST dataset
            output_dir (str): Directory to save output files
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Device for computations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_sample(self, sample_id=None):
        """
        Load a sample from the dataset.
        
        Args:
            sample_id (str, optional): ID of the sample to load
            
        Returns:
            dict: Dictionary containing the sample data
        """
        print(f"Loading sample data from {self.dataset_path}...")
        sample = sample_one_pair(self.dataset_path, sample_id)
        print(f"Loaded sample: {sample['sample_id']}")
        
        return sample
    
    def enhance_st_data(self, st_data, histology=None, method='bicubic', scale_factor=2):
        """
        Enhance ST data using the specified method.
        
        Args:
            st_data (numpy.ndarray): ST data to enhance
            histology (numpy.ndarray, optional): Histology image
            method (str): Enhancement method
            scale_factor (int): Scale factor for enhancement
            
        Returns:
            dict: Results dictionary
        """
        print(f"Enhancing ST data using method: {method}...")
        results = enhance_st_data(st_data, histology, method, scale_factor, self.device)
        
        return results
    
    def evaluate_enhancement(self, original_data, enhanced_data, method, downscale_factor=None):
        """
        Evaluate the enhancement results.
        
        Args:
            original_data (numpy.ndarray): Original ST data
            enhanced_data (numpy.ndarray): Enhanced ST data
            method (str): Enhancement method
            downscale_factor (float, optional): Factor to downscale enhanced data for comparison
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"Evaluating enhancement results for method: {method}...")
        
        # Start timing
        eval_start_time = time.time()
        
        # If downscale_factor is not provided, assume enhanced data has already been upscaled
        if downscale_factor is None:
            # Calculate the scale factor from the data shapes
            height_ratio = enhanced_data.shape[0] / original_data.shape[0]
            width_ratio = enhanced_data.shape[1] / original_data.shape[1]
            downscale_factor = min(height_ratio, width_ratio)
        
        # Get original and enhanced shapes
        orig_shape = original_data.shape
        enhanced_shape = enhanced_data.shape
        
        # Calculate spatial resolution metrics
        spatial_metrics = self._evaluate_spatial_resolution(original_data, enhanced_data, downscale_factor)
        
        # Calculate gene expression preservation metrics
        gene_metrics = self._evaluate_gene_expression(original_data, enhanced_data, downscale_factor)
        
        # Calculate computational efficiency metrics
        efficiency_metrics = self._evaluate_computational_efficiency(enhanced_data)
        
        # Calculate detail retention metrics
        detail_metrics = self._evaluate_detail_retention(original_data, enhanced_data, downscale_factor)
        
        # Calculate evaluation time
        eval_time = time.time() - eval_start_time
        
        return {
            'spatial_resolution': spatial_metrics,
            'gene_expression': gene_metrics,
            'computational_efficiency': efficiency_metrics,
            'detail_retention': detail_metrics,
            'evaluation_time': eval_time
        }
    
    def _evaluate_spatial_resolution(self, original_data, enhanced_data, scale_factor):
        """
        Evaluate spatial resolution improvement.
        
        Args:
            original_data (numpy.ndarray): Original ST data
            enhanced_data (numpy.ndarray): Enhanced ST data
            scale_factor (float): Factor by which the data was enhanced
            
        Returns:
            dict: Spatial resolution metrics
        """
        # Resize original data to match enhanced for comparison
        orig_upscaled = np.zeros_like(enhanced_data)
        for g in range(original_data.shape[2]):
            orig_upscaled[:, :, g] = resize(original_data[:, :, g], 
                                          enhanced_data[:, :, g].shape, 
                                          order=3, mode='edge')
        
        # Calculate mean squared error between bicubic interpolation and enhanced data
        bicubic_mse = mean_squared_error(orig_upscaled.flatten(), enhanced_data.flatten())
        
        # Calculate SSIM between original and enhanced data
        ssim_value = 0
        for g in range(original_data.shape[2]):
            ssim_value += ssim(
                resize(original_data[:, :, g], enhanced_data[:, :, g].shape, order=0, mode='edge'),
                enhanced_data[:, :, g],
                data_range=enhanced_data[:, :, g].max() - enhanced_data[:, :, g].min()
            )
        ssim_value /= original_data.shape[2]
        
        # Calculate effective resolution increase
        resolution_factor = (enhanced_data.shape[0] / original_data.shape[0] + 
                            enhanced_data.shape[1] / original_data.shape[1]) / 2
        
        return {
            'mse': bicubic_mse,
            'ssim': ssim_value,
            'resolution_factor': resolution_factor,
            'original_shape': original_data.shape,
            'enhanced_shape': enhanced_data.shape
        }
    
    def _evaluate_gene_expression(self, original_data, enhanced_data, scale_factor):
        """
        Evaluate gene expression accuracy preservation.
        
        Args:
            original_data (numpy.ndarray): Original ST data
            enhanced_data (numpy.ndarray): Enhanced ST data
            scale_factor (float): Factor by which the data was enhanced
            
        Returns:
            dict: Gene expression preservation metrics
        """
        # Downsample enhanced data to original size
        enhanced_downsampled = np.zeros_like(original_data)
        for g in range(original_data.shape[2]):
            enhanced_downsampled[:, :, g] = resize(enhanced_data[:, :, g], 
                                                original_data[:, :, g].shape, 
                                                order=3, mode='edge')
        
        # Calculate correlation between original and downsampled enhanced data
        gene_correlations = []
        for g in range(original_data.shape[2]):
            corr, _ = pearsonr(original_data[:, :, g].flatten(), enhanced_downsampled[:, :, g].flatten())
            gene_correlations.append(corr)
        
        avg_correlation = np.mean(gene_correlations)
        
        # Calculate gene expression preservation metric
        expression_mse = mean_squared_error(original_data.flatten(), enhanced_downsampled.flatten())
        expression_preservation = 1 / (1 + expression_mse)  # Higher is better
        
        return {
            'correlation': avg_correlation,
            'expression_preservation': expression_preservation,
            'gene_correlations': gene_correlations
        }
    
    def _evaluate_computational_efficiency(self, enhanced_data):
        """
        Evaluate computational efficiency.
        
        Args:
            enhanced_data (numpy.ndarray): Enhanced ST data
            
        Returns:
            dict: Computational efficiency metrics
        """
        # Calculate memory usage
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024**2  # in MB
        
        # Calculate enhanced data size
        data_size = enhanced_data.nbytes / 1024**2  # in MB
        
        return {
            'memory_usage_mb': memory_usage,
            'enhanced_data_size_mb': data_size
        }
    
    def _evaluate_detail_retention(self, original_data, enhanced_data, scale_factor):
        """
        Evaluate detail retention in spatial patterns.
        
        Args:
            original_data (numpy.ndarray): Original ST data
            enhanced_data (numpy.ndarray): Enhanced ST data
            scale_factor (float): Factor by which the data was enhanced
            
        Returns:
            dict: Detail retention metrics
        """
        # Calculate gradient magnitude for both original and enhanced
        orig_gradients = []
        enhanced_gradients = []
        
        for g in range(original_data.shape[2]):
            # Original gradients
            edge_x = sobel(original_data[:, :, g], axis=0)
            edge_y = sobel(original_data[:, :, g], axis=1)
            orig_grad = np.sqrt(edge_x**2 + edge_y**2)
            orig_gradients.append(np.mean(orig_grad))
            
            # Enhanced gradients
            edge_x = sobel(enhanced_data[:, :, g], axis=0)
            edge_y = sobel(enhanced_data[:, :, g], axis=1)
            enhanced_grad = np.sqrt(edge_x**2 + edge_y**2)
            enhanced_gradients.append(np.mean(enhanced_grad))
        
        # Calculate gradient similarity
        grad_similarity = np.mean([e/o if o > 0 else 1 for e, o in zip(enhanced_gradients, orig_gradients)])
        
        # Calculate edge preservation
        # Resize original gradients for comparison
        orig_upscaled_grad = np.zeros_like(enhanced_data)
        edge_preservation = 0
        
        for g in range(original_data.shape[2]):
            # Original edges
            edge_x = sobel(original_data[:, :, g], axis=0)
            edge_y = sobel(original_data[:, :, g], axis=1)
            orig_grad = np.sqrt(edge_x**2 + edge_y**2)
            
            # Resize original gradient
            orig_upscaled_grad[:, :, g] = resize(orig_grad, enhanced_data[:, :, g].shape, 
                                              order=1, mode='edge')
            
            # Enhanced edges
            edge_x = sobel(enhanced_data[:, :, g], axis=0)
            edge_y = sobel(enhanced_data[:, :, g], axis=1)
            enhanced_grad = np.sqrt(edge_x**2 + edge_y**2)
            
            # Calculate structural similarity of edges
            edge_sim = ssim(
                orig_upscaled_grad[:, :, g],
                enhanced_grad,
                data_range=max(orig_upscaled_grad[:, :, g].max(), enhanced_grad.max())
            )
            edge_preservation += edge_sim
        
        edge_preservation /= original_data.shape[2]
        
        return {
            'gradient_similarity': grad_similarity,
            'edge_preservation': edge_preservation
        }
    
    def visualize_results(self, original_data, enhanced_data, method, sample_id, gene_indices=None):
        """
        Visualize the enhancement results.
        
        Args:
            original_data (numpy.ndarray): Original ST data
            enhanced_data (numpy.ndarray): Enhanced ST data
            method (str): Enhancement method
            sample_id (str): Sample ID
            gene_indices (list, optional): Indices of genes to visualize
            
        Returns:
            list: Paths to visualization files
        """
        if gene_indices is None:
            gene_indices = [0, min(5, original_data.shape[2]-1), min(10, original_data.shape[2]-1)]
        
        visualization_paths = []
        
        for gene_idx in gene_indices:
            if gene_idx >= original_data.shape[2]:
                continue
                
            vis_path = visualize_enhancement(
                original_data, enhanced_data, method, sample_id, gene_idx, self.output_dir)
            
            if vis_path:
                visualization_paths.append(vis_path)
        
        return visualization_paths
    
    def save_results(self, results, sample_id, method):
        """
        Save analysis results to file.
        
        Args:
            results (dict): Analysis results
            sample_id (str): Sample ID
            method (str): Enhancement method
            
        Returns:
            str: Path to results file
        """
        results_file = os.path.join(self.output_dir, f"{sample_id}_{method}_results.txt")
        
        with open(results_file, 'w') as f:
            f.write(f"ENHANCEMENT RESULTS FOR SAMPLE {sample_id}\n")
            f.write(f"Method: {method}\n")
            f.write("=" * 60 + "\n\n")
            
            # Write enhancement results
            f.write("ENHANCEMENT DETAILS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Original shape: {results['original_shape']}\n")
            f.write(f"Enhanced shape: {results['enhanced_data'].shape}\n")
            f.write(f"Enhancement time: {results['enhancement_time']:.4f} seconds\n\n")
            
            # Write evaluation results
            f.write("EVALUATION METRICS\n")
            f.write("-" * 40 + "\n")
            
            # Spatial resolution
            f.write("Spatial Resolution:\n")
            for metric, value in results['metrics']['spatial_resolution'].items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.4f}\n")
                else:
                    f.write(f"  {metric}: {value}\n")
            f.write("\n")
            
            # Gene expression
            f.write("Gene Expression Preservation:\n")
            for metric, value in results['metrics']['gene_expression'].items():
                if metric != 'gene_correlations':
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric}: {value}\n")
            f.write("\n")
            
            # Computational efficiency
            f.write("Computational Efficiency:\n")
            for metric, value in results['metrics']['computational_efficiency'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
            
            # Detail retention
            f.write("Detail Retention:\n")
            for metric, value in results['metrics']['detail_retention'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
            
            # Timings
            f.write("TIMING INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Enhancement time: {results['enhancement_time']:.4f} seconds\n")
            f.write(f"Evaluation time: {results['metrics']['evaluation_time']:.4f} seconds\n")
            f.write(f"Total analysis time: {results['total_time']:.4f} seconds\n")
        
        print(f"Results saved to {results_file}")
        return results_file


def run_experimental_analysis(dataset_path, output_dir, method, sample_id=None):
    """
    Run experimental analysis for a given enhancement method.
    
    Args:
        dataset_path (str): Path to the dataset
        output_dir (str): Directory to save output files
        method (str): Enhancement method
        sample_id (str, optional): ID of the sample to analyze
        
    Returns:
        dict: Analysis results
    """
    start_time = time.time()
    
    # Create analyzer
    analyzer = ExperimentalSTAnalyzer(dataset_path, output_dir)
    
    # Load sample data
    sample = analyzer.load_sample(sample_id)
    sample_id = sample['sample_id']
    
    # Extract ST data and histology
    st_data = sample['st_data']
    histology = sample['histology']
    
    # Enhancement scale factor
    scale_factor = 4
    
    # Enhance ST data
    enhancement_results = analyzer.enhance_st_data(st_data, histology, method, scale_factor)
    enhanced_data = enhancement_results['enhanced_data']
    enhancement_time = enhancement_results['enhancement_time']
    
    # Evaluate enhancement
    evaluation_metrics = analyzer.evaluate_enhancement(st_data, enhanced_data, method, scale_factor)
    
    # Visualize results
    visualization_paths = analyzer.visualize_results(st_data, enhanced_data, method, sample_id)
    
    # Calculate total analysis time
    total_time = time.time() - start_time
    
    # Save results
    results = {
        'sample_id': sample_id,
        'enhancement_method': method,
        'original_shape': st_data.shape,
        'enhanced_data': enhanced_data,
        'enhancement_time': enhancement_time,
        'metrics': evaluation_metrics,
        'visualization_paths': visualization_paths,
        'total_time': total_time
    }
    
    results_file = analyzer.save_results(results, sample_id, method)
    
    # Prepare results for return (exclude large data arrays)
    return_results = {
        'sample_id': sample_id,
        'enhancement_method': method,
        'original_shape': st_data.shape,
        'enhanced_shape': enhanced_data.shape,
        'metrics': evaluation_metrics,
        'timings': {
            'enhancement_time': enhancement_time,
            'evaluation_time': evaluation_metrics['evaluation_time'],
            'total_analysis_time': total_time
        },
        'results_file': results_file,
        'visualization_paths': visualization_paths
    }
    
    return return_results


def compare_methods(results_dict, output_dir):
    """
    Compare different enhancement methods.
    
    Args:
        results_dict (dict): Dictionary of results for each method
        output_dir (str): Directory to save comparison results
    """
    # Extract metrics for comparison
    methods = list(results_dict.keys())
    sample_id = results_dict[methods[0]]['sample_id']
    
    # Create comparison tables
    comparison = {
        'spatial_resolution': {},
        'gene_expression': {},
        'computational_efficiency': {},
        'detail_retention': {}
    }
    
    for method in methods:
        for category in comparison.keys():
            comparison[category][method] = results_dict[method]['metrics'][category]
    
    # Save comparison to file
    comparison_file = os.path.join(output_dir, f"{sample_id}_method_comparison.txt")
    with open(comparison_file, 'w') as f:
        f.write(f"ENHANCEMENT METHOD COMPARISON FOR SAMPLE {sample_id}\n")
        f.write("="*60 + "\n\n")
        
        for category in comparison.keys():
            f.write(f"{category.upper()}:\n")
            
            # Get all metrics for this category
            all_metrics = set()
            for method_data in comparison[category].values():
                all_metrics.update(method_data.keys())
            
            # Create a table header
            f.write(f"{'Metric':<25}")
            for method in methods:
                f.write(f"{method:<15}")
            f.write("\n")
            
            f.write("-" * (25 + 15 * len(methods)) + "\n")
            
            # Fill in the table
            for metric in sorted(all_metrics):
                if metric in ['original_shape', 'enhanced_shape', 'gene_correlations']:
                    continue  # Skip complex structures
                
                f.write(f"{metric:<25}")
                for method in methods:
                    value = comparison[category][method].get(metric, "N/A")
                    if isinstance(value, float):
                        f.write(f"{value:<15.4f}")
                    else:
                        f.write(f"{value:<15}")
                f.write("\n")
            
            f.write("\n")
        
        # Compare timings
        f.write("TIMINGS:\n")
        f.write(f"{'Timing':<25}")
        for method in methods:
            f.write(f"{method:<15}")
        f.write("\n")
        
        f.write("-" * (25 + 15 * len(methods)) + "\n")
        
        for timing in ['enhancement_time', 'evaluation_time', 'total_analysis_time']:
            f.write(f"{timing:<25}")
            for method in methods:
                value = results_dict[method]['timings'][timing]
                f.write(f"{value:<15.4f}")
            f.write("\n")
        
        # Overall recommendation
        f.write("\nOVERALL RECOMMENDATION:\n")
        f.write("-" * 40 + "\n")
        
        # Choose best method based on combined metrics
        scores = {}
        metrics = {
            'spatial_resolution': {
                'mse': {'weight': 1.0, 'best': min},
                'ssim': {'weight': 1.0, 'best': max}
            },
            'gene_expression': {
                'correlation': {'weight': 1.5, 'best': max},
                'expression_preservation': {'weight': 1.5, 'best': max}
            },
            'computational_efficiency': {
                'memory_usage_mb': {'weight': 0.5, 'best': min}
            },
            'detail_retention': {
                'edge_preservation': {'weight': 1.0, 'best': max},
                'gradient_similarity': {'weight': 1.0, 'best': lambda x: abs(1-x)}
            }
        }
        
        for method in methods:
            scores[method] = 0
            
            for category, category_metrics in metrics.items():
                for metric_name, metric_info in category_metrics.items():
                    try:
                        value = comparison[category][method][metric_name]
                        # Normalize values across methods (0 to 1, where 1 is best)
                        all_values = [comparison[category][m][metric_name] for m in methods]
                        best_function = metric_info['best']
                        
                        if best_function == min:
                            # Lower is better
                            normalized = (max(all_values) - value) / (max(all_values) - min(all_values) + 1e-10)
                        elif best_function == max:
                            # Higher is better
                            normalized = (value - min(all_values)) / (max(all_values) - min(all_values) + 1e-10)
                        else:
                            # Custom function
                            values_transformed = [best_function(v) for v in all_values]
                            value_transformed = best_function(value)
                            normalized = (max(values_transformed) - value_transformed) / (max(values_transformed) - min(values_transformed) + 1e-10)
                        
                        scores[method] += normalized * metric_info['weight']
                    except (KeyError, ZeroDivisionError):
                        pass
        
        # Normalize final scores
        max_score = max(scores.values()) if scores else 1
        for method in scores:
            scores[method] = scores[method] / max_score * 100
            
        # Write recommendation
        best_method = max(scores.items(), key=lambda x: x[1])[0]
        f.write(f"Based on all metrics, the recommended method is: {best_method.upper()}\n\n")
        f.write("Method scores (higher is better):\n")
        for method, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {method:<15}: {score:.1f}/100\n")
    
    print(f"Method comparison saved to {comparison_file}")
    
    # Create visualization comparing methods
    plt.figure(figsize=(12, 8))
    
    # Plot scores
    plt.bar(methods, [scores[method] for method in methods])
    plt.title("Enhancement Method Comparison")
    plt.ylabel("Score (higher is better)")
    plt.ylim(0, 100)
    
    for i, method in enumerate(methods):
        plt.text(i, scores[method] + 2, f"{scores[method]:.1f}", ha='center')
    
    plt.tight_layout()
    comparison_chart = os.path.join(output_dir, f"{sample_id}_method_comparison.png")
    plt.savefig(comparison_chart)
    plt.close()
    
    return comparison_file


def run_method_comparison(dataset_path, output_dir, sample_id=None):
    """
    Run comparison of different enhancement methods.
    
    Args:
        dataset_path (str): Path to the dataset
        output_dir (str): Directory to save output files
        sample_id (str, optional): ID of the sample to analyze
        
    Returns:
        dict: Results for each method
    """
    # Methods to compare
    methods = ["bicubic", "srcnn", "histology_guided", "gene_vae"]
    all_results = {}
    
    # Run analysis for each method
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running analysis with method: {method}")
        print(f"{'='*50}")
        
        results = run_experimental_analysis(dataset_path, output_dir, method, sample_id)
        all_results[method] = results
    
    # Compare methods
    compare_methods(all_results, output_dir)
    
    return all_results


if __name__ == "__main__":
    # Example usage
    dataset_path = "/workspace/hest_analyze_dataset"
    output_dir = "/workspace/hest_analyze_691621c1-da00-4a19-9b48-549678439e15/results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run method comparison
    results = run_method_comparison(dataset_path, output_dir)
    
    # Print summary
    print("\nExperiment completed successfully.")
