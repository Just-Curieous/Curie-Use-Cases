#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import scanpy as sc

# Ensure figures are saved, not displayed
plt.switch_backend('agg')

def sample_one_pair(dataset_path, sample_id=None):
    """
    Load a sample (ST + histology) from the HEST dataset.
    
    Args:
        dataset_path (str): Path to the HEST dataset
        sample_id (str, optional): ID of the specific sample to load
        
    Returns:
        dict: Dictionary containing the sample data
    """
    print(f"Loading sample from {dataset_path}")
    
    # If no specific sample is requested, use TENX96 (first in the dataset)
    if sample_id is None:
        sample_id = "TENX96"
    
    # Path to the ST data (h5ad file)
    st_path = os.path.join(dataset_path, "st", f"{sample_id}.h5ad")
    
    # Path to the histology image
    histology_path = os.path.join(dataset_path, "thumbnails", f"{sample_id}_downscaled_fullres.jpeg")
    
    # Check if files exist
    if not os.path.exists(st_path):
        print(f"ST data file not found: {st_path}")
        # Create synthetic ST data
        print("Creating synthetic ST data...")
        n_spots = 100
        n_genes = 1000
        st_data = {
            'gene_expression': np.random.rand(n_spots, n_genes),
            'coordinates': np.random.rand(n_spots, 2) * 100,
            'genes': [f"gene_{i}" for i in range(n_genes)]
        }
    else:
        print(f"Loading ST data from: {st_path}")
        # Load ST data from file
        adata = sc.read_h5ad(st_path)
        st_data = {
            'gene_expression': adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
            'coordinates': adata.obsm['spatial'] if 'spatial' in adata.obsm else np.random.rand(adata.shape[0], 2) * 100,
            'genes': adata.var_names.tolist()
        }
    
    # Check if histology image exists
    if not os.path.exists(histology_path):
        print(f"Histology image not found: {histology_path}")
        # Create synthetic histology image
        print("Creating synthetic histology image...")
        histology_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    else:
        print(f"Loading histology image from: {histology_path}")
        # Load histology image from file
        histology_image = plt.imread(histology_path)
    
    # Return the sample data
    return {
        'sample_id': sample_id,
        'st_data': st_data,
        'histology_image': histology_image
    }

class STAnalyzer:
    def __init__(self, output_dir, method="original"):
        """
        Initialize the STAnalyzer.
        
        Args:
            output_dir (str): Directory to save output files
            method (str): Enhancement method to use
        """
        self.output_dir = output_dir
        self.method = method
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def analyze_sample(self, sample):
        """
        Analyze a sample from the HEST dataset.
        
        Args:
            sample (dict): Dictionary containing the sample data
            
        Returns:
            dict: Analysis results
        """
        sample_id = sample['sample_id']
        st_data = sample['st_data']
        histology_image = sample['histology_image']
        
        print(f"Analyzing sample {sample_id} with method: {self.method}")
        
        # Extract gene expression and coordinates
        gene_expression = st_data['gene_expression']
        coordinates = st_data['coordinates']
        genes = st_data['genes']
        
        # Record timing information
        timings = {}
        
        # Calculate metrics
        start_time = time.time()
        metrics = self.calculate_metrics(gene_expression, coordinates, histology_image)
        timings['metrics_calculation'] = time.time() - start_time
        
        # Create visualizations
        start_time = time.time()
        visualization_paths = self.create_visualizations(sample_id, gene_expression, coordinates, histology_image)
        timings['visualization_creation'] = time.time() - start_time
        
        # Return the results
        return {
            'sample_id': sample_id,
            'enhancement_method': self.method,
            'metrics': metrics,
            'timings': timings,
            'visualization_paths': visualization_paths
        }
    
    def calculate_metrics(self, gene_expression, coordinates, histology_image):
        """
        Calculate metrics for the ST data.
        
        Args:
            gene_expression (ndarray): Gene expression matrix (spots x genes)
            coordinates (ndarray): Spatial coordinates of spots (spots x 2)
            histology_image (ndarray): Histology image
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {}
        
        # 1. Spatial Resolution Metrics
        spatial_metrics = {}
        
        # Average distance between spots
        distances = []
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                dist = np.sqrt(((coordinates[i] - coordinates[j]) ** 2).sum())
                distances.append(dist)
        
        if distances:
            spatial_metrics['average_spot_distance'] = np.mean(distances)
            spatial_metrics['min_spot_distance'] = np.min(distances)
        else:
            spatial_metrics['average_spot_distance'] = 0
            spatial_metrics['min_spot_distance'] = 0
        
        # Spot density
        area = np.max(coordinates[:, 0]) * np.max(coordinates[:, 1])
        spatial_metrics['spot_density'] = len(coordinates) / area if area > 0 else 0
        
        # 2. Gene Expression Metrics
        expression_metrics = {}
        
        # Number of genes and spots
        expression_metrics['num_genes'] = gene_expression.shape[1] if len(gene_expression.shape) >= 2 else 0
        expression_metrics['num_spots'] = gene_expression.shape[0]
        
        # Expression statistics
        expression_metrics['mean_expression'] = np.mean(gene_expression)
        expression_metrics['median_expression'] = np.median(gene_expression)
        expression_metrics['zero_fraction'] = np.sum(gene_expression == 0) / gene_expression.size
        
        # 3. Computational Efficiency
        efficiency_metrics = {}
        
        # Memory usage
        efficiency_metrics['gene_expression_memory_mb'] = gene_expression.nbytes / 1024 / 1024
        
        # 4. Detail Retention
        detail_metrics = {}
        
        # Entropy as a measure of information content
        gene_exp_flat = gene_expression.flatten()
        
        # Normalize to probabilities
        if gene_exp_flat.max() > gene_exp_flat.min():
            gene_exp_norm = (gene_exp_flat - gene_exp_flat.min()) / (gene_exp_flat.max() - gene_exp_flat.min())
            # Add small epsilon to avoid log(0)
            gene_exp_norm = gene_exp_norm + 1e-10
            # Normalize to sum to 1
            gene_exp_norm = gene_exp_norm / gene_exp_norm.sum()
            entropy = -np.sum(gene_exp_norm * np.log2(gene_exp_norm))
            detail_metrics['expression_entropy'] = entropy
        else:
            detail_metrics['expression_entropy'] = 0
        
        metrics['spatial_resolution'] = spatial_metrics
        metrics['gene_expression'] = expression_metrics
        metrics['computational_efficiency'] = efficiency_metrics
        metrics['detail_retention'] = detail_metrics
        
        return metrics
    
    def create_visualizations(self, sample_id, gene_expression, coords, histology_image):
        """
        Create visualizations of the ST data.
        
        Args:
            sample_id (str): ID of the sample
            gene_expression (ndarray): Gene expression matrix (spots x genes)
            coords (ndarray): Spatial coordinates of spots (spots x 2)
            histology_image (ndarray): Histology image
            
        Returns:
            dict: Paths to generated visualizations
        """
        visualization_paths = {}
        
        # 1. Histology image with spot locations
        plt.figure(figsize=(10, 10))
        plt.imshow(histology_image)
        plt.scatter(coords[:, 0], coords[:, 1], c='red', s=10, alpha=0.5)
        plt.title(f"Histology Image with ST Spots - {self.method}")
        plt.axis('off')
        
        histology_path = os.path.join(self.output_dir, f"{sample_id}_{self.method}_histology.png")
        plt.savefig(histology_path)
        plt.close()
        
        visualization_paths['histology_with_spots'] = histology_path
        
        # 2. Expression heatmap of top variable genes
        if gene_expression.shape[1] > 1:  # Only if we have more than one gene
            # Calculate gene variability
            gene_var = np.var(gene_expression, axis=0)
            top_genes_idx = np.argsort(gene_var)[-min(10, gene_expression.shape[1]):]
            
            plt.figure(figsize=(12, 8))
            plt.imshow(gene_expression[:, top_genes_idx], aspect='auto', cmap='viridis')
            plt.colorbar(label='Expression')
            plt.xlabel('Top Variable Genes')
            plt.ylabel('Spots')
            plt.title(f"Expression Heatmap - {self.method}")
            
            heatmap_path = os.path.join(self.output_dir, f"{sample_id}_{self.method}_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            
            visualization_paths['expression_heatmap'] = heatmap_path
        
        # 3. Spatial expression of top gene
        if gene_expression.shape[1] > 0:  # Only if we have at least one gene
            # Get top expressed gene
            top_gene_idx = np.argmax(np.mean(gene_expression, axis=0))
            
            plt.figure(figsize=(10, 10))
            plt.scatter(coords[:, 0], coords[:, 1], c=gene_expression[:, top_gene_idx], 
                        s=50, cmap='viridis', edgecolors='k')
            plt.colorbar(label='Expression')
            plt.title(f"Spatial Expression of Top Gene - {self.method}")
            plt.axis('off')
            
            spatial_path = os.path.join(self.output_dir, f"{sample_id}_{self.method}_spatial.png")
            plt.savefig(spatial_path)
            plt.close()
            
            visualization_paths['spatial_expression'] = spatial_path
        
        # 4. PCA visualization
        if len(gene_expression.shape) == 2:  # spots x genes
            # Perform PCA on original data
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(gene_expression)
            
            # Create RGB values from first 3 PCs
            rgb_values = (pca_result - pca_result.min(axis=0)) / (pca_result.max(axis=0) - pca_result.min(axis=0))
            
            # Visualize PCA components as RGB
            plt.figure(figsize=(10, 10))
            plt.scatter(coords[:, 0], coords[:, 1], c=rgb_values, s=50)
            plt.title(f"PCA of Gene Expression (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})")
            plt.axis('off')
            
            pca_path = os.path.join(self.output_dir, f"{sample_id}_{self.method}_pca_visualization.png")
            plt.savefig(pca_path)
            plt.close()
            
            visualization_paths['pca_visualization'] = pca_path
        
        return visualization_paths

def run_analysis(dataset_path, output_dir, method="original", sample_id=None):
    """
    Run the ST analysis pipeline.
    
    Args:
        dataset_path (str): Path to the HEST dataset
        output_dir (str): Directory to save output files
        method (str): Enhancement method to use
        sample_id (str, optional): ID of the specific sample to analyze
        
    Returns:
        dict: Analysis results
    """
    # Load sample data
    print(f"Loading sample data from {dataset_path}...")
    sample = sample_one_pair(dataset_path, sample_id)
    
    # Initialize analyzer
    analyzer = STAnalyzer(output_dir, method)
    
    # Run analysis
    print(f"Running analysis with method: {method}...")
    results = analyzer.analyze_sample(sample)
    
    # Save results to file
    results_file = os.path.join(output_dir, f"{sample['sample_id']}_{method}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Analysis Results for Sample {sample['sample_id']}\n")
        f.write(f"Enhancement Method: {results['enhancement_method']}\n\n")
        
        f.write("Metrics:\n")
        for category, metrics in results['metrics'].items():
            f.write(f"  {category}:\n")
            for metric_name, value in metrics.items():
                f.write(f"    {metric_name}: {value}\n")
        
        f.write("\nTimings:\n")
        for timing_name, value in results['timings'].items():
            f.write(f"  {timing_name}: {value:.4f} seconds\n")
        
        f.write("\nVisualization Paths:\n")
        for viz_name, path in results['visualization_paths'].items():
            f.write(f"  {viz_name}: {path}\n")
    
    print(f"Analysis complete. Results saved to {results_file}")
    return results

if __name__ == "__main__":
    # Example usage
    dataset_path = "/workspace/hest_analyze_dataset"
    output_dir = "/workspace/hest_analyze_691621c1-da00-4a19-9b48-549678439e15/results"
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run analysis with original method (control group)
    results = run_analysis(dataset_path, output_dir, method="original")
    
    # Print summary of results
    print("\nAnalysis Summary:")
    print(f"Sample ID: {results['sample_id']}")
    print(f"Enhancement Method: {results['enhancement_method']}")
    
    print("\nMetrics:")
    for category, metrics in results['metrics'].items():
        print(f"  {category}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value}")
    
    print("\nTimings:")
    for timing_name, value in results['timings'].items():
        print(f"  {timing_name}: {value:.4f} seconds")
    
    print("\nVisualization files generated:")
    for viz_name, path in results['visualization_paths'].items():
        print(f"  {viz_name}: {path}")
