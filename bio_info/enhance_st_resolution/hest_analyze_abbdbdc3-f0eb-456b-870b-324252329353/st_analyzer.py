#!/usr/bin/env python3
"""
Spatial Transcriptomics (ST) Analyzer

This script analyzes spatial transcriptomics data from the HEST dataset without integrating histology data.
It serves as the control group experiment for evaluating the importance of histology integration.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
try:
    from sklearn.manifold import UMAP
except ImportError:
    UMAP = None
try:
    import torch
except ImportError:
    torch = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('st_analyzer')

class STAnalyzer:
    """Class for analyzing spatial transcriptomics data."""
    
    def __init__(self, output_dir):
        """Initialize with output directory for results."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {output_dir}")
    
    def analyze_sample(self, gene_expression, spatial_coords, sample_id):
        """Analyze a single ST sample."""
        logger.info(f"Analyzing sample: {sample_id}")
        start_time = time.time()
        
        # Create results dictionary
        results = {
            'sample_id': sample_id,
            'metrics': {},
            'visualizations': {}
        }
        
        # Normalize gene expression (log + 1)
        gene_expression_norm = np.log1p(gene_expression)
        
        # Calculate spatial resolution metrics
        logger.info("Calculating spatial resolution metrics...")
        spatial_metrics = self._calculate_spatial_metrics(spatial_coords)
        results['metrics']['spatial_resolution'] = spatial_metrics
        
        # Calculate gene expression metrics
        logger.info("Calculating gene expression metrics...")
        expr_metrics = self._calculate_gene_expression_metrics(gene_expression)
        results['metrics']['gene_expression'] = expr_metrics
        
        # Perform dimensionality reduction (PCA)
        logger.info("Performing dimensionality reduction...")
        pca = PCA(n_components=min(50, gene_expression_norm.shape[1]))
        pca_result = pca.fit_transform(gene_expression_norm)
        
        # Perform clustering
        logger.info("Performing clustering for cell type identification...")
        n_clusters = min(8, gene_expression.shape[0] // 5)  # Heuristic for number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_result[:, :20])
        
        # Calculate clustering metrics
        logger.info("Calculating clustering metrics...")
        clustering_metrics = self._calculate_clustering_metrics(pca_result[:, :20], cluster_labels)
        results['metrics']['cell_typing'] = clustering_metrics
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        viz_paths = self._generate_visualizations(
            sample_id, gene_expression_norm, 
            spatial_coords, cluster_labels, pca_result
        )
        results['visualizations'] = viz_paths
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        
        # Save results to file
        self._save_results(results, sample_id)
        
        return results
    
    def _calculate_spatial_metrics(self, spatial_coords):
        """Calculate metrics related to spatial resolution."""
        metrics = {}
        
        # Number of spots
        metrics['num_spots'] = len(spatial_coords)
        
        # Calculate area
        x_range = np.max(spatial_coords[:, 0]) - np.min(spatial_coords[:, 0])
        y_range = np.max(spatial_coords[:, 1]) - np.min(spatial_coords[:, 1])
        area = x_range * y_range
        metrics['area'] = area
        
        # Calculate spot density
        metrics['spot_density'] = len(spatial_coords) / area
        
        # Calculate nearest neighbor distances
        dist_matrix = distance_matrix(spatial_coords, spatial_coords)
        np.fill_diagonal(dist_matrix, np.inf)
        min_distances = np.min(dist_matrix, axis=1)
        
        metrics['mean_spot_distance'] = np.mean(min_distances)
        metrics['median_spot_distance'] = np.median(min_distances)
        metrics['min_spot_distance'] = np.min(min_distances)
        metrics['max_spot_distance'] = np.max(min_distances)
        
        # Coverage statistics (using avg distance as spot radius)
        avg_radius = np.median(min_distances) / 2
        spot_area = np.pi * (avg_radius ** 2) * len(spatial_coords)
        metrics['coverage_ratio'] = spot_area / area
        
        return metrics
    
    def _calculate_gene_expression_metrics(self, gene_expression):
        """Calculate metrics related to gene expression."""
        metrics = {}
        
        # Basic statistics
        metrics['genes_per_spot_mean'] = np.mean(np.sum(gene_expression > 0, axis=1))
        metrics['genes_per_spot_median'] = np.median(np.sum(gene_expression > 0, axis=1))
        metrics['counts_per_spot_mean'] = np.mean(np.sum(gene_expression, axis=1))
        metrics['counts_per_spot_median'] = np.median(np.sum(gene_expression, axis=1))
        metrics['zero_fraction'] = np.sum(gene_expression == 0) / gene_expression.size
        
        # Gene statistics
        gene_means = np.mean(gene_expression, axis=0)
        gene_vars = np.var(gene_expression, axis=0)
        metrics['mean_gene_expression'] = np.mean(gene_means)
        metrics['median_gene_expression'] = np.median(gene_means)
        
        # Coefficient of variation
        gene_cv = np.sqrt(gene_vars) / (gene_means + 1e-8)
        metrics['mean_gene_cv'] = np.mean(gene_cv)
        
        return metrics
    
    def _calculate_clustering_metrics(self, embedding, cluster_labels):
        """Calculate metrics related to cell type clustering."""
        metrics = {}
        
        # Number of clusters
        metrics['num_clusters'] = len(np.unique(cluster_labels))
        
        # Cluster sizes
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        metrics['min_cluster_size'] = np.min(counts)
        metrics['max_cluster_size'] = np.max(counts)
        metrics['mean_cluster_size'] = np.mean(counts)
        metrics['median_cluster_size'] = np.median(counts)
        
        # Try to calculate silhouette score
        try:
            from sklearn.metrics import silhouette_score
            metrics['silhouette_score'] = silhouette_score(embedding, cluster_labels)
        except:
            metrics['silhouette_score'] = float('nan')
        
        return metrics
    
    def _generate_visualizations(self, sample_id, gene_expression, 
                               spatial_coords, cluster_labels, pca_result):
        """Generate visualizations for the sample."""
        viz_paths = {}
        
        # Create spatial plot colored by clusters
        plt.figure(figsize=(10, 10))
        plt.scatter(
            spatial_coords[:, 0], spatial_coords[:, 1],
            c=cluster_labels, cmap='tab20', s=30, alpha=0.8
        )
        plt.title(f"Spatial Distribution of Clusters (Sample: {sample_id})")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.colorbar(label="Cluster")
        plt.grid(alpha=0.3)
        
        # Save spatial plot
        spatial_plot_path = os.path.join(self.output_dir, f"{sample_id}_spatial_clusters.png")
        plt.savefig(spatial_plot_path)
        plt.close()
        viz_paths['spatial_clusters'] = spatial_plot_path
        
        # Create PCA plot
        plt.figure(figsize=(10, 8))
        plt.scatter(
            pca_result[:, 0], pca_result[:, 1],
            c=cluster_labels, cmap='tab20', s=30, alpha=0.8
        )
        plt.title(f"PCA of Gene Expression (Sample: {sample_id})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(label="Cluster")
        plt.grid(alpha=0.3)
        
        # Save PCA plot
        pca_plot_path = os.path.join(self.output_dir, f"{sample_id}_pca.png")
        plt.savefig(pca_plot_path)
        plt.close()
        viz_paths['pca'] = pca_plot_path
        
        # Create gene expression heatmap (for top variable genes)
        gene_vars = np.var(gene_expression, axis=0)
        top_genes_idx = np.argsort(-gene_vars)[:20]  # Top 20 variable genes
        
        plt.figure(figsize=(12, 8))
        plt.pcolor(gene_expression[:, top_genes_idx], cmap='viridis')
        plt.title(f"Gene Expression Heatmap - Top 20 Variable Genes (Sample: {sample_id})")
        plt.xlabel("Gene Index")
        plt.ylabel("Spot Index")
        plt.colorbar(label="Expression")
        
        # Save heatmap
        heatmap_path = os.path.join(self.output_dir, f"{sample_id}_expression_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        viz_paths['expression_heatmap'] = heatmap_path
        
        return viz_paths
    
    def _save_results(self, results, sample_id):
        """Save analysis results to a text file."""
        output_file = os.path.join(self.output_dir, f"{sample_id}_analysis_results.txt")
        
        with open(output_file, 'w') as f:
            f.write(f"Spatial Transcriptomics Analysis Results - Sample: {sample_id}\n")
            f.write("=" * 60 + "\n\n")
            
            # Write processing time
            f.write(f"Processing Time: {results['processing_time']:.2f} seconds\n\n")
            
            # Write metrics
            f.write("Metrics:\n")
            f.write("-" * 40 + "\n")
            
            for category, metrics in results['metrics'].items():
                f.write(f"\n{category.replace('_', ' ').title()}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value}\n")
            
            # Write visualization paths
            f.write("\nVisualizations:\n")
            f.write("-" * 40 + "\n")
            
            for viz_name, path in results['visualizations'].items():
                f.write(f"  {viz_name}: {os.path.basename(path)}\n")
        
        logger.info(f"Results saved to: {output_file}")


def load_synthetic_data(n_spots=100, n_genes=200):
    """Generate synthetic ST data for testing."""
    logger.info("Generating synthetic ST data...")
    
    # Create spatial coordinates in a grid-like pattern
    side = int(np.ceil(np.sqrt(n_spots)))
    x_coords = np.repeat(np.arange(side), side)[:n_spots]
    y_coords = np.tile(np.arange(side), side)[:n_spots]
    spatial_coords = np.column_stack([x_coords, y_coords])
    
    # Generate gene expression with spatial patterns
    gene_expression = np.zeros((n_spots, n_genes))
    
    # Add baseline expression
    gene_expression += np.random.lognormal(0, 1, (1, n_genes))
    
    # Add spatial gradient for some genes
    for i in range(n_genes // 3):
        gene_expression[:, i] += x_coords / side * 5
    
    # Add spatial clusters for some genes
    cluster_centers = np.random.rand(5, 2) * side
    for i in range(n_genes // 3, 2 * n_genes // 3):
        for center in cluster_centers:
            dist = np.sqrt(np.sum((spatial_coords - center) ** 2, axis=1))
            gene_expression[:, i] += np.exp(-dist / 5) * 10
    
    # Add random noise
    gene_expression += np.random.lognormal(-1, 1, gene_expression.shape)
    
    # Ensure non-negative values
    gene_expression = np.maximum(0, gene_expression)
    
    # Sparsify (add zeros)
    mask = np.random.rand(*gene_expression.shape) < 0.8
    gene_expression[mask] = 0
    
    logger.info(f"Generated data with {n_spots} spots and {n_genes} genes")
    return gene_expression, spatial_coords


def load_data_from_path(dataset_path, sample_id=None):
    """Load data from the HEST dataset."""
    try:
        logger.info(f"Attempting to load data from {dataset_path}")
        
        # If a specific path to a file exists, try to load it
        if os.path.isfile(os.path.join(dataset_path, f"{sample_id}.npz")):
            data = np.load(os.path.join(dataset_path, f"{sample_id}.npz"))
            gene_expression = data['gene_expression']
            spatial_coords = data['spatial_coords']
            return gene_expression, spatial_coords, sample_id
        
        # Check if we have ST data in the st directory
        st_dir = os.path.join(dataset_path, "st")
        if os.path.isdir(st_dir):
            # Look for h5ad files
            import glob
            h5ad_files = glob.glob(os.path.join(st_dir, "*.h5ad"))
            
            if h5ad_files:
                try:
                    import scanpy as sc
                    adata = sc.read_h5ad(h5ad_files[0])
                    
                    # Extract gene expression and coordinates
                    gene_expression = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
                    spatial_coords = adata.obsm['spatial'] if 'spatial' in adata.obsm else np.array([adata.obs['x'], adata.obs['y']]).T
                    
                    sample_id = os.path.basename(h5ad_files[0]).split('.')[0]
                    logger.info(f"Loaded {sample_id} from AnnData file with {gene_expression.shape[0]} spots and {gene_expression.shape[1]} genes")
                    
                    return gene_expression, spatial_coords, sample_id
                except Exception as e:
                    logger.error(f"Failed to load AnnData file: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
    
    logger.warning("Could not load real data, falling back to synthetic data")
    
    # Fall back to synthetic data
    gene_expression, spatial_coords = load_synthetic_data()
    return gene_expression, spatial_coords, "synthetic_sample"


def main():
    """Main function to run the analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze spatial transcriptomics data without histology integration")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--sample_id", default=None, help="ID of the sample to analyze")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check hardware info
    logger.info("Hardware information:")
    if torch is not None:
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    gene_expression, spatial_coords, sample_id = load_data_from_path(
        args.dataset_path, args.sample_id
    )
    
    # Save summary file with dataset info
    with open(os.path.join(args.output_dir, "dataset_info.txt"), 'w') as f:
        f.write(f"Dataset path: {args.dataset_path}\n")
        f.write(f"Sample ID: {sample_id}\n")
        f.write(f"Number of spots: {gene_expression.shape[0]}\n")
        f.write(f"Number of genes: {gene_expression.shape[1]}\n")
        f.write(f"Sparsity: {np.mean(gene_expression == 0):.4f}\n")
    
    # Initialize analyzer
    analyzer = STAnalyzer(output_dir=args.output_dir)
    
    # Analyze the sample
    results = analyzer.analyze_sample(gene_expression, spatial_coords, sample_id)
    
    # Create summary file
    with open(os.path.join(args.output_dir, "analysis_summary.txt"), 'w') as f:
        f.write("Spatial Transcriptomics Analysis - Control Group Experiment\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample ID: {sample_id}\n")
        f.write(f"Number of spots: {gene_expression.shape[0]}\n")
        f.write(f"Number of genes: {gene_expression.shape[1]}\n")
        f.write(f"Analysis approach: ST data only (no histology integration)\n\n")
        
        f.write("Key Findings:\n")
        f.write("-" * 40 + "\n")
        
        # Spatial resolution
        spatial_metrics = results['metrics']['spatial_resolution']
        f.write(f"Spot density: {spatial_metrics['spot_density']:.4f} spots per unit area\n")
        f.write(f"Mean spot distance: {spatial_metrics['mean_spot_distance']:.4f}\n")
        f.write(f"Coverage ratio: {spatial_metrics['coverage_ratio']:.4f}\n\n")
        
        # Cell typing
        cell_metrics = results['metrics']['cell_typing']
        f.write(f"Number of identified cell types: {cell_metrics['num_clusters']}\n")
        f.write(f"Silhouette score: {cell_metrics['silhouette_score']:.4f}\n\n")
        
        # Expression statistics
        expr_metrics = results['metrics']['gene_expression']
        f.write(f"Mean genes per spot: {expr_metrics['genes_per_spot_mean']:.2f}\n")
        f.write(f"Mean counts per spot: {expr_metrics['counts_per_spot_mean']:.2f}\n")
        f.write(f"Data sparsity: {expr_metrics['zero_fraction']:.4f}\n\n")
        
        f.write("Analysis time: {:.2f} seconds\n".format(results['processing_time']))
        
    logger.info("Analysis completed successfully")


if __name__ == "__main__":
    main()
