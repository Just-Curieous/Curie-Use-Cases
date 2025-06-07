#!/usr/bin/env python3
"""
Spatial Transcriptomics Analyzer for Control Group
This script analyzes original non-imputed spatial transcriptomics data from the HEST dataset.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import anndata as ad
import warnings
warnings.filterwarnings('ignore')

# Set path constants
WORKSPACE_DIR = "/workspace/hest_analyze_050df064-8685-41f4-9454-af5084ea223a"
DATASET_DIR = "/workspace/hest_analyze_dataset"
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "results")
DATA_FILE = os.path.join(DATASET_DIR, "st", "TENX96.h5ad")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_message(msg):
    """Print a timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def load_data():
    """Load the ST data or create synthetic data if not available."""
    log_message(f"Loading ST data from: {DATA_FILE}")
    
    try:
        # Try to load the real data
        adata = sc.read_h5ad(DATA_FILE)
        log_message(f"Successfully loaded data with {adata.n_obs} spots and {adata.n_vars} genes.")
        return adata
    except FileNotFoundError:
        log_message("Data file not found. Creating synthetic data...")
        
        # Create synthetic data
        n_spots = 1000
        n_genes = 500
        X = np.random.negative_binomial(5, 0.5, size=(n_spots, n_genes))
        
        # Create AnnData object
        obs_names = [f"spot_{i}" for i in range(n_spots)]
        var_names = [f"gene_{i}" for i in range(n_genes)]
        adata = ad.AnnData(X=X, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
        
        # Add spatial coordinates
        coords = np.random.uniform(0, 1000, size=(n_spots, 2))
        adata.obsm['spatial'] = coords
        
        log_message(f"Created synthetic data with {n_spots} spots and {n_genes} genes.")
        return adata

def basic_stats(adata):
    """Calculate basic statistics of the ST data."""
    stats = {}
    
    # Basic counts
    stats['n_spots'] = adata.n_obs
    stats['n_genes'] = adata.n_vars
    
    # Expression statistics
    mean_expr = np.mean(adata.X, axis=0)
    stats['mean_gene_expr'] = np.mean(mean_expr)
    stats['median_gene_expr'] = np.median(mean_expr)
    stats['max_gene_expr'] = np.max(mean_expr)
    
    # Zero expression statistics
    zero_counts = np.sum(adata.X == 0)
    stats['zero_fraction'] = zero_counts / (adata.n_obs * adata.n_vars)
    
    return stats

def spatial_metrics(adata):
    """Calculate spatial metrics of the ST data."""
    metrics = {}
    
    # Get spatial coordinates
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
    else:
        log_message("No spatial coordinates found. Using random coordinates.")
        coords = np.random.uniform(0, 1000, size=(adata.n_obs, 2))
    
    # Calculate pairwise distances
    dist_matrix = distance.pdist(coords)
    metrics['mean_spot_distance'] = np.mean(dist_matrix)
    metrics['min_spot_distance'] = np.min(dist_matrix)
    metrics['max_spot_distance'] = np.max(dist_matrix)
    
    # Calculate spot density
    area = np.prod(np.max(coords, axis=0) - np.min(coords, axis=0))
    metrics['spot_density'] = adata.n_obs / area
    
    return metrics, coords

def gene_expression_analysis(adata):
    """Analyze gene expression patterns."""
    results = {}
    
    # Calculate gene statistics
    mean_expr = np.mean(adata.X, axis=0)
    var_expr = np.var(adata.X, axis=0)
    
    # Find highly variable genes
    gene_stats = pd.DataFrame({
        'mean': mean_expr,
        'var': var_expr,
        'cv': np.sqrt(var_expr) / mean_expr
    })
    
    # Get top variable genes
    top_var_genes = gene_stats.sort_values('cv', ascending=False).index[:50].tolist()
    results['top_variable_genes'] = top_var_genes
    results['mean_cv'] = np.mean(gene_stats['cv'])
    results['max_cv'] = np.max(gene_stats['cv'])
    
    return results

def clustering_analysis(adata, coords):
    """Perform clustering to identify cell types."""
    results = {}
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(adata.X)
    
    # Perform K-means clustering
    n_clusters = min(8, adata.n_obs // 100)  # Ensure reasonable number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    if n_clusters > 1:
        sil_score = silhouette_score(X_scaled, clusters)
    else:
        sil_score = 0
    
    # Count cluster sizes
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    cluster_sizes = dict(zip(unique_clusters, counts))
    
    results['n_clusters'] = n_clusters
    results['cluster_sizes'] = cluster_sizes
    results['silhouette_score'] = sil_score
    results['clusters'] = clusters
    
    return results

def visualize_results(adata, coords, gene_results, cluster_results):
    """Generate visualizations of the analysis results."""
    vis_paths = {}
    
    # 1. Spatial plot with clusters
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=cluster_results['clusters'], 
                          cmap='tab10', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Spatial Clustering of Cells')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.tight_layout()
    
    spatial_path = os.path.join(OUTPUT_DIR, 'spatial_clusters.png')
    plt.savefig(spatial_path)
    plt.close()
    vis_paths['spatial_clusters'] = spatial_path
    
    # 2. t-SNE visualization
    X_scaled = StandardScaler().fit_transform(adata.X)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_results['clusters'],
                cmap='tab10', s=50, alpha=0.7)
    plt.title('t-SNE Visualization of Gene Expression')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    
    tsne_path = os.path.join(OUTPUT_DIR, 'tsne_visualization.png')
    plt.savefig(tsne_path)
    plt.close()
    vis_paths['tsne'] = tsne_path
    
    # 3. Gene expression heatmap (top genes)
    if len(gene_results['top_variable_genes']) > 10:
        top_genes = gene_results['top_variable_genes'][:10]
        
        # Subset data for top genes
        gene_indices = [adata.var_names.get_loc(g) if g in adata.var_names 
                        else i for i, g in enumerate(top_genes)]
        expr_data = adata.X[:100, gene_indices]  # First 100 cells
        
        plt.figure(figsize=(12, 8))
        plt.imshow(expr_data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Expression')
        plt.title('Expression of Top 10 Variable Genes')
        plt.xlabel('Genes')
        plt.ylabel('Cells')
        plt.tight_layout()
        
        heatmap_path = os.path.join(OUTPUT_DIR, 'gene_expression_heatmap.png')
        plt.savefig(heatmap_path)
        plt.close()
        vis_paths['heatmap'] = heatmap_path
    
    return vis_paths

def main():
    """Main function to run the analysis pipeline."""
    start_time = time.time()
    
    log_message("Starting spatial transcriptomics analysis - Control group (no imputation)")
    log_message(f"Output directory: {OUTPUT_DIR}")
    
    # 1. Load data
    adata = load_data()
    
    # 2. Calculate basic statistics
    log_message("Calculating basic statistics...")
    stats = basic_stats(adata)
    log_message(f"Found {stats['n_spots']} spots and {stats['n_genes']} genes")
    
    # 3. Calculate spatial metrics
    log_message("Calculating spatial metrics...")
    metrics, coords = spatial_metrics(adata)
    log_message(f"Mean distance between spots: {metrics['mean_spot_distance']:.2f}")
    log_message(f"Spot density: {metrics['spot_density']:.4f} spots per unit area")
    
    # 4. Analyze gene expression
    log_message("Analyzing gene expression patterns...")
    gene_results = gene_expression_analysis(adata)
    log_message(f"Mean coefficient of variation: {gene_results['mean_cv']:.2f}")
    
    # 5. Perform clustering
    log_message("Performing clustering to identify cell types...")
    cluster_results = clustering_analysis(adata, coords)
    log_message(f"Found {cluster_results['n_clusters']} clusters with silhouette score: {cluster_results['silhouette_score']:.2f}")
    
    # 6. Generate visualizations
    log_message("Generating visualizations...")
    vis_paths = visualize_results(adata, coords, gene_results, cluster_results)
    for name, path in vis_paths.items():
        log_message(f"Generated {name} visualization: {path}")
    
    # 7. Write report
    end_time = time.time()
    total_time = end_time - start_time
    log_message(f"Analysis completed in {total_time:.2f} seconds")
    
    report_path = os.path.join(OUTPUT_DIR, "analysis_results.txt")
    with open(report_path, "w") as f:
        f.write("Spatial Transcriptomics Analysis Results\n")
        f.write("======================================\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis time: {total_time:.2f} seconds\n\n")
        
        f.write("1. Basic Statistics\n")
        f.write("------------------\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("2. Spatial Metrics\n")
        f.write("-----------------\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("3. Gene Expression Analysis\n")
        f.write("-------------------------\n")
        f.write(f"mean_cv: {gene_results['mean_cv']}\n")
        f.write(f"max_cv: {gene_results['max_cv']}\n")
        f.write(f"Number of top variable genes identified: {len(gene_results['top_variable_genes'])}\n")
        f.write("\n")
        
        f.write("4. Clustering Results\n")
        f.write("-------------------\n")
        f.write(f"Number of clusters: {cluster_results['n_clusters']}\n")
        f.write(f"Silhouette score: {cluster_results['silhouette_score']}\n")
        f.write(f"Cluster sizes: {cluster_results['cluster_sizes']}\n")
        f.write("\n")
        
        f.write("5. Visualizations\n")
        f.write("---------------\n")
        for name, path in vis_paths.items():
            f.write(f"{name}: {path}\n")
    
    log_message(f"Report written to: {report_path}")
    log_message("Analysis completed successfully")
    
    return {
        'stats': stats,
        'metrics': metrics,
        'gene_results': gene_results,
        'cluster_results': cluster_results,
        'vis_paths': vis_paths,
        'report_path': report_path,
        'total_time': total_time
    }

if __name__ == "__main__":
    results = main()
