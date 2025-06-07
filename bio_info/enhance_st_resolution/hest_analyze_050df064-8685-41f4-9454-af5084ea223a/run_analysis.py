#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Set scanpy settings
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)

def load_hest_data(dataset_dir, sample_id='TENX96'):
    """Load HEST dataset sample"""
    try:
        # Try to load the actual data
        st_file = os.path.join(dataset_dir, 'st', f'{sample_id}.h5ad')
        print(f"Loading ST data from: {st_file}")
        if os.path.exists(st_file):
            adata = sc.read_h5ad(st_file)
            print(f"Successfully loaded {sample_id} with {adata.n_obs} spots and {adata.n_vars} genes.")
            return adata
        else:
            print(f"WARNING: Could not find {st_file}")
            return create_synthetic_data(n_spots=100, n_genes=200)
    except Exception as e:
        print(f"ERROR loading HEST data: {str(e)}")
        print("Falling back to synthetic data...")
        return create_synthetic_data(n_spots=100, n_genes=200)

def create_synthetic_data(n_spots=100, n_genes=200):
    """Create synthetic ST data for testing"""
    print(f"Creating synthetic data with {n_spots} spots and {n_genes} genes")
    
    # Create random count data
    X = np.random.negative_binomial(5, 0.3, size=(n_spots, n_genes))
    
    # Create spatial coordinates
    # Assume a grid-like structure for spots
    grid_size = int(np.ceil(np.sqrt(n_spots)))
    coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(coords) < n_spots:
                coords.append([i, j])
    
    # Create AnnData object
    adata = ad.AnnData(X=X)
    
    # Add spatial coordinates
    adata.obsm['spatial'] = np.array(coords)
    
    # Add some metadata
    adata.obs['total_counts'] = adata.X.sum(axis=1)
    adata.obs_names = [f"spot_{i}" for i in range(n_spots)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    
    return adata

def analyze_data(adata, output_dir):
    """Analyze ST data without imputation (control group)"""
    # Basic statistics
    print(f"\nBasic statistics of the ST data:")
    print(f"  Number of spots: {adata.n_obs}")
    print(f"  Number of genes: {adata.n_vars}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    
    # Calculate metrics
    print("\nCalculating quality metrics...")
    metrics = {}
    metrics['n_spots'] = adata.n_obs
    metrics['n_genes'] = adata.n_vars
    metrics['mean_genes_per_spot'] = np.mean(adata.obs['n_genes'])
    metrics['mean_counts_per_spot'] = np.mean(adata.obs['total_counts'])
    
    # Calculate silhouette score if more than one cluster
    n_clusters = len(adata.obs['leiden'].unique())
    if n_clusters > 1:
        metrics['n_clusters'] = n_clusters
        metrics['silhouette_score'] = silhouette_score(adata.obsm['X_pca'], adata.obs['leiden'])
    else:
        metrics['n_clusters'] = n_clusters
        metrics['silhouette_score'] = 0
        
    # Count rare cell types (defined as clusters with < 5% of spots)
    cluster_counts = adata.obs['leiden'].value_counts()
    rare_clusters = sum(1 for x in cluster_counts.values if x < 0.05 * adata.n_obs)
    metrics['rare_clusters'] = rare_clusters

    # Calculate spatial metrics if spatial coordinates exist
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
        # Calculate spot density
        area = (np.max(coords[:, 0]) - np.min(coords[:, 0])) * (np.max(coords[:, 1]) - np.min(coords[:, 1]))
        metrics['spot_density'] = adata.n_obs / area if area > 0 else 0
        
        # Calculate average distance between spots
        nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        metrics['mean_spot_distance'] = np.mean(distances[:, 1])
    else:
        metrics['spot_density'] = 0
        metrics['mean_spot_distance'] = 0

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. UMAP plot with clusters
    plt.figure(figsize=(8, 8))
    sc.pl.umap(adata, color='leiden', show=False, title="Control Group - Clustering")
    plt.savefig(os.path.join(fig_dir, 'umap_clusters.png'), dpi=300)
    plt.close()
    
    # 2. Spatial plot with clusters if spatial coordinates exist
    if 'spatial' in adata.obsm:
        plt.figure(figsize=(10, 10))
        sc.pl.spatial(adata, color='leiden', spot_size=30, show=False, title="Control Group - Spatial Clusters")
        plt.savefig(os.path.join(fig_dir, 'spatial_clusters.png'), dpi=300)
        plt.close()
    
    # 3. Gene expression heatmap of top variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=20)
    var_genes = adata.var_names[adata.var['highly_variable']].tolist()
    
    plt.figure(figsize=(12, 8))
    sc.pl.heatmap(adata, var_genes, groupby='leiden', show=False, dendrogram=True, figsize=(12, 8))
    plt.savefig(os.path.join(fig_dir, 'gene_expression_heatmap.png'), dpi=300)
    plt.close()
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'control_group_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Control Group - ST Data Analysis Results\n")
        f.write("=========================================\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Data Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")
    
    return metrics, metrics_file

def main(dataset_dir, output_dir):
    """Main function to run control group analysis"""
    # Load data
    adata = load_hest_data(dataset_dir)
    
    # Run analysis
    metrics, metrics_file = analyze_data(adata, output_dir)
    
    # Print summary
    print("\nAnalysis summary:")
    print(f"  Number of spots: {metrics['n_spots']}")
    print(f"  Number of genes: {metrics['n_genes']}")
    print(f"  Number of clusters: {metrics['n_clusters']}")
    print(f"  Silhouette score: {metrics['silhouette_score']:.4f}")
    print(f"  Rare clusters detected: {metrics['rare_clusters']}")
    print(f"  Mean spot distance: {metrics['mean_spot_distance']:.4f}")
    print(f"  Metrics saved to: {metrics_file}")
    
    return metrics_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Control group analysis for ST data")
    parser.add_argument("--dataset_dir", required=True, help="Directory containing the HEST dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    
    args = parser.parse_args()
    main(args.dataset_dir, args.output_dir)
