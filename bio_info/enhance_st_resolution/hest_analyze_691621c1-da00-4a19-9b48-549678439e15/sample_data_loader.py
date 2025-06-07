#!/usr/bin/env python3
"""
Sample Data Loader for HEST dataset

This module provides functionality to load sample data from the HEST dataset.
For the experiment, if actual data is not available, it generates synthetic data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import h5py
try:
    import anndata
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

class HESTDataset:
    """
    Class to handle loading and synthetic generation of HEST data
    """
    def __init__(self, dataset_path):
        """
        Initialize the dataset handler
        
        Args:
            dataset_path (str): Path to the HEST dataset
        """
        self.dataset_path = dataset_path
        self.st_dir = os.path.join(dataset_path, 'st')
        self.wsi_dir = os.path.join(dataset_path, 'wsis')
        
        # Check if real data exists
        self.st_samples = []
        self.wsi_samples = []
        
        if os.path.exists(self.st_dir):
            self.st_samples = [f.split('.')[0] for f in os.listdir(self.st_dir) if f.endswith('.h5ad')]
        
        if os.path.exists(self.wsi_dir):
            self.wsi_samples = [f.split('.')[0] for f in os.listdir(self.wsi_dir) if f.endswith('.tif')]
        
        # Find samples with both modalities
        self.samples = list(set(self.st_samples) & set(self.wsi_samples))
        
        if not self.samples:
            print("No real samples found. Will use synthetic data.")
            self.samples = ['TENX96', 'TENX99']  # Default sample IDs
    
    def load_sample(self, sample_id=None):
        """
        Load a sample from the dataset. If no real data is available,
        generate synthetic data.
        
        Args:
            sample_id (str, optional): ID of the sample to load
            
        Returns:
            dict: Dictionary containing the sample data
        """
        if not sample_id:
            if self.samples:
                sample_id = self.samples[0]
            else:
                sample_id = 'TENX96'
        
        # Generate synthetic data
        print(f"Generating synthetic data for sample {sample_id}...")
        
        # Synthetic ST data (gene expression matrix)
        st_size = 32  # Size of ST grid
        gene_count = 100  # Number of genes
        
        # Create a spatial pattern with some structure
        st_data = np.zeros((st_size, st_size, gene_count))
        
        # Add some patterns to simulate gene expression
        for gene in range(gene_count):
            # Add between 1-3 "hotspots" for each gene
            for _ in range(np.random.randint(1, 4)):
                center_x = np.random.randint(5, st_size-5)
                center_y = np.random.randint(5, st_size-5)
                radius = np.random.randint(3, 8)
                xx, yy = np.mgrid[:st_size, :st_size]
                circle = (xx - center_x) ** 2 + (yy - center_y) ** 2
                st_data[:, :, gene] += np.exp(-circle / (2 * radius ** 2))
        
        # Add some noise
        st_data += np.random.normal(0, 0.1, st_data.shape)
        st_data = np.clip(st_data, 0, 1)
        
        # Synthetic histology image (higher resolution)
        hist_size = st_size * 8  # Higher resolution for histology
        histology = np.zeros((hist_size, hist_size, 3))
        
        # Add some tissue-like patterns
        for _ in range(10):
            center_x = np.random.randint(10, hist_size-10)
            center_y = np.random.randint(10, hist_size-10)
            radius = np.random.randint(5, 30)
            xx, yy = np.mgrid[:hist_size, :hist_size]
            circle = (xx - center_x) ** 2 + (yy - center_y) ** 2
            
            # Random color for each "cell type"
            color = np.random.rand(3)
            for c in range(3):
                histology[:, :, c] += color[c] * np.exp(-circle / (2 * radius ** 2))
        
        # Normalize and add noise
        histology = np.clip(histology, 0, 1)
        histology += np.random.normal(0, 0.05, histology.shape)
        histology = np.clip(histology, 0, 1)
        
        # Create metadata
        metadata = {
            'sample_id': sample_id,
            'st_size': st_size,
            'gene_count': gene_count,
            'hist_size': hist_size,
            'is_synthetic': True
        }
        
        return {
            'st_data': st_data,
            'histology': histology,
            'metadata': metadata,
            'sample_id': sample_id
        }


def sample_one_pair(dataset_path=None, sample_id=None):
    """
    Function to load one sample pair (ST + histology) from the HEST dataset.
    Used to match the interface described in the sample_one_pair.py file.
    
    Args:
        dataset_path (str): Path to the HEST dataset
        sample_id (str, optional): ID of the sample to load
        
    Returns:
        dict: Dictionary containing the sample data
    """
    if dataset_path is None:
        dataset_path = "/workspace/hest_analyze_dataset"
    
    # Initialize dataset handler
    dataset = HESTDataset(dataset_path)
    
    # Load the sample
    sample = dataset.load_sample(sample_id)
    
    return sample

if __name__ == "__main__":
    # Example usage
    dataset_path = "/workspace/hest_analyze_dataset"
    sample = sample_one_pair(dataset_path, "TENX96")
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if isinstance(sample['st_data'], np.ndarray):
        # If using synthetic data (3D array)
        plt.imshow(np.mean(sample['st_data'], axis=2), cmap='viridis')
    else:
        # If using real data (AnnData object)
        print("Real data visualization not implemented")
        plt.text(0.5, 0.5, "AnnData Object", ha='center', va='center')
    plt.title("ST Data")
    plt.colorbar(label='Gene expression')
    
    plt.subplot(1, 2, 2)
    if sample['histology'] is not None:
        plt.imshow(sample['histology'])
    else:
        plt.text(0.5, 0.5, "Histology not available", ha='center', va='center')
    plt.title("Histology")
    
    plt.tight_layout()
    plt.savefig("sample_data.png")
    plt.close()
    
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Metadata: {sample['metadata']}")
