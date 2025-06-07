import os
import random
from datasets import load_dataset
import pandas as pd
from PIL import Image
import numpy as np

def load_hest_sample(sample_id=None, random_sample=True):
    """
    Load one tissue sample from the HEST dataset.
    
    Args:
        sample_id (str, optional): Specific sample ID to load. If None, picks randomly.
        random_sample (bool): Whether to pick a random sample if sample_id is None.
    
    Returns:
        dict: Dictionary containing the sample data with ST and histology information
    """
    
    print("Loading HEST dataset from HuggingFace...")
    
    # Load the dataset

    local_dir='hest_data' # hest will be dowloaded to this folder

    ids_to_query = ['TENX96', 'TENX99'] # list of ids to query

    list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    dataset = load_dataset(
        'MahmoodLab/hest', 
        cache_dir=local_dir,
        patterns=list_patterns
    )
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # If no specific sample_id provided, pick one
    if sample_id is None:
        if random_sample:
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            print(f"Randomly selected sample at index {idx}")
        else:
            # Pick the first sample
            idx = 0
            sample = dataset[idx]
            print(f"Selected first sample at index {idx}")
    else:
        # Find sample by ID
        sample = None
        for i, s in enumerate(dataset):
            if s.get('sample_id') == sample_id or s.get('id') == sample_id:
                sample = s
                idx = i
                print(f"Found sample {sample_id} at index {idx}")
                break
        
        if sample is None:
            print(f"Sample {sample_id} not found. Using first sample instead.")
            sample = dataset[0]
            idx = 0
    
    return sample, idx

def display_sample_info(sample):
    """Display information about the selected sample."""
    
    print("\n" + "="*50)
    print("SAMPLE INFORMATION")
    print("="*50)
    
    # Print all available keys
    print("Available data fields:")
    for key in sample.keys():
        print(f"  - {key}: {type(sample[key])}")
    
    print("\nSample details:")
    
    # Common fields that might be present
    info_fields = ['sample_id', 'id', 'organ', 'tissue_type', 'technology', 
                   'species', 'age', 'sex', 'patient_id']
    
    for field in info_fields:
        if field in sample:
            print(f"  {field}: {sample[field]}")
    
    # Check for spatial transcriptomics data
    st_fields = ['adata', 'gene_expression', 'coordinates', 'spots']
    st_present = any(field in sample for field in st_fields)
    print(f"\nSpatial Transcriptomics data present: {st_present}")
    
    # Check for histology image
    hist_fields = ['image', 'histology_image', 'tissue_image']
    hist_present = any(field in sample for field in hist_fields)
    print(f"Histology image present: {hist_present}")
    
    return sample

def extract_sample_data(sample):
    """Extract and organize the key data components."""
    
    data = {
        'metadata': {},
        'spatial_transcriptomics': None,
        'histology_image': None,
        'coordinates': None
    }
    
    # Extract metadata
    metadata_fields = ['sample_id', 'id', 'organ', 'tissue_type', 'technology', 
                       'species', 'age', 'sex', 'patient_id']
    
    for field in metadata_fields:
        if field in sample:
            data['metadata'][field] = sample[field]
    
    # Extract spatial transcriptomics data
    if 'adata' in sample:
        data['spatial_transcriptomics'] = sample['adata']
    elif 'gene_expression' in sample:
        data['spatial_transcriptomics'] = sample['gene_expression']
    
    # Extract histology image
    if 'image' in sample:
        data['histology_image'] = sample['image']
    elif 'histology_image' in sample:
        data['histology_image'] = sample['histology_image']
    elif 'tissue_image' in sample:
        data['histology_image'] = sample['tissue_image']
    
    # Extract coordinates if available
    if 'coordinates' in sample:
        data['coordinates'] = sample['coordinates']
    elif 'spots' in sample:
        data['coordinates'] = sample['spots']
    
    return data

def save_sample_data(data, output_dir="hest_sample_output"):
    """Save the sample data to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving sample data to {output_dir}/")
    
    # Save metadata
    if data['metadata']:
        metadata_df = pd.DataFrame([data['metadata']])
        metadata_df.to_csv(f"{output_dir}/metadata.csv", index=False)
        print("  - metadata.csv")
    
    # Save histology image
    if data['histology_image'] is not None:
        if isinstance(data['histology_image'], Image.Image):
            data['histology_image'].save(f"{output_dir}/histology_image.png")
            print("  - histology_image.png")
        elif isinstance(data['histology_image'], np.ndarray):
            Image.fromarray(data['histology_image']).save(f"{output_dir}/histology_image.png")
            print("  - histology_image.png")
    
    # Save coordinates if available
    if data['coordinates'] is not None:
        if isinstance(data['coordinates'], (list, np.ndarray)):
            coords_df = pd.DataFrame(data['coordinates'])
            coords_df.to_csv(f"{output_dir}/coordinates.csv", index=False)
            print("  - coordinates.csv")
    
    # Note about spatial transcriptomics data
    if data['spatial_transcriptomics'] is not None:
        print("  - spatial_transcriptomics data available (use scanpy to process)")
    
    return output_dir

def main():
    """Main function to demonstrate usage."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    try:
        # Load one sample from HEST dataset
        sample, idx = load_hest_sample(random_sample=True)
        
        # Display sample information
        display_sample_info(sample)
        
        # Extract and organize data
        data = extract_sample_data(sample)
        
        # Save data to files
        output_dir = save_sample_data(data)
        
        print(f"\nSample successfully processed and saved to {output_dir}/")
        print("\nYou can now work with:")
        print("  - Metadata (sample info)")
        print("  - Histology image")
        print("  - Spatial transcriptomics data (if present)")
        print("  - Spatial coordinates (if present)")
        
        return data
        
    except Exception as e:
        print(f"Error processing HEST dataset: {str(e)}")
        print("Make sure you have the required packages installed:")
        print("  pip install datasets pandas pillow numpy scanpy")
        return None

if __name__ == "__main__":
    # Run the main function
    sample_data = main()
    
    # Example of how to access the data
    if sample_data:
        print("\nExample data access:")
        print(f"Sample metadata: {sample_data['metadata']}")
        if sample_data['histology_image']:
            print(f"Histology image type: {type(sample_data['histology_image'])}")
        if sample_data['spatial_transcriptomics']:
            print(f"ST data type: {type(sample_data['spatial_transcriptomics'])}")