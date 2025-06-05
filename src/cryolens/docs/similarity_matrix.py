#!/usr/bin/env python
"""
Generate visualizations for the similarity_matrix_test.md

This script creates visualizations for testing similarity matrices, including heatmaps and
projected image comparisons. It handles different Signal-to-Noise Ratio (SNR) levels for
different structures, ensuring that each structure is represented with both high and low
SNR samples when available. This is important because each structure typically has only
one SNR level, so the script randomly samples multiple structures and keeps the highest
and lowest SNR levels as needed.

For similar structure pairs, high SNR samples are used to better visualize similarities.
For dissimilar structure pairs, low SNR samples are used to better visualize differences.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from pathlib import Path
import argparse
import logging
import pandas as pd
import re
import subprocess
import torch

# Add cryolens to path if needed
CRYOLENS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if CRYOLENS_PATH not in sys.path:
    sys.path.append(CRYOLENS_PATH)

from cryolens.affinity import SimilarityCalculator

# Use CurriculumParquetDataset from cryolens
try:
    from cryolens.data.datasets import CurriculumParquetDataset
except ImportError:
    print("Error importing CurriculumParquetDataset. Make sure cryolens is in your Python path.")
    sys.exit(1)

# Replace StructureRandomParquetDataset with our implementation using CurriculumParquetDataset
class MultiSNRStructureDataset:
    """
    Extended dataset that loads multiple SNR levels for each structure.
    This handles the case where each structure only has one SNR level by randomly
    sampling multiple structures and choosing high and low SNR levels as needed.
    """
    
    def __init__(self, base_dir, box_size=48, device='cpu', augment=False):
        """
        Initialize dataset.
        
        Args:
            base_dir: Directory or parquet file containing structure data
            box_size: Size of volume box
            device: Device to load data to
            augment: Whether to apply data augmentation
        """
        self.base_dir = Path(base_dir)
        self.box_size = box_size
        self.device = device
        self.augment = augment
        self.structure_names = []
        self.datasets = []
        self.selected_parquets = []
        self.total_items = 0
        self.weights = []
        
        # Get structure IDs from command line
        structure_ids = ['4V1W', '1FA2', '6X1Q', '6EK0', '6SCJ', '6N4V']
        structure_names = ['apo-ferritin', 'beta-amylase', 'beta-galactoside', 'ribosome', 'thyroglobulin', 'virus-like-particle']
        self.name_to_pdb = dict(zip(structure_names, structure_ids))
        
        # Load structure parquets
        self._find_structure_parquets()
    
    def _find_structure_parquets(self):
        """Find parquet files for each structure with different SNR levels."""
        # Check if the path is a file or directory
        if self.base_dir.is_file() and self.base_dir.suffix == '.parquet':
            # Handle a single parquet file
            logger.info(f"Loading single parquet file: {self.base_dir}")
            self._load_single_parquet(self.base_dir)
        else:
            # Handle directory structure with subdirectories for each structure
            self._load_directory_structure()
    
    def _load_single_parquet(self, parquet_path):
        """Load a single parquet file and organize by structure."""
        try:
            # Load the parquet file
            df = pd.read_parquet(parquet_path)
            
            # Group by molecule ID
            if 'molecule_id' in df.columns:
                molecule_ids = df['molecule_id'].unique()
                
                logger.info(f"Found {len(molecule_ids)} unique molecule IDs in {parquet_path}")
                
                for molecule_id in molecule_ids:
                    if pd.isna(molecule_id):
                        continue
                        
                    # Extract data for this molecule
                    molecule_df = df[df['molecule_id'] == molecule_id].copy()
                    
                    # Skip if empty
                    if len(molecule_df) == 0:
                        continue
                    
                    # Process subvolumes if needed
                    if 'subvolume' in molecule_df.columns and isinstance(molecule_df['subvolume'].iloc[0], bytes):
                        molecule_df['subvolume'] = molecule_df['subvolume'].apply(self._process_subvolume)
                    
                    # Add SNR info if available
                    if 'source_type' in molecule_df.columns:
                        for i, source in enumerate(molecule_df['source_type'].unique()):
                            # Extract SNR from source if available
                            snr_match = re.search(r'snr[-_]?(\d+\.?\d*)', str(source).lower())
                            if snr_match:
                                snr_value = float(snr_match.group(1))
                                source_df = molecule_df[molecule_df['source_type'] == source].copy()
                                source_df['snr_level'] = snr_value
                                
                                # Store dataset
                                self.datasets.append(source_df)
                                self.selected_parquets.append(parquet_path)
                                self.structure_names.append(molecule_id)
                                self.total_items += len(source_df)
                                
                                logger.info(f"Added {len(source_df)} samples for {molecule_id} with SNR {snr_value}")
                    else:
                        # No SNR info, just add the molecule
                        self.datasets.append(molecule_df)
                        self.selected_parquets.append(parquet_path)
                        self.structure_names.append(molecule_id)
                        self.total_items += len(molecule_df)
                        logger.info(f"Added {len(molecule_df)} samples for {molecule_id}")
            else:
                logger.warning(f"No molecule_id column found in {parquet_path}")
                
            # Calculate weights
            if self.datasets:
                dataset_sizes = [len(df) for df in self.datasets]
                self.weights = [size / sum(dataset_sizes) for size in dataset_sizes]
            
        except Exception as e:
            logger.error(f"Error loading parquet file {parquet_path}: {e}")
    
    def _load_directory_structure(self):
        """Load data from structure directories with SNR subdirectories."""
        structure_dirs = [d for d in self.base_dir.glob("*") if d.is_dir()]
        
        if not structure_dirs:
            raise ValueError(f"No structure directories found in {self.base_dir}")
        
        self.datasets = []
        self.selected_parquets = []
        self.structure_names = []
        self.total_items = 0
        
        # Dictionary to track SNR levels per structure
        structure_snr_levels = {}
        
        # First pass: collect all structure directories and their SNR levels
        for struct_dir in structure_dirs:
            structure_name = struct_dir.name
            
            # Find SNR subdirectories
            snr_dirs = [d for d in struct_dir.glob("snr_*") if d.is_dir()]
            
            if not snr_dirs:
                logger.warning(f"No SNR directories found for structure {structure_name}")
                continue
            
            # Sort SNR directories by SNR value
            def extract_snr_from_dir(dirname):
                match = re.search(r'snr[-_]?(\d+\.?\d*)', str(dirname.name).lower())
                if match:
                    return float(match.group(1))
                return float('inf')  # Default value if SNR not found
            
            snr_dirs.sort(key=extract_snr_from_dir)
            structure_snr_levels[structure_name] = {
                "dirs": snr_dirs,
                "snr_values": [extract_snr_from_dir(d) for d in snr_dirs]
            }
            
        # Get sorted list of all available SNR values across all structures
        all_snr_values = []
        for struct_info in structure_snr_levels.values():
            all_snr_values.extend(struct_info["snr_values"])
        unique_snr_values = sorted(set(all_snr_values))
        
        if not unique_snr_values:
            raise ValueError("No SNR values found across any structures")
            
        # Determine highest and lowest SNR values
        lowest_snr = unique_snr_values[0] if unique_snr_values else None
        highest_snr = unique_snr_values[-1] if unique_snr_values else None
        
        logger.info(f"Found SNR range across all structures: {lowest_snr} to {highest_snr}")
        
        # Second pass: process each structure and add datasets with both high and low SNR when available
        for structure_name, struct_info in structure_snr_levels.items():
            snr_dirs = struct_info["dirs"]
            snr_values = struct_info["snr_values"]
            
            # Try to get both highest and lowest SNR for each structure when available
            if len(snr_values) > 0:  # At least one SNR level exists
                # Get the directory with the highest SNR value for this structure
                highest_idx = snr_values.index(max(snr_values))
                self._add_snr_dataset(structure_name, snr_dirs[highest_idx])
                
                # If multiple SNR levels exist, also add the lowest
                if len(snr_values) > 1:
                    lowest_idx = snr_values.index(min(snr_values))
                    # Only add if different from the highest (avoid duplicates)
                    if lowest_idx != highest_idx:
                        self._add_snr_dataset(structure_name, snr_dirs[lowest_idx])
        
        if not self.datasets:
            raise ValueError("No valid parquet files could be loaded")
            
        logger.info(f"Loaded {len(self.datasets)} datasets with {self.total_items} total samples")
            
        # Calculate weights proportional to dataset sizes
        dataset_sizes = [len(df) for df in self.datasets]
        self.weights = [size / sum(dataset_sizes) for size in dataset_sizes]
    
    def _process_subvolume(self, subvolume):
        """Process a subvolume from bytes to numpy array."""
        if isinstance(subvolume, bytes):
            try:
                # Look for shape information in the row
                if hasattr(self, 'current_shape') and self.current_shape is not None:
                    shape = self.current_shape
                else:
                    # Default to cubic box
                    shape = (self.box_size, self.box_size, self.box_size)
                
                # Convert bytes to array and reshape
                return np.frombuffer(subvolume, dtype=np.float32).reshape(shape)
            except Exception as e:
                logger.error(f"Error processing subvolume: {e}")
                return None
        return subvolume
    
    def _add_snr_dataset(self, structure_name, snr_dir):
        """Helper function to add a dataset for a specific structure and SNR level."""
        # Get all batch parquet files from the SNR directory
        batch_files = list(snr_dir.glob("batch_*.parquet"))
        
        if not batch_files:
            # Try to find any parquet files
            batch_files = list(snr_dir.glob("*.parquet"))
            if not batch_files:
                logger.warning(f"No batch files found in {snr_dir}")
                return
        
        # Extract SNR value from directory name
        snr_value = 'unknown'
        snr_match = re.search(r'snr[-_]?(\d+\.?\d*)', snr_dir.name.lower())
        if snr_match:
            snr_value = float(snr_match.group(1))
        
        # Randomly select ONE batch file
        selected_batch_file = np.random.choice(batch_files)
        
        # Skip metadata files
        if 'metadata' in selected_batch_file.name:
            # If we selected a metadata file, try to find a non-metadata file
            non_metadata_files = [f for f in batch_files if 'metadata' not in f.name]
            if non_metadata_files:
                selected_batch_file = np.random.choice(non_metadata_files)
            else:
                logger.warning(f"Only metadata files found in {snr_dir}")
                return
        
        try:
            # Load only the selected batch file
            df = pd.read_parquet(selected_batch_file)
            
            if 'molecule_id' not in df.columns:
                # Try to infer molecule ID from directory name
                df['molecule_id'] = structure_name
            
            # Add SNR level if not present
            if 'snr_level' not in df.columns and 'snr' not in df.columns and snr_value != 'unknown':
                df['snr_level'] = snr_value
            
            # Process subvolumes if in bytes format
            def process_subvolume(x):
                if isinstance(x, bytes):
                    try:
                        shape = None
                        if 'shape' in df.columns:
                            shape_idx = df.columns.get_loc('shape')
                            shape = df.iloc[0, shape_idx]
                        
                        # If shape is a list or array, use it, otherwise use default box size
                        if isinstance(shape, (list, np.ndarray)) and len(shape) == 3:
                            return np.frombuffer(x, dtype=np.float32).reshape(shape)
                        else:
                            # Assume cubic box
                            return np.frombuffer(x, dtype=np.float32).reshape((self.box_size, self.box_size, self.box_size))
                    except:
                        return None
                return x
            
            # Apply processing only if we have bytes data
            if 'subvolume' in df.columns and isinstance(df['subvolume'].iloc[0], bytes):
                df['subvolume'] = df['subvolume'].apply(process_subvolume)
                
                # Filter out invalid rows
                valid_mask = df['subvolume'].notna()
                invalid_count = (~valid_mask).sum()
                if invalid_count > 0:
                    logger.warning(f"Found {invalid_count} invalid subvolumes in {snr_dir}")
                    df = df[valid_mask].reset_index(drop=True)
            
            # Get SNR level from the dataframe if available
            if 'snr_level' in df.columns:
                snr_value = df['snr_level'].iloc[0]
            elif 'snr' in df.columns:
                snr_value = df['snr'].iloc[0]
            
            logger.info(f"Structure {structure_name}: Added SNR {snr_value}, batch file {selected_batch_file.name} with {len(df)} samples")
            
            self.datasets.append(df)
            self.selected_parquets.append(selected_batch_file)
            self.structure_names.append(structure_name)
            self.total_items += len(df)
            
        except Exception as e:
            logger.error(f"Error loading batch file {selected_batch_file}: {e}")
    
    def get_sample_by_index(self, index_tuple):
        """
        Get a sample by dataset and sample index.
        
        Args:
            index_tuple: Tuple of (dataset_index, sample_index)
        
        Returns:
            Tuple of (volume, molecule_id, snr_value)
        """
        dataset_idx, sample_idx = index_tuple
        
        if dataset_idx >= len(self.datasets):
            raise ValueError(f"Dataset index {dataset_idx} out of range (max {len(self.datasets)-1})")
            
        df = self.datasets[dataset_idx]
        if sample_idx >= len(df):
            raise ValueError(f"Sample index {sample_idx} out of range (max {len(df)-1})")
            
        row = df.iloc[sample_idx]
        
        # Get subvolume data
        subvolume = row['subvolume']
        if not isinstance(subvolume, np.ndarray):
            raise ValueError(f"Subvolume type {type(subvolume)} not supported")
            
        # Normalize subvolume
        subvolume = self._normalize_volume(subvolume)
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(subvolume).float().unsqueeze(0)
        
        # Get molecule ID
        molecule_id = row['molecule_id']
        
        # Get SNR value if available
        snr_value = None
        if 'snr_level' in row:
            snr_value = row['snr_level']
        elif 'snr' in row:
            snr_value = row['snr']
            
        return volume_tensor, molecule_id, snr_value
    
    def _normalize_volume(self, volume):
        """Normalize the volume using z-score normalization."""
        # Skip normalization if volume is empty or constant
        if volume.size == 0 or np.all(volume == volume.flat[0]):
            return volume
            
        # Z-score normalization (mean=0, std=1)
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 0:
            normalized = (volume - mean) / std
        else:
            normalized = volume - mean  # If std=0, just center the data
            
        return normalized

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_similarity_heatmap(similarity_matrix, molecule_order, output_path):
    """
    Plot the similarity matrix as a heatmap with dendrograms on both axes.
    
    Args:
        similarity_matrix: Numpy array containing similarity scores
        molecule_order: List of molecule IDs corresponding to the matrix indices
        output_path: Path to save the output image
    """
    # Convert the similarity matrix to a distance matrix
    # Since similarity values are between 0 and 1, we convert to distance as 1 - similarity
    distance_matrix = 1 - similarity_matrix
    
    # Perform hierarchical clustering
    # Convert to condensed form required by linkage
    condensed_dist = squareform(distance_matrix)
    
    # Calculate linkage matrix for clustering
    Z = hierarchy.linkage(condensed_dist, method='average')
    
    # Get the order of rows/columns for the clustered heatmap
    # This reorders the indices in a way that places similar structures together
    dendro_idx = hierarchy.dendrogram(Z, no_plot=True)['leaves']
    
    # Reorder matrix and labels
    reordered_matrix = similarity_matrix[dendro_idx, :][:, dendro_idx]
    reordered_molecules = [molecule_order[i] for i in dendro_idx]
    
    # Create a DataFrame for better labeling
    df = pd.DataFrame(reordered_matrix,
                     index=reordered_molecules, 
                     columns=reordered_molecules)
    
    # Set up figure and GridSpec with space for colorbar
    fig = plt.figure(figsize=(16, 14))
    
    # Define GridSpec with 2x3 layout:
    # - Top left: title area
    # - Top middle: column dendrogram
    # - Top right: empty space
    # - Bottom left: row dendrogram
    # - Bottom middle: heatmap
    # - Bottom right: colorbar
    gs = fig.add_gridspec(2, 3, 
                         width_ratios=[0.15, 0.7, 0.15], 
                         height_ratios=[0.15, 0.85],
                         wspace=0.01, hspace=0.01)
    
    # Add the dendrograms
    ax_row_dendrogram = fig.add_subplot(gs[1, 0])
    row_dendrogram = hierarchy.dendrogram(Z, orientation='left', ax=ax_row_dendrogram, 
                                        no_labels=True, color_threshold=0)
    ax_row_dendrogram.axis('off')
    
    ax_col_dendrogram = fig.add_subplot(gs[0, 1])
    col_dendrogram = hierarchy.dendrogram(Z, orientation='top', ax=ax_col_dendrogram, 
                                        no_labels=True, color_threshold=0)
    ax_col_dendrogram.axis('off')
    
    # Plot the heatmap
    ax_heatmap = fig.add_subplot(gs[1, 1])
    
    # Plot heatmap with reduced annotations for clarity
    n = similarity_matrix.shape[0]
    annot = True if n <= 10 else False
    sns_heatmap = sns.heatmap(df, annot=annot, cmap="viridis", fmt=".2f",
                linewidths=0, cbar=False, ax=ax_heatmap)
    
    # Add colorbar separately in its own axis
    ax_colorbar = fig.add_subplot(gs[1, 2])
    plt.colorbar(sns_heatmap.collections[0], cax=ax_colorbar)
    ax_colorbar.set_ylabel('Similarity Score')
    
    # Adjust labels for readability
    plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha='right',
            rotation_mode='anchor')
    plt.setp(ax_heatmap.get_yticklabels(), rotation=0)
    
    # Add a title to the upper left empty subplot
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, "Structure Similarity Matrix\nwith Hierarchical Clustering",
                ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Make sure the empty area in the top-right is properly handled
    ax_empty = fig.add_subplot(gs[0, 2])
    ax_empty.axis('off')
    
    # Save figure with tight layout
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved similarity heatmap with dendrograms to {output_path}")

def plot_similarity_distribution(similarity_matrix, output_path):
    """
    Plot the distribution of similarity scores.
    
    Args:
        similarity_matrix: Numpy array containing similarity scores
        output_path: Path to save the output image
    """
    # Get upper triangle values (excluding diagonal)
    n = similarity_matrix.shape[0]
    similarity_values = []
    
    for i in range(n):
        for j in range(i+1, n):
            similarity_values.append(similarity_matrix[i, j])
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram with KDE
    sns.histplot(similarity_values, bins=20, kde=True)
    
    plt.title("Distribution of Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved similarity distribution to {output_path}")

def find_extreme_pairs(similarity_matrix, molecule_order, n_pairs=2):
    """
    Find the most similar and most dissimilar pairs.
    
    Args:
        similarity_matrix: Numpy array containing similarity scores
        molecule_order: List of molecule IDs corresponding to the matrix indices
        n_pairs: Number of pairs to find
        
    Returns:
        tuple: (most_similar_pairs, most_dissimilar_pairs)
    """
    n = similarity_matrix.shape[0]
    
    # Create lists to store pairs and scores
    pairs = []
    scores = []
    
    # Get upper triangle values (excluding diagonal)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
            scores.append(similarity_matrix[i, j])
    
    # Convert to numpy array for easier indexing
    scores = np.array(scores)
    
    # Find indices of most similar pairs (highest scores)
    most_similar_indices = np.argsort(scores)[-n_pairs:][::-1]
    most_similar_pairs = [(pairs[idx], scores[idx]) for idx in most_similar_indices]
    
    # Find indices of most dissimilar pairs (lowest scores)
    most_dissimilar_indices = np.argsort(scores)[:n_pairs]
    most_dissimilar_pairs = [(pairs[idx], scores[idx]) for idx in most_dissimilar_indices]
    
    # Convert indices to molecule IDs
    similar_pairs = [((molecule_order[i], molecule_order[j]), score) 
                    for (i, j), score in most_similar_pairs]
    
    dissimilar_pairs = [((molecule_order[i], molecule_order[j]), score) 
                       for (i, j), score in most_dissimilar_pairs]
    
    return similar_pairs, dissimilar_pairs

def find_sample_by_snr(dataset, structure_idx, target_snr=10.0, snr_tolerance=1.0, prefer_high=True):
    """
    Find a sample from a structure with the specified SNR target.
    
    Args:
        dataset: Instance of MultiSNRStructureDataset
        structure_idx: Index of the structure to search
        target_snr: Target SNR value (default: 10.0)
        snr_tolerance: Tolerance around target SNR (default: 1.0)
        prefer_high: If True, prefer high SNR samples; if False, prefer low SNR samples
        
    Returns:
        tuple: (sample_index, sample_data) or (None, None) if not found
    """
    try:
        # Get datasets for this structure
        df = dataset.datasets[structure_idx]
        if len(df) == 0:
            return None, None
            
        # Look for SNR information in the directory structure
        structure_dir = dataset.selected_parquets[structure_idx].parent
        snr_match = re.search(r'snr[-_]?(\d+\.?\d*)', str(structure_dir).lower())
        snr_value = float(snr_match.group(1)) if snr_match else None
        
        # Check if this directory has the target SNR
        if snr_value is not None and abs(snr_value - target_snr) <= snr_tolerance:
            # This directory has an SNR close to our target, pick a random sample
            sample_idx = np.random.randint(0, len(df))
            sample, _, _ = dataset.get_sample_by_index((structure_idx, sample_idx))
            logger.info(f"Found {'high' if prefer_high else 'low'} SNR ({snr_value}) sample for structure {dataset.structure_names[structure_idx]}")
            return sample_idx, sample
        
        # If we reach here, we couldn't find a suitable sample
        logger.warning(f"Could not find {'high' if prefer_high else 'low'} SNR sample for structure {dataset.structure_names[structure_idx]}")
        return None, None
        
    except Exception as e:
        logger.error(f"Error finding high SNR sample: {e}")
        return None, None

# Cache for projections to avoid recomputation
_projection_cache = {}

def get_cached_projections(volume, cache_key=None):
    """
    Get projections for a volume, using cache to avoid recomputation.
    
    Args:
        volume: 3D numpy array
        cache_key: Optional cache key. If None, projections are computed but not cached.
        
    Returns:
        Tuple of (xy_proj, xz_proj, yz_proj) normalized projections
    """
    # Check cache first if key provided
    if cache_key and cache_key in _projection_cache:
        return _projection_cache[cache_key]
    
    # Create average projections
    xy_proj = np.mean(volume, axis=0)  # Average projection along z
    xz_proj = np.mean(volume, axis=1)  # Average projection along y
    yz_proj = np.mean(volume, axis=2)  # Average projection along x
    
    # Normalize projections
    def normalize(img):
        if img.max() == img.min():
            return np.zeros_like(img)
        return (img - img.min()) / (img.max() - img.min())
    
    projections = (normalize(xy_proj), normalize(xz_proj), normalize(yz_proj))
    
    # Cache if key provided
    if cache_key:
        _projection_cache[cache_key] = projections
    
    return projections

def plot_structure_pair(dataset, pair_info, output_path, is_similar=True, high_snr=10.0, low_snr=1.0):
    """
    Plot a pair of structures side by side, focusing on high SNR samples.
    
    Args:
        dataset: Instance of MultiSNRStructureDataset
        pair_info: Tuple containing ((mol_id1, mol_id2), similarity_score)
        output_path: Path to save the output image
        is_similar: Whether this is a similar pair (affects the title)
        high_snr: High SNR value to use (default: 10.0)
        low_snr: Low SNR value to use (default: 1.0)
    """
    (mol_id1, mol_id2), similarity_score = pair_info
    
    # Find the structure indices in the dataset
    try:
        idx1 = dataset.structure_names.index(mol_id1)
        idx2 = dataset.structure_names.index(mol_id2)
    except ValueError:
        logger.warning(f"Could not find structures {mol_id1} or {mol_id2} in dataset")
        return
    
    # Get sample data for visualization, using appropriate SNR level based on similarity
    try:
        # For similar pairs, use high SNR samples to better show similarities
        # For dissimilar pairs, use low SNR samples to better show differences
        target_snr = high_snr if is_similar else low_snr
        
        # Try to get samples with the target SNR first
        sample_idx1, sample1 = find_sample_by_snr(dataset, idx1, target_snr, prefer_high=is_similar)
        sample_idx2, sample2 = find_sample_by_snr(dataset, idx2, target_snr, prefer_high=is_similar)
        
        # Fall back to random samples if target SNR not available
        if sample1 is None:
            logger.warning(f"Using random sample for structure {mol_id1} ({target_snr} SNR not found)")
            df1 = dataset.datasets[idx1]
            if len(df1) == 0:
                logger.warning(f"No samples available for structure {mol_id1}")
                return
            sample_idx1 = np.random.randint(0, len(df1))
            sample1, _, _ = dataset.get_sample_by_index((idx1, sample_idx1))
            
        if sample2 is None:
            logger.warning(f"Using random sample for structure {mol_id2} ({target_snr} SNR not found)")
            df2 = dataset.datasets[idx2]
            if len(df2) == 0:
                logger.warning(f"No samples available for structure {mol_id2}")
                return
            sample_idx2 = np.random.randint(0, len(df2))
            sample2, _, _ = dataset.get_sample_by_index((idx2, sample_idx2))
        
        # Convert to numpy arrays
        volume1 = sample1[0].numpy()  # Remove channel dimension
        volume2 = sample2[0].numpy()  # Remove channel dimension
        
        # Create cache keys based on structure and sample indices
        cache_key1 = f"{mol_id1}_{idx1}_{sample_idx1}"
        cache_key2 = f"{mol_id2}_{idx2}_{sample_idx2}"
        
        # Get projections using cache
        xy1, xz1, yz1 = get_cached_projections(volume1, cache_key1)
        xy2, xz2, yz2 = get_cached_projections(volume2, cache_key2)
        
        # Create figure with 2 rows (one for each structure) and 3 columns (one for each projection)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Get SNR values from directory names
        structure_dir1 = dataset.selected_parquets[idx1].parent
        structure_dir2 = dataset.selected_parquets[idx2].parent
        snr_match1 = re.search(r'snr[-_]?(\d+\.?\d*)', str(structure_dir1).lower())
        snr_match2 = re.search(r'snr[-_]?(\d+\.?\d*)', str(structure_dir2).lower())
        snr_value1 = f"SNR {snr_match1.group(1)}" if snr_match1 else "Unknown SNR"
        snr_value2 = f"SNR {snr_match2.group(1)}" if snr_match2 else "Unknown SNR"
        
        # Plot structure 1
        axes[0, 0].imshow(xy1, cmap='gray')
        axes[0, 0].set_title("XY Average Projection")
        axes[0, 0].set_ylabel(f"{mol_id1}\n{snr_value1}", fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(xz1, cmap='gray')
        axes[0, 1].set_title("XZ Average Projection")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(yz1, cmap='gray')
        axes[0, 2].set_title("YZ Average Projection")
        axes[0, 2].axis('off')
        
        # Plot structure 2
        axes[1, 0].imshow(xy2, cmap='gray')
        axes[1, 0].set_ylabel(f"{mol_id2}\n{snr_value2}", fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(xz2, cmap='gray')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(yz2, cmap='gray')
        axes[1, 2].axis('off')
        
        # Set common title
        similarity_type = "Similar" if is_similar else "Dissimilar"
        plt.suptitle(f"{similarity_type} Structure Pair: {mol_id1} - {mol_id2}\nSimilarity Score: {similarity_score:.2f}", 
                    fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust for the title
        
        # Save figure
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved structure pair visualization to {output_path}")
    
    except Exception as e:
        logger.error(f"Error creating pair visualization: {e}")

def generate_markdown_tables(similar_pairs, dissimilar_pairs, output_dir):
    """
    Generate Markdown tables for similar and dissimilar structure pairs.
    
    Args:
        similar_pairs: List of similar structure pairs with similarity scores
        dissimilar_pairs: List of dissimilar structure pairs with similarity scores
        output_dir: Output directory path for images (needed for image paths in the table)
        
    Returns:
        tuple: (similar_table, dissimilar_table) as markdown strings
    """
    # Create markdown table for similar pairs
    similar_table = "| Structure Pair | Similarity Score | Visualization |\n"
    similar_table += "|----------------|------------------|---------------|\n"
    
    for i, ((struct1, struct2), score) in enumerate(similar_pairs):
        img_path = f"./figures/similar_pair_{i+1}.png"  # Relative path for markdown
        similar_table += f"| {struct1}-{struct2} | {score:.2f} | ![Similar Pair {i+1}]({img_path}) |\n"
    
    # Create markdown table for dissimilar pairs
    dissimilar_table = "| Structure Pair | Similarity Score | Visualization |\n"
    dissimilar_table += "|----------------|------------------|---------------|\n"
    
    for i, ((struct1, struct2), score) in enumerate(dissimilar_pairs):
        img_path = f"./figures/dissimilar_pair_{i+1}.png"  # Relative path for markdown
        dissimilar_table += f"| {struct1}-{struct2} | {score:.2f} | ![Dissimilar Pair {i+1}]({img_path}) |\n"
    
    return similar_table, dissimilar_table

def main():
    """Main function to generate visualizations."""
    parser = argparse.ArgumentParser(description="Generate similarity matrix visualizations for visual tests")
    parser.add_argument("--db_path", type=str, required=True, help="Path to similarity database")
    parser.add_argument("--parquet_dir", type=str, required=True, help="Path to parquet directory or file")
    parser.add_argument("--output_dir", type=str, default="./docs/figures", help="Output directory for visualizations")
    parser.add_argument("--high_snr", type=float, default=10.0, help="High SNR value for visualization (default: 10.0)")
    parser.add_argument("--low_snr", type=float, default=1.0, help="Low SNR value for visualization (default: 1.0)")
    parser.add_argument("--md_output", type=str, default="./docs/similarity_matrix.md", help="Output file for markdown")
    parser.add_argument("--copick_config", type=str, help="Path to Copick config file")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract structure IDs from Copick config if provided
    structure_ids = None
    if args.copick_config:
        try:
            import copick
            root = copick.from_file(args.copick_config)
            
            # Process pickable objects
            structure_ids = []
            structure_names = []
            
            if hasattr(root, 'pickable_objects'):
                for obj in root.pickable_objects:
                    if hasattr(obj, 'is_particle') and obj.is_particle:
                        if hasattr(obj, 'pdb_id') and hasattr(obj, 'name'):
                            structure_ids.append(obj.pdb_id)
                            structure_names.append(obj.name)
            
            logger.info(f"Loaded {len(structure_ids)} structure IDs from Copick config")
        except Exception as e:
            logger.warning(f"Error loading Copick config: {str(e)}")
            structure_ids = None
    
    # Initialize dataset for structure visualization with multiple SNR levels
    try:
        dataset = MultiSNRStructureDataset(
            base_dir=args.parquet_dir,
            box_size=48,
            device='cpu',
            augment=False
        )
    except Exception as e:
        logger.error(f"Error initializing dataset: {e}")
        # Try using CurriculumParquetDataset as fallback
        try:
            logger.info("Trying to load data using CurriculumParquetDataset...")
            structure_ids = ['4V1W', '1FA2', '6X1Q', '6EK0', '6SCJ', '6N4V']
            structure_names = ['apo-ferritin', 'beta-amylase', 'beta-galactoside', 'ribosome', 'thyroglobulin', 'virus-like-particle']
            name_to_pdb = dict(zip(structure_names, structure_ids))

            dataset = CurriculumParquetDataset(
                parquet_paths=[args.parquet_dir],
                name_to_pdb=name_to_pdb,
                box_size=48,
                device='cpu',
                augment=False
            )
            # Make the interface compatible with our custom class
            dataset.structure_names = structure_ids
        except Exception as nested_e:
            logger.error(f"Error loading fallback dataset: {nested_e}")
            sys.exit(1)
    
    # Get structure IDs from dataset
    if structure_ids is None:
        structure_ids = dataset.structure_names
        
    logger.info(f"Loaded dataset with {len(structure_ids)} structures")
    
    # Load similarity matrix
    calculator = SimilarityCalculator(args.db_path)
    similarity_matrix, molecule_order = calculator.load_matrix(structure_ids)
    
    logger.info(f"Loaded similarity matrix with shape {similarity_matrix.shape}")
    
    # Generate visualization paths
    matrix_path = output_dir / "similarity_matrix_heatmap.png"
    distribution_path = output_dir / "similarity_score_distribution.png"

    # Generate basic visualizations
    plot_similarity_heatmap(similarity_matrix, molecule_order, matrix_path)
    plot_similarity_distribution(similarity_matrix, distribution_path)
    
    # Find most similar and dissimilar pairs
    similar_pairs, dissimilar_pairs = find_extreme_pairs(similarity_matrix, molecule_order)
    
    # Visualize the extreme pairs using appropriate SNR samples
    # For similar pairs, use high SNR to better show similarities
    # For dissimilar pairs, use low SNR to better show differences
    for i, pair in enumerate(similar_pairs):
        output_path = output_dir / f"similar_pair_{i+1}.png"
        plot_structure_pair(dataset, pair, output_path, is_similar=True, high_snr=args.high_snr, low_snr=args.low_snr)
    
    for i, pair in enumerate(dissimilar_pairs):
        output_path = output_dir / f"dissimilar_pair_{i+1}.png"
        plot_structure_pair(dataset, pair, output_path, is_similar=False, high_snr=args.high_snr, low_snr=args.low_snr)
    
    # Generate markdown tables for the test document
    similar_table, dissimilar_table = generate_markdown_tables(similar_pairs, dissimilar_pairs, output_dir)
    
    # Write tables to output file
    tables_path = Path(args.md_output)
    
    # Create markdown content
    md_content = f"""# Similarity Matrix Analysis

This document presents visualizations and analyses of the structure similarity matrix generated for CryoLens.

## Similarity Matrix Heatmap

The following heatmap visualizes the similarity between different structures with hierarchical clustering:

![Similarity Matrix Heatmap]({Path(matrix_path).relative_to(tables_path.parent)})

## Similarity Score Distribution

The distribution of similarity scores across all structure pairs:

![Similarity Score Distribution]({Path(distribution_path).relative_to(tables_path.parent)})

# Similar Structures Table

{similar_table}

# Dissimilar Structures Table

{dissimilar_table}
"""
    
    # Create parent directory if needed
    tables_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write markdown file
    with open(tables_path, 'w') as f:
        f.write(md_content)
    
    logger.info(f"Wrote markdown content to {tables_path}")
    
    # Try to update the similarity_matrix_test.md file if it exists
    try:
        # Get the directory of this script
        script_dir = Path(__file__).parent
        
        # Define the path to the update script
        update_script = script_dir / "update_markdown_tables.py"
        md_file = script_dir / "similarity_matrix_test.md"
        
        if update_script.exists() and md_file.exists():
            # Run the update script
            cmd = [sys.executable, str(update_script), "--md_file", str(md_file), "--tables_file", str(tables_path)]
            logger.info(f"Running update script: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info("Markdown file updated successfully")
    except Exception as e:
        logger.warning(f"Error updating similarity_matrix_test.md file: {e}")
    
    logger.info("All visualizations generated successfully")

if __name__ == "__main__":
    main()