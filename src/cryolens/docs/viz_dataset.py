#!/usr/bin/env python
"""
Generate visualizations for CryoLens MLC Dataset
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse
import logging
import re
import json

from cryolens.data.datasets import CurriculumParquetDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_structure_ids_from_copick(config_path):
    """Get structure IDs and names from Copick config."""
    try:
        import copick
        root = copick.from_file(config_path)
        
        # Process pickable objects
        structure_ids = []
        structure_names = []
        
        if hasattr(root, 'pickable_objects'):
            for obj in root.pickable_objects:
                if hasattr(obj, 'is_particle') and obj.is_particle:
                    if hasattr(obj, 'pdb_id') and hasattr(obj, 'name'):
                        structure_ids.append(obj.pdb_id)
                        structure_names.append(obj.name)
        
        if not structure_ids:
            # Fallback values
            structure_ids = ['4V1W', '1FA2', '6X1Q', '6EK0', '6SCJ', '6N4V']
            structure_names = ['apo-ferritin', 'beta-amylase', 'beta-galactoside', 'ribosome', 'thyroglobulin', 'virus-like-particle']
            
        return structure_ids, structure_names
    except Exception as e:
        logger.warning(f"Error loading Copick config: {str(e)}")
        # Fallback values
        structure_ids = ['4V1W', '1FA2', '6X1Q', '6EK0', '6SCJ', '6N4V']
        structure_names = ['apo-ferritin', 'beta-amylase', 'beta-galactoside', 'ribosome', 'thyroglobulin', 'virus-like-particle']
        return structure_ids, structure_names

def plot_structure_distribution(dataset, output_path):
    """
    Plot the distribution of structures in the dataset.
    
    Args:
        dataset: Instance of CurriculumParquetDataset
        output_path: Path to save the output image
    """
    # Count structures by name
    structure_counts = {}
    
    # Extract unique molecule IDs from the dataframe
    unique_molecules = dataset.df['molecule_id'].unique()
    
    # Count occurrences of each molecule
    for molecule in unique_molecules:
        if pd.isna(molecule):
            molecule = "background"
            
        count = len(dataset.df[dataset.df['molecule_id'] == molecule]) if molecule != "background" else len(dataset.df[dataset.df['molecule_id'].isna()])
        structure_counts[molecule] = count
    
    # Sort by count
    sorted_structures = sorted(structure_counts.items(), key=lambda x: x[1], reverse=True)
    names = [name for name, _ in sorted_structures]
    counts = [count for _, count in sorted_structures]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=names, y=counts)
    plt.title(f"Structure Distribution in Dataset (Total: {len(sorted_structures)} structures)")
    plt.xlabel("Structure Name")
    plt.ylabel("Count")
    plt.xticks(rotation=90, ha="right", fontsize=8)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved structure distribution to {output_path}")

def plot_sample_structures(dataset, output_path, samples_per_structure=3):
    """
    Plot sample volumes from the dataset.
    
    Args:
        dataset: Instance of CurriculumParquetDataset
        output_path: Path to save the output image
        samples_per_structure: Number of samples to display per structure
    """
    # Get unique molecule IDs
    unique_molecules = [m for m in dataset.df['molecule_id'].unique()]
    
    # Add background
    if 'background' not in unique_molecules and any(pd.isna(m) for m in unique_molecules):
        unique_molecules = [m for m in unique_molecules if not pd.isna(m)]
        unique_molecules.append('background')
    
    # Limit to 4 structures
    n_structures = min(len(unique_molecules), 4)
    
    # Create a figure with 3 views (XY, XZ, YZ) for each sample
    fig, axes = plt.subplots(n_structures, samples_per_structure * 3, 
                            figsize=(samples_per_structure * 5, n_structures * 3))
    
    # Process each structure
    for i, molecule_name in enumerate(unique_molecules[:n_structures]):
        if n_structures == 1:
            row_axes = axes
        else:
            row_axes = axes[i]
        
        # Get dataset indices for this structure
        if molecule_name == 'background':
            structure_indices = dataset.df[dataset.df['molecule_id'].isna()].index.tolist()
        else:
            structure_indices = dataset.df[dataset.df['molecule_id'] == molecule_name].index.tolist()
        
        # Get samples for this structure
        for j in range(samples_per_structure):
            # Get a random dataset index for this structure
            if not structure_indices:
                continue
                
            structure_idx = np.random.choice(structure_indices)
            
            # Get sample using the get_sample_by_index method
            sample, _, source_type = dataset.get_sample_by_index(structure_idx)
            
            # Get SNR value from source type or filename
            snr_match = re.search(r'snr[-_]?(\d+\.?\d*)', str(source_type).lower())
            snr_value = f"SNR {snr_match.group(1)}" if snr_match else "Unknown SNR"
            
            # Convert to numpy array
            volume = sample[0].numpy()  # First dimension is the channel
            
            # Create projections
            xy_proj = np.max(volume, axis=0)  # Max projection along z
            xz_proj = np.max(volume, axis=1)  # Max projection along y
            yz_proj = np.max(volume, axis=2)  # Max projection along x
            
            # Normalize projections
            def normalize(img):
                if img.max() == img.min():
                    return np.zeros_like(img)
                return (img - img.min()) / (img.max() - img.min())
            
            xy_proj = normalize(xy_proj)
            xz_proj = normalize(xz_proj)
            yz_proj = normalize(yz_proj)
            
            # Plot projections
            col_idx = j * 3
            if n_structures == 1 and samples_per_structure == 1:
                # Single plot case
                axes[0].imshow(xy_proj, cmap='gray')
                axes[0].set_title(f"XY Projection\n{snr_value}")
                axes[0].axis('off')
                
                axes[1].imshow(xz_proj, cmap='gray')
                axes[1].set_title(f"XZ Projection\n{snr_value}")
                axes[1].axis('off')
                
                axes[2].imshow(yz_proj, cmap='gray')
                axes[2].set_title(f"YZ Projection\n{snr_value}")
                axes[2].axis('off')
            else:
                # Multiple plots case
                row_axes[col_idx].imshow(xy_proj, cmap='gray')
                row_axes[col_idx].set_title(f"XY\n{snr_value}")
                row_axes[col_idx].axis('off')
                
                row_axes[col_idx + 1].imshow(xz_proj, cmap='gray')
                row_axes[col_idx + 1].set_title(f"XZ\n{snr_value}")
                row_axes[col_idx + 1].axis('off')
                
                row_axes[col_idx + 2].imshow(yz_proj, cmap='gray')
                row_axes[col_idx + 2].set_title(f"YZ\n{snr_value}")
                row_axes[col_idx + 2].axis('off')
            
            # Add structure name to first column only
            if j == 0:
                if n_structures == 1:
                    row_axes[0].set_ylabel(molecule_name, fontsize=12, rotation=0, labelpad=40, va='center')
                else:
                    row_axes[0].set_ylabel(molecule_name, fontsize=10, rotation=0, labelpad=20, va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved sample structures to {output_path}")

def plot_all_molecule_projections(dataset, output_dir, samples_per_molecule=3):
    """
    Save 3 examples of each molecule ID (including background) with sum projections along each axis.
    
    Args:
        dataset: Instance of CurriculumParquetDataset
        output_dir: Path to the output directory
        samples_per_molecule: Number of samples to save per molecule
    """
    # Create output directory
    projections_dir = Path(output_dir) / "molecule_projections"
    projections_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique molecule IDs, including handling background
    unique_molecules = list(dataset.df['molecule_id'].dropna().unique())
    
    # Check if we have background samples (NaN values)
    has_background = dataset.df['molecule_id'].isna().any()
    
    # Process each molecule
    for molecule_name in unique_molecules + (['background'] if has_background else []):
        logger.info(f"Processing projections for {molecule_name}")
        
        # Get dataset indices for this molecule
        if molecule_name == 'background':
            molecule_indices = dataset.df[dataset.df['molecule_id'].isna()].index.tolist()
            safe_name = 'background'
        else:
            molecule_indices = dataset.df[dataset.df['molecule_id'] == molecule_name].index.tolist()
            # Create a safe filename from the molecule name
            safe_name = re.sub(r'[^\w\-]', '_', molecule_name)
        
        # Skip if no samples found
        if not molecule_indices:
            logger.warning(f"No samples found for molecule {molecule_name}")
            continue
        
        # Randomly sample up to samples_per_molecule indices
        sample_count = min(samples_per_molecule, len(molecule_indices))
        sampled_indices = np.random.choice(molecule_indices, size=sample_count, replace=False)
        
        # Process each sampled index
        for i, idx in enumerate(sampled_indices):
            # Get sample
            sample, _, source_type = dataset.get_sample_by_index(idx)
            
            # Convert to numpy array
            volume = sample[0].numpy()  # First dimension is the channel
            
            # Create sum projections (instead of max)
            xy_proj = np.sum(volume, axis=0)  # Sum projection along z
            xz_proj = np.sum(volume, axis=1)  # Sum projection along y
            yz_proj = np.sum(volume, axis=2)  # Sum projection along x
            
            # Normalize projections
            def normalize(img):
                if img.max() == img.min():
                    return np.zeros_like(img)
                return (img - img.min()) / (img.max() - img.min())
            
            xy_proj = normalize(xy_proj)
            xz_proj = normalize(xz_proj)
            yz_proj = normalize(yz_proj)
            
            # Get SNR value from source type or filename if available
            snr_match = re.search(r'snr[-_]?(\d+\.?\d*)', str(source_type).lower())
            snr_value = f"SNR {snr_match.group(1)}" if snr_match else "Unknown SNR"
            
            # Create figure with 3 views
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot projections
            axes[0].imshow(xy_proj, cmap='gray')
            axes[0].set_title(f"{safe_name} - XY Sum Projection")
            axes[0].axis('off')
            
            axes[1].imshow(xz_proj, cmap='gray')
            axes[1].set_title(f"{safe_name} - XZ Sum Projection")
            axes[1].axis('off')
            
            axes[2].imshow(yz_proj, cmap='gray')
            axes[2].set_title(f"{safe_name} - YZ Sum Projection")
            axes[2].axis('off')
            
            plt.suptitle(f"{safe_name} - Sample {i+1} - {snr_value}")
            plt.tight_layout()
            
            # Save figure
            output_path = projections_dir / f"{safe_name}_sample_{i+1}.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
            
    logger.info(f"Saved projections for all molecules to {projections_dir}")

def plot_density_distribution(dataset, output_path, n_samples=1000):
    """
    Plot the density distribution across samples.
    
    Args:
        dataset: Instance of CurriculumParquetDataset
        output_path: Path to save the output image
        n_samples: Number of samples to use for the distribution
    """
    # Collect density values
    densities = []
    structures = []
    
    # Limit sampling to keep it efficient
    actual_samples = min(n_samples, 100)
    
    # Get unique molecule IDs
    unique_molecules = [m for m in dataset.df['molecule_id'].unique()]
    
    # Collect samples from each structure (up to 10)
    for molecule_name in unique_molecules[:10]:
        # Handle background
        if pd.isna(molecule_name):
            molecule_name = "background"
            structure_indices = dataset.df[dataset.df['molecule_id'].isna()].index.tolist()
        else:
            structure_indices = dataset.df[dataset.df['molecule_id'] == molecule_name].index.tolist()
        
        if not structure_indices:
            continue
        
        # Limit samples per structure
        samples_per_structure = max(1, actual_samples // min(10, len(unique_molecules)))
        
        # Collect samples for this structure
        for _ in range(samples_per_structure):
            # Pick a random index for this structure
            structure_idx = np.random.choice(structure_indices)
            
            # Get sample using the get_sample_by_index method
            sample, _, _ = dataset.get_sample_by_index(structure_idx)
            volume = sample[0].numpy()  # First dimension is the channel
            
            # Calculate density (mean value)
            density = np.mean(volume)
            densities.append(density)
            structures.append(molecule_name)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Violin plot if we have enough data
    if len(densities) > 10:
        sns.violinplot(x=structures, y=densities)
    else:
        # Fallback to boxplot for small datasets
        sns.boxplot(x=structures, y=densities)
        
    plt.title("Density Distribution Across Structures")
    plt.xlabel("Structure Name")
    plt.ylabel("Mean Density")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved density distribution to {output_path}")

def find_snr_variations(dataset, molecule_name):
    """
    Find samples of a structure across different SNR levels.
    
    Args:
        dataset: Instance of CurriculumParquetDataset
        molecule_name: Name of the structure to find
        
    Returns:
        dict: Mapping from SNR level to dataset indices
    """
    # Get indices for this structure
    if molecule_name == 'background':
        structure_indices = dataset.df[dataset.df['molecule_id'].isna()].index.tolist()
    else:
        structure_indices = dataset.df[dataset.df['molecule_id'] == molecule_name].index.tolist()
    
    if not structure_indices:
        logger.warning(f"Structure {molecule_name} not found in dataset")
        return {}
    
    # Collect SNR levels
    snr_samples = {}
    
    for structure_idx in structure_indices:
        # Get source type for this sample
        _, _, source_type = dataset.get_sample_by_index(structure_idx)
        
        # Extract SNR value from source type
        snr_match = re.search(r'snr[-_]?(\d+\.?\d*)', str(source_type).lower())
        
        if snr_match:
            snr_value = float(snr_match.group(1))
            
            # Skip if we already have this SNR level
            if snr_value in snr_samples:
                continue
            
            # Store sample info
            snr_samples[snr_value] = structure_idx
    
    return snr_samples

def plot_snr_comparison(dataset, output_path, molecule_name=None):
    """
    Plot a comparison of the same structure across different SNR levels.
    
    Args:
        dataset: Instance of CurriculumParquetDataset
        output_path: Path to save the output image
        molecule_name: Specific structure to visualize (if None, picks one with most SNR variants)
    """
    # If no structure provided, find one with the most SNR variations
    if molecule_name is None:
        unique_molecules = [m for m in dataset.df['molecule_id'].unique()]
        if any(pd.isna(m) for m in unique_molecules):
            unique_molecules = [m for m in unique_molecules if not pd.isna(m)]
            unique_molecules.append('background')
        
        best_structure = None
        most_snr_levels = 0
        
        # Check the first 10 structures to find one with many SNR levels
        for struct in unique_molecules[:10]:
            snr_variations = find_snr_variations(dataset, struct)
            if len(snr_variations) > most_snr_levels:
                most_snr_levels = len(snr_variations)
                best_structure = struct
        
        if best_structure is None:
            logger.warning("Could not find a structure with multiple SNR levels")
            return
        
        molecule_name = best_structure
    
    # Find SNR variations for this structure
    snr_variations = find_snr_variations(dataset, molecule_name)
    
    if not snr_variations:
        logger.warning(f"No SNR variations found for structure {molecule_name}")
        return
    
    # Sort by SNR level
    snr_levels = sorted(snr_variations.keys())
    
    # Show all SNR levels, up to 6
    if len(snr_levels) > 6:
        # Pick highest, lowest, and evenly spaced middle values to get total of 6
        step = len(snr_levels) / 5
        indices = [int(i * step) for i in range(5)] + [len(snr_levels) - 1]
        snr_levels = [snr_levels[i] for i in indices]
    
    # Create figure
    fig, axes = plt.subplots(len(snr_levels), 3, figsize=(12, 3 * len(snr_levels)))
    
    # Process each SNR level
    for i, snr in enumerate(snr_levels):
        structure_idx = snr_variations[snr]
        
        # Get sample
        sample, _, _ = dataset.get_sample_by_index(structure_idx)
        volume = sample[0].numpy()  # First dimension is the channel
        
        # Create sum projections (instead of max)
        xy_proj = np.sum(volume, axis=0)
        xz_proj = np.sum(volume, axis=1)
        yz_proj = np.sum(volume, axis=2)
        
        # Normalize projections
        def normalize(img):
            if img.max() == img.min():
                return np.zeros_like(img)
            return (img - img.min()) / (img.max() - img.min())
        
        xy_proj = normalize(xy_proj)
        xz_proj = normalize(xz_proj)
        yz_proj = normalize(yz_proj)
        
        # Plot projections
        if len(snr_levels) == 1:
            row_axes = axes
        else:
            row_axes = axes[i]
        
        row_axes[0].imshow(xy_proj, cmap='gray')
        row_axes[0].set_title(f"XY Sum Projection")
        row_axes[0].set_ylabel(f"SNR {snr}", fontsize=12)
        row_axes[0].axis('off')
        
        row_axes[1].imshow(xz_proj, cmap='gray')
        row_axes[1].set_title(f"XZ Sum Projection")
        row_axes[1].axis('off')
        
        row_axes[2].imshow(yz_proj, cmap='gray')
        row_axes[2].set_title(f"YZ Sum Projection")
        row_axes[2].axis('off')
    
    plt.suptitle(f"Structure {molecule_name} Across Different SNR Levels", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for suptitle
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved SNR comparison to {output_path}")

def find_high_low_snr_samples(dataset, n_structures=3):
    """
    Find samples of different structures at both high and low SNR.
    
    Args:
        dataset: Instance of CurriculumParquetDataset
        n_structures: Number of different structures to find
        
    Returns:
        dict: Mapping from structure_name to dict of {'high': (snr, sample), 'low': (snr, sample)}
    """
    # Get unique molecule IDs
    unique_molecules = [m for m in dataset.df['molecule_id'].unique()]
    
    # Handle background
    has_background = any(pd.isna(m) for m in unique_molecules)
    unique_molecules = [m for m in unique_molecules if not pd.isna(m)]
    if has_background:
        unique_molecules.append('background')
    
    # Shuffle to randomize selection
    np.random.shuffle(unique_molecules)
    
    # Result dictionary
    structure_samples = {}
    
    # Process structures until we have enough
    for molecule_name in unique_molecules:
        # Skip if we already have enough structures
        if len(structure_samples) >= n_structures:
            break
        
        # Get SNR variations for this structure
        snr_variations = find_snr_variations(dataset, molecule_name)
        
        # Skip if not enough variations
        if len(snr_variations) < 2:
            continue
        
        # Sort by SNR level
        snr_levels = sorted(snr_variations.keys())
        
        # Get lowest and highest SNR
        low_snr = snr_levels[0]
        high_snr = snr_levels[-1]
        
        # Skip if SNR range is too small
        if high_snr - low_snr < 2.0:
            continue
        
        # Get samples
        low_idx = snr_variations[low_snr]
        high_idx = snr_variations[high_snr]
        
        low_sample, _, _ = dataset.get_sample_by_index(low_idx)
        high_sample, _, _ = dataset.get_sample_by_index(high_idx)
        
        # Store samples
        structure_samples[molecule_name] = {
            'low': (low_snr, low_sample),
            'high': (high_snr, high_sample)
        }
    
    return structure_samples

def plot_structure_snr_comparison(dataset, output_path_high, output_path_low):
    """
    Plot a comparison of different structures at high and low SNR.
    
    Args:
        dataset: Instance of CurriculumParquetDataset
        output_path_high: Path to save the high SNR comparison image
        output_path_low: Path to save the low SNR comparison image
    """
    # Find samples of different structures at high and low SNR
    n_structures = 3
    structure_samples = find_high_low_snr_samples(dataset, n_structures)
    
    if not structure_samples:
        logger.warning("Could not find enough structures with both high and low SNR samples")
        return
    
    # Create figures for high and low SNR
    fig_high, axes_high = plt.subplots(len(structure_samples), 3, figsize=(12, 3 * len(structure_samples)))
    fig_low, axes_low = plt.subplots(len(structure_samples), 3, figsize=(12, 3 * len(structure_samples)))
    
    # Process each structure
    for i, (molecule_name, samples) in enumerate(structure_samples.items()):
        # Process high SNR sample
        snr_high, high_sample = samples['high']
        volume_high = high_sample[0].numpy()  # First dimension is the channel
        
        # Process low SNR sample
        snr_low, low_sample = samples['low']
        volume_low = low_sample[0].numpy()  # First dimension is the channel
        
        # Create sum projections
        def get_sum_projections(volume):
            xy_proj = np.sum(volume, axis=0)  # Sum projection along z
            xz_proj = np.sum(volume, axis=1)  # Sum projection along y
            yz_proj = np.sum(volume, axis=2)  # Sum projection along x
            
            # Normalize projections
            def normalize(img):
                if img.max() == img.min():
                    return np.zeros_like(img)
                return (img - img.min()) / (img.max() - img.min())
            
            return normalize(xy_proj), normalize(xz_proj), normalize(yz_proj)
        
        xy_high, xz_high, yz_high = get_sum_projections(volume_high)
        xy_low, xz_low, yz_low = get_sum_projections(volume_low)
        
        # Plot high SNR projections
        if len(structure_samples) == 1:
            row_axes_high = axes_high
        else:
            row_axes_high = axes_high[i]
        
        row_axes_high[0].imshow(xy_high, cmap='gray')
        row_axes_high[0].set_title(f"XY Sum Projection")
        row_axes_high[0].set_ylabel(f"{molecule_name}\nSNR {snr_high}", fontsize=10)
        row_axes_high[0].axis('off')
        
        row_axes_high[1].imshow(xz_high, cmap='gray')
        row_axes_high[1].set_title(f"XZ Sum Projection")
        row_axes_high[1].axis('off')
        
        row_axes_high[2].imshow(yz_high, cmap='gray')
        row_axes_high[2].set_title(f"YZ Sum Projection")
        row_axes_high[2].axis('off')
        
        # Plot low SNR projections
        if len(structure_samples) == 1:
            row_axes_low = axes_low
        else:
            row_axes_low = axes_low[i]
        
        row_axes_low[0].imshow(xy_low, cmap='gray')
        row_axes_low[0].set_title(f"XY Sum Projection")
        row_axes_low[0].set_ylabel(f"{molecule_name}\nSNR {snr_low}", fontsize=10)
        row_axes_low[0].axis('off')
        
        row_axes_low[1].imshow(xz_low, cmap='gray')
        row_axes_low[1].set_title(f"XZ Sum Projection")
        row_axes_low[1].axis('off')
        
        row_axes_low[2].imshow(yz_low, cmap='gray')
        row_axes_low[2].set_title(f"YZ Sum Projection")
        row_axes_low[2].axis('off')
    
    # Set titles and save figures
    fig_high.suptitle(f"Different Structures at High SNR", fontsize=14)
    fig_high.tight_layout()
    fig_high.subplots_adjust(top=0.9)  # Adjust for suptitle
    
    fig_low.suptitle(f"Different Structures at Low SNR", fontsize=14)
    fig_low.tight_layout()
    fig_low.subplots_adjust(top=0.9)  # Adjust for suptitle
    
    fig_high.savefig(output_path_high, dpi=150)
    fig_low.savefig(output_path_low, dpi=150)
    
    plt.close(fig_high)
    plt.close(fig_low)
    
    logger.info(f"Saved structure comparison at high SNR to {output_path_high}")
    logger.info(f"Saved structure comparison at low SNR to {output_path_low}")

def plot_source_type_distribution(dataset, output_path):
    """
    Plot the distribution of source types in the dataset.
    
    Args:
        dataset: Instance of CurriculumParquetDataset
        output_path: Path to save the output image
    """
    if 'source_type' not in dataset.df.columns:
        logger.warning("source_type column not found in dataset")
        return
        
    # Count source types
    source_counts = dataset.df['source_type'].value_counts()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=source_counts.index, y=source_counts.values)
    plt.title(f"Source Type Distribution (Total: {len(source_counts)} types)")
    plt.xlabel("Source Type")
    plt.ylabel("Count")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved source type distribution to {output_path}")

def plot_curriculum_weights(curriculum_path, output_path):
    """
    Plot the weight distribution in the curriculum specification.
    
    Args:
        curriculum_path: Path to curriculum specification JSON file
        output_path: Path to save the output image
    """
    if not curriculum_path or not os.path.exists(curriculum_path):
        logger.warning(f"Curriculum file not found: {curriculum_path}")
        return
        
    try:
        # Load curriculum specification
        with open(curriculum_path, 'r') as f:
            curriculum = json.load(f)
            
        if not curriculum or not isinstance(curriculum, list):
            logger.warning("Invalid curriculum format")
            return
            
        # Create figure
        n_stages = len(curriculum)
        fig, axes = plt.subplots(n_stages, 1, figsize=(12, 4 * n_stages))
        
        # Handle single stage case
        if n_stages == 1:
            axes = [axes]
            
        # Plot weights for each stage
        for i, stage in enumerate(curriculum):
            weights = stage.get('weights', {})
            duration = stage.get('duration', float('inf'))
            
            if not weights:
                axes[i].text(0.5, 0.5, "No weights specified", 
                           ha='center', va='center', fontsize=14)
                axes[i].set_title(f"Stage {i+1} (Duration: {duration if duration != float('inf') else 'Infinite'})")
                axes[i].axis('off')
                continue
                
            # Sort by weight
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            keys = [k for k, _ in sorted_weights]
            values = [v for _, v in sorted_weights]
            
            # Plot
            axes[i].bar(keys, values)
            axes[i].set_title(f"Stage {i+1} (Duration: {duration if duration != float('inf') else 'Infinite'})")
            axes[i].set_ylabel("Weight")
            axes[i].tick_params(axis='x', rotation=90)
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved curriculum weights to {output_path}")
        
    except Exception as e:
        logger.warning(f"Error plotting curriculum weights: {str(e)}")

def main():
    """Main function to generate visualizations."""
    parser = argparse.ArgumentParser(description="Generate dataset visualizations for CryoLens MLC")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to parquet file")
    parser.add_argument("--copick_config", type=str, required=True, help="Path to Copick config file")
    parser.add_argument("--curriculum_path", type=str, default=None, help="Path to curriculum specification")
    parser.add_argument("--output_dir", type=str, default="./docs", help="Output directory for visualizations")
    parser.add_argument("--box_size", type=int, default=48, help="Box size for volumes")
    parser.add_argument("--normalization", type=str, default="z-score", choices=["z-score", "min-max", "percentile", "none"], 
                      help="Type of normalization to apply")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figures subdirectory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Get structure IDs and names from Copick config
    structure_ids, structure_names = get_structure_ids_from_copick(args.copick_config)
    
    # Create name to PDB mapping
    name_to_pdb = dict(zip(structure_names, structure_ids))
    
    # Load curriculum spec if provided
    curriculum_spec = None
    if args.curriculum_path and os.path.exists(args.curriculum_path):
        try:
            with open(args.curriculum_path, 'r') as f:
                curriculum_spec = json.load(f)
            logger.info(f"Loaded curriculum specification with {len(curriculum_spec)} stages")
        except Exception as e:
            logger.warning(f"Error loading curriculum specification: {str(e)}")
    
    # Initialize dataset
    logger.info(f"Initializing dataset from {args.parquet_path}")
    dataset = CurriculumParquetDataset(
        parquet_paths=[args.parquet_path],
        name_to_pdb=name_to_pdb,
        box_size=args.box_size,
        curriculum_spec=curriculum_spec,
        samples_per_epoch=1000,  # Use a smaller value for visualization
        device='cpu',
        augment=False,  # Disable augmentation for consistent visualization
        normalization=args.normalization
    )
    
    logger.info(f"Dataset loaded with {len(dataset.df)} samples")
    logger.info(f"Found {len(set(dataset.df['molecule_id'].dropna()))} unique structure names")
    
    # Generate visualization paths
    structure_dist_path = figures_dir / "dataset_viz_structure_distribution.png"
    sample_structures_path = figures_dir / "dataset_viz_sample_structures.png"
    density_dist_path = figures_dir / "dataset_viz_density_distribution.png"
    snr_comparison_path = figures_dir / "dataset_viz_snr_comparison.png"
    high_snr_path = figures_dir / "dataset_viz_structure_high_snr.png"
    low_snr_path = figures_dir / "dataset_viz_structure_low_snr.png"
    source_type_path = figures_dir / "dataset_viz_source_type_distribution.png"
    curriculum_path = figures_dir / "dataset_viz_curriculum_weights.png"
    
    # Generate basic visualizations
    plot_structure_distribution(dataset, structure_dist_path)
    plot_sample_structures(dataset, sample_structures_path)
    plot_density_distribution(dataset, density_dist_path)
    
    # Generate SNR comparison visualizations
    plot_snr_comparison(dataset, snr_comparison_path)
    plot_structure_snr_comparison(dataset, high_snr_path, low_snr_path)
    
    # Generate additional visualizations
    plot_source_type_distribution(dataset, source_type_path)
    if args.curriculum_path:
        plot_curriculum_weights(args.curriculum_path, curriculum_path)
    
    # Generate molecule projections (3 examples per molecule with sum projections)
    plot_all_molecule_projections(dataset, output_dir, samples_per_molecule=3)
    
    logger.info("All visualizations generated successfully")

if __name__ == "__main__":
    main()