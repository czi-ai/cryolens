"""
Evaluate classification performance with statistical validation.

This script evaluates classification using CryoLens and TomoTwin embeddings with
proper cross-validation and statistical testing. All methods are compared at the
same dimensionality (32D) for fair comparison.

Usage:
    python -m cryolens.scripts.evaluate_classification \\
        --config examples/classification_config.yaml \\
        --output-dir ./results/classification/

Configuration file should specify:
    - cryolens_embeddings: path to embeddings.h5
    - tomotwin_embeddings: path to tomotwin.parquet
    - tomotwin_coords: path to coords.csv
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from cryolens.evaluation.classification import (
    stratified_cross_validation,
    compute_per_class_metrics_from_predictions,
    compute_statistical_significance
)
from cryolens.evaluation.attention_fusion_cv import (
    stratified_cross_validation_with_attention,
    train_final_fusion_and_save
)


import torch


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def normalize_protein_name(name: str) -> str:
    """Normalize protein names to consistent format."""
    if pd.isna(name):
        return 'unknown'
    
    # Standardize to lowercase with underscores
    normalized = str(name).lower().replace('-', '_').replace(' ', '_')
    
    # Fix common naming inconsistencies
    name_corrections = {
        'beta_galactoside': 'beta_galactosidase',
        'betagalactoside': 'beta_galactosidase',
        'beta_amylase': 'beta_amylase',  # Keep as-is
        'betaamylase': 'beta_amylase',
        'apo_ferritin': 'apoferritin',
        'apoferritin': 'apoferritin',
        'virus_like_particle': 'virus_like_particle',  # Keep as-is
        'vlp': 'virus_like_particle',
        'ribosome': 'ribosome',  # Keep as-is
        'thyroglobulin': 'thyroglobulin',  # Keep as-is
    }
    
    return name_corrections.get(normalized, normalized)


def load_cryolens_embeddings(h5_path: Path, structural_dim: int = 32) -> Tuple[np.ndarray, List[str], List[str], List[Dict]]:
    """
    Load CryoLens embeddings from H5 file, extracting only structural dimensions.
    
    CryoLens uses a segmented decoder where the first 80% of latent dimensions
    (32D out of 40D) correspond to structural information. For fair comparison
    with TomoTwin's 32D embeddings, we extract only these structural dimensions.
    
    Args:
        h5_path: Path to HDF5 file with embeddings
        structural_dim: Number of structural dimensions to extract (default: 32)
        
    Returns:
        Tuple of (32D embeddings array, list of labels, list of run names, list of metadata dicts)
    """
    embeddings_list = []
    labels_list = []
    run_names_list = []
    metadata_list = []
    
    with h5py.File(h5_path, 'r') as f:
        embeddings_group = f['embeddings']
        metadata_group = f.get('metadata', {})
        
        for sample_id in embeddings_group.keys():
            sample_group = embeddings_group[sample_id]
            
            # Get mean embedding
            if 'mu' in sample_group:
                embedding = sample_group['mu'][:]
            else:
                embedding = sample_group[:]
            
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            # Extract only structural dimensions (first 32D of 40D)
            structural_embedding = embedding[:structural_dim]
            embeddings_list.append(structural_embedding)
            
            # Extract metadata from sample_group attributes
            sample_metadata = {
                'sample_id': sample_id,
                'coordinates': sample_group.attrs.get('coordinates', None),
                'object_name': sample_group.attrs.get('object_name', None),
                'picks_index': sample_group.attrs.get('picks_index', None),
                'point_index': sample_group.attrs.get('point_index', None),
                'run_name': sample_group.attrs.get('run_name', None),
                'voxel_spacing': sample_group.attrs.get('voxel_spacing', None)
            }
            
            # Convert numpy arrays to lists for JSON serialization
            if sample_metadata['coordinates'] is not None:
                sample_metadata['coordinates'] = sample_metadata['coordinates'].tolist()
            
            metadata_list.append(sample_metadata)
            
            # Get structure name from metadata or sample_id
            structure_name = None
            if sample_id in metadata_group:
                meta = metadata_group[sample_id]
                if 'object_type' in meta.attrs:
                    structure_name = meta.attrs['object_type']
            
            if structure_name is None:
                # Extract from sample_id format: "protein_run_picks_id"
                parts = sample_id.split('_')
                structure_name = parts[0] if len(parts) >= 1 else 'unknown'
            
            # Extract run name from sample_id
            # Format: "protein_run_picks_id" -> extract "run"
            if len(sample_id.split('_')) >= 2:
                run_name = sample_id.split('_')[1]
            else:
                run_name = 'unknown'
            
            labels_list.append(normalize_protein_name(structure_name))
            run_names_list.append(run_name)
    
    embeddings = np.array(embeddings_list, dtype=np.float32)
    print(f"  Extracted {structural_dim}D structural embeddings (from {embedding.shape[0]}D total)")
    print(f"  Unique runs: {len(set(run_names_list))}")
    
    return embeddings, labels_list, run_names_list, metadata_list


def load_tomotwin_embeddings(
    parquet_path: Path,
    coords_path: Path,
    embedding_dim: int = 32
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load TomoTwin embeddings from parquet and coordinates CSV.
    
    Args:
        parquet_path: Path to TomoTwin embeddings parquet file
        coords_path: Path to coordinates CSV file
        embedding_dim: Expected embedding dimension (default: 32)
        
    Returns:
        Tuple of (32D embeddings array, list of labels, list of run names)
    """
    embeddings_df = pd.read_parquet(parquet_path)
    coords_df = pd.read_csv(coords_path)
    
    # Extract embedding columns (numeric columns excluding filepath)
    embedding_cols = [
        col for col in embeddings_df.columns 
        if col != 'filepath' and embeddings_df[col].dtype in ['float64', 'float32', 'int64', 'int32']
    ]
    
    # Ensure we have the expected dimension
    if len(embedding_cols) < embedding_dim:
        raise ValueError(f"Expected {embedding_dim}D embeddings, found {len(embedding_cols)}D")
    
    # Take only first embedding_dim columns
    embedding_cols = embedding_cols[:embedding_dim]
    
    # Merge embeddings with coordinates
    coords_df['filepath_standardized'] = 'output/' + coords_df['filepath']
    
    merged_df = pd.merge(
        coords_df,
        embeddings_df,
        left_on='filepath_standardized',
        right_on='filepath',
        how='inner'
    )
    
    if len(merged_df) == 0:
        raise ValueError("No matching samples found after merging embeddings and coordinates")
    
    embeddings = merged_df[embedding_cols].values.astype(np.float32)
    labels = [normalize_protein_name(protein) for protein in merged_df['protein']]
    
    # Extract run names from filepath (format: runXXX/...)
    run_names = []
    for fp in merged_df['filepath_x']:  # Use filepath_x from coords_df
        parts = fp.split('/')
        if len(parts) >= 1:
            run_names.append(parts[0])
        else:
            run_names.append('unknown')
    
    print(f"  Unique runs: {len(set(run_names))}")
    
    return embeddings, labels, run_names


def align_embeddings(
    cl_embeddings: np.ndarray,
    cl_labels: List[str],
    cl_metadata: List[Dict],
    tt_embeddings: np.ndarray,
    tt_labels: List[str],
    random_seed: int = 171717
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict], List[str]]:
    """
    Align CryoLens and TomoTwin embeddings by matching structure labels.
    
    Both embeddings should be 32D at this point for fair comparison.
    Balances samples per class to ensure equal representation.
    
    Args:
        cl_embeddings: CryoLens embeddings (32D)
        cl_labels: CryoLens labels
        cl_metadata: CryoLens metadata
        tt_embeddings: TomoTwin embeddings (32D)
        tt_labels: TomoTwin labels
        random_seed: Random seed for sampling
        
    Returns:
        Tuple of (aligned CryoLens embeddings, aligned TomoTwin embeddings,
                 aligned labels, aligned metadata, list of common structures)
    """
    # Verify dimensions match
    if cl_embeddings.shape[1] != tt_embeddings.shape[1]:
        raise ValueError(
            f"Dimension mismatch: CryoLens {cl_embeddings.shape[1]}D vs "
            f"TomoTwin {tt_embeddings.shape[1]}D"
        )
    
    # Find common structures
    cl_structures = set(cl_labels)
    tt_structures = set(tt_labels)
    common_structures = sorted(cl_structures & tt_structures)
    
    if not common_structures:
        raise ValueError("No common structures found between CryoLens and TomoTwin")
    
    print(f"Found {len(common_structures)} common structures: {common_structures}")
    
    aligned_cl = []
    aligned_tt = []
    aligned_labels = []
    aligned_metadata = []
    
    np.random.seed(random_seed)
    
    for structure in common_structures:
        # Get indices for this structure
        cl_indices = np.array([i for i, label in enumerate(cl_labels) if label == structure])
        tt_indices = np.array([i for i, label in enumerate(tt_labels) if label == structure])
        
        # Balance samples (take minimum to ensure same count)
        n_samples = min(len(cl_indices), len(tt_indices))
        if n_samples == 0:
            continue
        
        # Randomly sample to balance
        cl_selected = np.random.choice(cl_indices, n_samples, replace=False)
        tt_selected = np.random.choice(tt_indices, n_samples, replace=False)
        
        aligned_cl.extend(cl_embeddings[cl_selected])
        aligned_tt.extend(tt_embeddings[tt_selected])
        aligned_labels.extend([structure] * n_samples)
        aligned_metadata.extend([cl_metadata[i] for i in cl_selected])
    
    return (
        np.array(aligned_cl, dtype=np.float32),
        np.array(aligned_tt, dtype=np.float32),
        aligned_labels,
        aligned_metadata,
        common_structures
    )


def create_fusion_embeddings(
    tt_embeddings: np.ndarray,
    cl_embeddings: np.ndarray,
    fusion_method: str = 'average',
    device: str = 'cpu'
) -> np.ndarray:
    """
    Create fused embeddings from TomoTwin and CryoLens.
    
    All fusion methods maintain 32D output for fair comparison.
    
    Args:
        tt_embeddings: TomoTwin embeddings (N, 32)
        cl_embeddings: CryoLens embeddings (N, 32)
        fusion_method: Fusion strategy ('average', 'concat', 'weighted', 'attention')
        device: Device for attention fusion ('cpu', 'cuda', 'mps')
        
    Returns:
        Fused embeddings (N, 32)
    """
    if fusion_method == 'average':
        # Simple element-wise average (stays 32D)
        return (tt_embeddings + cl_embeddings) / 2
    
    elif fusion_method == 'concat':
        # Concatenate to 64D (will be reduced by classifier)
        # Note: This gives classifier more capacity - not strictly fair
        return np.concatenate([tt_embeddings, cl_embeddings], axis=1)
    
    elif fusion_method == 'weighted':
        # Fixed weighted average (70-30 split favoring TomoTwin)
        return 0.7 * tt_embeddings + 0.3 * cl_embeddings
    
    elif fusion_method == 'attention':
        # Learned attention-based fusion (trains a small network)
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from torch.utils.data import TensorDataset, DataLoader
        import torch.optim as optim
        
        # This will be implemented by train_attention_fusion
        # which is called from main() before this function
        raise RuntimeError(
            "Attention fusion requires model training. "
            "This should be handled in main() before calling create_fusion_embeddings."
        )
    
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")


def format_class_name(name: str) -> str:
    """Format class names for display in figures."""
    # Map to display names
    display_names = {
        'beta_galactosidase': 'β-Galactosidase',
        'beta_amylase': 'β-Amylase',
        'apoferritin': 'Apoferritin',
        'apo_ferritin': 'Apoferritin',
        'virus_like_particle': 'Virus-Like Particle',
        'ribosome': 'Ribosome',
        'thyroglobulin': 'Thyroglobulin',
    }
    
    return display_names.get(name, name.replace('_', ' ').title())


def create_classification_figure(
    results: Dict,
    per_class_results: Dict,
    common_structures: List[str],
    output_path: Path,
    embedding_dim: int = 32
):
    """
    Create comprehensive classification figure with error bars.
    
    Args:
        results: Overall results dictionary
        per_class_results: Per-class results dictionary
        common_structures: List of evaluated structure names
        output_path: Path to save figure
        embedding_dim: Embedding dimension used (for title annotation)
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Overall performance comparison
    ax1 = fig.add_subplot(gs[0, :2])
    methods = ['TomoTwin', 'CryoLens', 'Fusion']
    map_means = [
        results['tomotwin']['mean_map'],
        results['cryolens']['mean_map'],
        results['fusion']['mean_map']
    ]
    map_stds = [
        results['tomotwin']['std_map'],
        results['cryolens']['std_map'],
        results['fusion']['std_map']
    ]
    
    x = np.arange(len(methods))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax1.bar(x, map_means, yerr=map_stds, capsize=5, alpha=0.8, color=colors)
    
    ax1.set_ylabel('Mean Average Precision', fontsize=12, fontweight='bold')
    ax1.set_title(f'Overall Classification Performance (10-fold CV, {embedding_dim}D)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, max(map_means) * 1.2])
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, map_means, map_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # Add significance stars
    if results['significance']['is_significant']:
        p = results['significance']['p_value']
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = ''
        
        if stars:
            y_pos = map_means[2] + map_stds[2] + 0.02
            ax1.text(2, y_pos, stars, ha='center', fontsize=20, fontweight='bold')
    
    # 2. Per-class breakdown (horizontal bars)
    ax2 = fig.add_subplot(gs[0, 2])
    
    classes = common_structures
    fusion_maps = [per_class_results['fusion'][c]['mean_map'] for c in classes]
    fusion_stds = [per_class_results['fusion'][c]['std_map'] for c in classes]
    
    # Sort by performance
    sorted_indices = np.argsort(fusion_maps)
    classes_sorted = [classes[i] for i in sorted_indices]
    maps_sorted = [fusion_maps[i] for i in sorted_indices]
    stds_sorted = [fusion_stds[i] for i in sorted_indices]
    
    y_pos = np.arange(len(classes_sorted))
    bars = ax2.barh(y_pos, maps_sorted, xerr=stds_sorted, alpha=0.7, capsize=3, color='#2ca02c')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([format_class_name(c) for c in classes_sorted], fontsize=9)
    ax2.set_xlabel('MAP', fontsize=11, fontweight='bold')
    ax2.set_title('Per-Class Performance\n(Fusion)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim([0, 1])
    
    # 3. Improvement comparison (TomoTwin vs Fusion)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Calculate improvements per class
    improvements = []
    tt_maps = []
    for class_name in classes:
        tt_map = per_class_results['tomotwin'][class_name]['mean_map']
        fusion_map = per_class_results['fusion'][class_name]['mean_map']
        tt_maps.append(tt_map)
        
        improvement = ((fusion_map - tt_map) / tt_map * 100) if tt_map > 0 else 0
        improvements.append(improvement)
    
    # Bar plot comparison
    x = np.arange(len(classes))
    width = 0.35
    
    tt_bars = ax3.bar(x - width/2, tt_maps, width, label='TomoTwin', alpha=0.8, color='#1f77b4')
    fusion_bars = ax3.bar(x + width/2, 
                          [per_class_results['fusion'][c]['mean_map'] for c in classes],
                          width, label='Fusion', alpha=0.8, color='#2ca02c')
    
    ax3.set_xlabel('Particle Class', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Average Precision', fontsize=12, fontweight='bold')
    ax3.set_title('Per-Class Comparison: TomoTwin vs Fusion', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([format_class_name(c) for c in classes], 
                        rotation=45, ha='right', fontsize=10)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1])
    
    # Add improvement percentages above fusion bars
    for i, (improvement, fusion_bar) in enumerate(zip(improvements, fusion_bars)):
        height = fusion_bar.get_height()
        color = 'green' if improvement > 0 else 'red'
        ax3.text(fusion_bar.get_x() + fusion_bar.get_width()/2., height + 0.02,
                f'+{improvement:.0f}%' if improvement > 0 else f'{improvement:.0f}%',
                ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
    
    # Overall title with p-value
    p_val = results['significance']['p_value']
    improvement_pct = results['significance']['mean_improvement'] / results['tomotwin']['mean_map'] * 100
    
    plt.suptitle(
        f'Classification Performance with Feature Fusion ({embedding_dim}D embeddings)\n'
        f'Overall Improvement: {improvement_pct:+.1f}% (p={p_val:.4f}{"***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""})',
        fontsize=16, fontweight='bold'
    )
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=Path, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for results')
    parser.add_argument('--n-folds', type=int, default=10,
                       help='Number of cross-validation folds (default: 10)')
    parser.add_argument('--random-seed', type=int, default=171717,
                       help='Random seed for reproducibility (default: 171717)')
    parser.add_argument('--embedding-dim', type=int, default=32,
                       help='Embedding dimension to use (default: 32)')
    parser.add_argument('--fusion-method', type=str, default='average',
                       choices=['average', 'concat', 'weighted', 'attention'],
                       help='Fusion method (default: average)')
    parser.add_argument('--attention-epochs', type=int, default=20,
                       help='Number of epochs for attention fusion training (default: 20)')
    parser.add_argument('--balance-classes', action='store_true',
                       help='Undersample majority classes for balanced evaluation')
    parser.add_argument('--save-fused-embeddings', type=Path, default=None,
                       help='Path to save final fused embeddings (only for attention fusion)')
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"CLASSIFICATION EVALUATION ({args.embedding_dim}D)")
    print("="*70)
    print(f"Fusion method: {args.fusion_method}")
    print(f"Random seed: {args.random_seed}")
    print(f"Cross-validation folds: {args.n_folds}")
    
    print("\nLoading embeddings...")
    
    # Load CryoLens embeddings (extract structural dimensions only)
    print("  Loading CryoLens embeddings...")
    cl_embeddings, cl_labels, cl_runs, cl_metadata = load_cryolens_embeddings(
        Path(config['cryolens_embeddings']),
        structural_dim=args.embedding_dim
    )
    print(f"    Loaded {len(cl_embeddings)} CryoLens samples ({args.embedding_dim}D)")
    
    # Load TomoTwin embeddings
    print("  Loading TomoTwin embeddings...")
    tt_embeddings, tt_labels, tt_runs = load_tomotwin_embeddings(
        Path(config['tomotwin_embeddings']),
        Path(config['tomotwin_coords']),
        embedding_dim=args.embedding_dim
    )
    print(f"    Loaded {len(tt_embeddings)} TomoTwin samples ({args.embedding_dim}D)")
    
    # Align embeddings
    print("\nAligning embeddings...")
    aligned_cl, aligned_tt, aligned_labels, aligned_metadata, common_structures = align_embeddings(
        cl_embeddings, cl_labels, cl_metadata, tt_embeddings, tt_labels, args.random_seed
    )
    
    print(f"  Aligned {len(aligned_labels)} samples ({args.embedding_dim}D each)")
    print(f"  Classes: {common_structures}")
    
    # Report class distribution
    print(f"\n  Class distribution:")
    class_counts = {}
    for class_name in common_structures:
        count = sum(1 for label in aligned_labels if label == class_name)
        class_counts[class_name] = count
        print(f"    {class_name:20s}: {count:4d} samples ({count/len(aligned_labels)*100:5.1f}%)")
    
    # Check for severe imbalance
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 3:
        print(f"\n  ⚠ WARNING: Class imbalance detected! (ratio: {imbalance_ratio:.1f}x)")
        print(f"    Most common: {max_count} samples")
        print(f"    Least common: {min_count} samples")
        print(f"    This may inflate MAP for majority classes.")
    
    # Balance classes if requested
    if args.balance_classes:
        print(f"\n  Balancing classes by undersampling to {min_count} samples each...")
        
        balanced_tt = []
        balanced_cl = []
        balanced_labels = []
        balanced_metadata = []
        
        np.random.seed(args.random_seed)
        
        for class_name in common_structures:
            # Get indices for this class
            class_indices = [i for i, label in enumerate(aligned_labels) if label == class_name]
            
            # Undersample to min_count
            if len(class_indices) > min_count:
                selected_indices = np.random.choice(class_indices, min_count, replace=False)
            else:
                selected_indices = class_indices
            
            balanced_tt.extend(aligned_tt[selected_indices])
            balanced_cl.extend(aligned_cl[selected_indices])
            balanced_labels.extend([class_name] * len(selected_indices))
            balanced_metadata.extend([aligned_metadata[i] for i in selected_indices])
        
        # Convert to arrays
        aligned_tt = np.array(balanced_tt, dtype=np.float32)
        aligned_cl = np.array(balanced_cl, dtype=np.float32)
        aligned_labels = balanced_labels
        aligned_metadata = balanced_metadata
        
        print(f"  Balanced to {len(aligned_labels)} total samples ({min_count} per class)")
    
    # Evaluate each method
    print(f"\nRunning {args.n_folds}-fold cross-validation...")
    
    results = {}
    
    print(f"  Evaluating TomoTwin ({args.embedding_dim}D)...")
    results['tomotwin'] = stratified_cross_validation(
        aligned_tt, aligned_labels, n_folds=args.n_folds, random_seed=args.random_seed,
        return_predictions=True
    )
    
    print(f"  Evaluating CryoLens ({args.embedding_dim}D structural)...")
    results['cryolens'] = stratified_cross_validation(
        aligned_cl, aligned_labels, n_folds=args.n_folds, random_seed=args.random_seed,
        return_predictions=True
    )
    
    # Handle fusion method
    if args.fusion_method == 'attention':
        # Determine device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        print(f"  Evaluating Attention Fusion (32D, nested CV to prevent leakage)...")
        print(f"    Training fusion inside each fold on {device}...")
        results['fusion'] = stratified_cross_validation_with_attention(
            aligned_tt,
            aligned_cl,
            aligned_labels,
            n_folds=args.n_folds,
            n_epochs=args.attention_epochs,
            random_seed=args.random_seed,
            device=device,
            return_predictions=True
        )
    else:
        # Simple fusion methods
        print(f"  Evaluating Fusion ({args.fusion_method})...")
        fusion_embeddings = create_fusion_embeddings(
            aligned_tt, aligned_cl, fusion_method=args.fusion_method
        )
        print(f"    Fusion dimension: {fusion_embeddings.shape[1]}D")
        
        results['fusion'] = stratified_cross_validation(
            fusion_embeddings, aligned_labels, n_folds=args.n_folds, random_seed=args.random_seed,
            return_predictions=True
        )
    
    # Statistical significance
    print("\nComputing statistical significance...")
    results['significance'] = compute_statistical_significance(
        results['tomotwin']['map_per_fold'],
        results['fusion']['map_per_fold']
    )
    
    # Per-class metrics (fast - reuses predictions from CV)
    print("\nComputing per-class metrics (from saved predictions)...")
    
    per_class_results = {
        'tomotwin': compute_per_class_metrics_from_predictions(
            results['tomotwin']['predictions'], common_structures
        ),
        'cryolens': compute_per_class_metrics_from_predictions(
            results['cryolens']['predictions'], common_structures
        ),
        'fusion': compute_per_class_metrics_from_predictions(
            results['fusion']['predictions'], common_structures
        )
    }
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nOverall Performance ({args.n_folds}-fold CV, {args.embedding_dim}D):")
    print(f"  TomoTwin MAP:  {results['tomotwin']['mean_map']:.3f} ± {results['tomotwin']['std_map']:.3f}")
    print(f"  CryoLens MAP:  {results['cryolens']['mean_map']:.3f} ± {results['cryolens']['std_map']:.3f}")
    print(f"  Fusion MAP:    {results['fusion']['mean_map']:.3f} ± {results['fusion']['std_map']:.3f}")
    
    improvement_pct = results['significance']['mean_improvement'] / results['tomotwin']['mean_map'] * 100
    print(f"\nImprovement:     {improvement_pct:+.1f}%")
    print(f"p-value:         {results['significance']['p_value']:.4f}")
    print(f"Significant:     {results['significance']['is_significant']} (α=0.05)")
    
    print(f"\nPer-Class Results (Fusion):")
    for class_name in common_structures:
        metrics = per_class_results['fusion'][class_name]
        print(f"  {class_name:20s}: MAP={metrics['mean_map']:.3f}±{metrics['std_map']:.3f}")
    
    # Save results
    output_json = args.output_dir / 'classification_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if key == 'significance':
            results_serializable[key] = value
        else:
            # Skip predictions (not needed in saved results)
            results_serializable[key] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in value.items()
                if k not in ['predictions', 'label_encoder']
            }
    
    with open(output_json, 'w') as f:
        json.dump({
            'overall': results_serializable,
            'per_class': per_class_results,
            'config': config,
            'common_structures': common_structures,
            'n_samples': len(aligned_labels),
            'n_folds': args.n_folds,
            'random_seed': args.random_seed,
            'embedding_dim': args.embedding_dim,
            'fusion_method': args.fusion_method
        }, f, indent=2)
    
    print(f"\nSaved results to {output_json}")
    
    # Create figure
    print("\nGenerating figure...")
    figure_path = args.output_dir / 'classification_performance.png'
    create_classification_figure(
        results,
        per_class_results,
        common_structures,
        figure_path,
        embedding_dim=args.embedding_dim
    )
    
    # Save fused embeddings if requested (only for attention fusion)
    if args.save_fused_embeddings and args.fusion_method == 'attention':
        print("\n" + "="*70)
        print("TRAINING FINAL FUSION MODEL AND SAVING EMBEDDINGS")
        print("="*70)
        
        # Determine device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        # Train final model and save embeddings
        fused_embeddings, metadata = train_final_fusion_and_save(
            aligned_tt,
            aligned_cl,
            aligned_labels,
            args.save_fused_embeddings,
            sample_metadata=aligned_metadata,
            n_epochs=args.attention_epochs,
            random_seed=args.random_seed,
            device=device,
            verbose=True
        )
        
        print(f"\nFused embeddings metadata:")
        print(f"  Output path: {metadata['output_path']}")
        print(f"  Samples: {metadata['n_samples']}")
        print(f"  Embedding dimension: {metadata['embedding_dim']}D")
        print(f"  Classes: {metadata['class_names']}")
    elif args.save_fused_embeddings and args.fusion_method != 'attention':
        print(f"\n⚠ WARNING: --save-fused-embeddings only works with attention fusion")
        print(f"  Current fusion method: {args.fusion_method}")
        print(f"  Skipping fused embeddings save.")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
