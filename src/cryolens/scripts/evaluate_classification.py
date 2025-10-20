"""
Evaluate classification performance with statistical validation.

This script evaluates classification using CryoLens and TomoTwin embeddings with
proper cross-validation and statistical testing.

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
    compute_per_class_metrics,
    compute_statistical_significance
)


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def normalize_protein_name(name: str) -> str:
    """Normalize protein names to consistent format."""
    if pd.isna(name):
        return 'unknown'
    return str(name).lower().replace('-', '_').replace(' ', '_')


def load_cryolens_embeddings(h5_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load CryoLens embeddings from H5 file.
    
    Args:
        h5_path: Path to HDF5 file with embeddings
        
    Returns:
        Tuple of (embeddings array, list of labels)
    """
    embeddings_list = []
    labels_list = []
    
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
            
            embeddings_list.append(embedding)
            
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
            
            labels_list.append(normalize_protein_name(structure_name))
    
    return np.array(embeddings_list, dtype=np.float32), labels_list


def load_tomotwin_embeddings(
    parquet_path: Path,
    coords_path: Path
) -> Tuple[np.ndarray, List[str]]:
    """
    Load TomoTwin embeddings from parquet and coordinates CSV.
    
    Args:
        parquet_path: Path to TomoTwin embeddings parquet file
        coords_path: Path to coordinates CSV file
        
    Returns:
        Tuple of (embeddings array, list of labels)
    """
    embeddings_df = pd.read_parquet(parquet_path)
    coords_df = pd.read_csv(coords_path)
    
    # Extract embedding columns (numeric columns excluding filepath)
    embedding_cols = [
        col for col in embeddings_df.columns 
        if col != 'filepath' and embeddings_df[col].dtype in ['float64', 'float32', 'int64', 'int32']
    ]
    
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
    
    return embeddings, labels


def align_embeddings(
    cl_embeddings: np.ndarray,
    cl_labels: List[str],
    tt_embeddings: np.ndarray,
    tt_labels: List[str],
    random_seed: int = 171717
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Align CryoLens and TomoTwin embeddings by matching structure labels.
    
    Balances samples per class to ensure fair comparison.
    
    Args:
        cl_embeddings: CryoLens embeddings
        cl_labels: CryoLens labels
        tt_embeddings: TomoTwin embeddings
        tt_labels: TomoTwin labels
        random_seed: Random seed for sampling
        
    Returns:
        Tuple of (aligned CryoLens embeddings, aligned TomoTwin embeddings,
                 aligned labels, list of common structures)
    """
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
    
    return (
        np.array(aligned_cl, dtype=np.float32),
        np.array(aligned_tt, dtype=np.float32),
        aligned_labels,
        common_structures
    )


def create_classification_figure(
    results: Dict,
    per_class_results: Dict,
    common_structures: List[str],
    output_path: Path
):
    """
    Create comprehensive classification figure with error bars.
    
    Args:
        results: Overall results dictionary
        per_class_results: Per-class results dictionary
        common_structures: List of evaluated structure names
        output_path: Path to save figure
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
    ax1.set_title('Overall Classification Performance (10-fold CV)', fontsize=14, fontweight='bold')
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
    ax2.set_yticklabels([c.replace('_', ' ').title() for c in classes_sorted], fontsize=9)
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
    ax3.set_xticklabels([c.replace('_', ' ').title() for c in classes], 
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
        f'Classification Performance with Feature Fusion\n'
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
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading embeddings...")
    
    # Load CryoLens embeddings
    print("  Loading CryoLens embeddings...")
    cl_embeddings, cl_labels = load_cryolens_embeddings(
        Path(config['cryolens_embeddings'])
    )
    print(f"    Loaded {len(cl_embeddings)} CryoLens embeddings")
    
    # Load TomoTwin embeddings
    print("  Loading TomoTwin embeddings...")
    tt_embeddings, tt_labels = load_tomotwin_embeddings(
        Path(config['tomotwin_embeddings']),
        Path(config['tomotwin_coords'])
    )
    print(f"    Loaded {len(tt_embeddings)} TomoTwin embeddings")
    
    # Align embeddings
    print("\nAligning embeddings...")
    aligned_cl, aligned_tt, aligned_labels, common_structures = align_embeddings(
        cl_embeddings, cl_labels, tt_embeddings, tt_labels, args.random_seed
    )
    
    print(f"  Aligned {len(aligned_labels)} samples")
    print(f"  Classes: {common_structures}")
    
    # Create fusion embeddings
    print("\nCreating fusion embeddings...")
    fusion_embeddings = np.concatenate([aligned_tt, aligned_cl], axis=1)
    print(f"  Fusion dimension: {fusion_embeddings.shape[1]}")
    
    # Evaluate each method
    print(f"\nRunning {args.n_folds}-fold cross-validation...")
    
    results = {}
    
    print("  Evaluating TomoTwin...")
    results['tomotwin'] = stratified_cross_validation(
        aligned_tt, aligned_labels, n_folds=args.n_folds, random_seed=args.random_seed
    )
    
    print("  Evaluating CryoLens...")
    results['cryolens'] = stratified_cross_validation(
        aligned_cl, aligned_labels, n_folds=args.n_folds, random_seed=args.random_seed
    )
    
    print("  Evaluating Fusion...")
    results['fusion'] = stratified_cross_validation(
        fusion_embeddings, aligned_labels, n_folds=args.n_folds, random_seed=args.random_seed
    )
    
    # Statistical significance
    print("\nComputing statistical significance...")
    results['significance'] = compute_statistical_significance(
        results['tomotwin']['map_per_fold'],
        results['fusion']['map_per_fold']
    )
    
    # Per-class metrics
    print("\nComputing per-class metrics...")
    
    per_class_results = {
        'tomotwin': compute_per_class_metrics(
            aligned_tt, aligned_labels, common_structures, args.n_folds, args.random_seed
        ),
        'cryolens': compute_per_class_metrics(
            aligned_cl, aligned_labels, common_structures, args.n_folds, args.random_seed
        ),
        'fusion': compute_per_class_metrics(
            fusion_embeddings, aligned_labels, common_structures, args.n_folds, args.random_seed
        )
    }
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nOverall Performance ({args.n_folds}-fold CV):")
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
            results_serializable[key] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in value.items()
            }
    
    with open(output_json, 'w') as f:
        json.dump({
            'overall': results_serializable,
            'per_class': per_class_results,
            'config': config,
            'common_structures': common_structures,
            'n_samples': len(aligned_labels),
            'n_folds': args.n_folds,
            'random_seed': args.random_seed
        }, f, indent=2)
    
    print(f"\nSaved results to {output_json}")
    
    # Create figure
    print("\nGenerating figure...")
    figure_path = args.output_dir / 'classification_performance.png'
    create_classification_figure(
        results,
        per_class_results,
        common_structures,
        figure_path
    )
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
