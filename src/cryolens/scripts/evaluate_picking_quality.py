"""
Evaluate particle picking quality assessment using reconstruction distances.

This script demonstrates CryoLens's ability to identify contaminated particle
sets by measuring distances between individual reconstructions and averaged
reconstructions. Contamination is detected through MSE and Missing Wedge Loss
distance metrics.

Features:
- Caches reconstructions for reuse across contamination scenarios
- Generates plots progressively as scenarios complete
- Supports resuming from partial runs

Usage:
    python -m cryolens.scripts.evaluate_picking_quality \
        --checkpoint models/cryolens_epoch_2600.pt \
        --copick-config ml_challenge_experimental.json \
        --output-dir results/picking_quality/ \
        --structure-pairs ribosome,thyroglobulin beta-galactoside,virus-like-particle
"""

import argparse
import json
import numpy as np
import torch
import h5py
import mrcfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from cryolens.utils.checkpoint_loading import load_vae_model
from cryolens.inference.pipeline import InferencePipeline
from cryolens.data import CopickDataLoader
from cryolens.evaluation.ood_reconstruction import (
    generate_resampled_reconstructions,
    align_volume,
    normalize_volume_zscore
)
from cryolens.training.losses import MissingWedgeLoss, NormalizedMSELoss
from cryolens.evaluation.fsc import apply_soft_mask


# Contamination ratios to test
CONTAMINATION_RATIOS = [
    (100, 0),   # Pure X
    (99, 1),    # 1% contamination
    (90, 10),   # 10% contamination
    (50, 50),   # 50/50 mix
    (10, 90),   # 90% contamination
    (1, 99),    # 99% contamination
    (0, 100),   # Pure Y
]


def load_or_generate_reconstructions(
    particles: List[np.ndarray],
    structure_name: str,
    cache_dir: Path,
    model: torch.nn.Module,
    pipeline: InferencePipeline,
    device: torch.device
) -> List[np.ndarray]:
    """
    Load reconstructions from cache or generate and cache them.
    
    Parameters
    ----------
    particles : List[np.ndarray]
        Input particles
    structure_name : str
        Name of structure (for cache file naming)
    cache_dir : Path
        Directory for cache files
    model : torch.nn.Module
        Trained model
    pipeline : InferencePipeline
        Inference pipeline
    device : torch.device
        Computation device
        
    Returns
    -------
    List[np.ndarray]
        Aligned reconstructions
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{structure_name}_reconstructions.h5"
    
    # Try to load from cache
    if cache_file.exists():
        print(f"    Loading cached reconstructions from {cache_file}")
        try:
            reconstructions = []
            with h5py.File(cache_file, 'r') as f:
                n_particles = len([k for k in f.keys() if k.startswith('recon_')])
                for i in range(n_particles):
                    reconstructions.append(f[f'recon_{i:03d}'][:])
            
            if len(reconstructions) == len(particles):
                print(f"    Successfully loaded {len(reconstructions)} cached reconstructions")
                return reconstructions
            else:
                print(f"    Cache has {len(reconstructions)} reconstructions but need {len(particles)}, regenerating...")
        except Exception as e:
            print(f"    Error loading cache: {e}, regenerating...")
    
    # Generate reconstructions
    print(f"    Generating and caching reconstructions...")
    all_reconstructions = []
    reference_reconstruction = None
    
    for idx, particle in enumerate(tqdm(particles, desc="    Processing", leave=False)):
        # Generate reconstruction (single sample for deterministic results)
        recons = generate_resampled_reconstructions(
            particle, model, pipeline, device,
            n_samples=1,
            noise_level=0.0,
            target_shape=(48, 48, 48)
        )
        
        # Normalize
        recon_norm = normalize_volume_zscore(recons[0])
        
        # Set reference from first particle
        if idx == 0:
            reference_reconstruction = recon_norm
            aligned_recon = recon_norm
        else:
            # Align to reference
            aligned_recon, _ = align_volume(
                reference_reconstruction, recon_norm,
                n_angles=24, refine=True
            )
        
        all_reconstructions.append(aligned_recon)
    
    # Save to cache
    print(f"    Saving {len(all_reconstructions)} reconstructions to cache...")
    with h5py.File(cache_file, 'w') as f:
        f.attrs['structure_name'] = structure_name
        f.attrs['n_particles'] = len(all_reconstructions)
        for i, recon in enumerate(all_reconstructions):
            f.create_dataset(f'recon_{i:03d}', data=recon, compression='gzip')
    
    return all_reconstructions


def sample_particles_from_runs(
    copick_loader: CopickDataLoader,
    structure_name: str,
    n_particles: int,
    voxel_size: float = 10.0,
    box_size: int = 48,
    random_seed: int = 42
) -> List[np.ndarray]:
    """
    Sample particles randomly from multiple runs until reaching target count.
    
    Parameters
    ----------
    copick_loader : CopickDataLoader
        Copick data loader
    structure_name : str
        Name of structure to sample
    n_particles : int
        Target number of particles
    voxel_size : float
        Voxel size in Angstroms
    box_size : int
        Box size for particles
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    List[np.ndarray]
        List of sampled particles
    """
    np.random.seed(random_seed)
    
    # Load more particles than needed to allow random selection
    data = copick_loader.load_particles(
        structure_filter=[structure_name],
        max_particles_per_structure=n_particles + 50,  # Extra buffer
        target_voxel_spacing=voxel_size,
        box_size=box_size,
        normalize=False,
        verbose=False
    )
    
    if structure_name not in data or len(data[structure_name]['particles']) == 0:
        raise ValueError(f"No particles found for {structure_name}")
    
    all_particles = data[structure_name]['particles']
    
    # Randomly sample exactly n_particles
    if len(all_particles) < n_particles:
        raise ValueError(
            f"Not enough particles for {structure_name}: "
            f"found {len(all_particles)}, need {n_particles}"
        )
    
    indices = np.random.choice(len(all_particles), n_particles, replace=False)
    sampled_particles = [all_particles[i] for i in indices]
    
    # Apply masking
    sampled_particles = [
        apply_soft_mask(p, radius=22, soft_edge=5)
        for p in sampled_particles
    ]
    
    return sampled_particles


def compute_reconstruction_distances(
    reconstructions: List[np.ndarray],
    average_reconstruction: np.ndarray,
    mse_loss: NormalizedMSELoss,
    mwl_loss: MissingWedgeLoss,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MSE and Missing Wedge Loss distances from each reconstruction to average.
    
    Parameters
    ----------
    reconstructions : List[np.ndarray]
        Individual particle reconstructions
    average_reconstruction : np.ndarray
        Averaged reconstruction
    mse_loss : NormalizedMSELoss
        MSE loss function
    mwl_loss : MissingWedgeLoss
        Missing wedge loss function
    device : torch.device
        Computation device
        
    Returns
    -------
    mse_distances : np.ndarray
        MSE distances (N,)
    mwl_distances : np.ndarray
        Missing wedge loss distances (N,)
    """
    mse_distances = []
    mwl_distances = []
    
    # Convert average to tensor
    avg_tensor = torch.from_numpy(average_reconstruction).float().unsqueeze(0).unsqueeze(0).to(device)
    
    for recon in reconstructions:
        # Convert to tensor
        recon_tensor = torch.from_numpy(recon).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Compute MSE
        with torch.no_grad():
            mse_dist = mse_loss(recon_tensor, avg_tensor).item()
            mwl_dist = mwl_loss(recon_tensor, avg_tensor).item()
        
        mse_distances.append(mse_dist)
        mwl_distances.append(mwl_dist)
    
    return np.array(mse_distances), np.array(mwl_distances)


def create_scenario_plot(
    result: Dict,
    output_path: Path,
    mse_distances_x: np.ndarray,
    mse_distances_y: np.ndarray,
    mwl_distances_x: np.ndarray,
    mwl_distances_y: np.ndarray
):
    """Create immediate plot for a single contamination scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    structure_x = result['structure_x']
    structure_y = result['structure_y']
    ratio_x = result['ratio_x']
    ratio_y = result['ratio_y']
    
    # MSE violin plot
    ax = axes[0]
    if len(mse_distances_x) > 0 and len(mse_distances_y) > 0:
        parts = ax.violinplot(
            [mse_distances_x, mse_distances_y],
            positions=[0, 1],
            showmeans=True,
            showmedians=True
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([structure_x, structure_y])
        ax.set_ylabel('MSE Distance to Average', fontweight='bold')
        ax.set_title(f'MSE Distances\nCohen\'s d = {result["mse"]["cohens_d"]:.2f}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # MWL violin plot
    ax = axes[1]
    if len(mwl_distances_x) > 0 and len(mwl_distances_y) > 0:
        parts = ax.violinplot(
            [mwl_distances_x, mwl_distances_y],
            positions=[0, 1],
            showmeans=True,
            showmedians=True
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([structure_x, structure_y])
        ax.set_ylabel('Missing Wedge Loss Distance to Average', fontweight='bold')
        ax.set_title(f'MWL Distances\nCohen\'s d = {result["mwl"]["cohens_d"]:.2f}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(
        f'{structure_x} vs {structure_y}: {ratio_x}/{ratio_y} Contamination',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_contamination_scenario(
    reconstructions_x: List[np.ndarray],
    reconstructions_y: List[np.ndarray],
    structure_x: str,
    structure_y: str,
    ratio_x: int,
    ratio_y: int,
    mse_loss: NormalizedMSELoss,
    mwl_loss: MissingWedgeLoss,
    device: torch.device,
    output_dir: Path
) -> Dict:
    """
    Analyze single contamination scenario using cached reconstructions.
    
    Parameters
    ----------
    reconstructions_x : List[np.ndarray]
        Cached reconstructions from structure X
    reconstructions_y : List[np.ndarray]
        Cached reconstructions from structure Y
    structure_x : str
        Name of structure X
    structure_y : str
        Name of structure Y
    ratio_x : int
        Number of X particles
    ratio_y : int
        Number of Y particles
    mse_loss : NormalizedMSELoss
        MSE loss function
    mwl_loss : MissingWedgeLoss
        Missing wedge loss function
    device : torch.device
        Computation device
    output_dir : Path
        Output directory
        
    Returns
    -------
    Dict
        Analysis results
    """
    print(f"  Analyzing {structure_x}:{ratio_x} / {structure_y}:{ratio_y}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already computed
    results_file = output_dir / 'results.json'
    if results_file.exists():
        print(f"    Scenario already analyzed, loading existing results...")
        with open(results_file, 'r') as f:
            return json.load(f)
    
    # Select reconstructions according to ratio
    selected_x = reconstructions_x[:ratio_x] if ratio_x > 0 else []
    selected_y = reconstructions_y[:ratio_y] if ratio_y > 0 else []
    
    all_reconstructions = selected_x + selected_y
    
    # Compute average reconstruction
    print(f"    Computing average reconstruction...")
    average_reconstruction = np.mean(all_reconstructions, axis=0)
    
    # Compute distances
    print(f"    Computing distances...")
    mse_distances, mwl_distances = compute_reconstruction_distances(
        all_reconstructions,
        average_reconstruction,
        mse_loss,
        mwl_loss,
        device
    )
    
    # Split distances by class
    mse_x = mse_distances[:ratio_x] if ratio_x > 0 else np.array([])
    mse_y = mse_distances[ratio_x:] if ratio_y > 0 else np.array([])
    mwl_x = mwl_distances[:ratio_x] if ratio_x > 0 else np.array([])
    mwl_y = mwl_distances[ratio_x:] if ratio_y > 0 else np.array([])
    
    # Compute statistics
    results = {
        'structure_x': structure_x,
        'structure_y': structure_y,
        'ratio_x': ratio_x,
        'ratio_y': ratio_y,
        'mse': {
            'x_mean': float(np.mean(mse_x)) if len(mse_x) > 0 else None,
            'x_std': float(np.std(mse_x)) if len(mse_x) > 0 else None,
            'y_mean': float(np.mean(mse_y)) if len(mse_y) > 0 else None,
            'y_std': float(np.std(mse_y)) if len(mse_y) > 0 else None,
            'cohens_d': float(
                (np.mean(mse_x) - np.mean(mse_y)) /
                np.sqrt((np.std(mse_x)**2 + np.std(mse_y)**2) / 2)
            ) if len(mse_x) > 0 and len(mse_y) > 0 else None,
        },
        'mwl': {
            'x_mean': float(np.mean(mwl_x)) if len(mwl_x) > 0 else None,
            'x_std': float(np.std(mwl_x)) if len(mwl_x) > 0 else None,
            'y_mean': float(np.mean(mwl_y)) if len(mwl_y) > 0 else None,
            'y_std': float(np.std(mwl_y)) if len(mwl_y) > 0 else None,
            'cohens_d': float(
                (np.mean(mwl_x) - np.mean(mwl_y)) /
                np.sqrt((np.std(mwl_x)**2 + np.std(mwl_y)**2) / 2)
            ) if len(mwl_x) > 0 and len(mwl_y) > 0 else None,
        }
    }
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save distance arrays
    np.save(output_dir / 'mse_distances_x.npy', mse_x)
    np.save(output_dir / 'mse_distances_y.npy', mse_y)
    np.save(output_dir / 'mwl_distances_x.npy', mwl_x)
    np.save(output_dir / 'mwl_distances_y.npy', mwl_y)
    
    # Save average reconstruction
    with mrcfile.new(output_dir / 'average_reconstruction.mrc', overwrite=True) as mrc:
        mrc.set_data(average_reconstruction.astype(np.float32))
    
    # Create immediate plot for this scenario
    print(f"    Creating scenario plot...")
    create_scenario_plot(
        results,
        output_dir / 'scenario_plot.png',
        mse_x, mse_y, mwl_x, mwl_y
    )
    
    print(f"    Results saved to {output_dir}")
    
    return results


def create_pair_summary_plot(
    results_list: List[Dict],
    output_path: Path
):
    """Create summary plot for a single structure pair across all contamination levels."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    structure_x = results_list[0]['structure_x']
    structure_y = results_list[0]['structure_y']
    
    # Prepare data for plotting
    contamination_levels = []
    mse_data_x, mse_data_y = [], []
    mwl_data_x, mwl_data_y = [], []
    
    for result in results_list:
        ratio_x = result['ratio_x']
        ratio_y = result['ratio_y']
        
        if ratio_x == 0 or ratio_y == 0:
            continue  # Skip pure cases for this plot
        
        label = f"{ratio_x}/{ratio_y}"
        contamination_levels.append(label)
        
        mse_data_x.append(result['mse'])
        mse_data_y.append(result['mse'])
        mwl_data_x.append(result['mwl'])
        mwl_data_y.append(result['mwl'])
    
    # Plot MSE distributions
    ax = axes[0]
    x_pos = np.arange(len(contamination_levels))
    width = 0.35
    
    x_means = [d['x_mean'] for d in mse_data_x]
    y_means = [d['y_mean'] for d in mse_data_y]
    x_stds = [d['x_std'] for d in mse_data_x]
    y_stds = [d['y_std'] for d in mse_data_y]
    
    ax.bar(x_pos - width/2, x_means, width, yerr=x_stds,
           label=structure_x, alpha=0.8, capsize=5)
    ax.bar(x_pos + width/2, y_means, width, yerr=y_stds,
           label=structure_y, alpha=0.8, capsize=5)
    
    ax.set_xlabel('Contamination Ratio (X/Y)', fontweight='bold')
    ax.set_ylabel('MSE Distance to Average', fontweight='bold')
    ax.set_title('MSE Distance Distributions', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(contamination_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot MWL distributions
    ax = axes[1]
    x_means = [d['x_mean'] for d in mwl_data_x]
    y_means = [d['y_mean'] for d in mwl_data_y]
    x_stds = [d['x_std'] for d in mwl_data_x]
    y_stds = [d['y_std'] for d in mwl_data_y]
    
    ax.bar(x_pos - width/2, x_means, width, yerr=x_stds,
           label=structure_x, alpha=0.8, capsize=5)
    ax.bar(x_pos + width/2, y_means, width, yerr=y_stds,
           label=structure_y, alpha=0.8, capsize=5)
    
    ax.set_xlabel('Contamination Ratio (X/Y)', fontweight='bold')
    ax.set_ylabel('Missing Wedge Loss Distance to Average', fontweight='bold')
    ax.set_title('Missing Wedge Loss Distance Distributions', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(contamination_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(
        f'Distance Distributions: {structure_x} vs {structure_y}',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_global_summary_plots(
    all_results: Dict[str, List[Dict]],
    output_dir: Path
):
    """Create global summary plots across all structure pairs."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    
    # Separability analysis
    create_separability_figure(all_results, figures_dir / 'separability_analysis.png')
    
    # Contamination heatmap
    create_contamination_heatmap(all_results, figures_dir / 'contamination_heatmap.png')


def create_separability_figure(
    all_results: Dict[str, List[Dict]],
    save_path: Path
):
    """Create figure showing Cohen's d effect sizes for separability."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect data
    pair_names = []
    mse_cohens_d = defaultdict(list)
    mwl_cohens_d = defaultdict(list)
    contamination_labels = []
    
    for pair_name, results_list in all_results.items():
        pair_names.append(pair_name)
        
        for result in results_list:
            ratio_x = result['ratio_x']
            ratio_y = result['ratio_y']
            
            if ratio_x == 0 or ratio_y == 0:
                continue
            
            label = f"{ratio_x}/{ratio_y}"
            if label not in contamination_labels:
                contamination_labels.append(label)
            
            if result['mse']['cohens_d'] is not None:
                mse_cohens_d[label].append(abs(result['mse']['cohens_d']))
            if result['mwl']['cohens_d'] is not None:
                mwl_cohens_d[label].append(abs(result['mwl']['cohens_d']))
    
    # Plot MSE Cohen's d
    ax = axes[0]
    x_pos = np.arange(len(contamination_labels))
    width = 0.8 / len(pair_names)
    
    for i, pair_name in enumerate(pair_names):
        values = [
            np.mean([d for d in mse_cohens_d[label]]) if label in mse_cohens_d else 0
            for label in contamination_labels
        ]
        ax.bar(x_pos + i * width, values, width, label=pair_name, alpha=0.8)
    
    ax.set_xlabel('Contamination Ratio', fontweight='bold')
    ax.set_ylabel("Cohen's d (Effect Size)", fontweight='bold')
    ax.set_title('MSE Separability', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos + width * (len(pair_names) - 1) / 2)
    ax.set_xticklabels(contamination_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Large effect')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium effect')
    
    # Plot MWL Cohen's d
    ax = axes[1]
    for i, pair_name in enumerate(pair_names):
        values = [
            np.mean([d for d in mwl_cohens_d[label]]) if label in mwl_cohens_d else 0
            for label in contamination_labels
        ]
        ax.bar(x_pos + i * width, values, width, label=pair_name, alpha=0.8)
    
    ax.set_xlabel('Contamination Ratio', fontweight='bold')
    ax.set_ylabel("Cohen's d (Effect Size)", fontweight='bold')
    ax.set_title('Missing Wedge Loss Separability', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos + width * (len(pair_names) - 1) / 2)
    ax.set_xticklabels(contamination_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.suptitle(
        'Class Separability Analysis (Higher = Better Discrimination)',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_contamination_heatmap(
    all_results: Dict[str, List[Dict]],
    save_path: Path
):
    """Create heatmap showing effect sizes across pairs and contamination levels."""
    # Prepare data for heatmap
    contamination_labels = ['99/1', '90/10', '50/50', '10/90', '1/99']
    pair_names = list(all_results.keys())
    
    mse_matrix = np.zeros((len(pair_names), len(contamination_labels)))
    mwl_matrix = np.zeros((len(pair_names), len(contamination_labels)))
    
    for i, pair_name in enumerate(pair_names):
        results_list = all_results[pair_name]
        for result in results_list:
            ratio_x = result['ratio_x']
            ratio_y = result['ratio_y']
            label = f"{ratio_x}/{ratio_y}"
            
            if label in contamination_labels:
                j = contamination_labels.index(label)
                if result['mse']['cohens_d'] is not None:
                    mse_matrix[i, j] = abs(result['mse']['cohens_d'])
                if result['mwl']['cohens_d'] is not None:
                    mwl_matrix[i, j] = abs(result['mwl']['cohens_d'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MSE heatmap
    ax = axes[0]
    im = ax.imshow(mse_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(contamination_labels)))
    ax.set_yticks(np.arange(len(pair_names)))
    ax.set_xticklabels(contamination_labels)
    ax.set_yticklabels([p.replace('_', '\n') for p in pair_names])
    ax.set_xlabel('Contamination Ratio', fontweight='bold')
    ax.set_ylabel('Structure Pair', fontweight='bold')
    ax.set_title('MSE Effect Sizes', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(pair_names)):
        for j in range(len(contamination_labels)):
            text = ax.text(j, i, f'{mse_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label="Cohen's d")
    
    # MWL heatmap
    ax = axes[1]
    im = ax.imshow(mwl_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(contamination_labels)))
    ax.set_yticks(np.arange(len(pair_names)))
    ax.set_xticklabels(contamination_labels)
    ax.set_yticklabels([p.replace('_', '\n') for p in pair_names])
    ax.set_xlabel('Contamination Ratio', fontweight='bold')
    ax.set_ylabel('Structure Pair', fontweight='bold')
    ax.set_title('Missing Wedge Loss Effect Sizes', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(pair_names)):
        for j in range(len(contamination_labels)):
            text = ax.text(j, i, f'{mwl_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label="Cohen's d")
    
    plt.suptitle(
        'Contamination Detection Performance',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--copick-config',
        type=str,
        required=True,
        help='Path to Copick configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--structure-pairs',
        nargs='+',
        required=True,
        help='Structure pairs to test (format: structure1,structure2)'
    )
    parser.add_argument(
        '--n-particles',
        type=int,
        default=100,
        help='Number of particles per structure (default: 100)'
    )
    parser.add_argument(
        '--voxel-size',
        type=float,
        default=10.0,
        help='Voxel size in Angstroms (default: 10.0)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("PARTICLE PICKING QUALITY ASSESSMENT")
    print("="*70)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Copick config:  {args.copick_config}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Particles:      {args.n_particles} per structure")
    print(f"Voxel size:     {args.voxel_size}Å")
    print(f"Random seed:    {args.random_seed}")
    print("="*70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading model...")
    model, config = load_vae_model(
        args.checkpoint,
        device=device,
        load_config=True,
        strict_loading=False
    )
    model.eval()
    print("  Model loaded successfully")
    
    # Create pipeline
    pipeline = InferencePipeline(
        model=model,
        device=device,
        normalization_method=config.get('normalization', 'z-score')
    )
    
    # Initialize Copick loader
    print(f"\nInitializing Copick data loader...")
    copick_loader = CopickDataLoader(args.copick_config)
    print("  Copick loader initialized")
    
    # Initialize loss functions
    mse_loss = NormalizedMSELoss(volume_size=48).to(device)
    mwl_loss = MissingWedgeLoss(volume_size=48, wedge_angle=90.0, weight_factor=0.3).to(device)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / 'reconstruction_cache'
    
    # Parse structure pairs
    structure_pairs = []
    for pair_str in args.structure_pairs:
        s1, s2 = pair_str.split(',')
        structure_pairs.append((s1.strip(), s2.strip()))
    
    print(f"\nStructure pairs to evaluate: {len(structure_pairs)}")
    for s1, s2 in structure_pairs:
        print(f"  {s1} vs {s2}")
    
    # Process each structure pair
    all_results = {}
    
    for structure_x, structure_y in structure_pairs:
        pair_name = f"{structure_x}_{structure_y}"
        print(f"\n{'='*70}")
        print(f"Processing pair: {structure_x} vs {structure_y}")
        print(f"{'='*70}")
        
        # Sample particles
        print(f"Sampling {args.n_particles} particles per structure...")
        try:
            particles_x = sample_particles_from_runs(
                copick_loader, structure_x, args.n_particles,
                args.voxel_size, random_seed=args.random_seed
            )
            print(f"  Loaded {len(particles_x)} particles for {structure_x}")
            
            particles_y = sample_particles_from_runs(
                copick_loader, structure_y, args.n_particles,
                args.voxel_size, random_seed=args.random_seed + 1  # Different seed
            )
            print(f"  Loaded {len(particles_y)} particles for {structure_y}")
        except Exception as e:
            print(f"  Error loading particles: {e}")
            continue
        
        # Generate or load cached reconstructions
        print(f"Loading or generating reconstructions...")
        reconstructions_x = load_or_generate_reconstructions(
            particles_x, structure_x, cache_dir,
            model, pipeline, device
        )
        reconstructions_y = load_or_generate_reconstructions(
            particles_y, structure_y, cache_dir,
            model, pipeline, device
        )
        
        # Analyze each contamination ratio
        pair_results = []
        for ratio_x, ratio_y in CONTAMINATION_RATIOS:
            scenario_dir = args.output_dir / pair_name / f"contamination_{ratio_x}_{ratio_y}"
            
            result = analyze_contamination_scenario(
                reconstructions_x, reconstructions_y,
                structure_x, structure_y,
                ratio_x, ratio_y,
                mse_loss, mwl_loss,
                device,
                scenario_dir
            )
            
            pair_results.append(result)
        
        all_results[pair_name] = pair_results
        
        # Create pair summary plot immediately
        print(f"Creating pair summary plot...")
        pair_summary_path = args.output_dir / pair_name / f'{pair_name}_summary.png'
        create_pair_summary_plot(pair_results, pair_summary_path)
        
        # Update global summary plots progressively
        print(f"Updating global summary plots...")
        create_global_summary_plots(all_results, args.output_dir)
    
    # Save overall summary
    summary_path = args.output_dir / 'contamination_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSaved summary to {summary_path}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output_dir}")
    print(f"Figures saved to: {args.output_dir / 'figures'}")
    print("="*70)


if __name__ == '__main__':
    main()
