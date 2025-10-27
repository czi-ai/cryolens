#!/usr/bin/env python3
"""
Evaluate pairwise alignment with comprehensive visual debugging.

This script evaluates alignment quality between reconstructions and provides:
- 10 sample visualizations with orthoviews (input, reconstruction, aligned)
- Alignment scores for each particle
- Ability to specify template particle for alignment
- Comprehensive metrics and visualizations
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mrcfile
import numpy as np
import torch
import zarr
from scipy import ndimage
from scipy.ndimage import zoom
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Import CryoLens components
try:
    from cryolens.utils.checkpoint_loading import load_vae_model
    from cryolens.inference.pipeline import InferencePipeline
    from cryolens.evaluation.fsc import apply_soft_mask
    from cryolens.utils.normalization import denormalize_volume
    from cryolens.splats.alignment_methods import align_volumes, apply_rotation_to_volume
except ImportError as e:
    print(f"Error importing CryoLens: {e}")
    print("Make sure CryoLens is installed and accessible")
    sys.exit(1)


def load_sampled_data(input_dir: Path) -> Tuple[zarr.Array, Dict]:
    """Load uniformly sampled particles and metadata."""
    print(f"Loading sampled data from {input_dir}...")
    
    zarr_path = input_dir / "subvolumes.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Subvolumes not found at {zarr_path}")
    
    subvolumes = zarr.open(str(zarr_path), mode='r')
    
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"  Loaded {len(metadata['samples'])} samples")
    print(f"  Subvolume shape: {subvolumes.shape}")
    
    return subvolumes, metadata


def load_mrc_structure(mrc_path: Path, box_size: int = 48) -> np.ndarray:
    """Load and resize MRC structure file."""
    print(f"Loading reference structure from {mrc_path}...")
    
    with mrcfile.open(mrc_path, mode='r') as mrc:
        data = mrc.data.copy()
    
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    
    if data.shape != (box_size, box_size, box_size):
        zoom_factors = [box_size / s for s in data.shape]
        data = zoom(data, zoom_factors, order=3)
    
    print(f"  Loaded structure with shape {data.shape}")
    
    return data.astype(np.float32)


def normalize_volume_zscore(volume: np.ndarray) -> np.ndarray:
    """Z-score normalization"""
    mean_val = np.mean(volume)
    std_val = np.std(volume)
    if std_val > 0:
        return (volume - mean_val) / std_val
    return volume - mean_val


def center_crop_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center crop volume to target shape."""
    if volume.shape == target_shape:
        return volume
    
    crop_slices = []
    for current, target in zip(volume.shape, target_shape):
        if current > target:
            start = (current - target) // 2
            crop_slices.append(slice(start, start + target))
        else:
            crop_slices.append(slice(None))
    
    return volume[tuple(crop_slices)]


def generate_reconstruction(
    particle: np.ndarray,
    model: torch.nn.Module,
    pipeline: InferencePipeline,
    device: torch.device,
    target_shape: Tuple[int, int, int] = (48, 48, 48)
) -> np.ndarray:
    """Generate a single reconstruction from a particle."""
    decoder = model.decoder
    
    results = pipeline.process_volume(
        particle,
        return_embeddings=True,
        return_reconstruction=False
    )
    
    mu = results['embeddings']
    
    mu_tensor = torch.tensor(mu, dtype=torch.float32).unsqueeze(0).to(device)
    pose = torch.tensor(
        results['pose'] if results['pose'] is not None else np.array([1.0, 0.0, 0.0, 0.0]),
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    global_weight = torch.tensor(
        results['global_weight'] if results['global_weight'] is not None else np.array([1.0]),
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstruction = decoder(mu_tensor, pose, global_weight, use_final_convolution=True)
    
    reconstruction_np = reconstruction.cpu().numpy()[0, 0]
    reconstruction_np = center_crop_volume(reconstruction_np, target_shape)
    reconstruction_np = denormalize_volume(reconstruction_np, results['normalization_stats'])
    
    # CRITICAL: Negate reconstruction to match expected contrast
    reconstruction_np = -reconstruction_np
    
    return reconstruction_np


def align_to_template(
    reconstructions: List[np.ndarray],
    template_idx: int,
    n_angles: int = 12,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Align all reconstructions to a specified template reconstruction.
    
    Args:
        reconstructions: List of reconstruction volumes
        template_idx: Index of reconstruction to use as template
        n_angles: Angular sampling for alignment
        verbose: Print progress
    
    Returns:
        aligned_rotations: Rotation matrices (N, 3, 3)
        alignment_scores: Alignment quality scores (N,)
        aligned_volumes: List of aligned reconstructions
    """
    if verbose:
        print(f"Aligning {len(reconstructions)} reconstructions to template (particle {template_idx})...")
    
    template = reconstructions[template_idx]
    template_norm = normalize_volume_zscore(template)
    
    aligned_rotations = []
    alignment_scores = []
    aligned_volumes = []
    
    for i, recon in enumerate(tqdm(reconstructions, desc="Aligning", disable=not verbose)):
        if i == template_idx:
            # Template aligns to itself with identity
            aligned_rotations.append(np.eye(3))
            alignment_scores.append(1.0)
            aligned_volumes.append(recon.copy())
        else:
            recon_norm = normalize_volume_zscore(recon)
            
            aligned, score, rotation = align_volumes(
                template_norm,
                recon_norm,
                method='cross_correlation',
                angular_step=360.0 / n_angles,
                n_iterations=50
            )
            
            aligned_rotations.append(rotation)
            alignment_scores.append(score)
            aligned_volumes.append(aligned)
    
    return np.array(aligned_rotations), np.array(alignment_scores), aligned_volumes


def plot_orthoview(volume: np.ndarray, ax_xy: plt.Axes, ax_xz: plt.Axes, ax_yz: plt.Axes, 
                   title: str = "", vmin: float = None, vmax: float = None):
    """Plot orthogonal views of a 3D volume."""
    z_mid, y_mid, x_mid = [s // 2 for s in volume.shape]
    
    # XY plane (Z slice)
    im1 = ax_xy.imshow(volume[z_mid, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    ax_xy.set_title(f'{title}\nXY (Z={z_mid})', fontsize=8)
    ax_xy.axis('off')
    
    # XZ plane (Y slice)
    im2 = ax_xz.imshow(volume[:, y_mid, :], cmap='gray', vmin=vmin, vmax=vmax)
    ax_xz.set_title(f'XZ (Y={y_mid})', fontsize=8)
    ax_xz.axis('off')
    
    # YZ plane (X slice)
    im3 = ax_yz.imshow(volume[:, :, x_mid], cmap='gray', vmin=vmin, vmax=vmax)
    ax_yz.set_title(f'YZ (X={x_mid})', fontsize=8)
    ax_yz.axis('off')
    
    return im1, im2, im3


def create_sample_visualization(
    particles: List[np.ndarray],
    reconstructions: List[np.ndarray],
    aligned_volumes: List[np.ndarray],
    alignment_scores: np.ndarray,
    sample_indices: List[int],
    output_dir: Path,
    dpi: int = 150
):
    """
    Create detailed visualizations for selected samples.
    
    Args:
        particles: List of input particles
        reconstructions: List of reconstructions
        aligned_volumes: List of aligned reconstructions
        alignment_scores: Alignment scores for each particle
        sample_indices: Which particles to visualize
        output_dir: Where to save figures
        dpi: Figure resolution
    """
    print(f"\nCreating visualizations for {len(sample_indices)} samples...")
    
    for idx in tqdm(sample_indices, desc="Creating visualizations"):
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        particle = particles[idx]
        recon = reconstructions[idx]
        aligned = aligned_volumes[idx]
        score = alignment_scores[idx]
        
        # Compute consistent intensity range across all volumes for this particle
        vmin = min(np.min(particle), np.min(recon), np.min(aligned))
        vmax = max(np.max(particle), np.max(recon), np.max(aligned))
        
        # Row 1: Input particle orthoviews
        ax_input_xy = fig.add_subplot(gs[0, 0])
        ax_input_xz = fig.add_subplot(gs[1, 0])
        ax_input_yz = fig.add_subplot(gs[2, 0])
        plot_orthoview(particle, ax_input_xy, ax_input_xz, ax_input_yz, 
                      f"Input Particle {idx}", vmin, vmax)
        
        # Row 2: Reconstruction orthoviews
        ax_recon_xy = fig.add_subplot(gs[0, 1])
        ax_recon_xz = fig.add_subplot(gs[1, 1])
        ax_recon_yz = fig.add_subplot(gs[2, 1])
        plot_orthoview(recon, ax_recon_xy, ax_recon_xz, ax_recon_yz,
                      f"Reconstruction {idx}", vmin, vmax)
        
        # Row 3: Aligned reconstruction orthoviews
        ax_aligned_xy = fig.add_subplot(gs[0, 2])
        ax_aligned_xz = fig.add_subplot(gs[1, 2])
        ax_aligned_yz = fig.add_subplot(gs[2, 2])
        plot_orthoview(aligned, ax_aligned_xy, ax_aligned_xz, ax_aligned_yz,
                      f"Aligned {idx}\nScore: {score:.4f}", vmin, vmax)
        
        # Row 4: Summary statistics
        ax_stats = fig.add_subplot(gs[3, :])
        ax_stats.axis('off')
        
        # Compute correlation between input and aligned
        input_flat = particle.flatten()
        aligned_flat = aligned.flatten()
        corr, _ = pearsonr(input_flat, aligned_flat)
        
        stats_text = (
            f"Particle {idx} Statistics:\n"
            f"{'='*50}\n\n"
            f"Alignment Score:        {score:.4f}\n"
            f"Input-Aligned Corr:     {corr:.4f}\n\n"
            f"Input Range:            [{np.min(particle):.3f}, {np.max(particle):.3f}]\n"
            f"Reconstruction Range:   [{np.min(recon):.3f}, {np.max(recon):.3f}]\n"
            f"Aligned Range:          [{np.min(aligned):.3f}, {np.max(aligned):.3f}]\n\n"
            f"Input Mean/Std:         {np.mean(particle):.3f} ± {np.std(particle):.3f}\n"
            f"Recon Mean/Std:         {np.mean(recon):.3f} ± {np.std(recon):.3f}\n"
            f"Aligned Mean/Std:       {np.mean(aligned):.3f} ± {np.std(aligned):.3f}"
        )
        
        ax_stats.text(0.1, 0.5, stats_text, ha='left', va='center',
                     fontsize=10, family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(f'Particle {idx} - Complete Pipeline Visualization',
                    fontsize=14, fontweight='bold')
        
        output_path = output_dir / f"particle_{idx:03d}_orthoviews.png"
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved visualizations to {output_dir}")


def plot_alignment_scores_distribution(
    alignment_scores: np.ndarray,
    template_idx: int,
    output_path: Path,
    dpi: int = 150
):
    """Plot distribution of alignment scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(alignment_scores, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax1.axvline(np.mean(alignment_scores), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(alignment_scores):.4f}')
    ax1.axvline(np.median(alignment_scores), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(alignment_scores):.4f}')
    ax1.set_xlabel('Alignment Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Alignment Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Scores by particle index
    ax2.scatter(range(len(alignment_scores)), alignment_scores, alpha=0.6, s=50)
    ax2.scatter([template_idx], [alignment_scores[template_idx]], color='red', s=200,
               marker='*', label=f'Template (particle {template_idx})', zorder=10)
    ax2.axhline(np.mean(alignment_scores), color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Particle Index', fontsize=12)
    ax2.set_ylabel('Alignment Score', fontsize=12)
    ax2.set_title('Alignment Scores by Particle', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved alignment score plot to {output_path}")


def compute_pairwise_correlations(
    aligned_volumes: List[np.ndarray],
    verbose: bool = True
) -> np.ndarray:
    """Compute pairwise correlations between all aligned volumes."""
    n = len(aligned_volumes)
    correlations = np.zeros((n, n))
    
    if verbose:
        print(f"\nComputing pairwise correlations for {n} volumes...")
    
    for i in tqdm(range(n), disable=not verbose):
        vol_i_flat = aligned_volumes[i].flatten()
        for j in range(i, n):
            vol_j_flat = aligned_volumes[j].flatten()
            corr, _ = pearsonr(vol_i_flat, vol_j_flat)
            correlations[i, j] = corr
            correlations[j, i] = corr
    
    return correlations


def plot_correlation_matrix(
    correlations: np.ndarray,
    template_idx: int,
    output_path: Path,
    dpi: int = 150
):
    """Plot heatmap of pairwise correlations."""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    im = ax.imshow(correlations, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Highlight template row/column
    ax.axhline(template_idx - 0.5, color='red', linewidth=2, alpha=0.7)
    ax.axhline(template_idx + 0.5, color='red', linewidth=2, alpha=0.7)
    ax.axvline(template_idx - 0.5, color='red', linewidth=2, alpha=0.7)
    ax.axvline(template_idx + 0.5, color='red', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Particle Index', fontsize=12)
    ax.set_ylabel('Particle Index', fontsize=12)
    ax.set_title(f'Pairwise Correlations (Template: Particle {template_idx})',
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='Pearson Correlation')
    
    # Add text annotation for template
    ax.text(template_idx, -1, f'Template\n(P{template_idx})', 
           ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved correlation matrix to {output_path}")


def save_alignment_scores_to_csv(
    alignment_scores: np.ndarray,
    output_path: Path
):
    """Save alignment scores to CSV file."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['particle_index', 'alignment_score'])
        for i, score in enumerate(alignment_scores):
            writer.writerow([i, score])
    
    print(f"  Saved alignment scores to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate alignment quality with comprehensive visual debugging"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with subvolumes.zarr and metadata.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="CryoLens checkpoint path"
    )
    parser.add_argument(
        "--reference-structure",
        type=Path,
        required=True,
        help="Reference structure MRC file (not used for alignment, only for masking)"
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=25,
        help="Number of particles to evaluate"
    )
    parser.add_argument(
        "--template-particle",
        type=int,
        default=1,
        help="Index of particle to use as alignment template (0-indexed, default=1 for 2nd particle)"
    )
    parser.add_argument(
        "--n-angles",
        type=int,
        default=36,
        help="Angular sampling for alignment search"
    )
    parser.add_argument(
        "--n-visualize",
        type=int,
        default=10,
        help="Number of particles to create detailed visualizations for"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("ALIGNMENT EVALUATION WITH VISUAL DEBUGGING")
    print("="*80)
    print(f"Input dir:         {args.input_dir}")
    print(f"Checkpoint:        {args.checkpoint}")
    print(f"Reference:         {args.reference_structure}")
    print(f"N particles:       {args.n_particles}")
    print(f"Template particle: {args.template_particle}")
    print(f"N angles:          {args.n_angles}")
    print(f"N visualizations:  {args.n_visualize}")
    print(f"Output:            {args.output_dir}")
    print("="*80)
    
    # Validate template particle index
    if args.template_particle < 0 or args.template_particle >= args.n_particles:
        print(f"\nERROR: Template particle index {args.template_particle} is out of range [0, {args.n_particles-1}]")
        return 1
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    print("\nLoading CryoLens model...")
    try:
        model, config = load_vae_model(
            args.checkpoint,
            device=device,
            load_config=True,
            strict_loading=False
        )
        model.eval()
        print("  Model loaded successfully")
    except Exception as e:
        print(f"  Error loading model: {e}")
        return 1
    
    normalization = config.get('normalization', 'z-score')
    pipeline = InferencePipeline(
        model=model,
        device=device,
        normalization_method=normalization
    )
    
    try:
        subvolumes, metadata = load_sampled_data(args.input_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    try:
        reference_structure = load_mrc_structure(args.reference_structure)
    except Exception as e:
        print(f"Error loading reference structure: {e}")
        return 1
    
    print("\nApplying soft masks to particles...")
    masked_particles = []
    for i in tqdm(range(args.n_particles), desc="Masking particles"):
        particle = subvolumes[i]
        masked = apply_soft_mask(particle, radius=22, soft_edge=5)
        masked_particles.append(masked)
    
    print("\nGenerating reconstructions...")
    reconstructions = []
    
    for particle in tqdm(masked_particles, desc="Reconstructing"):
        recon = generate_reconstruction(
            particle, model, pipeline, device, target_shape=(48, 48, 48)
        )
        reconstructions.append(recon)
    
    print(f"  Generated {len(reconstructions)} reconstructions")
    
    print(f"\nAligning all reconstructions to particle {args.template_particle}...")
    aligned_rotations, alignment_scores, aligned_volumes = align_to_template(
        reconstructions,
        template_idx=args.template_particle,
        n_angles=args.n_angles,
        verbose=args.verbose
    )
    
    print(f"\n  Aligned {len(aligned_rotations)} reconstructions")
    print(f"  Template particle: {args.template_particle}")
    print(f"  Mean alignment score: {np.mean(alignment_scores):.4f}")
    print(f"  Median alignment score: {np.median(alignment_scores):.4f}")
    print(f"  Score range: [{np.min(alignment_scores):.4f}, {np.max(alignment_scores):.4f}]")
    print(f"  Template score: {alignment_scores[args.template_particle]:.4f}")
    
    # Select particles to visualize (evenly spaced + template + best/worst)
    n_viz = min(args.n_visualize, args.n_particles)
    
    # Always include template
    viz_indices = [args.template_particle]
    
    # Add best and worst alignment scores (excluding template)
    non_template_indices = [i for i in range(args.n_particles) if i != args.template_particle]
    non_template_scores = [(i, alignment_scores[i]) for i in non_template_indices]
    non_template_scores.sort(key=lambda x: x[1])
    
    if len(non_template_scores) > 0:
        viz_indices.append(non_template_scores[-1][0])  # Best
        if len(non_template_scores) > 1:
            viz_indices.append(non_template_scores[0][0])  # Worst
    
    # Add evenly spaced samples
    step = max(1, args.n_particles // (n_viz - len(viz_indices)))
    for i in range(0, args.n_particles, step):
        if i not in viz_indices:
            viz_indices.append(i)
        if len(viz_indices) >= n_viz:
            break
    
    viz_indices = sorted(list(set(viz_indices)))[:n_viz]
    
    print(f"\nCreating detailed visualizations for {len(viz_indices)} particles:")
    print(f"  Indices: {viz_indices}")
    print(f"  (Template: {args.template_particle})")
    
    create_sample_visualization(
        masked_particles,
        reconstructions,
        aligned_volumes,
        alignment_scores,
        viz_indices,
        args.output_dir
    )
    
    print("\nPlotting alignment score distribution...")
    plot_alignment_scores_distribution(
        alignment_scores,
        args.template_particle,
        args.output_dir / "alignment_scores_distribution.png"
    )
    
    print("\nComputing pairwise correlations...")
    correlations = compute_pairwise_correlations(aligned_volumes, verbose=args.verbose)
    
    print(f"  Mean correlation: {np.mean(correlations):.4f}")
    print(f"  Median correlation: {np.median(correlations):.4f}")
    print(f"  Correlation range: [{np.min(correlations):.4f}, {np.max(correlations):.4f}]")
    
    print("\nPlotting correlation matrix...")
    plot_correlation_matrix(
        correlations,
        args.template_particle,
        args.output_dir / "pairwise_correlations.png"
    )
    
    print("\nSaving results to HDF5...")
    h5_path = args.output_dir / "alignment_evaluation.h5"
    
    with h5py.File(h5_path, 'w') as f:
        f.attrs['n_particles'] = args.n_particles
        f.attrs['template_particle'] = args.template_particle
        f.attrs['n_angles'] = args.n_angles
        f.attrs['checkpoint'] = args.checkpoint
        
        f.create_dataset('aligned_rotations', data=aligned_rotations, compression='gzip')
        f.create_dataset('alignment_scores', data=alignment_scores, compression='gzip')
        f.create_dataset('pairwise_correlations', data=correlations, compression='gzip')
        f.create_dataset('reference_structure', data=reference_structure, compression='gzip')
        
        # Save a subset of volumes
        vols_grp = f.create_group('volumes')
        for idx in viz_indices:
            vols_grp.create_dataset(f'particle_{idx:03d}', data=masked_particles[idx], compression='gzip')
            vols_grp.create_dataset(f'reconstruction_{idx:03d}', data=reconstructions[idx], compression='gzip')
            vols_grp.create_dataset(f'aligned_{idx:03d}', data=aligned_volumes[idx], compression='gzip')
    
    print(f"  Saved to {h5_path}")
    
    print("\nSaving alignment scores to CSV...")
    csv_path = args.output_dir / "alignment_scores.csv"
    save_alignment_scores_to_csv(alignment_scores, csv_path)
    
    print("\nSaving metrics to JSON...")
    json_path = args.output_dir / "metrics.json"
    
    json_output = {
        'n_particles': args.n_particles,
        'template_particle': args.template_particle,
        'n_angles': args.n_angles,
        'checkpoint': args.checkpoint,
        'input_dir': str(args.input_dir),
        'reference_structure': str(args.reference_structure),
        'alignment_scores': {
            'mean': float(np.mean(alignment_scores)),
            'median': float(np.median(alignment_scores)),
            'std': float(np.std(alignment_scores)),
            'min': float(np.min(alignment_scores)),
            'max': float(np.max(alignment_scores)),
            'template_score': float(alignment_scores[args.template_particle])
        },
        'pairwise_correlations': {
            'mean': float(np.mean(correlations)),
            'median': float(np.median(correlations)),
            'std': float(np.std(correlations)),
            'min': float(np.min(correlations)),
            'max': float(np.max(correlations))
        },
        'visualized_particles': viz_indices
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"  Saved to {json_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"\nKey outputs:")
    print(f"  - {len(viz_indices)} detailed orthoview visualizations")
    print(f"  - Alignment score distribution plot")
    print(f"  - Pairwise correlation matrix")
    print(f"  - HDF5 file with all data")
    print(f"  - CSV file with alignment scores")
    print(f"  - JSON file with summary metrics")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
