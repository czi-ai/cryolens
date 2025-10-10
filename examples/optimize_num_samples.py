#!/usr/bin/env python3
"""
Optimize num_samples parameter for best resolution in CryoLens affinity reconstruction.

This script sweeps across different num_samples values and tracks FSC resolution
to find the optimal balance between quality and computational cost.

Examples
--------
Basic optimization sweep:
    python optimize_num_samples.py \\
        --particles ./example_data/ribosome_particles.zarr \\
        --ground-truth ./references/ribosome.mrc \\
        --checkpoint-epoch 2600 \\
        --sample-range 1 50 5

Custom sample points:
    python optimize_num_samples.py \\
        --particles ./example_data/ribosome_particles.zarr \\
        --ground-truth ./references/ribosome.mrc \\
        --checkpoint-epoch 2600 \\
        --sample-points 1 5 10 15 20 25 30 40 50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import mrcfile
except ImportError:
    print("Error: mrcfile is not installed. Install with: pip install mrcfile")
    sys.exit(1)

from cryolens.inference.pipeline import create_inference_pipeline
from reconstruct_particles_affinity import (
    reconstruct_with_affinity_output,
    crop_to_size,
    normalize_volume_zscore,
    apply_soft_mask,
    align_to_reference,
    calculate_fsc,
)


def optimize_num_samples(
    pipeline,
    particles: np.ndarray,
    ground_truth: np.ndarray,
    sample_points: List[int],
    batch_size: int = 8,
    box_size: int = 48,
    voxel_spacing: float = 10.0,
    n_alignment_angles: int = 24,
) -> Tuple[List[float], List[float], int]:
    """
    Test different num_samples values and return resolution results.
    
    Parameters
    ----------
    pipeline : InferencePipeline
        CryoLens inference pipeline
    particles : np.ndarray
        Particle volumes (N, D, D, D)
    ground_truth : np.ndarray
        Ground truth structure
    sample_points : List[int]
        List of num_samples values to test
    batch_size : int
        Batch size for inference
    box_size : int
        Target box size
    voxel_spacing : float
        Voxel spacing in Angstroms
    n_alignment_angles : int
        Number of angles for alignment
        
    Returns
    -------
    resolutions : List[float]
        FSC=0.5 resolution for each sample point
    correlations : List[float]
        Average correlation with ground truth for each sample point
    best_num_samples : int
        Optimal num_samples value
    """
    resolutions = []
    correlations = []
    
    # Ensure ground truth matches particle size
    if ground_truth.shape != (box_size, box_size, box_size):
        if all(gt_dim <= box_size for gt_dim in ground_truth.shape):
            # Pad ground truth
            pad_width = []
            for gt_dim in ground_truth.shape:
                pad_before = (box_size - gt_dim) // 2
                pad_after = box_size - gt_dim - pad_before
                pad_width.append((pad_before, pad_after))
            ground_truth = np.pad(ground_truth, pad_width, mode='constant', constant_values=0)
        else:
            ground_truth = crop_to_size(ground_truth, (box_size, box_size, box_size))
    
    gt_normalized = normalize_volume_zscore(ground_truth)
    gt_masked = apply_soft_mask(gt_normalized, radius=22, soft_edge=5)
    
    print(f"\n{'='*80}")
    print(f"Testing {len(sample_points)} different num_samples values")
    print(f"Sample points: {sample_points}")
    print(f"{'='*80}\n")
    
    for num_samples in sample_points:
        print(f"\n{'─'*80}")
        print(f"Testing num_samples = {num_samples}")
        print(f"{'─'*80}")
        
        # Reconstruct with current num_samples
        mean_recons, _ = reconstruct_with_affinity_output(
            pipeline=pipeline,
            particles=particles,
            batch_size=batch_size,
            num_samples=num_samples,
            box_size=box_size,
        )
        
        # Align all reconstructions to ground truth
        print(f"Aligning {len(mean_recons)} reconstructions...")
        aligned_reconstructions = []
        alignment_correlations = []
        
        for recon in tqdm(mean_recons, desc="Aligning"):
            recon_normalized = normalize_volume_zscore(recon)
            recon_masked = apply_soft_mask(recon_normalized, radius=22, soft_edge=5)
            aligned, params = align_to_reference(
                gt_masked, recon_masked, n_alignment_angles, refine=True
            )
            aligned_reconstructions.append(aligned)
            alignment_correlations.append(params['correlation'])
        
        # Average aligned reconstructions
        global_mean = np.mean(aligned_reconstructions, axis=0)
        avg_correlation = np.mean(alignment_correlations)
        
        # Calculate FSC
        _, _, resolution_at_half = calculate_fsc(
            global_mean, ground_truth, voxel_spacing
        )
        
        resolutions.append(resolution_at_half)
        correlations.append(avg_correlation)
        
        print(f"Resolution (FSC=0.5): {resolution_at_half:.2f} Å")
        print(f"Average correlation: {avg_correlation:.4f}")
    
    # Find best num_samples (lowest resolution value = highest resolution)
    best_idx = np.argmin(resolutions)
    best_num_samples = sample_points[best_idx]
    best_resolution = resolutions[best_idx]
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"Best num_samples: {best_num_samples}")
    print(f"Best resolution: {best_resolution:.2f} Å")
    print(f"{'='*80}\n")
    
    return resolutions, correlations, best_num_samples


def plot_optimization_results(
    sample_points: List[int],
    resolutions: List[float],
    correlations: List[float],
    output_path: Path,
    structure_name: str,
):
    """Create visualization of optimization results."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Resolution plot
    ax1.plot(sample_points, resolutions, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(min(resolutions), color='r', linestyle='--', alpha=0.5,
                label=f'Best: {min(resolutions):.1f} Å')
    best_idx = np.argmin(resolutions)
    ax1.plot(sample_points[best_idx], resolutions[best_idx], 'r*', 
             markersize=20, label=f'Optimal: {sample_points[best_idx]} samples')
    ax1.set_xlabel('Number of Samples per Particle', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Resolution at FSC=0.5 (Å)', fontsize=14, fontweight='bold')
    ax1.set_title(f'{structure_name} - Resolution vs Num Samples', 
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=12)
    ax1.invert_yaxis()  # Lower resolution values are better
    
    # Correlation plot
    ax2.plot(sample_points, correlations, 'go-', linewidth=2, markersize=8)
    ax2.axhline(max(correlations), color='r', linestyle='--', alpha=0.5,
                label=f'Best: {max(correlations):.4f}')
    best_corr_idx = np.argmax(correlations)
    ax2.plot(sample_points[best_corr_idx], correlations[best_corr_idx], 'r*',
             markersize=20, label=f'Peak: {sample_points[best_corr_idx]} samples')
    ax2.set_xlabel('Number of Samples per Particle', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Correlation Coefficient', fontsize=14, fontweight='bold')
    ax2.set_title(f'{structure_name} - Correlation vs Num Samples',
                  fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved optimization plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize num_samples for best CryoLens affinity reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input options
    parser.add_argument(
        '--particles',
        type=str,
        required=True,
        help='Path to extracted particles zarr file',
    )
    parser.add_argument(
        '--ground-truth',
        type=str,
        required=True,
        help='Path to ground truth MRC for FSC calculation',
    )
    
    # Model options
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        help='Path to checkpoint file (overrides --checkpoint-epoch)',
    )
    parser.add_argument(
        '--checkpoint-epoch',
        type=int,
        default=2600,
        help='Checkpoint epoch to use (default: 2600)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference (default: cuda if available)',
    )
    
    # Optimization options
    parser.add_argument(
        '--sample-range',
        type=int,
        nargs=3,
        metavar=('START', 'STOP', 'STEP'),
        help='Range of num_samples to test: start stop step (e.g., 1 50 5)',
    )
    parser.add_argument(
        '--sample-points',
        type=int,
        nargs='+',
        help='Specific num_samples values to test (e.g., 1 5 10 20 30)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8)',
    )
    parser.add_argument(
        '--n-alignment-angles',
        type=int,
        default=24,
        help='Number of angles for alignment search (default: 24)',
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='./optimization_results',
        help='Output directory (default: ./optimization_results)',
    )
    
    args = parser.parse_args()
    
    # Determine sample points to test
    if args.sample_points:
        sample_points = sorted(args.sample_points)
    elif args.sample_range:
        start, stop, step = args.sample_range
        sample_points = list(range(start, stop + 1, step))
    else:
        # Default: test common values
        sample_points = [1, 5, 10, 15, 20, 25, 30, 40, 50]
    
    print(f"Will test num_samples values: {sample_points}")
    
    # Get checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        from cryolens.data import fetch_checkpoint
        checkpoint_path = fetch_checkpoint(epoch=args.checkpoint_epoch)
    
    # Create inference pipeline
    print(f"\nLoading model from: {checkpoint_path}")
    device = torch.device(args.device)
    pipeline = create_inference_pipeline(checkpoint_path, device=device)
    print(f"Model loaded successfully on {device}")
    
    # Load particles from zarr
    import zarr
    print(f"\nLoading particles from: {args.particles}")
    root = zarr.open(args.particles, mode='r')
    particles = root['particles'][:]
    
    # Load metadata
    metadata_path = Path(args.particles) / 'metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    structure_name = metadata['structure']
    voxel_spacing = metadata.get('voxel_spacing', 10.0)
    box_size = particles.shape[1]
    
    print(f"Loaded {len(particles)} particles for {structure_name}")
    print(f"Voxel spacing: {voxel_spacing} Å")
    print(f"Box size: {box_size}^3")
    
    # Load ground truth
    print(f"\nLoading ground truth from: {args.ground_truth}")
    with mrcfile.open(args.ground_truth) as mrc:
        ground_truth = mrc.data.copy()
    
    # Run optimization
    resolutions, correlations, best_num_samples = optimize_num_samples(
        pipeline=pipeline,
        particles=particles,
        ground_truth=ground_truth,
        sample_points=sample_points,
        batch_size=args.batch_size,
        box_size=box_size,
        voxel_spacing=voxel_spacing,
        n_alignment_angles=args.n_alignment_angles,
    )
    
    # Save results
    output_dir = Path(args.output) / structure_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save optimization data
    results = {
        'structure': structure_name,
        'n_particles': len(particles),
        'voxel_spacing': voxel_spacing,
        'sample_points': sample_points,
        'resolutions': resolutions,
        'correlations': correlations,
        'best_num_samples': best_num_samples,
        'best_resolution': float(min(resolutions)),
    }
    
    results_path = output_dir / 'optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")
    
    # Save detailed CSV
    csv_path = output_dir / 'optimization_data.csv'
    with open(csv_path, 'w') as f:
        f.write('num_samples,resolution_angstroms,correlation\n')
        for ns, res, corr in zip(sample_points, resolutions, correlations):
            f.write(f'{ns},{res:.4f},{corr:.6f}\n')
    print(f"Saved: {csv_path}")
    
    # Plot results
    plot_path = output_dir / 'optimization_plot.png'
    plot_optimization_results(
        sample_points, resolutions, correlations, plot_path, structure_name
    )
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Num Samples':<15} {'Resolution (Å)':<20} {'Correlation':<15}")
    print(f"{'-'*80}")
    for ns, res, corr in zip(sample_points, resolutions, correlations):
        marker = " ← BEST" if ns == best_num_samples else ""
        print(f"{ns:<15} {res:<20.2f} {corr:<15.4f}{marker}")
    print(f"{'='*80}\n")
    
    print(f"\n🎯 RECOMMENDATION: Use num_samples={best_num_samples} for optimal resolution")
    print(f"   This achieves {min(resolutions):.2f} Å resolution\n")
    
    print("Optimization complete!")


if __name__ == '__main__':
    main()
