#!/usr/bin/env python3
"""
Optimize number of particles for best resolution in CryoLens affinity reconstruction.

This script sweeps across different numbers of particles and tracks FSC resolution
to understand how averaging more particles improves reconstruction quality.

Examples
--------
Basic optimization sweep:
    python optimize_num_particles.py \\
        --particles ./example_data/ribosome_particles.zarr \\
        --ground-truth ./references/ribosome.mrc \\
        --checkpoint-epoch 2600 \\
        --num-samples 1 \\
        --particle-range 1 100 5

Custom particle counts:
    python optimize_num_particles.py \\
        --particles ./example_data/ribosome_particles.zarr \\
        --ground-truth ./references/ribosome.mrc \\
        --checkpoint-epoch 2600 \\
        --num-samples 1 \\
        --particle-counts 1 5 10 20 30 40 50 75 100
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


def optimize_num_particles(
    pipeline,
    particles: np.ndarray,
    ground_truth: np.ndarray,
    particle_counts: List[int],
    num_samples: int = 1,
    n_trials: int = 1,
    batch_size: int = 8,
    box_size: int = 48,
    voxel_spacing: float = 10.0,
    n_alignment_angles: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Test different numbers of particles and return resolution results.
    
    Parameters
    ----------
    pipeline : InferencePipeline
        CryoLens inference pipeline
    particles : np.ndarray
        Particle volumes (N, D, D, D)
    ground_truth : np.ndarray
        Ground truth structure
    particle_counts : List[int]
        List of particle counts to test
    num_samples : int
        Number of samples per particle (fixed for all tests)
    n_trials : int
        Number of random trials for each particle count
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
    resolutions_mean : np.ndarray
        Mean FSC=0.5 resolution for each particle count across trials
    resolutions_std : np.ndarray
        Standard deviation of resolution across trials
    correlations_mean : np.ndarray
        Mean correlation for each particle count across trials
    correlations_std : np.ndarray
        Standard deviation of correlation across trials
    best_num_particles : int
        Optimal number of particles
    """
    max_particles = len(particles)
    
    # Validate particle counts
    particle_counts = [n for n in particle_counts if n <= max_particles]
    if not particle_counts:
        raise ValueError(f"No valid particle counts (max available: {max_particles})")
    
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
    
    # Reconstruct ALL particles once (to avoid recomputing)
    print(f"\n{'='*80}")
    print(f"Reconstructing all {max_particles} particles with num_samples={num_samples}")
    print(f"{'='*80}\n")
    
    mean_recons, _ = reconstruct_with_affinity_output(
        pipeline=pipeline,
        particles=particles,
        batch_size=batch_size,
        num_samples=num_samples,
        box_size=box_size,
    )
    
    # Now test different particle counts with multiple trials
    print(f"\n{'='*80}")
    print(f"Testing {len(particle_counts)} different particle counts")
    print(f"Particle counts: {particle_counts}")
    print(f"Number of trials per count: {n_trials}")
    print(f"{'='*80}\n")
    
    # Store results for all trials
    # Shape: (len(particle_counts), n_trials)
    all_resolutions = []
    all_correlations = []
    
    for n_particles in particle_counts:
        print(f"\n{'─'*80}")
        print(f"Testing with {n_particles} particles ({n_trials} trials)")
        print(f"{'─'*80}")
        
        trial_resolutions = []
        trial_correlations = []
        
        for trial_idx in range(n_trials):
            # Randomly sample n_particles without replacement
            if n_trials > 1:
                particle_indices = np.random.choice(
                    max_particles, size=n_particles, replace=False
                )
                selected_recons = mean_recons[particle_indices]
                trial_label = f"Trial {trial_idx+1}/{n_trials}"
            else:
                # For single trial, just use first n_particles (deterministic)
                selected_recons = mean_recons[:n_particles]
                trial_label = "Processing"
            
            # Align all selected reconstructions to ground truth
            aligned_reconstructions = []
            alignment_correlations = []
            
            desc = f"{trial_label}" if n_trials > 1 else "Aligning"
            for recon in tqdm(selected_recons, desc=desc, leave=False):
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
            
            trial_resolutions.append(resolution_at_half)
            trial_correlations.append(avg_correlation)
        
        # Store results for this particle count
        all_resolutions.append(trial_resolutions)
        all_correlations.append(trial_correlations)
        
        # Print summary for this particle count
        mean_res = np.mean(trial_resolutions)
        std_res = np.std(trial_resolutions)
        mean_corr = np.mean(trial_correlations)
        std_corr = np.std(trial_correlations)
        
        if n_trials > 1:
            print(f"Resolution (FSC=0.5): {mean_res:.2f} ± {std_res:.2f} Å")
            print(f"Average correlation: {mean_corr:.4f} ± {std_corr:.4f}")
        else:
            print(f"Resolution (FSC=0.5): {mean_res:.2f} Å")
            print(f"Average correlation: {mean_corr:.4f}")
    
    # Convert to numpy arrays and compute statistics
    all_resolutions = np.array(all_resolutions)  # Shape: (n_particle_counts, n_trials)
    all_correlations = np.array(all_correlations)
    
    resolutions_mean = np.mean(all_resolutions, axis=1)
    resolutions_std = np.std(all_resolutions, axis=1)
    correlations_mean = np.mean(all_correlations, axis=1)
    correlations_std = np.std(all_correlations, axis=1)
    
    # Find best num_particles (lowest mean resolution value = highest resolution)
    best_idx = np.argmin(resolutions_mean)
    best_num_particles = particle_counts[best_idx]
    best_resolution = resolutions_mean[best_idx]
    best_resolution_std = resolutions_std[best_idx]
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"Best num_particles: {best_num_particles}")
    if n_trials > 1:
        print(f"Best resolution: {best_resolution:.2f} ± {best_resolution_std:.2f} Å")
    else:
        print(f"Best resolution: {best_resolution:.2f} Å")
    print(f"{'='*80}\n")
    
    return resolutions_mean, resolutions_std, correlations_mean, correlations_std, best_num_particles


def plot_optimization_results(
    particle_counts: List[int],
    resolutions_mean: np.ndarray,
    resolutions_std: np.ndarray,
    correlations_mean: np.ndarray,
    correlations_std: np.ndarray,
    output_path: Path,
    structure_name: str,
    n_trials: int,
):
    """Create visualization of optimization results."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Resolution plot
    ax1.errorbar(particle_counts, resolutions_mean, yerr=resolutions_std,
                 fmt='bo-', linewidth=2, markersize=8, capsize=5, capthick=2,
                 label='Mean ± Std' if n_trials > 1 else 'Resolution')
    ax1.axhline(min(resolutions_mean), color='r', linestyle='--', alpha=0.5,
                label=f'Best: {min(resolutions_mean):.1f} Å')
    best_idx = np.argmin(resolutions_mean)
    ax1.plot(particle_counts[best_idx], resolutions_mean[best_idx], 'r*', 
             markersize=20, label=f'Optimal: {particle_counts[best_idx]} particles')
    ax1.set_xlabel('Number of Particles Averaged', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Resolution at FSC=0.5 (Å)', fontsize=14, fontweight='bold')
    title = f'{structure_name} - Resolution vs Num Particles'
    if n_trials > 1:
        title += f' ({n_trials} trials)'
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=12)
    ax1.invert_yaxis()  # Lower resolution values are better
    
    # Correlation plot
    ax2.errorbar(particle_counts, correlations_mean, yerr=correlations_std,
                 fmt='go-', linewidth=2, markersize=8, capsize=5, capthick=2,
                 label='Mean ± Std' if n_trials > 1 else 'Correlation')
    ax2.axhline(max(correlations_mean), color='r', linestyle='--', alpha=0.5,
                label=f'Best: {max(correlations_mean):.4f}')
    best_corr_idx = np.argmax(correlations_mean)
    ax2.plot(particle_counts[best_corr_idx], correlations_mean[best_corr_idx], 'r*',
             markersize=20, label=f'Peak: {particle_counts[best_corr_idx]} particles')
    ax2.set_xlabel('Number of Particles Averaged', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Correlation Coefficient', fontsize=14, fontweight='bold')
    title = f'{structure_name} - Correlation vs Num Particles'
    if n_trials > 1:
        title += f' ({n_trials} trials)'
    ax2.set_title(title, fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved optimization plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize number of particles for best CryoLens affinity reconstruction",
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
        '--num-samples',
        type=int,
        default=1,
        help='Fixed number of samples per particle (default: 1)',
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=1,
        help='Number of random trials for each particle count (default: 1)',
    )
    parser.add_argument(
        '--particle-range',
        type=int,
        nargs=3,
        metavar=('START', 'STOP', 'STEP'),
        help='Range of particle counts to test: start stop step (e.g., 1 100 5)',
    )
    parser.add_argument(
        '--particle-counts',
        type=int,
        nargs='+',
        help='Specific particle counts to test (e.g., 1 5 10 20 50 100)',
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
    
    # Determine particle counts to test
    if args.particle_counts:
        particle_counts = sorted(args.particle_counts)
    elif args.particle_range:
        start, stop, step = args.particle_range
        particle_counts = list(range(start, stop + 1, step))
    else:
        # Default: test specific values that show the progression well
        particle_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 
                          45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    print(f"Will test particle counts: {particle_counts}")
    
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
    
    # Filter particle counts to available range
    max_particles = len(particles)
    particle_counts = [n for n in particle_counts if n <= max_particles]
    if not particle_counts:
        print(f"ERROR: No valid particle counts (max available: {max_particles})")
        sys.exit(1)
    
    if max(particle_counts) > max_particles:
        print(f"WARNING: Limiting to {max_particles} particles (all available)")
    
    # Load ground truth
    print(f"\nLoading ground truth from: {args.ground_truth}")
    with mrcfile.open(args.ground_truth) as mrc:
        ground_truth = mrc.data.copy()
    
    # Run optimization
    resolutions_mean, resolutions_std, correlations_mean, correlations_std, best_num_particles = optimize_num_particles(
        pipeline=pipeline,
        particles=particles,
        ground_truth=ground_truth,
        particle_counts=particle_counts,
        num_samples=args.num_samples,
        n_trials=args.n_trials,
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
        'total_particles_available': len(particles),
        'num_samples_per_particle': args.num_samples,
        'n_trials': args.n_trials,
        'voxel_spacing': voxel_spacing,
        'particle_counts': particle_counts,
        'resolutions_mean': resolutions_mean.tolist(),
        'resolutions_std': resolutions_std.tolist(),
        'correlations_mean': correlations_mean.tolist(),
        'correlations_std': correlations_std.tolist(),
        'best_num_particles': best_num_particles,
        'best_resolution': float(min(resolutions_mean)),
    }
    
    results_path = output_dir / 'particle_optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")
    
    # Save detailed CSV
    csv_path = output_dir / 'particle_optimization_data.csv'
    with open(csv_path, 'w') as f:
        if args.n_trials > 1:
            f.write('num_particles,resolution_mean,resolution_std,correlation_mean,correlation_std\n')
            for np_val, res_mean, res_std, corr_mean, corr_std in zip(
                particle_counts, resolutions_mean, resolutions_std, 
                correlations_mean, correlations_std
            ):
                f.write(f'{np_val},{res_mean:.4f},{res_std:.4f},{corr_mean:.6f},{corr_std:.6f}\n')
        else:
            f.write('num_particles,resolution_angstroms,correlation\n')
            for np_val, res, corr in zip(particle_counts, resolutions_mean, correlations_mean):
                f.write(f'{np_val},{res:.4f},{corr:.6f}\n')
    print(f"Saved: {csv_path}")
    
    # Plot results
    plot_path = output_dir / 'particle_optimization_plot.png'
    plot_optimization_results(
        particle_counts, resolutions_mean, resolutions_std,
        correlations_mean, correlations_std, plot_path, structure_name, args.n_trials
    )
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"PARTICLE OPTIMIZATION SUMMARY TABLE")
    if args.n_trials > 1:
        print(f"(Mean ± Std across {args.n_trials} trials)")
    print(f"{'='*80}")
    if args.n_trials > 1:
        print(f"{'Num Particles':<15} {'Resolution (Å)':<25} {'Correlation':<20}")
        print(f"{'-'*80}")
        for np_val, res_mean, res_std, corr_mean, corr_std in zip(
            particle_counts, resolutions_mean, resolutions_std,
            correlations_mean, correlations_std
        ):
            marker = " ← BEST" if np_val == best_num_particles else ""
            print(f"{np_val:<15} {res_mean:>6.2f} ± {res_std:<5.2f}          "
                  f"{corr_mean:>6.4f} ± {corr_std:<6.4f}{marker}")
    else:
        print(f"{'Num Particles':<15} {'Resolution (Å)':<20} {'Correlation':<15}")
        print(f"{'-'*80}")
        for np_val, res, corr in zip(particle_counts, resolutions_mean, correlations_mean):
            marker = " ← BEST" if np_val == best_num_particles else ""
            print(f"{np_val:<15} {res:<20.2f} {corr:<15.4f}{marker}")
    print(f"{'='*80}\n")
    
    # Calculate improvement metrics
    initial_res = resolutions_mean[0]
    initial_std = resolutions_std[0]
    best_res = min(resolutions_mean)
    best_idx = np.argmin(resolutions_mean)
    best_std = resolutions_std[best_idx]
    improvement = initial_res - best_res
    percent_improvement = (improvement / initial_res) * 100
    
    print(f"\n📊 IMPROVEMENT ANALYSIS:")
    if args.n_trials > 1:
        print(f"   Initial (1 particle): {initial_res:.2f} ± {initial_std:.2f} Å")
        print(f"   Best ({best_num_particles} particles): {best_res:.2f} ± {best_std:.2f} Å")
        print(f"   Improvement: {improvement:.2f} Å ({percent_improvement:.1f}%)")
    else:
        print(f"   Initial (1 particle): {initial_res:.2f} Å")
        print(f"   Best ({best_num_particles} particles): {best_res:.2f} Å")
        print(f"   Improvement: {improvement:.2f} Å ({percent_improvement:.1f}%)")
    
    print(f"\n🎯 RECOMMENDATION: Use {best_num_particles} particles for optimal resolution")
    if args.n_trials > 1:
        print(f"   This achieves {best_res:.2f} ± {best_std:.2f} Å resolution\n")
    else:
        print(f"   This achieves {best_res:.2f} Å resolution\n")
    
    print("Optimization complete!")


if __name__ == '__main__':
    main()
