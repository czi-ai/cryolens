#!/usr/bin/env python3
"""
Reconstruct particles using CryoLens.

This script loads particle subvolumes (from extract_copick_particles.py or
direct Copick access), performs reconstruction using a pre-trained CryoLens
model, and saves the results with optional FSC analysis.

Examples
--------
Reconstruct from extracted particles:
    python reconstruct_particles.py \\
        --particles ./example_data/ribosome_particles.zarr \\
        --checkpoint-epoch 2600 \\
        --output ./reconstructions/

Reconstruct with uncertainty estimation:
    python reconstruct_particles.py \\
        --particles ./example_data/ribosome_particles.zarr \\
        --checkpoint-epoch 2600 \\
        --num-samples 10 \\
        --output ./reconstructions/

Reconstruct with FSC analysis:
    python reconstruct_particles.py \\
        --particles ./example_data/ribosome_particles.zarr \\
        --checkpoint-epoch 2600 \\
        --ground-truth ./references/ribosome.mrc \\
        --output ./reconstructions/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import mrcfile
except ImportError:
    print("Error: mrcfile is not installed. Install with: pip install mrcfile")
    sys.exit(1)

from cryolens.inference.pipeline import create_inference_pipeline


def reconstruct_with_uncertainty(
    pipeline,
    particles: np.ndarray,
    batch_size: int = 8,
    num_samples: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Reconstruct particles with optional uncertainty estimation.
    
    Parameters
    ----------
    pipeline : InferencePipeline
        CryoLens inference pipeline
    particles : np.ndarray
        Particle volumes (N, D, D, D)
    batch_size : int
        Batch size for inference
    num_samples : int
        Number of reconstructions per particle (for uncertainty)
        
    Returns
    -------
    mean_recons : np.ndarray
        Mean reconstructions (N, D, D, D)
    std_recons : np.ndarray or None
        Standard deviation across samples (N, D, D, D) if num_samples > 1
    """
    n_particles = len(particles)
    all_reconstructions = []
    
    # Process in batches
    for i in tqdm(range(0, n_particles, batch_size), desc="Reconstructing"):
        batch_particles = particles[i:i+batch_size]
        batch_recons = []
        
        # Multiple samples per particle if requested
        for sample_idx in range(num_samples):
            batch_results = []
            
            for particle in batch_particles:
                # Add small noise for uncertainty estimation
                if num_samples > 1:
                    noise_scale = 0.05 * particle.std()
                    noisy_particle = particle + np.random.randn(*particle.shape) * noise_scale
                else:
                    noisy_particle = particle
                
                # Reconstruct
                result = pipeline.process_volume(
                    noisy_particle,
                    return_embeddings=False,
                    return_reconstruction=True,
                    use_identity_pose=True
                )
                batch_results.append(result['reconstruction'])
            
            batch_recons.append(np.stack(batch_results))
        
        # Stack samples: (num_samples, batch_size, D, D, D)
        batch_recons = np.stack(batch_recons)
        # Transpose to: (batch_size, num_samples, D, D, D)
        batch_recons = np.transpose(batch_recons, (1, 0, 2, 3, 4))
        all_reconstructions.append(batch_recons)
    
    # Concatenate all batches: (N, num_samples, D, D, D)
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    
    # Calculate mean and std
    mean_recons = all_reconstructions.mean(axis=1)  # (N, D, D, D)
    
    if num_samples > 1:
        std_recons = all_reconstructions.std(axis=1)  # (N, D, D, D)
    else:
        std_recons = None
    
    return mean_recons, std_recons


def calculate_fsc(
    volume1: np.ndarray,
    volume2: np.ndarray,
    voxel_size: float = 1.0,
    mask_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate Fourier Shell Correlation between two volumes.
    
    Parameters
    ----------
    volume1, volume2 : np.ndarray
        3D volumes to compare
    voxel_size : float
        Voxel size in Angstroms
    mask_radius : float, optional
        Spherical mask radius in voxels
        
    Returns
    -------
    resolution_angstroms : np.ndarray
        Resolution in Angstroms for each shell
    fsc : np.ndarray
        FSC values
    resolution_at_half : float
        Resolution at FSC=0.5 (Angstroms)
    """
    # Ensure volumes have the same size by cropping to smaller dimensions
    if volume1.shape != volume2.shape:
        target_shape = tuple(min(s1, s2) for s1, s2 in zip(volume1.shape, volume2.shape))
        
        def crop_center(volume, target_shape):
            """Crop volume to target shape from center."""
            starts = tuple((s - t) // 2 for s, t in zip(volume.shape, target_shape))
            slices = tuple(slice(start, start + t) for start, t in zip(starts, target_shape))
            return volume[slices]
        
        volume1 = crop_center(volume1, target_shape)
        volume2 = crop_center(volume2, target_shape)
        print(f"Cropped volumes to {target_shape} for FSC")
    
    # Apply spherical mask if requested
    if mask_radius is not None:
        center = np.array(volume1.shape) // 2
        z, y, x = np.ogrid[:volume1.shape[0], :volume1.shape[1], :volume1.shape[2]]
        dist = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
        
        # Soft Gaussian mask
        edge_width = 3
        mask = np.exp(-0.5 * ((dist - mask_radius) / edge_width)**2)
        mask[dist <= mask_radius] = 1.0
        mask[dist > mask_radius + 3*edge_width] = 0.0
        
        volume1 = volume1 * mask
        volume2 = volume2 * mask
    
    # Compute 3D FFT
    fft1 = np.fft.fftshift(np.fft.fftn(volume1))
    fft2 = np.fft.fftshift(np.fft.fftn(volume2))
    
    # Get volume dimensions
    nz, ny, nx = volume1.shape
    
    # Create frequency grid
    z_freq = np.fft.fftshift(np.fft.fftfreq(nz))
    y_freq = np.fft.fftshift(np.fft.fftfreq(ny))
    x_freq = np.fft.fftshift(np.fft.fftfreq(nx))
    
    # Create 3D meshgrid
    fz, fy, fx = np.meshgrid(z_freq, y_freq, x_freq, indexing='ij')
    
    # Calculate radial frequency (in units of 1/pixel)
    freq_radius = np.sqrt(fx**2 + fy**2 + fz**2)
    
    # Maximum frequency is Nyquist
    max_freq = 0.5
    
    # Create radial bins
    n_bins = min(volume1.shape) // 2
    freq_bins = np.linspace(0, max_freq, n_bins + 1)
    
    # Calculate FSC for each shell
    fsc = np.zeros(n_bins)
    
    for i in range(n_bins):
        # Define shell
        inner = freq_bins[i]
        outer = freq_bins[i + 1]
        
        # Select voxels in this shell
        shell_mask = (freq_radius >= inner) & (freq_radius < outer)
        
        if not np.any(shell_mask):
            continue
        
        # Get FFT values in this shell
        fft1_shell = fft1[shell_mask]
        fft2_shell = fft2[shell_mask]
        
        # Calculate correlation
        numerator = np.abs(np.sum(fft1_shell * np.conj(fft2_shell)))
        denom1 = np.sum(np.abs(fft1_shell)**2)
        denom2 = np.sum(np.abs(fft2_shell)**2)
        
        if denom1 > 0 and denom2 > 0:
            fsc[i] = numerator / np.sqrt(denom1 * denom2)
    
    # Convert frequency bins to resolution in Angstroms
    # freq (1/pixel) * (1 pixel / voxel_size Angstrom) = 1/Angstrom
    # resolution = 1 / freq
    freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2
    
    # Avoid division by zero for DC component
    freq_centers[0] = freq_bins[1] / 2 if freq_bins[1] > 0 else 1e-10
    
    # Convert to resolution in Angstroms: resolution = voxel_size / freq
    resolution_angstroms = voxel_size / freq_centers
    
    # Find resolution at FSC = 0.5
    idx_half = np.where(fsc < 0.5)[0]
    if len(idx_half) > 0:
        # Linear interpolation between points
        idx = idx_half[0]
        if idx > 0 and fsc[idx-1] > 0.5:
            # Interpolate
            frac = (0.5 - fsc[idx]) / (fsc[idx-1] - fsc[idx])
            resolution_at_half = resolution_angstroms[idx] + frac * (resolution_angstroms[idx-1] - resolution_angstroms[idx])
        else:
            resolution_at_half = resolution_angstroms[idx]
    else:
        resolution_at_half = resolution_angstroms[-1]  # Better than Nyquist
    
    return resolution_angstroms, fsc, resolution_at_half


def save_results(
    mean_reconstructions: np.ndarray,
    output_dir: Path,
    structure_name: str,
    voxel_spacing: float = 10.0,
    std_reconstructions: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    num_samples: int = 1,
):
    """
    Save reconstruction results.
    
    Parameters
    ----------
    mean_reconstructions : np.ndarray
        Mean reconstructed volumes (N, D, D, D)
    output_dir : Path
        Output directory
    structure_name : str
        Structure name
    voxel_spacing : float
        Voxel spacing in Angstroms
    std_reconstructions : np.ndarray, optional
        Uncertainty maps (N, D, D, D)
    ground_truth : np.ndarray, optional
        Ground truth structure for FSC
    num_samples : int
        Number of samples used per particle
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_particles = len(mean_reconstructions)
    
    print(f"\nSaving results to {output_dir}")
    
    # Average across all particles
    global_mean = mean_reconstructions.mean(axis=0)
    
    # Save mean reconstruction
    mean_path = output_dir / f'{structure_name}_mean_reconstruction.mrc'
    with mrcfile.new(str(mean_path), overwrite=True) as mrc:
        mrc.set_data(global_mean.astype(np.float32))
        mrc.voxel_size = voxel_spacing
    print(f"Saved: {mean_path}")
    
    # Save uncertainty if available
    if std_reconstructions is not None:
        # Average uncertainty across particles
        global_uncertainty = std_reconstructions.mean(axis=0)
        uncertainty_path = output_dir / f'{structure_name}_uncertainty.mrc'
        with mrcfile.new(str(uncertainty_path), overwrite=True) as mrc:
            mrc.set_data(global_uncertainty.astype(np.float32))
            mrc.voxel_size = voxel_spacing
        print(f"Saved: {uncertainty_path}")
    
    # Calculate FSC if ground truth provided
    results = {
        'structure': structure_name,
        'n_particles': n_particles,
        'num_samples_per_particle': num_samples,
        'voxel_spacing': voxel_spacing,
    }
    
    if ground_truth is not None:
        print("\nCalculating FSC with ground truth...")
        mask_radius = global_mean.shape[0] // 2 - 5
        resolution_angstroms, fsc, resolution_at_half = calculate_fsc(
            global_mean, ground_truth, voxel_spacing, mask_radius
        )
        
        # Save FSC curve
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(resolution_angstroms, fsc, 'b-', linewidth=2, label='FSC')
        ax.axhline(0.5, color='r', linestyle='--', linewidth=1.5, label='FSC=0.5 threshold')
        ax.axvline(resolution_at_half, color='g', linestyle='--', linewidth=1.5,
                  label=f'Resolution: {resolution_at_half:.1f} Å')
        
        ax.set_xlabel('Resolution (Å)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Fourier Shell Correlation', fontsize=14, fontweight='bold')
        ax.set_title(f'{structure_name} FSC Curve', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(fontsize=12, loc='upper right')
        ax.set_xlim([resolution_angstroms[-1], resolution_angstroms[0]])  # Reverse x-axis (high to low res)
        ax.set_ylim([0, 1.05])
        ax.tick_params(labelsize=11)
        
        # Add Nyquist line
        nyquist_res = 2 * voxel_spacing
        ax.axvline(nyquist_res, color='orange', linestyle=':', linewidth=1.5, 
                  label=f'Nyquist: {nyquist_res:.1f} Å', alpha=0.7)
        ax.legend(fontsize=12, loc='upper right')
        
        fsc_plot_path = output_dir / f'{structure_name}_fsc.png'
        plt.savefig(fsc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fsc_plot_path}")
        
        results['resolution_at_fsc05'] = float(resolution_at_half)
        results['nyquist_limit'] = float(nyquist_res)
        
        # Save FSC data
        fsc_data_path = output_dir / f'{structure_name}_fsc.npz'
        np.savez(fsc_data_path, 
                resolution_angstroms=resolution_angstroms, 
                fsc=fsc, 
                resolution_at_half=resolution_at_half,
                nyquist=nyquist_res)
        print(f"Saved: {fsc_data_path}")
    
    # Generate orthoviews
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    center = global_mean.shape[0] // 2
    
    # Reconstruction views
    axes[0, 0].imshow(global_mean[center, :, :], cmap='gray')
    axes[0, 0].set_title('Reconstruction XY', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(global_mean[:, center, :], cmap='gray')
    axes[0, 1].set_title('Reconstruction XZ', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(global_mean[:, :, center], cmap='gray')
    axes[0, 2].set_title('Reconstruction YZ', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    if ground_truth is not None:
        # Crop ground truth to match if needed
        if ground_truth.shape != global_mean.shape:
            def crop_center(volume, target_shape):
                starts = tuple((s - t) // 2 for s, t in zip(volume.shape, target_shape))
                slices = tuple(slice(start, start + t) for start, t in zip(starts, target_shape))
                return volume[slices]
            ground_truth_cropped = crop_center(ground_truth, global_mean.shape)
        else:
            ground_truth_cropped = ground_truth
        
        center_gt = ground_truth_cropped.shape[0] // 2
        
        axes[1, 0].imshow(ground_truth_cropped[center_gt, :, :], cmap='gray')
        axes[1, 0].set_title('Ground Truth XY', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(ground_truth_cropped[:, center_gt, :], cmap='gray')
        axes[1, 1].set_title('Ground Truth XZ', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(ground_truth_cropped[:, :, center_gt], cmap='gray')
        axes[1, 2].set_title('Ground Truth YZ', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
    else:
        # Hide ground truth row if not available
        for ax in axes[1, :]:
            ax.axis('off')
    
    plt.suptitle(f'{structure_name} - Orthogonal Views', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    ortho_path = output_dir / f'{structure_name}_orthoviews.png'
    plt.savefig(ortho_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {ortho_path}")
    
    # Save summary JSON
    summary_path = output_dir / f'{structure_name}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"Summary for {structure_name}:")
    print(f"  Particles: {n_particles}")
    print(f"  Samples per particle: {num_samples}")
    if ground_truth is not None:
        print(f"  Resolution (FSC=0.5): {results['resolution_at_fsc05']:.1f} Å")
        print(f"  Nyquist limit: {results['nyquist_limit']:.1f} Å")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct particles using CryoLens",
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
    
    # Reconstruction options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8)',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of reconstructions per particle for uncertainty (default: 1)',
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='./reconstructions',
        help='Output directory (default: ./reconstructions)',
    )
    parser.add_argument(
        '--ground-truth',
        type=str,
        help='Path to ground truth MRC for FSC calculation',
    )
    
    args = parser.parse_args()
    
    # Get checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        from cryolens.data import fetch_checkpoint
        checkpoint_path = fetch_checkpoint(epoch=args.checkpoint_epoch)
    
    # Create inference pipeline
    print(f"Loading model from: {checkpoint_path}")
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
    
    print(f"Loaded {len(particles)} particles for {structure_name}")
    print(f"Voxel spacing: {voxel_spacing} Å")
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        print(f"\nLoading ground truth from: {args.ground_truth}")
        with mrcfile.open(args.ground_truth) as mrc:
            ground_truth = mrc.data.copy()
    
    # Reconstruct with uncertainty estimation
    print(f"\nReconstructing {len(particles)} particles...")
    if args.num_samples > 1:
        print(f"Using {args.num_samples} samples per particle for uncertainty estimation")
    
    mean_recons, std_recons = reconstruct_with_uncertainty(
        pipeline=pipeline,
        particles=particles,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
    
    # Save results
    output_dir = Path(args.output) / structure_name
    save_results(
        mean_reconstructions=mean_recons,
        output_dir=output_dir,
        structure_name=structure_name,
        voxel_spacing=voxel_spacing,
        std_reconstructions=std_recons,
        ground_truth=ground_truth,
        num_samples=args.num_samples,
    )
    
    print("Reconstruction complete!")


if __name__ == '__main__':
    main()
