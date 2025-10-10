#!/usr/bin/env python3
"""
Reconstruct particles using CryoLens with affinity output (Gaussian splats without final convolution).

This script uses only the affinity segment output (Gaussian splats) without the final
convolutional layer for reconstruction and alignment, matching the paper's claim that
alignment can be performed equivalently on Gaussian splat representations.

Examples
--------
Reconstruct from extracted particles using affinity output:
    python reconstruct_particles_affinity.py \\
        --particles ./example_data/ribosome_particles.zarr \\
        --checkpoint-epoch 2600 \\
        --num-samples 25 \\
        --output ./reconstructions/

Reconstruct with FSC analysis:
    python reconstruct_particles_affinity.py \\
        --particles ./example_data/ribosome_particles.zarr \\
        --checkpoint-epoch 2600 \\
        --num-samples 25 \\
        --ground-truth ./references/ribosome.mrc \\
        --output ./reconstructions/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from tqdm import tqdm
from scipy import ndimage
from scipy.fft import fftn, fftshift

try:
    import mrcfile
except ImportError:
    print("Error: mrcfile is not installed. Install with: pip install mrcfile")
    sys.exit(1)

from cryolens.inference.pipeline import create_inference_pipeline
from cryolens.utils.normalization import normalize_volume, denormalize_volume


def crop_to_size(volume: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Crop volume to target shape from center."""
    if volume.shape == target_shape:
        return volume
    
    crop_slices = []
    for actual, target in zip(volume.shape, target_shape):
        if actual > target:
            start = (actual - target) // 2
            end = start + target
            crop_slices.append(slice(start, end))
        else:
            crop_slices.append(slice(None))
    
    return volume[tuple(crop_slices)]


def apply_soft_mask(volume: np.ndarray, radius: float = 20, soft_edge: float = 5) -> np.ndarray:
    """Apply soft spherical mask with Gaussian edge"""
    shape = volume.shape
    center = np.array(shape) // 2
    
    nz, ny, nx = shape
    z, y, x = np.ogrid[:nz, :ny, :nx]
    r = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    
    mask = np.ones_like(r, dtype=float)
    transition_zone = (r > radius) & (r < radius + soft_edge)
    mask[transition_zone] = np.exp(-(r[transition_zone] - radius)**2 / (2 * (soft_edge/3)**2))
    mask[r >= radius + soft_edge] = 0
    
    return volume * mask


def normalize_volume_zscore(volume: np.ndarray) -> np.ndarray:
    """Z-score normalization"""
    mean_val = np.mean(volume)
    std_val = np.std(volume)
    if std_val > 0:
        return (volume - mean_val) / std_val
    return volume - mean_val


def compute_3d_cross_correlation(vol1: np.ndarray, vol2: np.ndarray) -> np.ndarray:
    """Compute 3D cross-correlation using FFT"""
    from scipy.fft import ifftn
    fft1 = fftn(vol1)
    fft2 = fftn(vol2)
    cross_corr = np.real(ifftn(fft1 * np.conj(fft2)))
    return fftshift(cross_corr)


def align_to_reference(
    reference: np.ndarray, 
    volume: np.ndarray, 
    n_angles: int = 24,
    refine: bool = True
) -> Tuple[np.ndarray, dict]:
    """Enhanced alignment to reference with optional refinement"""
    
    best_corr = -np.inf
    best_aligned = volume.copy()
    best_params = {}
    
    # Coarse search
    for axis_config in [(0, 1), (0, 2), (1, 2)]:
        angles = np.linspace(0, 360, n_angles, endpoint=False)
        
        for angle in angles:
            rotated = ndimage.rotate(volume, angle, axes=axis_config, reshape=False, order=1)
            cross_corr = compute_3d_cross_correlation(reference, rotated)
            max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
            center = np.array(cross_corr.shape) // 2
            shift = np.array(max_idx) - center
            aligned = ndimage.shift(rotated, shift, order=1)
            corr = np.corrcoef(reference.flatten(), aligned.flatten())[0, 1]
            
            if corr > best_corr:
                best_corr = corr
                best_aligned = aligned.copy()
                best_params = {
                    'rotation_axes': axis_config,
                    'angle': angle,
                    'shift': shift.tolist(),
                    'correlation': corr
                }
    
    # Fine refinement
    if refine and best_params:
        refined_angles = np.linspace(
            best_params['angle'] - 10,
            best_params['angle'] + 10,
            21
        )
        for angle in refined_angles:
            rotated = ndimage.rotate(volume, angle, axes=best_params['rotation_axes'], 
                                    reshape=False, order=1)
            cross_corr = compute_3d_cross_correlation(reference, rotated)
            max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
            center = np.array(cross_corr.shape) // 2
            shift = np.array(max_idx) - center
            aligned = ndimage.shift(rotated, shift, order=1)
            corr = np.corrcoef(reference.flatten(), aligned.flatten())[0, 1]
            
            if corr > best_corr:
                best_corr = corr
                best_aligned = aligned.copy()
                best_params['angle'] = angle
                best_params['shift'] = shift.tolist()
                best_params['correlation'] = corr
    
    return best_aligned, best_params


def reconstruct_with_affinity_output(
    pipeline,
    particles: np.ndarray,
    batch_size: int = 8,
    num_samples: int = 1,
    box_size: int = 48,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Reconstruct particles using affinity output (Gaussian splats without final convolution).
    
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
    box_size : int
        Target box size for reconstruction
        
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
                
                # Normalize
                normalized, norm_stats = normalize_volume(
                    noisy_particle,
                    method=pipeline.normalization_method,
                    return_stats=True
                )
                
                # Convert to tensor
                volume_tensor = torch.tensor(normalized, dtype=torch.float32)
                volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0).to(pipeline.device)
                
                with torch.no_grad():
                    # Encode
                    mu, log_var, pose, global_weight = pipeline.model.encode(volume_tensor)
                    
                    # Decode with identity pose and NO final convolution
                    identity_pose = torch.zeros(1, 4, device=pipeline.device)
                    identity_pose[:, 0] = 1.0
                    identity_global_weight = torch.ones(1, 1, device=pipeline.device)
                    
                    # Get reconstruction without final convolution (affinity output)
                    # CRITICAL: use_final_convolution=False gives us the Gaussian splats
                    reconstruction = pipeline.model.decoder(
                        mu, identity_pose, identity_global_weight,
                        use_final_convolution=False
                    )
                    
                    # Remove batch and channel dimensions
                    reconstruction_np = reconstruction.cpu().numpy()[0, 0]
                    
                    # Crop to original size if needed
                    if reconstruction_np.shape != (box_size, box_size, box_size):
                        reconstruction_np = crop_to_size(reconstruction_np, (box_size, box_size, box_size))
                    
                    # Denormalize
                    reconstruction_np = denormalize_volume(reconstruction_np, norm_stats)
                
                # IMPORTANT: Don't invert when using splats only
                # (only invert when using full pipeline with final convolution)
                batch_results.append(reconstruction_np)
            
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
    # Ensure volumes have the same size
    if volume1.shape != volume2.shape:
        target_shape = tuple(min(s1, s2) for s1, s2 in zip(volume1.shape, volume2.shape))
        volume1 = crop_to_size(volume1, target_shape)
        volume2 = crop_to_size(volume2, target_shape)
        print(f"Cropped volumes to {target_shape} for FSC")
    
    # Apply soft mask (CRITICAL for good FSC)
    if mask_radius is None:
        mask_radius = volume1.shape[0] // 2 - 5
    
    volume1 = apply_soft_mask(volume1, radius=mask_radius, soft_edge=5)
    volume2 = apply_soft_mask(volume2, radius=mask_radius, soft_edge=5)
    
    # Compute 3D FFT
    fft1 = fftshift(fftn(volume1))
    fft2 = fftshift(fftn(volume2))
    
    # Get dimensions
    nz, ny, nx = volume1.shape
    center = np.array([nz//2, ny//2, nx//2])
    
    # Create radial distance map
    z, y, x = np.ogrid[:nz, :ny, :nx]
    r = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    
    max_radius = min(center)
    n_shells = int(max_radius)
    
    fsc_values = []
    resolutions = []
    
    for i in range(1, n_shells):
        mask = (r >= i - 0.5) & (r < i + 0.5)
        if np.sum(mask) < 10:
            continue
        
        fft1_shell = fft1[mask]
        fft2_shell = fft2[mask]
        
        numerator = np.real(np.sum(fft1_shell * np.conj(fft2_shell)))
        denominator = np.sqrt(np.sum(np.abs(fft1_shell)**2) * np.sum(np.abs(fft2_shell)**2))
        
        fsc = numerator / denominator if denominator > 0 else 0
        fsc_values.append(fsc)
        
        resolution = (nx * voxel_size) / (2.0 * i)
        resolutions.append(resolution)
    
    fsc_array = np.array(fsc_values)
    res_array = np.array(resolutions)
    
    # Find resolution at FSC=0.5 using linear interpolation
    resolution_at_half = res_array[0] if len(res_array) > 0 else 240.0
    
    if len(fsc_array) > 1:
        idx_above = np.where(fsc_array >= 0.5)[0]
        idx_below = np.where(fsc_array < 0.5)[0]
        
        if len(idx_above) > 0 and len(idx_below) > 0:
            idx1 = idx_above[-1]
            idx2 = idx_below[0]
            
            if idx2 == idx1 + 1:
                fsc1, fsc2 = fsc_array[idx1], fsc_array[idx2]
                res1, res2 = res_array[idx1], res_array[idx2]
                
                t = (0.5 - fsc1) / (fsc2 - fsc1)
                resolution_at_half = res1 + t * (res2 - res1)
            else:
                resolution_at_half = res_array[idx_below[0]]
        elif len(idx_below) > 0:
            resolution_at_half = res_array[0]
        else:
            resolution_at_half = res_array[-1] if len(res_array) > 0 else 240.0
    
    return res_array, fsc_array, resolution_at_half


def save_results(
    mean_reconstructions: np.ndarray,
    output_dir: Path,
    structure_name: str,
    voxel_spacing: float = 10.0,
    std_reconstructions: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    num_samples: int = 1,
    n_alignment_angles: int = 24,
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
    n_alignment_angles : int
        Number of angles for alignment search
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_particles = len(mean_reconstructions)
    
    print(f"\nSaving results to {output_dir}")
    
    # Process reconstructions with alignment if ground truth provided
    if ground_truth is not None:
        print("\nAligning reconstructions to ground truth...")
        
        # Normalize ground truth
        gt_normalized = normalize_volume_zscore(ground_truth)
        gt_masked = apply_soft_mask(gt_normalized, radius=22, soft_edge=5)
        
        # Align all reconstructions
        aligned_reconstructions = []
        for recon in tqdm(mean_reconstructions, desc="Aligning"):
            recon_normalized = normalize_volume_zscore(recon)
            recon_masked = apply_soft_mask(recon_normalized, radius=22, soft_edge=5)
            aligned, _ = align_to_reference(gt_masked, recon_masked, n_alignment_angles, refine=True)
            aligned_reconstructions.append(aligned)
        
        # Average aligned reconstructions
        global_mean = np.mean(aligned_reconstructions, axis=0)
    else:
        # Simple average without alignment
        global_mean = mean_reconstructions.mean(axis=0)
    
    # Save mean reconstruction
    mean_path = output_dir / f'{structure_name}_affinity_mean_reconstruction.mrc'
    with mrcfile.new(str(mean_path), overwrite=True) as mrc:
        mrc.set_data(global_mean.astype(np.float32))
        mrc.voxel_size = voxel_spacing
    print(f"Saved: {mean_path}")
    
    # Save uncertainty if available
    if std_reconstructions is not None:
        global_uncertainty = std_reconstructions.mean(axis=0)
        uncertainty_path = output_dir / f'{structure_name}_affinity_uncertainty.mrc'
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
        'reconstruction_mode': 'affinity_only',
    }
    
    if ground_truth is not None:
        print("\nCalculating FSC with ground truth...")
        
        # Calculate FSC
        resolution_angstroms, fsc, resolution_at_half = calculate_fsc(
            global_mean, ground_truth, voxel_spacing
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
        ax.set_title(f'{structure_name} FSC Curve (Affinity Output)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(fontsize=12, loc='upper right')
        ax.set_xlim([resolution_angstroms[-1], resolution_angstroms[0]])
        ax.set_ylim([0, 1.05])
        ax.tick_params(labelsize=11)
        
        nyquist_res = 2 * voxel_spacing
        ax.axvline(nyquist_res, color='orange', linestyle=':', linewidth=1.5, 
                  label=f'Nyquist: {nyquist_res:.1f} Å', alpha=0.7)
        ax.legend(fontsize=12, loc='upper right')
        
        fsc_plot_path = output_dir / f'{structure_name}_affinity_fsc.png'
        plt.savefig(fsc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fsc_plot_path}")
        
        results['resolution_at_fsc05'] = float(resolution_at_half)
        results['nyquist_limit'] = float(nyquist_res)
        
        # Save FSC data
        fsc_data_path = output_dir / f'{structure_name}_affinity_fsc.npz'
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
    
    # Create subplot grid based on ground truth availability
    if ground_truth is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.reshape(1, -1)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    center = global_mean.shape[0] // 2
    
    def normalize_for_display(vol):
        vmin, vmax = np.percentile(vol, [2, 98])
        if vmax == vmin:
            vmax = vmin + 1
        return np.clip((vol - vmin) / (vmax - vmin), 0, 1)
    
    # Reconstruction views
    axes[0, 0].imshow(normalize_for_display(global_mean[center, :, :]), cmap='gray')
    axes[0, 0].set_title('Affinity Reconstruction XY', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(normalize_for_display(global_mean[:, center, :]), cmap='gray')
    axes[0, 1].set_title('Affinity Reconstruction XZ', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(normalize_for_display(global_mean[:, :, center]), cmap='gray')
    axes[0, 2].set_title('Affinity Reconstruction YZ', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    if ground_truth is not None:
        if ground_truth.shape != global_mean.shape:
            ground_truth_cropped = crop_to_size(ground_truth, global_mean.shape)
        else:
            ground_truth_cropped = ground_truth
        
        center_gt = ground_truth_cropped.shape[0] // 2
        
        axes[1, 0].imshow(normalize_for_display(ground_truth_cropped[center_gt, :, :]), cmap='gray')
        axes[1, 0].set_title('Ground Truth XY', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(normalize_for_display(ground_truth_cropped[:, center_gt, :]), cmap='gray')
        axes[1, 1].set_title('Ground Truth XZ', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(normalize_for_display(ground_truth_cropped[:, :, center_gt]), cmap='gray')
        axes[1, 2].set_title('Ground Truth YZ', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
    
    plt.suptitle(f'{structure_name} - Orthogonal Views (Affinity Output)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    ortho_path = output_dir / f'{structure_name}_affinity_orthoviews.png'
    plt.savefig(ortho_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {ortho_path}")
    
    # Save summary JSON
    summary_path = output_dir / f'{structure_name}_affinity_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"Summary for {structure_name} (Affinity Output):")
    print(f"  Particles: {n_particles}")
    print(f"  Samples per particle: {num_samples}")
    if ground_truth is not None:
        print(f"  Resolution (FSC=0.5): {results['resolution_at_fsc05']:.1f} Å")
        print(f"  Nyquist limit: {results['nyquist_limit']:.1f} Å")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct particles using CryoLens affinity output (without final convolution)",
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
    box_size = particles.shape[1]  # Assume cubic
    
    print(f"Loaded {len(particles)} particles for {structure_name}")
    print(f"Voxel spacing: {voxel_spacing} Å")
    print(f"Box size: {box_size}^3")
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        print(f"\nLoading ground truth from: {args.ground_truth}")
        with mrcfile.open(args.ground_truth) as mrc:
            ground_truth = mrc.data.copy()
            # Crop to match box size if needed
            if ground_truth.shape != (box_size, box_size, box_size):
                ground_truth = crop_to_size(ground_truth, (box_size, box_size, box_size))
    
    # Reconstruct using affinity output
    print(f"\nReconstructing {len(particles)} particles using affinity output (Gaussian splats)...")
    print("CRITICAL: Using use_final_convolution=False to get Gaussian splats")
    print("CRITICAL: NOT inverting reconstructions (only invert for full pipeline)")
    if args.num_samples > 1:
        print(f"Using {args.num_samples} samples per particle for uncertainty estimation")
    
    mean_recons, std_recons = reconstruct_with_affinity_output(
        pipeline=pipeline,
        particles=particles,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        box_size=box_size,
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
        n_alignment_angles=args.n_alignment_angles,
    )
    
    print("Reconstruction complete!")


if __name__ == '__main__':
    main()
