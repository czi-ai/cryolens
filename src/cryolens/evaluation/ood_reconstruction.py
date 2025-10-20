"""
Minimal OOD reconstruction evaluation for paper figures.

This module provides a simple pipeline to:
1. Load particles from Copick
2. Generate reconstructions with uncertainty estimation
3. Align and average reconstructions
4. Compute FSC metrics
5. Create paper figures

The implementation focuses on simplicity and reproducibility rather than
extensive abstraction or features.

Note: This implementation assumes a segmented Gaussian splat decoder.
"""

import numpy as np
import torch
import h5py
import mrcfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from scipy import ndimage
from scipy.fft import fftn, ifftn, fftshift
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from cryolens.data import CopickDataLoader
from cryolens.inference.pipeline import InferencePipeline
from cryolens.evaluation.fsc import compute_fsc_with_threshold, apply_soft_mask
from cryolens.utils.normalization import normalize_volume, denormalize_volume


def load_mrc_structure(mrc_path: Path, box_size: int = 48) -> np.ndarray:
    """
    Load and resize MRC structure file.
    
    Parameters
    ----------
    mrc_path : Path
        Path to MRC file
    box_size : int
        Target box size
        
    Returns
    -------
    np.ndarray
        Loaded and resized structure (box_size^3)
    """
    with mrcfile.open(mrc_path, mode='r') as mrc:
        data = mrc.data.copy()
    
    # Add dimension if 2D
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    
    # Resize if needed
    if data.shape != (box_size, box_size, box_size):
        zoom_factors = [box_size / s for s in data.shape]
        data = zoom(data, zoom_factors, order=3)
    
    return data.astype(np.float32)


def normalize_volume_zscore(volume: np.ndarray) -> np.ndarray:
    """Z-score normalization"""
    mean_val = np.mean(volume)
    std_val = np.std(volume)
    if std_val > 0:
        return (volume - mean_val) / std_val
    return volume - mean_val


def compute_3d_cross_correlation(vol1: np.ndarray, vol2: np.ndarray) -> np.ndarray:
    """
    Compute 3D cross-correlation using FFT.
    
    Parameters
    ----------
    vol1 : np.ndarray
        First volume
    vol2 : np.ndarray
        Second volume
        
    Returns
    -------
    np.ndarray
        Cross-correlation volume
    """
    fft1 = fftn(vol1)
    fft2 = fftn(vol2)
    cross_corr = np.real(ifftn(fft1 * np.conj(fft2)))
    return fftshift(cross_corr)


def align_volume(
    reference: np.ndarray,
    volume: np.ndarray,
    n_angles: int = 24,
    refine: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Align volume to reference using cross-correlation.
    
    Performs coarse rotational search followed by optional refinement.
    Rotation is performed about the center of the volume.
    
    Parameters
    ----------
    reference : np.ndarray
        Reference volume
    volume : np.ndarray
        Volume to align
    n_angles : int
        Number of angles for coarse search
    refine : bool
        Whether to perform fine refinement
        
    Returns
    -------
    aligned_volume : np.ndarray
        Aligned volume
    correlation : float
        Correlation coefficient with reference
        
    Notes
    -----
    The rotation center is explicitly set to the center of the volume.
    scipy.ndimage.rotate uses the volume center by default when reshape=False,
    but we ensure this explicitly for clarity.
    """
    best_corr = -np.inf
    best_aligned = volume.copy()
    best_angle = 0
    best_axes = (0, 1)
    
    # Coarse search over 3 axis pairs
    for axis_config in [(0, 1), (0, 2), (1, 2)]:
        angles = np.linspace(0, 360, n_angles, endpoint=False)
        
        for angle in angles:
            # Rotate about volume center
            # reshape=False: keep output size same as input (implicitly uses center)
            # order=1: linear interpolation (faster, sufficient for alignment)
            # prefilter=False: skip spline prefiltering for speed
            rotated = ndimage.rotate(
                volume, 
                angle, 
                axes=axis_config, 
                reshape=False, 
                order=1,
                prefilter=False
            )
            
            # Find best shift using cross-correlation
            cross_corr = compute_3d_cross_correlation(reference, rotated)
            max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
            center = np.array(cross_corr.shape) // 2
            shift = np.array(max_idx) - center
            
            # Apply shift
            aligned = ndimage.shift(rotated, shift, order=1)
            
            # Compute correlation
            corr = np.corrcoef(reference.flatten(), aligned.flatten())[0, 1]
            
            if corr > best_corr:
                best_corr = corr
                best_aligned = aligned.copy()
                best_angle = angle
                best_axes = axis_config
    
    # Fine refinement around best angle
    if refine:
        refined_angles = np.linspace(best_angle - 10, best_angle + 10, 21)
        
        for angle in refined_angles:
            rotated = ndimage.rotate(
                volume,
                angle,
                axes=best_axes,
                reshape=False,
                order=1,
                prefilter=False
            )
            cross_corr = compute_3d_cross_correlation(reference, rotated)
            max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
            center = np.array(cross_corr.shape) // 2
            shift = np.array(max_idx) - center
            aligned = ndimage.shift(rotated, shift, order=1)
            corr = np.corrcoef(reference.flatten(), aligned.flatten())[0, 1]
            
            if corr > best_corr:
                best_corr = corr
                best_aligned = aligned.copy()
    
    return best_aligned, best_corr


def center_crop_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Center crop volume to target shape.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume
    target_shape : Tuple[int, int, int]
        Target shape
        
    Returns
    -------
    np.ndarray
        Cropped volume
    """
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


def generate_resampled_reconstructions(
    particle: np.ndarray,
    model: torch.nn.Module,
    pipeline: InferencePipeline,
    device: torch.device,
    n_samples: int = 10,
    noise_level: float = 0.05,
    target_shape: Tuple[int, int, int] = (48, 48, 48)
) -> List[np.ndarray]:
    """
    Generate multiple reconstructions for uncertainty estimation.
    
    Adds small Gaussian noise to input and resamples from latent space
    to estimate reconstruction uncertainty.
    
    Note: Assumes segmented Gaussian splat decoder.
    
    Parameters
    ----------
    particle : np.ndarray
        Input particle (48^3)
    model : torch.nn.Module
        Trained CryoLens model with segmented decoder
    pipeline : InferencePipeline
        Inference pipeline
    device : torch.device
        Computation device
    n_samples : int
        Number of resampled reconstructions
    noise_level : float
        Noise standard deviation (as fraction of particle std)
    target_shape : Tuple[int, int, int]
        Output shape
        
    Returns
    -------
    List[np.ndarray]
        List of reconstructed volumes
    """
    reconstructions = []
    decoder = model.decoder
    
    for i in range(n_samples):
        # Add noise to input (except first sample)
        if i > 0:
            noisy_particle = particle + np.random.randn(*particle.shape) * noise_level * np.std(particle)
        else:
            noisy_particle = particle
        
        # Process through pipeline
        results = pipeline.process_volume(
            noisy_particle,
            return_embeddings=True,
            return_reconstruction=False
        )
        
        mu = results['embeddings']
        
        # Add latent noise for resampling (except first)
        if i > 0 and results.get('log_var') is not None:
            std = np.exp(0.5 * results['log_var'])
            eps = np.random.randn(*std.shape)
            mu = mu + eps * std * 0.5
        
        # Convert to tensors
        mu_tensor = torch.tensor(mu, dtype=torch.float32).unsqueeze(0).to(device)
        pose = torch.tensor(
            results['pose'] if results['pose'] is not None else np.array([1.0, 0.0, 0.0, 0.0]),
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        global_weight = torch.tensor(
            results['global_weight'] if results['global_weight'] is not None else np.array([1.0]),
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        
        # Decode with splats only (use_final_convolution=True for CTF-like layer)
        with torch.no_grad():
            reconstruction = decoder(mu_tensor, pose, global_weight, use_final_convolution=True)
        
        reconstruction_np = reconstruction.cpu().numpy()[0, 0]
        reconstruction_np = center_crop_volume(reconstruction_np, target_shape)
        reconstruction_np = denormalize_volume(reconstruction_np, results['normalization_stats'])
        
        reconstructions.append(reconstruction_np)
    
    return reconstructions


def evaluate_ood_structure(
    structure_name: str,
    model: torch.nn.Module,
    pipeline: InferencePipeline,
    copick_loader: CopickDataLoader,
    ground_truth_path: Path,
    output_dir: Path,
    device: torch.device,
    n_particles: int = 25,
    n_resamples: int = 10,
    particle_counts: List[int] = [5, 10, 15, 20, 25],
    voxel_size: float = 10.0
) -> Dict:
    """
    Evaluate single structure OOD reconstruction performance.
    
    This follows a two-stage alignment process:
    1. Align all reconstructions to first particle (common reference frame)
    2. Align averaged and individual reconstructions to GT for evaluation
    
    Parameters
    ----------
    structure_name : str
        Name of structure to evaluate
    model : torch.nn.Module
        Trained model with segmented decoder
    pipeline : InferencePipeline
        Inference pipeline
    copick_loader : CopickDataLoader
        Copick data loader
    ground_truth_path : Path
        Path to ground truth MRC file
    output_dir : Path
        Output directory for this structure
    device : torch.device
        Computation device
    n_particles : int
        Number of particles to process
    n_resamples : int
        Number of resamples per particle for uncertainty
    particle_counts : List[int]
        Particle counts to evaluate
    voxel_size : float
        Voxel size in Angstroms
        
    Returns
    -------
    Dict
        Evaluation results including metrics and paths
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {structure_name}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load ground truth
    print("Loading ground truth...")
    ground_truth = load_mrc_structure(ground_truth_path)
    ground_truth = apply_soft_mask(ground_truth, radius=22, soft_edge=5)
    gt_normalized = normalize_volume_zscore(ground_truth)
    
    # 2. Load particles from Copick
    print(f"Loading particles from Copick...")
    data = copick_loader.load_particles(
        structure_filter=[structure_name],
        max_particles_per_structure=n_particles + 5,
        target_voxel_spacing=voxel_size,
        box_size=48,
        normalize=False,
        verbose=False
    )
    
    if structure_name not in data or len(data[structure_name]['particles']) == 0:
        print(f"No particles found for {structure_name}")
        return {'error': 'No particles found'}
    
    particles = data[structure_name]['particles'][:n_particles]
    print(f"Loaded {len(particles)} particles")
    
    # CRITICAL: Apply masking to particles before processing
    # This ensures consistent signal processing and prevents artifacts
    # in both reconstruction and FSC computation
    print("Applying soft masks to particles...")
    particles = [apply_soft_mask(p, radius=22, soft_edge=5) for p in particles]
    
    # STAGE 1: Generate reconstructions and align to first particle
    print("\nSTAGE 1: Generating reconstructions and aligning to common reference...")
    all_reconstructions = []
    reference_reconstruction = None
    
    for idx, particle in enumerate(tqdm(particles, desc="Processing particles")):
        # Generate n_resamples reconstructions with noise
        recons = generate_resampled_reconstructions(
            particle, model, pipeline, device,
            n_samples=n_resamples,
            noise_level=0.05,
            target_shape=(48, 48, 48)
        )
        
        # Normalize reconstructions
        recons_norm = [normalize_volume_zscore(r) for r in recons]
        
        # Set reference from first particle's mean reconstruction
        if idx == 0:
            reference_reconstruction = np.mean(recons_norm, axis=0)
            print(f"  Using first particle as alignment reference")
            # First particle is already aligned to itself
            aligned_recons = recons_norm
        else:
            # Align each resample to the reference
            aligned_recons = []
            for recon_norm in recons_norm:
                aligned, _ = align_volume(reference_reconstruction, recon_norm, n_angles=24, refine=True)
                aligned_recons.append(aligned)
        
        # Average the aligned resamples
        mean_recon = np.mean(aligned_recons, axis=0)
        all_reconstructions.append(mean_recon)
    
    print(f"  All {len(all_reconstructions)} reconstructions aligned to common reference")
    
    # STAGE 2: Align to ground truth for evaluation
    print("\nSTAGE 2: Aligning to ground truth for evaluation...")
    
    # Align all individual reconstructions to GT
    gt_aligned_reconstructions = []
    for recon in tqdm(all_reconstructions, desc="Aligning to GT"):
        gt_aligned, _ = align_volume(gt_normalized, recon, n_angles=24, refine=True)
        gt_aligned_reconstructions.append(gt_aligned)
    
    # Compute metrics vs particle count using GT-aligned reconstructions
    print("Computing metrics vs particle count...")
    metrics = {}
    
    for n in particle_counts:
        if n > len(gt_aligned_reconstructions):
            continue
        
        # Average first n GT-aligned reconstructions
        avg = np.mean(gt_aligned_reconstructions[:n], axis=0)
        
        # Compute FSC with masking
        # CRITICAL: Apply masking during FSC computation to prevent edge artifacts
        # mask_radius=20.0 matches the working implementation for consistency
        _, _, resolution = compute_fsc_with_threshold(
            gt_normalized, avg,
            voxel_size=voxel_size,
            threshold=0.5,
            mask_radius=20.0,  # CRITICAL: Apply mask during FSC
            soft_edge=5.0
        )
        
        # Compute correlation
        correlation = np.corrcoef(
            gt_normalized.flatten(),
            avg.flatten()
        )[0, 1]
        
        metrics[n] = {
            'resolution': float(resolution),
            'correlation': float(correlation),
            'average': avg
        }
        
        print(f"  n={n:2d}: resolution={resolution:.1f}Å, correlation={correlation:.3f}")
    
    # Save results
    print("Saving results...")
    h5_path = output_dir / f"{structure_name}_results.h5"
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('ground_truth', data=ground_truth, compression='gzip')
        f.attrs['structure_name'] = structure_name
        f.attrs['n_particles'] = len(particles)
        f.attrs['voxel_size'] = voxel_size
        
        for n, m in metrics.items():
            grp = f.create_group(f'n_{n:03d}')
            grp.attrs['resolution'] = m['resolution']
            grp.attrs['correlation'] = m['correlation']
            grp.create_dataset('average', data=m['average'], compression='gzip')
    
    print(f"  Saved to {h5_path}")
    
    # Create figure
    print("Creating figure...")
    fig_path = output_dir / f"{structure_name}_results.png"
    create_ood_figure(structure_name, ground_truth, metrics, voxel_size, fig_path)
    print(f"  Saved figure to {fig_path}")
    
    return {
        'structure_name': structure_name,
        'metrics': metrics,
        'h5_path': str(h5_path),
        'figure_path': str(fig_path)
    }


def create_ood_figure(
    structure_name: str,
    ground_truth: np.ndarray,
    metrics_vs_count: Dict,
    voxel_size: float,
    save_path: Path,
    dpi: int = 150
):
    """
    Create simple 3-panel figure: GT, reconstruction, resolution plot.
    
    Parameters
    ----------
    structure_name : str
        Name of structure
    ground_truth : np.ndarray
        Ground truth volume
    metrics_vs_count : Dict
        Metrics at different particle counts
    voxel_size : float
        Voxel size in Angstroms
    save_path : Path
        Path to save figure
    dpi : int
        Figure DPI
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    z_slice = 24  # Central slice
    nyquist = voxel_size * 2  # Nyquist limit
    
    # Normalize for display
    def norm_for_display(vol):
        vmin, vmax = np.percentile(vol, [2, 98])
        return np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    # Panel 1: Ground truth
    ax = axes[0]
    ax.imshow(norm_for_display(ground_truth[z_slice]), cmap='gray')
    ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Panel 2: Best reconstruction
    max_n = max(metrics_vs_count.keys())
    best_recon = metrics_vs_count[max_n]['average']
    best_res = metrics_vs_count[max_n]['resolution']
    best_corr = metrics_vs_count[max_n]['correlation']
    
    ax = axes[1]
    ax.imshow(norm_for_display(best_recon[z_slice]), cmap='gray')
    ax.set_title(
        f'Reconstruction (n={max_n})\n{best_res:.1f}Å, r={best_corr:.3f}',
        fontsize=12,
        fontweight='bold'
    )
    ax.axis('off')
    
    # Panel 3: Resolution vs count
    ax = axes[2]
    counts = sorted(metrics_vs_count.keys())
    resolutions = [metrics_vs_count[n]['resolution'] for n in counts]
    correlations = [metrics_vs_count[n]['correlation'] for n in counts]
    
    # Plot resolution on left y-axis
    color = 'tab:blue'
    ax.set_xlabel('Number of Particles', fontsize=11)
    ax.set_ylabel('Resolution (Å)', color=color, fontsize=11)
    line1 = ax.plot(counts, resolutions, 'o-', color=color, linewidth=2, markersize=8, label='Resolution')
    ax.axhline(y=nyquist, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Nyquist ({nyquist:.0f}Å)')
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True, alpha=0.3)
    
    # Plot correlation on right y-axis
    ax2 = ax.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Correlation', color=color, fontsize=11)
    line2 = ax2.plot(counts, correlations, 's-', color=color, linewidth=2, markersize=8, label='Correlation')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 1])
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=9)
    
    ax.set_title('Quality vs Particle Count', fontsize=12, fontweight='bold')
    
    # Overall title
    plt.suptitle(
        f'{structure_name.replace("-", " ").title()} - OOD Reconstruction',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
