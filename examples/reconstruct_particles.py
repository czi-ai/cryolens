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

Reconstruct directly from Copick:
    python reconstruct_particles.py \\
        --copick-config mlc_experimental_publictest \\
        --structure ribosome \\
        --num-particles 30 \\
        --checkpoint-epoch 2600 \\
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


def load_model(checkpoint_path: str, device: str = 'cuda') -> torch.nn.Module:
    """
    Load CryoLens model from checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file
    device : str
        Device to load model on ('cuda' or 'cpu')
        
    Returns
    -------
    model : torch.nn.Module
        Loaded model in eval mode
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Import model classes
    from cryolens.models.vae import AffinityVAE
    from cryolens.models.encoders import Encoder3D
    from cryolens.models.decoders import SegmentedGaussianSplatDecoder
    
    # Get config from checkpoint or use defaults
    if 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters']
        latent_dims = config.get('latent_dims', 40)
        num_splats = config.get('num_splats', 768)
        box_size = config.get('box_size', 48)
        latent_ratio = config.get('latent_ratio', 0.8)
    else:
        # Defaults matching paper
        latent_dims = 40
        num_splats = 768
        box_size = 48
        latent_ratio = 0.8
    
    print(f"Model config: latent_dims={latent_dims}, num_splats={num_splats}, "
          f"box_size={box_size}, latent_ratio={latent_ratio}")
    
    # Create model architecture
    encoder = Encoder3D(
        input_shape=(box_size, box_size, box_size),
        layer_channels=(8, 16, 32, 64)
    )
    
    decoder = SegmentedGaussianSplatDecoder(
        (box_size, box_size, box_size),
        latent_dims=latent_dims,
        n_splats=num_splats,
        output_channels=1,
        device=device,
        splat_sigma_range=(0.005, 0.1),
        padding=9,
        latent_ratio=latent_ratio
    )
    
    model = AffinityVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dims=latent_dims,
        pose_channels=4,
    )
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    
    return model


def reconstruct_particles(
    model: torch.nn.Module,
    particles: np.ndarray,
    device: str = 'cuda',
    batch_size: int = 8,
    num_samples: int = 1,
) -> np.ndarray:
    """
    Reconstruct particles using CryoLens model.
    
    Parameters
    ----------
    model : torch.nn.Module
        CryoLens model
    particles : np.ndarray
        Particle volumes (N, D, D, D)
    device : str
        Device for inference
    batch_size : int
        Batch size for inference
    num_samples : int
        Number of reconstructions per particle (for uncertainty)
        
    Returns
    -------
    reconstructions : np.ndarray
        Reconstructed volumes (N, num_samples, D, D, D)
    """
    n_particles = len(particles)
    reconstructions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, n_particles, batch_size), desc="Reconstructing"):
            batch_particles = particles[i:i+batch_size]
            
            # Convert to tensor
            batch_tensor = torch.from_numpy(batch_particles).float()
            batch_tensor = batch_tensor.unsqueeze(1)  # Add channel dim
            batch_tensor = batch_tensor.to(device)
            
            # Multiple samples per particle if requested
            batch_recons = []
            for _ in range(num_samples):
                # Add small noise for uncertainty estimation
                if num_samples > 1:
                    noise_scale = 0.05 * batch_tensor.std()
                    noisy_input = batch_tensor + torch.randn_like(batch_tensor) * noise_scale
                else:
                    noisy_input = batch_tensor
                
                # Reconstruct
                recon, _, _, _, _, _ = model(noisy_input)
                batch_recons.append(recon.squeeze(1).cpu().numpy())
            
            # Stack samples
            batch_recons = np.stack(batch_recons, axis=1)  # (batch, samples, D, D, D)
            reconstructions.append(batch_recons)
    
    reconstructions = np.concatenate(reconstructions, axis=0)
    return reconstructions


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
    frequencies : np.ndarray
        Spatial frequencies (1/Angstrom)
    fsc : np.ndarray
        FSC values
    resolution : float
        Resolution at FSC=0.5 (Angstroms)
    """
    from scipy import ndimage
    
    # Apply spherical mask if requested
    if mask_radius is not None:
        center = np.array(volume1.shape) // 2
        z, y, x = np.ogrid[:volume1.shape[0], :volume1.shape[1], :volume1.shape[2]]
        dist = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
        
        # Soft mask with Gaussian edge
        edge_width = 5
        mask = np.exp(-((dist - mask_radius) / edge_width)**2)
        mask[dist <= mask_radius] = 1
        mask[dist > mask_radius + 3*edge_width] = 0
        
        volume1 = volume1 * mask
        volume2 = volume2 * mask
    
    # Fourier transform
    fft1 = np.fft.fftn(volume1)
    fft2 = np.fft.fftn(volume2)
    
    # Calculate radial distances
    center = np.array(volume1.shape) // 2
    z, y, x = np.ogrid[:volume1.shape[0], :volume1.shape[1], :volume1.shape[2]]
    radius = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    radius = radius.astype(int)
    
    # Calculate FSC for each shell
    max_radius = min(center)
    fsc = np.zeros(max_radius)
    
    for r in range(max_radius):
        mask_shell = (radius == r)
        if not np.any(mask_shell):
            continue
        
        numerator = np.sum(fft1[mask_shell] * np.conj(fft2[mask_shell]))
        denom1 = np.sum(np.abs(fft1[mask_shell])**2)
        denom2 = np.sum(np.abs(fft2[mask_shell])**2)
        
        if denom1 > 0 and denom2 > 0:
            fsc[r] = np.abs(numerator) / np.sqrt(denom1 * denom2)
    
    # Convert radius to spatial frequency
    frequencies = np.arange(max_radius) / (volume1.shape[0] * voxel_size)
    
    # Find resolution at FSC=0.5
    crossing_idx = np.where(fsc < 0.5)[0]
    if len(crossing_idx) > 0:
        # Linear interpolation
        idx = crossing_idx[0]
        if idx > 0:
            resolution = 1.0 / np.interp(0.5, [fsc[idx], fsc[idx-1]], 
                                        [frequencies[idx], frequencies[idx-1]])
        else:
            resolution = 1.0 / frequencies[idx]
    else:
        resolution = float('inf')
    
    return frequencies, fsc, resolution


def save_results(
    reconstructions: np.ndarray,
    particles: np.ndarray,
    output_dir: Path,
    structure_name: str,
    voxel_spacing: float = 10.0,
    ground_truth: Optional[np.ndarray] = None,
):
    """
    Save reconstruction results.
    
    Parameters
    ----------
    reconstructions : np.ndarray
        Reconstructed volumes (N, num_samples, D, D, D)
    particles : np.ndarray
        Original particles
    output_dir : Path
        Output directory
    structure_name : str
        Structure name
    voxel_spacing : float
        Voxel spacing in Angstroms
    ground_truth : np.ndarray, optional
        Ground truth structure for FSC
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_particles, num_samples = reconstructions.shape[:2]
    
    print(f"\nSaving results to {output_dir}")
    
    # Average reconstructions across samples and particles
    mean_recon = reconstructions.mean(axis=(0, 1))  # Average over particles and samples
    
    # Save mean reconstruction
    mean_path = output_dir / f'{structure_name}_mean_reconstruction.mrc'
    with mrcfile.new(str(mean_path), overwrite=True) as mrc:
        mrc.set_data(mean_recon.astype(np.float32))
        mrc.voxel_size = voxel_spacing
    print(f"Saved: {mean_path}")
    
    # Calculate and save uncertainty if multiple samples
    if num_samples > 1:
        std_recon = reconstructions.std(axis=1).mean(axis=0)  # Std across samples, mean across particles
        uncertainty_path = output_dir / f'{structure_name}_uncertainty.mrc'
        with mrcfile.new(str(uncertainty_path), overwrite=True) as mrc:
            mrc.set_data(std_recon.astype(np.float32))
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
        mask_radius = mean_recon.shape[0] // 2 - 5
        frequencies, fsc, resolution = calculate_fsc(
            mean_recon, ground_truth, voxel_spacing, mask_radius
        )
        
        # Save FSC curve
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(1/frequencies[1:], fsc[1:], 'b-', linewidth=2)
        ax.axhline(0.5, color='r', linestyle='--', label='FSC=0.5')
        ax.axvline(resolution, color='g', linestyle='--', 
                  label=f'Resolution={resolution:.1f}Å')
        ax.set_xlabel('Resolution (Å)', fontsize=12)
        ax.set_ylabel('FSC', fontsize=12)
        ax.set_title(f'{structure_name} FSC Curve', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1])
        
        fsc_plot_path = output_dir / f'{structure_name}_fsc.png'
        plt.savefig(fsc_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fsc_plot_path}")
        
        results['resolution_at_fsc05'] = float(resolution)
        
        # Save FSC data
        fsc_data_path = output_dir / f'{structure_name}_fsc.npz'
        np.savez(fsc_data_path, frequencies=frequencies, fsc=fsc, resolution=resolution)
        print(f"Saved: {fsc_data_path}")
    
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
        print(f"  Resolution (FSC=0.5): {resolution:.1f} Å")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct particles using CryoLens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--particles',
        type=str,
        help='Path to extracted particles zarr file',
    )
    input_group.add_argument(
        '--copick-config',
        type=str,
        help='Copick config name or path for direct extraction',
    )
    
    # Copick-specific options
    parser.add_argument(
        '--structure',
        type=str,
        help='Structure name (required if using --copick-config)',
    )
    parser.add_argument(
        '--num-particles',
        type=int,
        default=30,
        help='Number of particles to extract from Copick (default: 30)',
    )
    parser.add_argument(
        '--voxel-spacing',
        type=float,
        default=10.0,
        help='Voxel spacing for Copick extraction (default: 10.0)',
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
    
    # Validate Copick arguments
    if args.copick_config and not args.structure:
        parser.error("--structure is required when using --copick-config")
    
    # Get checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        from cryolens.data import fetch_checkpoint
        checkpoint_path = fetch_checkpoint(epoch=args.checkpoint_epoch)
    
    # Load model
    model = load_model(checkpoint_path, device=args.device)
    
    # Load particles
    if args.particles:
        # Load from zarr
        import zarr
        print(f"Loading particles from: {args.particles}")
        root = zarr.open(args.particles, mode='r')
        particles = root['particles'][:]
        
        # Load metadata
        metadata_path = Path(args.particles) / 'metadata.json'
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        structure_name = metadata['structure']
        voxel_spacing = metadata.get('voxel_spacing', 10.0)
        
        print(f"Loaded {len(particles)} particles for {structure_name}")
        
    else:
        # Extract from Copick
        print(f"Extracting particles from Copick project...")
        
        # Import and run extraction
        from cryolens.data import get_copick_config
        import copick
        
        # Get config
        try:
            config_path = get_copick_config(args.copick_config)
        except ValueError:
            config_path = args.copick_config
        
        # This would need the extract_particles_from_run function
        # For now, suggest using extract_copick_particles.py first
        print("Error: Direct Copick extraction not yet implemented.")
        print("Please use extract_copick_particles.py first, then pass --particles")
        return 1
        
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        print(f"Loading ground truth from: {args.ground_truth}")
        with mrcfile.open(args.ground_truth) as mrc:
            ground_truth = mrc.data.copy()
    
    # Reconstruct
    print(f"\nReconstructing {len(particles)} particles...")
    reconstructions = reconstruct_particles(
        model=model,
        particles=particles,
        device=args.device,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
    
    # Save results
    output_dir = Path(args.output) / structure_name
    save_results(
        reconstructions=reconstructions,
        particles=particles,
        output_dir=output_dir,
        structure_name=structure_name,
        voxel_spacing=voxel_spacing,
        ground_truth=ground_truth,
    )
    
    print("Reconstruction complete!")


if __name__ == '__main__':
    main()
