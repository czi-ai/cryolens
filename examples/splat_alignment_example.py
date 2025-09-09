"""
Example script demonstrating the splat extraction and alignment workflow.

This example shows how to:
1. Process multiple volumes through CryoLens
2. Extract Gaussian splat parameters (structure-only)
3. Align volumes using splat-based alignment
4. Compute averaged reconstruction
"""

import numpy as np
from pathlib import Path

# Import CryoLens modules
from cryolens.utils.checkpoint_loading import load_vae_model
from cryolens.inference import InferencePipeline
from cryolens.splats import align_and_average_volumes, align_to_ground_truth_poses


def main():
    """Main example workflow."""
    
    # Configuration
    model_path = "path/to/model_checkpoint.pt"  # Update with actual path
    device = "cuda"  # or "cpu"
    
    # Load model
    print("Loading model...")
    model, config = load_vae_model(model_path, device=device)
    
    # Create inference pipeline
    pipeline = InferencePipeline(model, device=device)
    
    # Load your volumes (example with random data)
    n_volumes = 10
    box_size = 48
    volumes = np.random.randn(n_volumes, box_size, box_size, box_size).astype(np.float32)
    
    # You would load real data like:
    # volumes = load_particles_from_copick(...)  # Your data loading function
    
    print(f"Processing {n_volumes} volumes...")
    
    # Process volumes and extract splat parameters
    results = []
    reconstructions = []
    
    for i, volume in enumerate(volumes):
        print(f"  Processing volume {i+1}/{n_volumes}")
        
        # Process with splat extraction (structure-only by default)
        result = pipeline.process_volume(
            volume,
            return_embeddings=True,
            return_reconstruction=True,
            return_splat_params=True,  # Enable splat extraction
            splat_segment='affinity'    # Only extract structure splats
        )
        
        results.append(result)
        reconstructions.append(result['reconstruction'])
    
    # Extract splat parameters for alignment
    splat_params_list = [r['splat_params'] for r in results]
    
    print("\nAligning volumes using Gaussian splats...")
    
    # Align all volumes to the first one (template_idx=0)
    averaged_volume, aligned_volumes, rotation_matrices = align_and_average_volumes(
        np.array(reconstructions),
        splat_params_list,
        template_idx=0,
        method='pca',  # or 'icp'
        coordinate_transform=True  # Transform from [-1,1] to voxel space
    )
    
    print(f"Alignment complete!")
    print(f"  Averaged volume shape: {averaged_volume.shape}")
    print(f"  Number of aligned volumes: {len(aligned_volumes)}")
    
    # If you have ground truth poses, you can also align to them
    if False:  # Set to True if you have ground truth
        ground_truth_poses = np.random.randn(n_volumes, 3, 3)  # Your GT poses
        
        # Convert rotation matrices to poses if needed
        recovered_poses = np.array(rotation_matrices)
        
        # Align to ground truth
        aligned_poses, global_rotation, metrics = align_to_ground_truth_poses(
            recovered_poses,
            ground_truth_poses,
            method='kabsch'
        )
        
        print(f"\nPose alignment to ground truth:")
        print(f"  Mean angular error: {metrics['mean_angular_error']:.1f}°")
        print(f"  Std angular error: {metrics['std_angular_error']:.1f}°")
    
    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save averaged volume
    np.save(output_dir / "averaged_reconstruction.npy", averaged_volume)
    
    # Save aligned volumes
    for i, aligned_vol in enumerate(aligned_volumes):
        np.save(output_dir / f"aligned_volume_{i:03d}.npy", aligned_vol)
    
    # Save rotation matrices
    np.save(output_dir / "alignment_rotations.npy", np.array(rotation_matrices))
    
    print(f"\nResults saved to {output_dir}")
    
    # Visualization (optional)
    try:
        import matplotlib.pyplot as plt
        
        # Show central slices
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original (first volume)
        axes[0].imshow(reconstructions[0][box_size//2], cmap='gray')
        axes[0].set_title('Original (First Volume)')
        axes[0].axis('off')
        
        # Aligned (first volume after alignment - should be unchanged as template)
        axes[1].imshow(aligned_volumes[0][box_size//2], cmap='gray')
        axes[1].set_title('Aligned (Template)')
        axes[1].axis('off')
        
        # Averaged
        axes[2].imshow(averaged_volume[box_size//2], cmap='gray')
        axes[2].set_title('Averaged Reconstruction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "alignment_comparison.png", dpi=150)
        plt.show()
        
        print("Visualization saved!")
        
    except ImportError:
        print("Matplotlib not available, skipping visualization")


if __name__ == "__main__":
    main()
