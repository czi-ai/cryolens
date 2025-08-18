import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pytorch_lightning.callbacks import Callback
from itertools import islice
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters"""
    every_n_epochs: int = 50                # How often to generate visualizations
    num_batches: int = 10                   # Number of batches to sample (for non-memoized sampling)
    max_samples_per_mol: int = 10           # Maximum samples to show per molecule
    samples_per_mol_per_source: int = 5     # Number of samples to collect per molecule per source type
    fig_width: int = 15                     # Width of the figure in inches
    dpi: int = 150                          # Resolution of the output images
    save_numpy_arrays: bool = True          # Whether to save raw numpy arrays
    organize_by_source: bool = True         # Whether to organize visualizations by source type
    
class VolumeProjector:
    """Handles 3D volume projection operations"""
    
    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        """Normalize image to [0,1] range"""
        if img.max() == img.min():
            return np.zeros_like(img)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    @classmethod
    def get_projections(cls, volume: torch.Tensor) -> Dict[str, np.ndarray]:
        """Generate sum projections of the volume"""
        if torch.is_tensor(volume):
            volume = volume.cpu().numpy()
            
        projections = {
            'top': cls.normalize(volume.sum(axis=-1)),
            'side': cls.normalize(volume.sum(axis=0)),
            'front': cls.normalize(volume.sum(axis=1))
        }
        return projections

class VisualizationPlotter:
    """Handles the plotting of visualizations with source type information"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        
    def setup_figure(self, n_molecules: int) -> Tuple[plt.Figure, plt.GridSpec]:
        """Setup the figure and grid for plotting with minimal whitespace"""
        # Handle edge case of no molecules
        if n_molecules <= 0:
            raise ValueError(f"Cannot create figure with {n_molecules} molecules")
            
        # Calculate dimensions based on content
        # Each molecule now needs 2 rows per sample (2 rows x max_samples_per_mol)
        n_rows = n_molecules * self.config.max_samples_per_mol * 2
        n_cols = 9  # 3 types per row * 3 views
        
        # Calculate figure dimensions maintaining reasonable proportions
        # Base the figure size on the desired width of each subplot
        desired_subplot_width = 1.5  # inches
        total_width = desired_subplot_width * n_cols
        
        # Scale the figure width to maintain readability
        scale_factor = self.config.fig_width / total_width
        actual_subplot_width = desired_subplot_width * scale_factor
        
        # Set subplot height equal to width for square appearance
        subplot_height = actual_subplot_width
        
        # Calculate total figure height
        fig_height = subplot_height * n_rows
        
        # Create figure and grid with appropriate spacing
        fig = plt.figure(figsize=(self.config.fig_width, fig_height))
        gs = fig.add_gridspec(n_rows, n_cols, 
                            hspace=0.3,
                            wspace=0.1)
        return fig, gs
        
    def plot_molecule(self, fig: plt.Figure, gs: plt.GridSpec, 
                     mol_data: List, mol_idx: int, mol_id: int) -> None:
        """Plot visualizations for a single molecule with source type and pose information"""
        # Each molecule's samples now take 2 rows per sample
        base_row = mol_idx * self.config.max_samples_per_mol * 2
        
        # Group samples by source type for better organization
        samples_by_source = defaultdict(list)
        for sample in mol_data:
            source_type = sample.get('source_type', 'unknown')
            samples_by_source[source_type].append(sample)
        
        # Add molecule header with count of source types
        source_counts = {source: len(samples) for source, samples in samples_by_source.items()}
        source_info = ", ".join([f"{source}: {count}" for source, count in source_counts.items()])
        plt.figtext(0.02, 1 - (base_row - 0.5) / gs.get_geometry()[0],
                   f'Molecule {mol_id} ({source_info})', ha='left', va='bottom', fontsize=10)
        
        # Flatten and sort samples by source type for display
        sorted_samples = []
        for source_type in sorted(samples_by_source.keys()):
            sorted_samples.extend(samples_by_source[source_type])
        
        # Plot up to max_samples_per_mol samples
        for sample_idx, sample in enumerate(sorted_samples[:self.config.max_samples_per_mol]):
            # Calculate the base row for this sample (2 rows per sample)
            sample_base_row = base_row + sample_idx * 2
            
            # First row: Input, Combined Splats, Background Splats
            row1_volumes = {
                'Input': sample['input'][0],
                'Combined Splats': sample['raw_splats'][0],
                'Background Splats': sample.get('segment_free', np.zeros_like(sample['raw_splats']))[0]
            }
            
            # Second row: Affinity Splats, Combined+Conv, Affinity+Conv
            row2_volumes = {
                'Affinity Splats': sample.get('segment_affinity', np.zeros_like(sample['raw_splats']))[0],
                'Combined Output': sample['output'][0],
                'Affinity Output': sample.get('segment_affinity_conv', np.zeros_like(sample['output']))[0]
            }
            
            # Add source type label for the first column
            source_type = sample.get('source_type', 'unknown')
            
            # Prepare pose information if available
            pose_info = {}
            if 'true_pose' in sample:
                pose_info['true_pose'] = sample['true_pose']
            if 'pred_pose' in sample:
                pose_info['pred_pose'] = sample['pred_pose']
            if 'pose_error' in sample:
                pose_info['pose_error'] = sample['pose_error']
            
            # Plot first row with pose information
            self._plot_volumes(fig, gs, row1_volumes, sample_base_row, 
                             source_label=source_type, row_label="Row 1", pose_info=pose_info)
            
            # Plot second row
            self._plot_volumes(fig, gs, row2_volumes, sample_base_row + 1, row_label="Row 2")
    
    def _plot_volumes(self, fig: plt.Figure, gs: plt.GridSpec, 
                     volumes: Dict, row: int, source_label: str = None, row_label: str = None,
                     pose_info: Dict = None) -> None:
        """Plot volume projections with source type and pose information"""
        for vol_idx, (vol_name, vol_data) in enumerate(volumes.items()):
            projections = VolumeProjector.get_projections(vol_data)
            base_col = vol_idx * 3
            
            for view_idx, (view_name, proj) in enumerate(projections.items()):
                ax = fig.add_subplot(gs[row, base_col + view_idx])
                ax.imshow(proj, cmap='gray')
                ax.axis('off')
                
                # Add volume title (always for the center view)
                if view_idx == 1:  # Center view
                    ax.set_title(vol_name, fontsize=9, pad=2)
                
                # Add row labels (for the first column only)
                if view_idx == 0 and vol_idx == 0:
                    # Add small text label in the corner for source type
                    if source_label:
                        ax.text(0.02, 0.98, source_label, transform=ax.transAxes,
                              fontsize=6, va='top', ha='left', 
                              bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
                    
                    # Add pose information if available
                    if pose_info and vol_idx == 0:  # Only show on first volume
                        pose_text = []
                        if 'true_pose' in pose_info:
                            true_pose = pose_info['true_pose']
                            if true_pose is not None:
                                pose_text.append(f"GT: [{true_pose[0]:.2f}, {true_pose[1]:.2f}, {true_pose[2]:.2f}, {true_pose[3]:.2f}]")
                        if 'pred_pose' in pose_info:
                            pred_pose = pose_info['pred_pose']
                            if pred_pose is not None:
                                pose_text.append(f"Pred: [{pred_pose[0]:.2f}, {pred_pose[1]:.2f}, {pred_pose[2]:.2f}, {pred_pose[3]:.2f}]")
                        if 'pose_error' in pose_info:
                            pose_text.append(f"Error: {pose_info['pose_error']:.3f} rad")
                        
                        if pose_text:
                            ax.text(0.02, 0.02, '\n'.join(pose_text), transform=ax.transAxes,
                                  fontsize=5, va='bottom', ha='left',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
                
                # First time through, add view labels to the top of each column
                if row == 0:
                    if view_idx == 0:
                        ax.set_title('Top', fontsize=8, pad=2)
                    elif view_idx == 1:
                        ax.set_title(vol_name, fontsize=9, pad=2)
                    elif view_idx == 2:
                        ax.set_title('Front', fontsize=8, pad=2)

class VisualizationCallback(Callback):
    """PyTorch Lightning callback for visualizing 3D volumes during training with memoized samples"""
    
    def __init__(self, config: VisualizationConfig = None, rank: int = 0, **kwargs):
        super().__init__()
        if config is None and kwargs:
            config = VisualizationConfig(**{
                k: v for k, v in kwargs.items() 
                if hasattr(VisualizationConfig, k)
            })
        self.config = config or VisualizationConfig()
        self.plotter = VisualizationPlotter(self.config)
        self.samples_per_mol_per_source = getattr(self.config, 'samples_per_mol_per_source', 5)
        # Number of structures to visualize (first N + random M)
        self.num_first_structures = 4
        self.num_random_structures = 6
        # Store rank to ensure only rank 0 saves visualizations
        self.rank = rank
        
    def _collect_samples_with_segments(self, dataset, pl_module) -> Dict[int, List]:
        """Collect samples using memoized visualization samples, limiting to a subset of structures"""
        all_samples = defaultdict(list)
        
        # Check if dataset has return_poses enabled
        dataset_has_poses = hasattr(dataset, 'return_poses') and dataset.return_poses
        
        # Get memoized visualization samples
        viz_samples = dataset.get_visualization_samples(self.samples_per_mol_per_source)
        
        # Check if we got any samples
        if not viz_samples:
            print(f"Warning: No visualization samples returned from dataset")
            return all_samples
        
        # Limit to a subset of structures to improve performance
        # Group by structure (source_type)
        structures = set(source for _, source in viz_samples.keys())
        structures = list(structures)
        
        # Select subset of structures (first N + random M)
        selected_structures = []
        if len(structures) <= (self.num_first_structures + self.num_random_structures):
            # If we have fewer structures than the limit, use all of them
            selected_structures = structures
        else:
            # Take the first N structures
            selected_structures.extend(structures[:self.num_first_structures])
            
            # Take M random structures from the remaining ones
            remaining = structures[self.num_first_structures:]
            random_selection = np.random.choice(
                remaining, 
                size=self.num_random_structures, 
                replace=False
            )
            selected_structures.extend(random_selection)
        
        # Filter viz_samples to only include selected structures
        filtered_viz_samples = {}
        for (mol_idx, source_type), indices in viz_samples.items():
            if source_type in selected_structures:
                filtered_viz_samples[(mol_idx, source_type)] = indices
        
        # Use the filtered samples
        viz_samples = filtered_viz_samples
        
        with torch.no_grad():
            # Process each (mol_idx, source_type) group
            for (mol_idx, source_type_key), indices in viz_samples.items():
                if mol_idx < 0:  # Skip background or unknown
                    continue
                    
                # Process samples for this molecule and source
                for idx in indices:
                    # Get sample directly using the memoized index
                    try:
                        sample_data = dataset.get_sample_by_index(idx)
                        
                        # The dataset.get_sample_by_index returns:
                        # - 3 values when return_poses=False: (volume, mol_id, source_type) 
                        # - 4 values when return_poses=True: (volume, mol_id, pose, source_type)
                        pose_data = None
                        source_type = None
                        
                        if dataset_has_poses and len(sample_data) == 4:
                            # With pose data: (volume, mol_id, pose, source_type)
                            subvolume, mol_id, pose_data, source_type = sample_data
                        elif not dataset_has_poses and len(sample_data) == 3:
                            # No pose data: (volume, mol_id, source_type)
                            subvolume, mol_id, source_type = sample_data
                        elif len(sample_data) == 3 and dataset_has_poses:
                            # Edge case: poses enabled but this sample doesn't have pose
                            # This shouldn't happen with our fixes but handle it anyway
                            subvolume, mol_id, source_type = sample_data
                            pose_data = None
                        elif len(sample_data) == 4 and not dataset_has_poses:
                            # Edge case: poses disabled but got 4 values anyway
                            # Take what we need
                            subvolume, mol_id, _, source_type = sample_data
                        else:
                            # Unexpected format
                            print(f"Warning: Unexpected sample format with {len(sample_data)} values (dataset_has_poses={dataset_has_poses})")
                            continue
                    except ValueError as e:
                        print(f"Warning: Error collecting visualization sample: {e}")
                        print(f"  Sample data length: {len(sample_data) if 'sample_data' in locals() else 'unknown'}")
                        print(f"  dataset_has_poses: {dataset_has_poses}")
                        continue
                    
                    # Skip if molecule ID is invalid
                    if mol_id == -1:
                        continue
                    
                    # Move to device
                    subvolume = subvolume.to(pl_module.device)
                    mol_id = mol_id.to(pl_module.device)
                    if pose_data is not None and not isinstance(pose_data, torch.Tensor):
                        pose_data = torch.tensor(pose_data, dtype=torch.float32)
                    if pose_data is not None:
                        pose_data = pose_data.to(pl_module.device)
                    
                    # Get standard output
                    output = pl_module(subvolume.unsqueeze(0))  # Add batch dimension
                    reconstructed, z, generated_pose, global_weight, mu, log_var = output
                    
                    # Use generated pose
                    pose = generated_pose
                    
                    # Calculate pose error if we have ground truth
                    pose_error = None
                    if pose_data is not None and pose is not None:
                        try:
                            # Calculate geodesic distance between poses if we have the loss function
                            if hasattr(pl_module, 'geodesic_pose_loss'):
                                with torch.no_grad():
                                    # Ensure both poses have batch dimension
                                    true_pose_batch = pose_data.unsqueeze(0) if pose_data.dim() == 1 else pose_data
                                    pred_pose_batch = pose.unsqueeze(0) if pose.dim() == 1 else pose
                                    pose_error = pl_module.geodesic_pose_loss(pred_pose_batch, true_pose_batch)
                                    pose_error = pose_error.item()
                        except Exception as e:
                            print(f"Error computing pose error: {e}")
                            pose_error = None
                    
                    # Make sure pose is not None before trying to decode splats
                    if pose is None:
                        # Create default pose (zero rotation around Z axis)
                        batch_size = z.shape[0]
                        pose = torch.zeros((batch_size, 4), device=z.device)
                        pose[:, 3] = 1.0  # z-axis
                    
                    # Get raw gaussian splats (combined splats with no convolution)
                    try:
                        splats, weights, sigmas = pl_module.model.decoder.decode_splats(z, pose)
                        raw_splats = pl_module.model.decoder._splatter(
                            splats, weights, sigmas,
                            splat_sigma_range=pl_module.model.decoder._splat_sigma_range
                        )
                    except Exception as e:
                        print(f"Error in decode_splats: {str(e)}")
                        # Fallback to an empty tensor with the right shape
                        raw_splats = torch.zeros_like(reconstructed)
                    
                    # Get segmented visualizations
                    try:
                        # Attempt to get segmented visualizations (should return list of two segments)
                        segment_output = pl_module.model.decoder.forward(
                            z, pose, global_weight=global_weight, use_final_convolution=True, segment_visualization=True
                        )
                        
                        segment_affinity_conv = None
                        segment_free_conv = None
                        
                        if isinstance(segment_output, list) and len(segment_output) == 2:
                            # These are the outputs with convolution applied
                            segment_affinity_conv = segment_output[0]
                            segment_free_conv = segment_output[1]
                    except (TypeError, AttributeError, RuntimeError) as e:
                        print(f"Segment visualization error: {str(e)}")
                        segment_affinity_conv = None
                        segment_free_conv = None
                    
                    # Create sample data with all components
                    # Use source_type from the loop variable if not returned from get_sample_by_index
                    if source_type is None:
                        source_type = source_type_key
                    
                    sample_data = {
                        'input': subvolume.cpu().numpy(),
                        'raw_splats': raw_splats[0].cpu().numpy(),
                        'output': reconstructed[0].cpu().numpy(),
                        'source_type': source_type
                    }
                    
                    # Now get the separate segments without convolution
                    try:
                        if hasattr(pl_module.model.decoder, 'affinity_segment_size'):
                            decoder = pl_module.model.decoder
                            
                            # Split latent vector into affinity and free segments
                            affinity_dims = decoder.affinity_segment_size
                            affinity_z = z[:, :affinity_dims]
                            free_z = z[:, affinity_dims:]
                            
                            # Process affinity segment (with pose)
                            affinity_centroids = decoder.affinity_centroids(affinity_z).view(z.shape[0], 3, -1)
                            affinity_weights = decoder.affinity_weights(affinity_z)
                            affinity_sigmas = decoder.affinity_sigmas(affinity_z)
                            
                            # Apply pose transformation to affinity segment
                            batch_size = z.shape[0]
                            if pose.shape[-1] == 1:
                                # Create the axis part of the pose
                                default_axis = decoder._default_axis
                                pose_expanded = torch.concat(
                                    [pose, torch.tile(default_axis, (batch_size, 1)).to(z.device)],
                                    axis=-1,
                                )
                            else:
                                pose_expanded = pose
                                
                            # Get rotation matrices
                            from cryolens.models.decoders.transforms import (
                                axis_angle_to_quaternion,
                                quaternion_to_rotation_matrix
                            )
                            quaternions = axis_angle_to_quaternion(pose_expanded, normalize=True)
                            rotation_matrices = quaternion_to_rotation_matrix(quaternions)
                            
                            # Apply rotation only to affinity splats
                            rotated_affinity_splats = torch.matmul(
                                rotation_matrices,
                                affinity_centroids,
                            )
                            
                            # Use only needed spatial dimensions
                            rotated_affinity_splats = rotated_affinity_splats[:, :decoder._ndim, :]
                            
                            # Scale for padding
                            padded_shape = tuple(s + 2 * decoder._padding for s in decoder._shape)
                            scale_factors = torch.tensor(
                                [s2/s1 for s1, s2 in zip(decoder._shape, padded_shape)],
                                device=z.device
                            )
                            rotated_affinity_splats = rotated_affinity_splats * scale_factors.view(1, -1, 1)
                            
                            # Render affinity splats only (no convolution)
                            affinity_splats = decoder._splatter(
                                rotated_affinity_splats, affinity_weights, affinity_sigmas,
                                splat_sigma_range=decoder._splat_sigma_range
                            )
                            
                            # Process free segment (no pose transformation)
                            free_centroids = decoder.free_centroids(free_z).view(z.shape[0], 3, -1)
                            free_weights = decoder.free_weights(free_z)
                            free_sigmas = decoder.free_sigmas(free_z)
                            
                            # Use only needed spatial dimensions 
                            free_splats = free_centroids[:, :decoder._ndim, :]
                            
                            # Scale for padding
                            free_splats = free_splats * scale_factors.view(1, -1, 1)
                            
                            # Render free splats only (no convolution)
                            free_splats_rendered = decoder._splatter(
                                free_splats, free_weights, free_sigmas,
                                splat_sigma_range=decoder._splat_sigma_range
                            )
                            
                            # Add segmented data to sample
                            sample_data['segment_affinity'] = affinity_splats[0].cpu().numpy()
                            sample_data['segment_free'] = free_splats_rendered[0].cpu().numpy()
                            
                            if segment_affinity_conv is not None:
                                sample_data['segment_affinity_conv'] = segment_affinity_conv[0].cpu().numpy()
                            if segment_free_conv is not None:
                                sample_data['segment_free_conv'] = segment_free_conv[0].cpu().numpy()
                    except Exception as e:
                        print(f"Error getting raw segmented splats: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                    # Store by molecule ID for grouping
                    all_samples[mol_id.item()].append(sample_data)
                    
        return all_samples
    
    def _save_numpy_arrays(self, all_samples, epoch, save_dir):
        """Save the raw 3D particle data as numpy arrays with source type information"""
        # Only rank 0 should save numpy arrays and create directories
        if self.rank != 0:
            return
            
        # Create the arrays directory if it doesn't exist
        arrays_dir = os.path.join(save_dir, "numpy_arrays")
        os.makedirs(arrays_dir, exist_ok=True)
        
        # Determine the array dimensions for each molecule
        for mol_id, samples in all_samples.items():
            # Skip if no samples
            if not samples:
                continue
                
            # Get sample dimensions from the first sample
            first_sample = samples[0]
            
            # Group samples by source type
            samples_by_source = defaultdict(list)
            for sample in samples:
                source_type = sample.get('source_type', 'unknown')
                samples_by_source[source_type].append(sample)
            
            # Check if we have segment visualizations available
            has_segments = 'segment_affinity' in first_sample and 'segment_free' in first_sample
            has_conv_segments = 'segment_affinity_conv' in first_sample
            
            # Get the target dimensions from the input (which is always correctly sized)
            target_shape = first_sample['input'].shape[1:]  # Should be (48, 48, 48)
            n_channels = first_sample['output'].shape[0]
            
            # Process each source type separately
            for source_type, source_samples in samples_by_source.items():
                # Count available samples
                n_samples = len(source_samples)
                
                # Initialize arrays with the target dimensions
                input_array = np.zeros((n_samples, first_sample['input'].shape[0]) + target_shape, dtype=np.float32)
                raw_splats_array = np.zeros((n_samples, first_sample['raw_splats'].shape[0]) + target_shape, dtype=np.float32)
                
                # For combined array, determine how many channels are needed
                n_segment_channels = 2  # Default: input + output
                if has_segments:
                    n_segment_channels += 2  # Add segment_affinity + segment_free
                if has_conv_segments:
                    n_segment_channels += 2  # Add segment_affinity_conv + segment_free_conv
                
                combined_array = np.zeros((n_samples, n_segment_channels, n_channels) + target_shape, dtype=np.float32)
                
                # Populate the arrays
                for i, sample in enumerate(source_samples):
                    # Handle input (should already be the target size)
                    input_array[i] = sample['input']
                    
                    # Handle raw_splats (check if it needs cropping)
                    raw_splat_shape = sample['raw_splats'].shape[1:]
                    if raw_splat_shape == target_shape:
                        # Already the right size, no cropping needed
                        raw_splats_array[i] = sample['raw_splats']
                    else:
                        # Needs cropping - calculate the right crop size
                        pad_each_side = [(dim_splat - dim_target) // 2 for dim_splat, dim_target in zip(raw_splat_shape, target_shape)]
                        
                        # Create slices for each dimension
                        slices = tuple(slice(pad, -pad if pad > 0 else None) for pad in pad_each_side)
                        raw_splats_array[i] = sample['raw_splats'][:, slices[0], slices[1], slices[2]]
                    
                    # Add to combined array
                    combined_array[i, 0] = sample['input']  # Input
                    
                    # Handle output (check if it needs cropping)
                    output_shape = sample['output'].shape[1:]
                    if output_shape == target_shape:
                        # Already the right size
                        combined_array[i, 1] = sample['output']
                    else:
                        # Needs cropping
                        pad_each_side = [(dim_out - dim_target) // 2 for dim_out, dim_target in zip(output_shape, target_shape)]
                        slices = tuple(slice(pad, -pad if pad > 0 else None) for pad in pad_each_side)
                        combined_array[i, 1] = sample['output'][:, slices[0], slices[1], slices[2]]
                    
                    # Handle segments if available
                    if has_segments:
                        # Process segment_affinity
                        affinity_shape = sample['segment_affinity'].shape[1:]
                        if affinity_shape == target_shape:
                            combined_array[i, 2] = sample['segment_affinity']
                        else:
                            pad_each_side = [(dim_aff - dim_target) // 2 for dim_aff, dim_target in zip(affinity_shape, target_shape)]
                            slices = tuple(slice(pad, -pad if pad > 0 else None) for pad in pad_each_side)
                            combined_array[i, 2] = sample['segment_affinity'][:, slices[0], slices[1], slices[2]]
                        
                        # Process segment_free
                        free_shape = sample['segment_free'].shape[1:]
                        if free_shape == target_shape:
                            combined_array[i, 3] = sample['segment_free']
                        else:
                            pad_each_side = [(dim_free - dim_target) // 2 for dim_free, dim_target in zip(free_shape, target_shape)]
                            slices = tuple(slice(pad, -pad if pad > 0 else None) for pad in pad_each_side)
                            combined_array[i, 3] = sample['segment_free'][:, slices[0], slices[1], slices[2]]
                    
                    # Handle conv segments if available
                    if has_conv_segments:
                        # Use appropriate index based on whether we have regular segments
                        idx_offset = 4 if has_segments else 2
                        
                        # Process segment_affinity_conv
                        if 'segment_affinity_conv' in sample:
                            affinity_conv_shape = sample['segment_affinity_conv'].shape[1:]
                            if affinity_conv_shape == target_shape:
                                combined_array[i, idx_offset] = sample['segment_affinity_conv']
                            else:
                                pad_each_side = [(dim_aff - dim_target) // 2 for dim_aff, dim_target in zip(affinity_conv_shape, target_shape)]
                                slices = tuple(slice(pad, -pad if pad > 0 else None) for pad in pad_each_side)
                                combined_array[i, idx_offset] = sample['segment_affinity_conv'][:, slices[0], slices[1], slices[2]]
                        
                        # Process segment_free_conv
                        if 'segment_free_conv' in sample:
                            free_conv_shape = sample['segment_free_conv'].shape[1:]
                            if free_conv_shape == target_shape:
                                combined_array[i, idx_offset + 1] = sample['segment_free_conv']
                            else:
                                pad_each_side = [(dim_free - dim_target) // 2 for dim_free, dim_target in zip(free_conv_shape, target_shape)]
                                slices = tuple(slice(pad, -pad if pad > 0 else None) for pad in pad_each_side)
                                combined_array[i, idx_offset + 1] = sample['segment_free_conv'][:, slices[0], slices[1], slices[2]]
                
                # Save the arrays for this molecule and source type
                mol_arrays_dir = os.path.join(arrays_dir, f"mol_{mol_id}")
                os.makedirs(mol_arrays_dir, exist_ok=True)
                
                # Include source type in filename to distinguish between sources
                source_suffix = source_type.replace(":", "_").replace("/", "_")
                np.save(os.path.join(mol_arrays_dir, f"input_{source_suffix}_epoch_{epoch}.npy"), input_array)
                np.save(os.path.join(mol_arrays_dir, f"raw_splats_{source_suffix}_epoch_{epoch}.npy"), raw_splats_array)
                np.save(os.path.join(mol_arrays_dir, f"combined_data_{source_suffix}_epoch_{epoch}.npy"), combined_array)
                
                # Save metadata about the arrays
                with open(os.path.join(mol_arrays_dir, f"metadata_{source_suffix}_epoch_{epoch}.txt"), "w") as f:
                    f.write(f"Input array shape: {input_array.shape}\n")
                    f.write(f"Raw splats array shape: {raw_splats_array.shape}\n")
                    f.write(f"Combined array shape: {combined_array.shape}\n")
                    f.write(f"Number of samples: {n_samples}\n")
                    f.write(f"Target dimensions: {target_shape}\n")
                    f.write(f"Source type: {source_type}\n")
                    f.write(f"Has segments: {has_segments}\n")
                    f.write(f"Has conv segments: {has_conv_segments}\n")
                    
                    # Document what's in each channel
                    channels_info = ["input", "combined_output"]
                    if has_segments:
                        channels_info.extend(["segment_affinity", "segment_free"])
                    if has_conv_segments:
                        channels_info.extend(["segment_affinity_conv", "segment_free_conv"])
                    
                    f.write(f"Combined channels: {channels_info}\n")
    
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.config.every_n_epochs != 0:
            return
            
        # Only rank 0 should generate visualizations
        if self.rank != 0 or trainer.global_rank != 0:
            return
            
        # Get dataloader
        dataloader = trainer.train_dataloader
        if isinstance(dataloader, list):
            dataloader = dataloader[0]
        
        # Get the dataset from the dataloader
        dataset = dataloader.dataset
        
        # Collect samples with segments using memoized indices
        all_samples = self._collect_samples_with_segments(dataset, pl_module)
        
        # Create visualizations directory
        viz_dir = os.path.join(pl_module.checkpoint_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Save numpy arrays if enabled
        if self.config.save_numpy_arrays:
            self._save_numpy_arrays(all_samples, trainer.current_epoch, viz_dir)
        
        # Only create the standard visualization to save time and disk space
        self._create_standard_visualization(all_samples, trainer.current_epoch, viz_dir, pl_module)
        
        # Skip source-specific visualizations to improve performance
        # self._create_source_specific_visualizations(all_samples, trainer.current_epoch, viz_dir, pl_module)
    
    def _create_standard_visualization(self, all_samples, epoch, viz_dir, pl_module):
        """Create a standard visualization with all samples"""
        # Skip if no samples collected
        if not all_samples or len(all_samples) == 0:
            print(f"Warning: No samples collected for visualization at epoch {epoch}")
            return
            
        # Create visualization
        fig, gs = self.plotter.setup_figure(len(all_samples))
        
        # Plot each molecule
        for mol_idx, (mol_id, samples) in enumerate(sorted(all_samples.items())):
            self.plotter.plot_molecule(fig, gs, samples, mol_idx, mol_id)
        
        # Add row descriptors on the left margin
        plt.figtext(0.01, 0.75, "Row 1: Input / Combined Splats / Background Splats", 
                  ha='left', va='center', fontsize=10, rotation=90)
        plt.figtext(0.01, 0.25, "Row 2: Affinity Splats / Combined Output / Affinity Output", 
                  ha='left', va='center', fontsize=10, rotation=90)
        
        # Save with minimal padding
        fig_path = os.path.join(viz_dir, f'3d_visualization_epoch_{epoch}.png')
        plt.savefig(fig_path, dpi=self.config.dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Log to MLflow
        try:
            if hasattr(pl_module, 'mlflow_logger') and pl_module.mlflow_logger is not None:
                pl_module.mlflow_logger.experiment.log_artifact(
                    pl_module.mlflow_logger.run_id,
                    fig_path
                )
        except Exception as e:
            print(f"Error logging visualization artifact: {str(e)}")
    
    def _create_source_specific_visualizations(self, all_samples, epoch, viz_dir, pl_module):
        """Create separate visualizations for each source type"""
        # Group samples by source type
        samples_by_source = defaultdict(lambda: defaultdict(list))
        
        for mol_id, samples in all_samples.items():
            for sample in samples:
                source_type = sample.get('source_type', 'unknown')
                samples_by_source[source_type][mol_id].append(sample)
        
        # Create a visualization for each source type
        for source_type, mol_samples in samples_by_source.items():
            # Skip if empty
            if not mol_samples:
                continue
                
            # Create a clean source name for the filename
            source_name = source_type.replace(":", "_").replace("/", "_")
            
            # Create visualization for this source type
            fig, gs = self.plotter.setup_figure(len(mol_samples))
            
            # Plot each molecule
            for mol_idx, (mol_id, samples) in enumerate(sorted(mol_samples.items())):
                self.plotter.plot_molecule(fig, gs, samples, mol_idx, mol_id)
            
            # Add row descriptors on the left margin
            plt.figtext(0.01, 0.75, "Row 1: Input / Combined Splats / Background Splats", 
                    ha='left', va='center', fontsize=10, rotation=90)
            plt.figtext(0.01, 0.25, "Row 2: Affinity Splats / Combined Output / Affinity Output", 
                    ha='left', va='center', fontsize=10, rotation=90)
            
            # Save with source type in filename
            fig_path = os.path.join(viz_dir, f'3d_visualization_{source_name}_epoch_{epoch}.png')
            plt.savefig(fig_path, dpi=self.config.dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            # Log to MLflow
            try:
                if hasattr(pl_module, 'mlflow_logger') and pl_module.mlflow_logger is not None:
                    pl_module.mlflow_logger.experiment.log_artifact(
                        pl_module.mlflow_logger.run_id,
                        fig_path
                    )
            except Exception as e:
                print(f"Error logging source-specific visualization: {str(e)}")
