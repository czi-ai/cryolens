"""
Gaussian splat decoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List

from .base import BaseDecoder, SoftStep, Negate
from .splatting import GaussianSplatRenderer
from .utils import CartesianAxes, SpatialDims
from .transforms import (
    axis_angle_to_quaternion,
    quaternion_to_rotation_matrix
)


class GaussianSplatDecoder(BaseDecoder):
    """Differentiable Gaussian splat decoder with padding for convolution edge effects."""

    def __init__(
        self,
        shape: Tuple[int],
        *,
        n_splats: int = 128,
        latent_dims: int = 8,
        output_channels: Optional[int] = None,
        splat_sigma_range: Tuple[float, float] = (0.02, 0.1),
        default_axis: CartesianAxes = CartesianAxes.Z,
        device: torch.device = torch.device("cpu"),
        chunk_size: int = 0,
        padding: int = 4
    ):
        super().__init__()

        self._device = device
        self._shape = shape
        self._ndim = len(shape)
        self._output_channels = output_channels
        self._splat_sigma_range = splat_sigma_range
        self._default_axis = default_axis
        self._chunk_size = chunk_size
        self._padding = padding

        if len(shape) not in (SpatialDims.TWO, SpatialDims.THREE):
            raise ValueError("Only 2D or 3D rotations are currently supported")

        # Register networks and move to specified device
        self.centroids = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(n_splats * 3, n_splats * 3),
            torch.nn.Tanh(),
        ).to(device)

        self.weights = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats),
            torch.nn.Tanh(),
            SoftStep(k=10.0),
        ).to(device)

        self.sigmas = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats),
            torch.nn.Sigmoid(),
        ).to(device)

        # Calculate padded shape for renderer
        padded_shape = tuple(s + 2 * padding for s in shape)

        # Initialize renderer with padded shape
        if chunk_size == 0:
            self._splatter = GaussianSplatRenderer(
                padded_shape,
                device=device,
            ).to(device)
        else:
            self._splatter = ChunkedGaussianSplatRenderer(
                padded_shape,
                chunk_size=chunk_size,
                device=device,
            ).to(device)

        # Add final conv decoder if needed
        if output_channels is not None:
            conv = (
                torch.nn.Conv3d
                if self._ndim == SpatialDims.THREE
                else torch.nn.Conv2d
            )

            if self.use_histogram:
                # If using histograms, we'll create a modified decoder that incorporates histogram information
                self._decoder_init = torch.nn.Sequential(
                    Negate(),
                    conv(1, 1, kernel_size=1),
                ).to(device)
                
                # The final conv will use 1 + histogram_bins input channels instead of just 1
                self._decoder_final = conv(1 + self.histogram_bins, output_channels, kernel_size=9, padding="valid").to(device)
            else:
                # Standard decoder without histogram
                self._decoder = torch.nn.Sequential(
                    Negate(),
                    conv(1, 1, kernel_size=1),
                    conv(1, output_channels, kernel_size=9, padding="valid"),
                ).to(device)

    def configure_renderer(
        self,
        shape: Tuple[int],
        *,
        splat_sigma_range: Tuple[float, float] = (0.02, 0.1),
        default_axis: CartesianAxes = CartesianAxes.Z,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Reconfigure the renderer."""
        self._shape = shape
        self._default_axis = default_axis.as_tensor()
        padded_shape = tuple(s + 2 * self._padding for s in shape)
        self._splatter = GaussianSplatRenderer(
            padded_shape,
            device=device,
        )
        self._splat_sigma_range = splat_sigma_range

    def decode_splats(
        self, z: torch.Tensor, pose: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """Decode the splats to retrieve the coordinates, weights and sigmas."""
        if pose.shape[-1] not in (1, 4):
            raise ValueError(
                "Pose needs to be either a single angle rotation about the "
                "`default_axis` or a full angle-axis representation in 3D. "
            )

        # Ensure all network components are on the same device as input
        device = z.device
        if next(self.parameters()).device != device:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                print(f"Moving GaussianSplatDecoder components to {device}")
            self.to(device)
        
        # Ensure pose is on the same device as z
        if pose.device != device:
            pose = pose.to(device)
        
        splats = self.centroids(z).view(z.shape[0], 3, -1)
        weights = self.weights(z)
        sigmas = self.sigmas(z)

        # Get batch size
        batch_size = z.shape[0]

        # Handle single dimension pose
        if pose.shape[-1] == 1:
            # Ensure default_axis is on the correct device
            if isinstance(self._default_axis, torch.Tensor):
                default_axis = self._default_axis.to(device)
            else:
                default_axis = CartesianAxes.as_tensor(self._default_axis, device=device)
            
            pose = torch.concat(
                [
                    pose,
                    torch.tile(default_axis, (batch_size, 1)),
                ],
                axis=-1,
            )

        # Convert axis angles to quaternions
        assert pose.shape[-1] == 4, pose.shape
        quaternions = axis_angle_to_quaternion(pose, normalize=True)

        # Convert quaternions to rotation matrices
        rotation_matrices = quaternion_to_rotation_matrix(quaternions)

        # Use torch.matmul for more efficient and numerically stable matrix multiplication
        rotated_splats = torch.matmul(rotation_matrices, splats)

        # Use only the required spatial dimensions
        rotated_splats = rotated_splats[:, :self._ndim, :]

        # Scale splats to account for padding
        padded_shape = tuple(s + 2 * self._padding for s in self._shape)
        scale_factors = torch.tensor(
            [s2/s1 for s1, s2 in zip(self._shape, padded_shape)],
            device=device
        )
        rotated_splats = rotated_splats * scale_factors.view(1, -1, 1)

        return rotated_splats, weights, sigmas

    def forward(
        self,
        z: torch.Tensor,
        pose: torch.Tensor,
        global_weight: torch.Tensor,
        *,
        use_final_convolution: bool = True,
    ) -> torch.Tensor:
        """Decode the latents to an image volume given an explicit transform.

        Parameters
        ----------
        z : tensor
            An (N, D) tensor specifying the D dimensional latent encodings for
            the minibatch of N images.
        pose : tensor
            An (N, 1 | 4) tensor specifying the pose in terms of a single
            rotation (assumed around the z-axis) or a full axis-angle rotation.
        use_final_convolution: bool
            Whether to apply the final convolutional layers to recover the image.
            This can be useful to inspect the underlying structure in a trained
            model.

        Returns
        -------
        x : tensor
            The decoded image from the latents and pose.
        """
        # Ensure all model components are on the same device as input
        device = z.device
        if next(self.parameters()).device != device:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                print(f"Moving GaussianSplatDecoder to {device} in forward()")
            self.to(device)
            
        # Ensure pose is on the correct device
        if pose.device != device:
            pose = pose.to(device)
        if global_weight.device != device:
            global_weight = global_weight.to(device)

        # Decode the splats from the latents and pose
        splats, weights, sigmas = self.decode_splats(z, pose)

        # Apply global weighting to splats
        global_weight_expanded = global_weight.expand_as(weights)  # Broadcast to match weights
        scaled_weights = weights * torch.sigmoid(global_weight_expanded)  # Apply sigmoid for 0-1 range
        
        # Apply the gaussian splat renderer with scaled weights
        x = self._splatter(
            splats, scaled_weights, sigmas, splat_sigma_range=self._splat_sigma_range
        )

        # Apply final convolution if needed
        if self._output_channels is not None and use_final_convolution:
            x = self._decoder(x)

        return x


class SegmentedGaussianSplatDecoder(BaseDecoder):
    """Gaussian splat decoder with latent dimensions separated into two segments.
    
    This decoder divides the latent space into two segments based on the latent_ratio:
    1. The first segment corresponds to dimensions regularized by AffinityCosineLoss
    2. The second segment contains the remaining "free" dimensions
    
    Each segment generates its own set of Gaussians, which are combined before
    the final convolution is applied. The total number of Gaussians is distributed
    proportionally between the two segments based on the latent_ratio.
    
    Parameters
    ----------
    shape : tuple
        A tuple describing the output shape of the image data. Can be 2- or 3-
        dimensional. For example: (32, 32, 32)
    latent_dims : int
        The dimensions of the latent representation.
    n_splats : int
        The total number of Gaussians to generate, distributed proportionally between segments.
    output_channels : int, optional
        The number of output channels in the final image volume.
    splat_sigma_range : tuple[float]
        The minimum and maximum sigma values for each splat.
    latent_ratio : float
        The ratio of latent dimensions used by AffinityCosineLoss. This defines
        the split between the two segments and how splats are distributed.
    default_axis : CartesianAxes
        Default cartesian axis for rotation.
    device : torch.device
        The device to use for computation.
    padding : int
        Padding for convolution edge effects.
    use_histogram : bool
        Whether to use histogram information during the final convolution.
    histogram_bins : int
        Number of bins to use for the histogram.
    """
    
    def __init__(
        self,
        shape: tuple,
        latent_dims: int = 8,
        n_splats: int = 128,
        output_channels: int = None,
        splat_sigma_range: tuple = (0.005, 0.1),  
        latent_ratio: float = 0.75,  
        default_axis: CartesianAxes = CartesianAxes.Z,
        device: torch.device = torch.device("cpu"),
        padding: int = 9,
        use_histogram: bool = False,
        histogram_bins: int = 32
    ):
        super().__init__()
        
        self._device = device
        self._shape = shape
        self._ndim = len(shape)
        self._output_channels = output_channels
        self._splat_sigma_range = splat_sigma_range
        
        # Store default_axis as tensor - will be moved to the correct device as needed
        self._default_axis = CartesianAxes.as_tensor(default_axis, device=device)
            
        self._padding = padding
        self.latent_ratio = latent_ratio
        self.use_histogram = use_histogram
        self.histogram_bins = histogram_bins
        
        # Define segment sizes based on latent_ratio
        self.affinity_segment_size = int(latent_dims * latent_ratio)  # Dimensions controlled by AffinityCosineLoss
        self.free_segment_size = latent_dims - self.affinity_segment_size  # Remaining free dimensions
        
        # Distribute splats proportionally based on latent_ratio
        self.affinity_n_splats = int(n_splats * latent_ratio)
        self.free_n_splats = n_splats - self.affinity_n_splats
                
        # Create separate networks for the two segments
        
        # Networks for affinity-regularized segment - 3 layers each
        self.affinity_centroids = nn.Sequential(
            nn.Linear(self.affinity_segment_size, self.affinity_n_splats * 4),
            nn.ReLU(),
            nn.Linear(self.affinity_n_splats * 4, self.affinity_n_splats * 3),
            nn.ReLU(),
            nn.Linear(self.affinity_n_splats * 3, self.affinity_n_splats * 3),
            nn.Tanh(),
        ).to(device)
        
        self.affinity_weights = nn.Sequential(
            nn.Linear(self.affinity_segment_size, self.affinity_n_splats * 2),
            nn.ReLU(),
            nn.Linear(self.affinity_n_splats * 2, self.affinity_n_splats),
            nn.Tanh(),
            SoftStep(k=10.0),
        ).to(device)
        
        self.affinity_sigmas = nn.Sequential(
            nn.Linear(self.affinity_segment_size, self.affinity_n_splats * 2),
            nn.ReLU(),
            nn.Linear(self.affinity_n_splats * 2, self.affinity_n_splats),
            nn.Sigmoid(),
        ).to(device)
        
        # Networks for free segment - 3 layers each
        self.free_centroids = nn.Sequential(
            nn.Linear(self.free_segment_size, self.free_n_splats * 4),
            nn.ReLU(),
            nn.Linear(self.free_n_splats * 4, self.free_n_splats * 3),
            nn.ReLU(),
            nn.Linear(self.free_n_splats * 3, self.free_n_splats * 3),
            nn.Tanh(),
        ).to(device)
        
        self.free_weights = nn.Sequential(
            nn.Linear(self.free_segment_size, self.free_n_splats * 2),
            nn.ReLU(),
            nn.Linear(self.free_n_splats * 2, self.free_n_splats),
            nn.Tanh(),
            SoftStep(k=10.0),
        ).to(device)
        
        self.free_sigmas = nn.Sequential(
            nn.Linear(self.free_segment_size, self.free_n_splats * 2),
            nn.ReLU(),
            nn.Linear(self.free_n_splats * 2, self.free_n_splats),
            nn.Sigmoid(),
        ).to(device)
            
        # Calculate padded shape for renderer
        padded_shape = tuple(s + 2 * padding for s in shape)
            
        # Initialize renderer with padded shape
        self._splatter = GaussianSplatRenderer(
            padded_shape,
            device=device,
        ).to(device)
            
        # Add final conv decoder if needed
        if output_channels is not None:
            conv = (
                torch.nn.Conv3d
                if self._ndim == SpatialDims.THREE
                else torch.nn.Conv2d
            )

            if self.use_histogram:
                # If using histograms, we'll create a modified decoder that incorporates histogram information
                self._decoder_init = torch.nn.Sequential(
                    Negate(),
                    conv(1, 1, kernel_size=1),
                ).to(device)
                
                # The final conv will use 1 + histogram_bins input channels instead of just 1
                self._decoder_final = conv(1 + self.histogram_bins, output_channels, kernel_size=9, padding="valid").to(device)
            else:
                # Standard decoder without histogram
                self._decoder = torch.nn.Sequential(
                    Negate(),
                    conv(1, 1, kernel_size=1),
                    conv(1, output_channels, kernel_size=9, padding="valid"),
                ).to(device)
            
    def decode_splats(
        self, z: torch.Tensor, pose: torch.Tensor, global_weight: Optional[torch.Tensor] = None,
        for_visualization: bool = False
    ) -> tuple[torch.Tensor]:
        """Decode the splats to retrieve the coordinates, weights and sigmas from both segments.
        
        Only applies pose transformation to splats from the affinity segment.
        The free segment's splats remain static (no pose transformation).
        
        Note on coordinate system:
        The returned splat coordinates are in the renderer's coordinate system [X, Y, Z].
        However, volume indexing follows NumPy convention: volume[Z, Y, X].
        For visualization, you may need to remap coordinates as [2, 0, 1] to align
        splats with volume slices.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape (batch_size, latent_dims)
        pose : torch.Tensor
            Pose tensor of shape (batch_size, 1 | 4)
        global_weight : torch.Tensor, optional
            Global weight tensor of shape (batch_size, 1) for amplitude scaling
        for_visualization : bool, optional
            If True, only apply renderer padding (not convolution padding) for raw splat visualization
            
        Returns
        -------
        tuple[torch.Tensor]
            Tuple of (splats, weights, sigmas) tensors
        """
        if pose.shape[-1] not in (1, 4):
            raise ValueError(
                "Pose needs to be either a single angle rotation about the "
                "`default_axis` or a full angle-axis representation in 3D. "
            )

        # Move inputs to correct device
        device = z.device
        
        # Ensure all network components are on the same device as the input
        if next(self.parameters()).device != device:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                print(f"Moving SegmentedGaussianSplatDecoder components to {device}")
            self.to(device)
        
        # Ensure pose is on the same device as z
        if pose.device != device:
            pose = pose.to(device)
        
        # Ensure global_weight is on the same device as z if provided
        if global_weight is not None and global_weight.device != device:
            global_weight = global_weight.to(device)
        
        # Split latent vector into two segments
        affinity_z = z[:, :self.affinity_segment_size]  # Regularized by affinity loss
        free_z = z[:, self.affinity_segment_size:]      # Free segment
        
        # Generate parameters for affinity segment (will have pose applied)
        affinity_splats = self.affinity_centroids(affinity_z).view(z.shape[0], 3, -1)
        affinity_weights = self.affinity_weights(affinity_z)
        affinity_sigmas = self.affinity_sigmas(affinity_z)
        
        # Generate parameters for free segment (no pose)
        free_splats = self.free_centroids(free_z).view(z.shape[0], 3, -1)
        free_weights = self.free_weights(free_z)
        free_sigmas = self.free_sigmas(free_z)
        
        # Get batch size
        batch_size = z.shape[0]

        # Handle single dimension pose for affinity segment
        if pose.shape[-1] == 1:
            # Create the axis part of the pose using the cached default_axis
            # Ensure default_axis is on the correct device
            default_axis = self._default_axis.to(device)
            pose_expanded = torch.concat(
                [
                    pose,
                    torch.tile(default_axis, (batch_size, 1)),
                ],
                axis=-1,
            )
        else:
            pose_expanded = pose

        # Convert axis angles to quaternions
        quaternions = axis_angle_to_quaternion(pose_expanded, normalize=True)

        # Convert quaternions to rotation matrices
        rotation_matrices = quaternion_to_rotation_matrix(quaternions)

        # Use torch.matmul for more efficient and numerically stable matrix multiplication
        # ONLY rotate the affinity segment splats
        rotated_affinity_splats = torch.matmul(
            rotation_matrices,
            affinity_splats,
        )

        # Use only the required spatial dimensions for both segments
        rotated_affinity_splats = rotated_affinity_splats[:, :self._ndim, :]
        free_splats = free_splats[:, :self._ndim, :]
        
        # Scale for padding
        # When used for visualization (raw splats without convolution), only use renderer padding
        # When used in forward pass with convolution, include convolution padding
        if for_visualization:
            # Only renderer padding for raw splat visualization
            padded_shape = tuple(s + 2 * self._padding for s in self._shape)
        else:
            # Include convolution padding for the full forward pass
            conv_padding = 4  # For 9x9 kernel with padding="valid"
            total_padding = self._padding + conv_padding
            padded_shape = tuple(s + 2 * total_padding for s in self._shape)
        scale_factors = torch.tensor(
            [s2/s1 for s1, s2 in zip(self._shape, padded_shape)],
            device=device  # Use the same device as input tensors
        )
        
        # Apply scaling to both segments
        rotated_affinity_splats = rotated_affinity_splats * scale_factors.view(1, -1, 1)
        free_splats = free_splats * scale_factors.view(1, -1, 1)
        
        # Now concatenate the pose-transformed affinity splats with the untransformed free splats
        splats = torch.cat([rotated_affinity_splats, free_splats], dim=2)
        weights = torch.cat([affinity_weights, free_weights], dim=1)
        sigmas = torch.cat([affinity_sigmas, free_sigmas], dim=1)

        # Apply global weight scaling if provided
        if global_weight is not None:
            # Apply sigmoid to ensure global_weight is in [0, 1] range
            # Shape: (batch_size, 1) -> (batch_size, total_splats)
            global_weight_sigmoid = torch.sigmoid(global_weight)
            global_weight_expanded = global_weight_sigmoid.expand_as(weights)
            weights = weights * global_weight_expanded

        return splats, weights, sigmas
    
    def forward(
        self,
        z: torch.Tensor,
        pose: torch.Tensor,
        global_weight: torch.Tensor,
        *,
        use_final_convolution: bool = True,
        segment_visualization: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Decode the latents to an image volume.
        
        Parameters
        ----------
        z : tensor
            An (N, D) tensor specifying the D dimensional latent encodings.
        pose : tensor
            An (N, 1 | 4) tensor specifying the pose (only applied to affinity segment).
        global_weight : tensor
            An (N, 1) tensor specifying global amplitude scaling for all splats.
        use_final_convolution: bool
            Whether to apply the final convolutional layers.
        segment_visualization: bool
            If True, returns separate outputs for affinity and free segments for visualization.
        """
        # Ensure all model components are on the same device as input
        device = z.device
        if next(self.parameters()).device != device:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                print(f"Moving SegmentedGaussianSplatDecoder to {device} in forward()")
            self.to(device)
            
        # Ensure pose and global_weight are on the correct device
        if pose.device != device:
            pose = pose.to(device)
        if global_weight.device != device:
            global_weight = global_weight.to(device)
        
        # Calculate total padding needed
        conv_padding = 4  # For the 9x9 convolution
        total_padding = self._padding + conv_padding
        
        # Calculate expanded shape for rendering
        expanded_shape = tuple(s + 2 * total_padding for s in self._shape)
        
        # Store original shape for later restoration
        original_shape = self._shape
        
        # Reconfigure renderer for expanded shape if needed
        # The renderer needs to match the actual rendering dimensions
        if self._splatter._shape != expanded_shape:
            self._splatter = GaussianSplatRenderer(
                expanded_shape,
                device=device,
            ).to(device)
        
        if segment_visualization:
            # Generate and render separate outputs for each segment
            segment_outputs = []
            
            # Split latent vector into two segments
            affinity_z = z[:, :self.affinity_segment_size]
            free_z = z[:, self.affinity_segment_size:]
            
            # Process affinity segment with pose
            affinity_splats = self.affinity_centroids(affinity_z).view(z.shape[0], 3, -1)
            affinity_weights = self.affinity_weights(affinity_z)
            affinity_sigmas = self.affinity_sigmas(affinity_z)
            
            # Get batch size
            batch_size = z.shape[0]

            # Handle single dimension pose for affinity segment
            if pose.shape[-1] == 1:
                # Ensure default_axis is on the correct device
                default_axis = self._default_axis.to(device)
                pose_expanded = torch.concat(
                    [
                        pose,
                        torch.tile(default_axis, (batch_size, 1)),
                    ],
                    axis=-1,
                )
            else:
                pose_expanded = pose

            # Convert axis angles to quaternions
            quaternions = axis_angle_to_quaternion(pose_expanded, normalize=True)
            
            # Convert quaternions to rotation matrices
            rotation_matrices = quaternion_to_rotation_matrix(quaternions)
            
            # Rotate the affinity segment points
            rotated_affinity_splats = torch.matmul(
                rotation_matrices,
                affinity_splats,
            )
            
            # Use only the required spatial dimensions
            rotated_affinity_splats = rotated_affinity_splats[:, :self._ndim, :]
            
            # Scale for padding - use the device of the input tensors
            scale_factors = torch.tensor(
                [s2/s1 for s1, s2 in zip(original_shape, expanded_shape)],
                device=device
            )
            rotated_affinity_splats = rotated_affinity_splats * scale_factors.view(1, -1, 1)
            
            # Apply global weight scaling to affinity weights
            global_weight_sigmoid = torch.sigmoid(global_weight)
            affinity_global_weight = global_weight_sigmoid.expand(batch_size, self.affinity_n_splats)
            affinity_weights_scaled = affinity_weights * affinity_global_weight
            
            # Render affinity segment
            affinity_output = self._splatter(
                rotated_affinity_splats, affinity_weights_scaled, affinity_sigmas, 
                splat_sigma_range=self._splat_sigma_range
            )
            
            # Process free segment WITHOUT pose
            free_splats = self.free_centroids(free_z).view(z.shape[0], 3, -1)
            free_weights = self.free_weights(free_z)
            free_sigmas = self.free_sigmas(free_z)
            
            # Use only required spatial dimensions for free segment
            free_splats = free_splats[:, :self._ndim, :]
            
            # Scale for padding (free segment)
            free_splats = free_splats * scale_factors.view(1, -1, 1)
            
            # Apply global weight scaling to free weights
            free_global_weight = global_weight_sigmoid.expand(batch_size, self.free_n_splats)
            free_weights_scaled = free_weights * free_global_weight
            
            # Render free segment
            free_output = self._splatter(
                free_splats, free_weights_scaled, free_sigmas,
                splat_sigma_range=self._splat_sigma_range
            )
            
            # Reset shape
            self._shape = original_shape
            
            # Apply convolution if requested
            if use_final_convolution and self._output_channels is not None:
                # Apply separately to each segment for visualization
                if self.use_histogram and hasattr(self, '_decoder_init') and hasattr(self, '_decoder_final'):
                    # Process affinity output with histogram
                    affinity_init = self._decoder_init(affinity_output)
                    affinity_hist = self._compute_histograms(affinity_output, self.histogram_bins)
                    affinity_with_hist = torch.cat([affinity_init, affinity_hist], dim=1)
                    affinity_output = self._decoder_final(affinity_with_hist)
                    
                    # Process free output with histogram
                    free_init = self._decoder_init(free_output)
                    free_hist = self._compute_histograms(free_output, self.histogram_bins)
                    free_with_hist = torch.cat([free_init, free_hist], dim=1)
                    free_output = self._decoder_final(free_with_hist)
                else:
                    # Standard path without histogram
                    affinity_output = self._decoder(affinity_output)
                    free_output = self._decoder(free_output)
                
                # Calculate crop to match input shape exactly
                # The convolution output includes padding from both rendering and convolution
                affinity_spatial_shape = affinity_output.shape[2:]
                free_spatial_shape = free_output.shape[2:]
                
                # Calculate cropping for affinity output
                affinity_crop_start = tuple((s - t) // 2 for s, t in zip(affinity_spatial_shape, original_shape))
                affinity_crop_end = tuple(start + size for start, size in zip(affinity_crop_start, original_shape))
                affinity_slices = tuple(slice(start, end) for start, end in zip(affinity_crop_start, affinity_crop_end))
                affinity_output = affinity_output[(slice(None), slice(None)) + affinity_slices]
                
                # Calculate cropping for free output
                free_crop_start = tuple((s - t) // 2 for s, t in zip(free_spatial_shape, original_shape))
                free_crop_end = tuple(start + size for start, size in zip(free_crop_start, original_shape))
                free_slices = tuple(slice(start, end) for start, end in zip(free_crop_start, free_crop_end))
                free_output = free_output[(slice(None), slice(None)) + free_slices]
                
                # Verify output shape
                expected_shape = (affinity_output.shape[0], self._output_channels) + original_shape
                assert affinity_output.shape == expected_shape, f"Output shape {affinity_output.shape} != expected {expected_shape}"
                assert free_output.shape == expected_shape, f"Output shape {free_output.shape} != expected {expected_shape}"
            
            segment_outputs = [affinity_output, free_output]
            return segment_outputs
            
        else:
            # Regular path: combine all splats with appropriate pose application and global weight scaling
            # Note: decode_splats already uses the correct expanded shape scaling
            splats, weights, sigmas = self.decode_splats(z, pose, global_weight)
            
            # Ensure renderer is configured for expanded shape
            if self._splatter._shape != expanded_shape:
                self._splatter = GaussianSplatRenderer(
                    expanded_shape,
                    device=device,
                ).to(device)
            
            # Apply the gaussian splat renderer
            x = self._splatter(
                splats, weights, sigmas, splat_sigma_range=self._splat_sigma_range
            )
            
            # Apply final convolution if needed
            if self._output_channels is not None and use_final_convolution:
                if self.use_histogram and hasattr(self, '_decoder_init') and hasattr(self, '_decoder_final'):
                    # Apply the initial part of the decoder
                    x_init = self._decoder_init(x)
                    
                    # Compute histograms of the input volume
                    histograms = self._compute_histograms(x, self.histogram_bins)
                    
                    # Concatenate initial feature with histogram features
                    x_with_hist = torch.cat([x_init, histograms], dim=1)
                    
                    # Apply the final convolution with the combined features
                    x = self._decoder_final(x_with_hist)
                else:
                    # Standard path without histogram
                    x = self._decoder(x)
                
                # Calculate crop to match input shape exactly
                # The convolution output includes padding from both rendering and convolution
                spatial_shape = x.shape[2:]
                
                # Calculate cropping to center the result
                crop_start = tuple((s - t) // 2 for s, t in zip(spatial_shape, original_shape))
                crop_end = tuple(start + size for start, size in zip(crop_start, original_shape))
                slices = tuple(slice(start, end) for start, end in zip(crop_start, crop_end))
                x = x[(slice(None), slice(None)) + slices]
                
                # Verify output shape
                expected_shape = (x.shape[0], self._output_channels) + original_shape
                assert x.shape == expected_shape, f"Output shape {x.shape} != expected {expected_shape}"
                
                return x
            else:
                return x
    
    def _compute_histograms(self, x: torch.Tensor, num_bins: int = 32) -> torch.Tensor:
        """Compute the histogram of values in the input volume.
        
        This creates a channel-wise histogram representation that can be used as
        additional context for the final convolution layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, *spatial_dims)
        num_bins : int
            Number of histogram bins
            
        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, num_bins, *spatial_dims) containing spatial
            histograms where each spatial position has a local histogram computed
            from a neighborhood.
        """
        batch_size, channels, *spatial_dims = x.shape
        device = x.device
        
        # Define min and max values for histogram bins
        # Use a fixed range of [-1, 1] to match the common range of normalized data
        min_val, max_val = -1.0, 1.0
        
        # Create histogram bins
        bins = torch.linspace(min_val, max_val, num_bins+1, device=device)
        bin_width = (max_val - min_val) / num_bins
        
        # Initialize output tensor for histograms
        # Each spatial location will have a histogram
        histograms = torch.zeros(batch_size, num_bins, *spatial_dims, device=device)
        
        # Create a sliding window view for efficient computation
        # For 3D data, we'll use a 3x3x3 neighborhood
        # For 2D data, we'll use a 3x3 neighborhood
        kernel_size = 3
        padding = kernel_size // 2
        
        # Pad the input for sliding window
        if len(spatial_dims) == 3:
            # 3D case
            padded_x = F.pad(x, (padding, padding, padding, padding, padding, padding))
            
            # For each bin, compute the count of values falling within that bin
            for b in range(num_bins):
                # Lower and upper bounds for this bin
                lower = bins[b]
                upper = bins[b+1]
                
                # Create binary mask for values in this bin
                if b == num_bins - 1:  # Include the upper bound in the last bin
                    bin_mask = (padded_x >= lower) & (padded_x <= upper)
                else:
                    bin_mask = (padded_x >= lower) & (padded_x < upper)
                
                # Convert to float for convolution
                bin_mask = bin_mask.float()
                
                # Use 3D convolution to count occurrences in the neighborhood
                # Create a kernel of ones for counting
                kernel = torch.ones(1, channels, kernel_size, kernel_size, kernel_size, device=device)
                
                # Apply convolution to count values in each neighborhood
                bin_counts = F.conv3d(bin_mask, kernel, padding=0)
                
                # Normalize by the total possible counts
                max_counts = channels * kernel_size**3
                bin_counts = bin_counts / max_counts
                
                # Store in the output tensor
                histograms[:, b] = bin_counts.squeeze(1)
                
        elif len(spatial_dims) == 2:
            # 2D case
            padded_x = F.pad(x, (padding, padding, padding, padding))
            
            # For each bin, compute the count of values falling within that bin
            for b in range(num_bins):
                # Lower and upper bounds for this bin
                lower = bins[b]
                upper = bins[b+1]
                
                # Create binary mask for values in this bin
                if b == num_bins - 1:  # Include the upper bound in the last bin
                    bin_mask = (padded_x >= lower) & (padded_x <= upper)
                else:
                    bin_mask = (padded_x >= lower) & (padded_x < upper)
                
                # Convert to float for convolution
                bin_mask = bin_mask.float()
                
                # Use 2D convolution to count occurrences in the neighborhood
                # Create a kernel of ones for counting
                kernel = torch.ones(1, channels, kernel_size, kernel_size, device=device)
                
                # Apply convolution to count values in each neighborhood
                bin_counts = F.conv2d(bin_mask, kernel, padding=0)
                
                # Normalize by the total possible counts
                max_counts = channels * kernel_size**2
                bin_counts = bin_counts / max_counts
                
                # Store in the output tensor
                histograms[:, b] = bin_counts.squeeze(1)
        
        return histograms
    
    def configure_renderer(
        self,
        shape: tuple,
        *,
        splat_sigma_range: tuple = (0.005, 0.1),
        default_axis: CartesianAxes = CartesianAxes.Z,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Reconfigure the renderer."""
        self._shape = shape
        
        # Store default_axis as tensor
        self._default_axis = CartesianAxes.as_tensor(default_axis, device=device)
            
        padded_shape = tuple(s + 2 * self._padding for s in shape)
        self._splatter = GaussianSplatRenderer(
            padded_shape,
            device=device,
        )
        self._splat_sigma_range = splat_sigma_range