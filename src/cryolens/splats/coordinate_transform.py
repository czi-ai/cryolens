"""
Coordinate transformation utilities for Gaussian splats.

This module provides functions to correctly transform Gaussian splat coordinates
from the decoder's coordinate system to the volume's coordinate system.

The key insight is that with meshgrid indexing="xy", there's a cyclic permutation:
- Splat X coordinate controls Y position in volume
- Splat Y coordinate controls Z position in volume  
- Splat Z coordinate controls X position in volume
"""

import numpy as np
from typing import Tuple, Optional


def transform_splat_coordinates(
    centroids: np.ndarray,
    volume_shape: Tuple[int, ...] = (48, 48, 48),
    renderer_padding: int = 9,
    apply_cyclic_permutation: bool = True
) -> np.ndarray:
    """
    Transform Gaussian splat coordinates from decoder space to volume space.
    
    This function handles the coordinate system mismatch between the decoder's
    output (which uses meshgrid with indexing="xy") and the numpy volume indexing.
    
    Parameters
    ----------
    centroids : np.ndarray
        Raw splat centroids from decoder, shape (num_splats, 3) or (batch, num_splats, 3)
        These are in normalized [-1, 1] space for the PADDED volume
    volume_shape : Tuple[int, ...]
        Original volume shape (before padding), typically (48, 48, 48)
    renderer_padding : int
        Padding used by the renderer (usually 9 for SegmentedGaussianSplatDecoder)
    apply_cyclic_permutation : bool
        Whether to apply the discovered cyclic permutation fix
        
    Returns
    -------
    np.ndarray
        Transformed centroids in volume voxel coordinates [Z, Y, X] for numpy indexing
    """
    # Handle batch dimension
    if centroids.ndim == 3:
        batch_size = centroids.shape[0]
        transformed_centroids = []
        for i in range(batch_size):
            transformed = transform_splat_coordinates(
                centroids[i], volume_shape, renderer_padding, apply_cyclic_permutation
            )
            transformed_centroids.append(transformed)
        return np.stack(transformed_centroids)
    
    # Calculate padded shape
    padded_shape = tuple(s + 2 * renderer_padding for s in volume_shape)
    
    # centroids are in normalized [-1, 1] space
    # Convert to padded voxel coordinates
    positions_padded = np.zeros_like(centroids)
    for i in range(3):
        positions_padded[:, i] = (centroids[:, i] + 1.0) * (padded_shape[i] / 2.0)
    
    # Subtract padding to get coordinates in original volume space
    positions_original = positions_padded - renderer_padding
    
    if apply_cyclic_permutation:
        # Apply the cyclic permutation discovered from diagnostics:
        # Splat coordinates [X, Y, Z] actually control [Y, Z, X] in the volume
        # 
        # For numpy/matplotlib [Z, Y, X] convention:
        # - Splat Y (index 1) -> Volume Z (position 0)
        # - Splat X (index 0) -> Volume Y (position 1)
        # - Splat Z (index 2) -> Volume X (position 2)
        positions_final = np.stack([
            positions_original[:, 1],  # Splat Y -> Volume Z
            positions_original[:, 0],  # Splat X -> Volume Y  
            positions_original[:, 2]   # Splat Z -> Volume X
        ], axis=1)
    else:
        # Direct mapping without permutation (original incorrect mapping)
        # This is kept for comparison/debugging purposes
        positions_final = np.stack([
            positions_original[:, 2],  # Z
            positions_original[:, 1],  # Y
            positions_original[:, 0]   # X
        ], axis=1)
    
    return positions_final


def extract_splat_positions_from_decoder(
    model: 'torch.nn.Module',
    z: 'torch.Tensor',
    pose: 'torch.Tensor',
    apply_coordinate_fix: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract splat positions from a CryoLens decoder with correct coordinate mapping.
    
    Parameters
    ----------
    model : torch.nn.Module
        CryoLens model with decoder
    z : torch.Tensor
        Latent encoding
    pose : torch.Tensor
        Pose parameters
    apply_coordinate_fix : bool
        Whether to apply the coordinate transformation fix
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (positions, weights) where positions are in volume voxel coordinates
    """
    import torch
    
    with torch.no_grad():
        decoder = model.decoder
        
        # Get splats from decoder
        if hasattr(decoder, 'affinity_segment_size'):
            # SegmentedGaussianSplatDecoder
            splats, weights, sigmas = decoder.decode_splats(z, pose, for_visualization=True)
        else:
            # Regular GaussianSplatDecoder
            splats, weights, sigmas = decoder.decode_splats(z, pose)
        
        # Get dimensions
        volume_shape = decoder._shape  # (48, 48, 48)
        renderer_padding = decoder._padding
        
        # splats are shape [batch_size, 3, num_splats] in normalized [-1, 1] space
        positions_norm = splats[0].cpu().numpy().T  # Transpose to [num_splats, 3]
        
        # Transform coordinates
        positions_final = transform_splat_coordinates(
            positions_norm,
            volume_shape,
            renderer_padding,
            apply_cyclic_permutation=apply_coordinate_fix
        )
        
        weights_np = weights[0].cpu().numpy().flatten()
        
        return positions_final, weights_np
