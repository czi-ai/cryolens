"""
Gaussian splat extraction utilities for CryoLens models.

This module provides functions for extracting and manipulating Gaussian splat
parameters from CryoLens VAE decoders.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def extract_gaussian_splats(
    model: torch.nn.Module,
    volumes: np.ndarray,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    batch_size: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Gaussian splat parameters from the VAE decoder.
    
    Parameters
    ----------
    model : torch.nn.Module
        CryoLens VAE model
    volumes : np.ndarray
        Input volumes of shape (N, D, H, W)
    config : Dict[str, Any]
        Model configuration
    device : torch.device, optional
        Device for computation
    batch_size : int
        Batch size for processing
    
    Returns
    -------
    Tuple containing:
        - centroids: (N, num_splats, 3) - 3D positions of splats
        - sigmas: (N, num_splats) - standard deviations
        - weights: (N, num_splats) - weights/amplitudes
        - embeddings: (N, latent_dims) - latent embeddings
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_centroids = []
    all_sigmas = []
    all_weights = []
    all_embeddings = []
    
    # Get normalization method
    norm_method = config.get('normalization', 'z-score')
    if hasattr(model, '_normalization_method'):
        norm_method = model._normalization_method
    
    with torch.no_grad():
        for i in range(0, len(volumes), batch_size):
            batch_volumes = volumes[i:i+batch_size]
            
            # Normalize volumes
            from cryolens.utils.normalization import normalize_volume
            normalized_batch = []
            for vol in batch_volumes:
                vol_norm = normalize_volume(vol, method=norm_method)
                normalized_batch.append(vol_norm)
            
            # Convert to tensor
            batch_tensor = torch.tensor(np.stack(normalized_batch), dtype=torch.float32)
            batch_tensor = batch_tensor.unsqueeze(1).to(device)
            
            # Encode
            mu, log_var, pose, global_weight = model.encode(batch_tensor)
            
            # Use identity pose to get canonical splats
            identity_pose = torch.zeros(batch_tensor.shape[0], 4, device=device)
            identity_pose[:, 0] = 1.0  # Set quaternion w component
            identity_global_weight = torch.ones(batch_tensor.shape[0], 1, device=device)
            
            # Extract splat parameters from decoder
            decoder = model.decoder
            
            # Check if segmented decoder
            if hasattr(decoder, 'affinity_centroids'):
                # SegmentedGaussianSplatDecoder
                centroids, sigmas, weights = _extract_segmented_splats(
                    decoder, mu, identity_pose, identity_global_weight
                )
            else:
                # Standard GaussianSplatDecoder
                centroids, sigmas, weights = _extract_standard_splats(
                    decoder, mu, identity_pose, identity_global_weight
                )
            
            # Store results
            all_centroids.append(centroids.cpu().numpy())
            all_sigmas.append(sigmas.cpu().numpy())
            all_weights.append(weights.cpu().numpy())
            all_embeddings.append(mu.cpu().numpy())
            
            logger.debug(f"Extracted splats from batch {i//batch_size + 1}/{(len(volumes) + batch_size - 1)//batch_size}")
    
    # Concatenate results
    centroids = np.concatenate(all_centroids, axis=0)
    sigmas = np.concatenate(all_sigmas, axis=0)
    weights = np.concatenate(all_weights, axis=0)
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Reshape centroids to (N, num_splats, 3) for 3D positions
    n_samples = centroids.shape[0]
    num_splats = sigmas.shape[1]
    centroids = centroids.reshape(n_samples, num_splats, 3)
    
    # The centroids from the networks are in normalized [-1, 1] space
    # Convert to voxel coordinates [0, 48] for a standard 48x48x48 volume
    decoder = model.decoder
    volume_shape = decoder._shape if hasattr(decoder, '_shape') else (48, 48, 48)
    
    # Convert from normalized to voxel space
    for i in range(3):
        centroids[:, :, i] = (centroids[:, :, i] + 1.0) * (volume_shape[i] / 2.0)
    
    # Apply coordinate permutation fix
    # The meshgrid with indexing="xy" creates a cyclic permutation:
    # Splat X -> Volume Y, Splat Y -> Volume Z, Splat Z -> Volume X
    centroids_fixed = centroids.copy()
    centroids_fixed[:, :, 0] = centroids[:, :, 1]  # Splat Y -> dim 0 (Z in volume)
    centroids_fixed[:, :, 1] = centroids[:, :, 0]  # Splat X -> dim 1 (Y in volume)
    centroids_fixed[:, :, 2] = centroids[:, :, 2]  # Splat Z -> dim 2 (X in volume)
    
    return centroids_fixed, sigmas, weights, embeddings


def _extract_segmented_splats(
    decoder: torch.nn.Module,
    mu: torch.Tensor,
    pose: torch.Tensor,
    global_weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract splats from SegmentedGaussianSplatDecoder."""
    # Split latent based on decoder's configuration
    affinity_segment_size = decoder.affinity_segment_size
    free_segment_size = decoder.free_segment_size
    
    mu_affinity = mu[:, :affinity_segment_size]
    mu_free = mu[:, affinity_segment_size:]
    
    # Get affinity centroids
    affinity_centroids = decoder.affinity_centroids(mu_affinity)
    free_centroids = decoder.free_centroids(mu_free)
    centroids = torch.cat([affinity_centroids, free_centroids], dim=1)
    
    # Get sigmas
    affinity_sigmas = decoder.affinity_sigmas(mu_affinity)
    free_sigmas = decoder.free_sigmas(mu_free)
    sigmas = torch.cat([affinity_sigmas, free_sigmas], dim=1)
    
    # Get weights
    affinity_weights = decoder.affinity_weights(mu_affinity)
    free_weights = decoder.free_weights(mu_free)
    weights = torch.cat([affinity_weights, free_weights], dim=1)
    
    # Apply global weight scaling
    weights = weights * global_weight
    
    return centroids, sigmas, weights


def _extract_standard_splats(
    decoder: torch.nn.Module,
    mu: torch.Tensor,
    pose: torch.Tensor,
    global_weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract splats from standard GaussianSplatDecoder."""
    centroids = decoder.centroids(mu)
    sigmas = decoder.sigmas(mu)
    weights = decoder.weights(mu)
    
    # Apply global weight scaling
    weights = weights * global_weight
    
    return centroids, sigmas, weights


def render_gaussian_splats(
    centroids: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray,
    volume_shape: Tuple[int, int, int] = (48, 48, 48),
    apply_rotation: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Render Gaussian splats to a volume.
    
    Parameters
    ----------
    centroids : np.ndarray
        Splat positions of shape (num_splats, 3) or (N, num_splats, 3)
    sigmas : np.ndarray
        Splat standard deviations of shape (num_splats,) or (N, num_splats)
    weights : np.ndarray
        Splat weights of shape (num_splats,) or (N, num_splats)
    volume_shape : Tuple[int, int, int]
        Shape of output volume
    apply_rotation : np.ndarray, optional
        Rotation matrix to apply to centroids
        
    Returns
    -------
    np.ndarray
        Rendered volume(s)
    """
    # Handle batch dimension
    if centroids.ndim == 3:
        # Batch processing
        batch_size = centroids.shape[0]
        volumes = []
        for i in range(batch_size):
            vol = render_gaussian_splats(
                centroids[i],
                sigmas[i] if sigmas.ndim > 1 else sigmas,
                weights[i] if weights.ndim > 1 else weights,
                volume_shape,
                apply_rotation
            )
            volumes.append(vol)
        return np.stack(volumes)
    
    # Apply rotation if provided
    if apply_rotation is not None:
        centroids = (apply_rotation @ centroids.T).T
    
    # Create coordinate grid
    grid_coords = np.meshgrid(
        np.arange(volume_shape[0]),
        np.arange(volume_shape[1]),
        np.arange(volume_shape[2]),
        indexing='ij'
    )
    grid = np.stack(grid_coords, axis=-1)  # Shape: (D, H, W, 3)
    
    # Initialize volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    # Render each splat
    for i in range(len(centroids)):
        # Compute distances from grid points to splat center
        distances = np.linalg.norm(grid - centroids[i], axis=-1)
        
        # Compute Gaussian
        gaussian = weights[i] * np.exp(-0.5 * (distances / sigmas[i]) ** 2)
        
        # Add to volume
        volume += gaussian
    
    return volume


def compare_splat_sets(
    splats1: Tuple[np.ndarray, np.ndarray, np.ndarray],
    splats2: Tuple[np.ndarray, np.ndarray, np.ndarray],
    weight_threshold: float = 0.01
) -> Dict[str, float]:
    """
    Compare two sets of Gaussian splats.
    
    Parameters
    ----------
    splats1 : Tuple[np.ndarray, np.ndarray, np.ndarray]
        First set of splats (centroids, sigmas, weights)
    splats2 : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Second set of splats (centroids, sigmas, weights)
    weight_threshold : float
        Minimum weight to consider splat significant
        
    Returns
    -------
    Dict[str, float]
        Comparison metrics
    """
    centroids1, sigmas1, weights1 = splats1
    centroids2, sigmas2, weights2 = splats2
    
    # Filter significant splats
    mask1 = weights1 > weight_threshold
    mask2 = weights2 > weight_threshold
    
    # Compute metrics
    metrics = {}
    
    # Centroid distance
    if mask1.any() and mask2.any():
        from scipy.spatial.distance import cdist
        distances = cdist(centroids1[mask1], centroids2[mask2])
        metrics['min_centroid_distance'] = float(np.min(distances))
        metrics['mean_centroid_distance'] = float(np.mean(distances))
    
    # Weight statistics
    metrics['weight_diff_mean'] = float(np.mean(np.abs(weights1 - weights2)))
    metrics['weight_diff_max'] = float(np.max(np.abs(weights1 - weights2)))
    
    # Sigma statistics
    metrics['sigma_diff_mean'] = float(np.mean(np.abs(sigmas1 - sigmas2)))
    metrics['sigma_diff_max'] = float(np.max(np.abs(sigmas1 - sigmas2)))
    
    return metrics
