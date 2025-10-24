"""
Unified alignment methods for reconstruction evaluation.

This module provides multiple alignment methods for aligning reconstructed
volumes, including both volume-based and splat-based approaches.

Available methods:
- cross_correlation: Volume-based alignment with coarse-to-fine search (default)
- fourier: Fourier-based phase correlation (fast but less accurate)
- gradient_descent: Gradient descent refinement (most accurate, slowest)
- ransac_icp: RANSAC+ICP on Gaussian splats (fast, requires splats)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.ndimage import affine_transform, rotate as ndimage_rotate
from scipy.optimize import minimize
from scipy.fft import fftn, ifftn, fftshift
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


# Default configurations for each method
DEFAULT_CONFIGS = {
    'cross_correlation': {
        'angular_step': 30.0,
        'n_iterations': 50
    },
    'fourier': {
        'angular_step': 15.0
    },
    'gradient_descent': {
        'max_iter': 50,
        'initial_method': 'cross_correlation'  # Warm start
    },
    'ransac_icp': {
        # Optimized parameters from Optuna tuning
        'weight_percentile': 48.3,
        'sphere_radius': 15.2,
        'ransac_iterations': 252,
        'ransac_inlier_threshold': 1.63,
        'icp_iterations': 17,
        'icp_threshold': 3.19,
        'min_samples': 10,
        'max_samples': 36,
        'use_adaptive_threshold': True,
        'adaptive_threshold_percentile': 82,
        'outlier_rejection_percentile': 99.6,
        'icp_convergence_threshold': 0.000153,
        'box_size': 48,
        'volume_center': None  # Will be computed as box_size / 2
    }
}


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to zero mean and unit variance."""
    vol_normalized = volume - np.mean(volume)
    std = np.std(vol_normalized)
    if std > 0:
        vol_normalized /= std
    return vol_normalized


def apply_rotation_to_volume(volume: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Apply rotation to a 3D volume around its center.
    
    Parameters
    ----------
    volume : np.ndarray
        3D volume to rotate
    rotation_matrix : np.ndarray
        3x3 rotation matrix
        
    Returns
    -------
    np.ndarray
        Rotated volume
    """
    # Check if this is identity
    if np.allclose(rotation_matrix, np.eye(3)):
        return volume.copy()
    
    center = np.array(volume.shape) / 2.0
    
    # For scipy's affine_transform, we need the inverse transformation
    inv_rotation = rotation_matrix.T
    
    # Calculate the offset
    offset = center - inv_rotation @ center
    
    # Apply transformation
    rotated_volume = affine_transform(
        volume,
        inv_rotation,
        offset=offset,
        order=1,  # Linear interpolation
        mode='constant',
        cval=0
    )
    
    return rotated_volume


def cross_correlation_3d(vol1: np.ndarray, vol2: np.ndarray) -> float:
    """
    Compute normalized cross-correlation between two volumes.
    Higher values indicate better alignment.
    """
    vol1_norm = normalize_volume(vol1)
    vol2_norm = normalize_volume(vol2)
    
    # Compute correlation
    correlation = np.sum(vol1_norm * vol2_norm) / vol1_norm.size
    return correlation


def align_cross_correlation(
    template: np.ndarray,
    target: np.ndarray,
    angular_step: float = 30.0,
    n_iterations: int = 50
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Find best rotation using cross-correlation optimization.
    Uses coarse-to-fine search strategy.
    
    Parameters
    ----------
    template : np.ndarray
        Template volume
    target : np.ndarray
        Target volume to align
    angular_step : float
        Angular step for coarse search in degrees
    n_iterations : int
        Number of iterations (not currently used)
        
    Returns
    -------
    aligned_volume : np.ndarray
        Aligned target volume
    score : float
        Alignment quality score
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    """
    template_norm = normalize_volume(template)
    best_rotation = np.eye(3)
    best_score = -np.inf
    
    # Coarse search
    angles_coarse = np.arange(0, 360, angular_step)
    
    for rx in angles_coarse[::3]:  # Sample every 3rd angle for speed
        for ry in angles_coarse[::3]:
            for rz in angles_coarse[::3]:
                # Create rotation matrix
                rx_rad, ry_rad, rz_rad = np.radians([rx, ry, rz])
                
                # Rotation matrices
                Rx = np.array([[1, 0, 0],
                              [0, np.cos(rx_rad), -np.sin(rx_rad)],
                              [0, np.sin(rx_rad), np.cos(rx_rad)]])
                
                Ry = np.array([[np.cos(ry_rad), 0, np.sin(ry_rad)],
                              [0, 1, 0],
                              [-np.sin(ry_rad), 0, np.cos(ry_rad)]])
                
                Rz = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0],
                              [np.sin(rz_rad), np.cos(rz_rad), 0],
                              [0, 0, 1]])
                
                R = Rz @ Ry @ Rx
                
                # Apply rotation and compute correlation
                rotated = apply_rotation_to_volume(target, R)
                score = cross_correlation_3d(template_norm, rotated)
                
                if score > best_score:
                    best_score = score
                    best_rotation = R
    
    # Fine search around best rotation
    if angular_step > 5:
        # Extract approximate angles from best rotation
        best_ry = np.arcsin(-best_rotation[2, 0])
        best_rx = np.arctan2(best_rotation[2, 1], best_rotation[2, 2])
        best_rz = np.arctan2(best_rotation[1, 0], best_rotation[0, 0])
        
        angles_fine = np.linspace(-angular_step/2, angular_step/2, 5)
        
        for drx in angles_fine:
            for dry in angles_fine:
                for drz in angles_fine:
                    rx_rad = best_rx + np.radians(drx)
                    ry_rad = best_ry + np.radians(dry)
                    rz_rad = best_rz + np.radians(drz)
                    
                    Rx = np.array([[1, 0, 0],
                                  [0, np.cos(rx_rad), -np.sin(rx_rad)],
                                  [0, np.sin(rx_rad), np.cos(rx_rad)]])
                    
                    Ry = np.array([[np.cos(ry_rad), 0, np.sin(ry_rad)],
                                  [0, 1, 0],
                                  [-np.sin(ry_rad), 0, np.cos(ry_rad)]])
                    
                    Rz = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0],
                                  [np.sin(rz_rad), np.cos(rz_rad), 0],
                                  [0, 0, 1]])
                    
                    R = Rz @ Ry @ Rx
                    
                    rotated = apply_rotation_to_volume(target, R)
                    score = cross_correlation_3d(template_norm, rotated)
                    
                    if score > best_score:
                        best_score = score
                        best_rotation = R
    
    aligned = apply_rotation_to_volume(target, best_rotation)
    return aligned, best_score, best_rotation


def align_fourier(
    template: np.ndarray,
    target: np.ndarray,
    angular_step: float = 15.0
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Find best rotation using Fourier-based phase correlation.
    This is faster but less accurate than cross-correlation.
    
    Parameters
    ----------
    template : np.ndarray
        Template volume
    target : np.ndarray
        Target volume to align
    angular_step : float
        Angular step for search in degrees
        
    Returns
    -------
    aligned_volume : np.ndarray
        Aligned target volume
    score : float
        Alignment quality score (cross-correlation for consistency)
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    """
    # Normalize volumes
    template_norm = normalize_volume(template)
    target_norm = normalize_volume(target)
    
    best_rotation = np.eye(3)
    best_score = -np.inf
    
    # Use coarser sampling for Fourier method (it's faster per iteration)
    angles = np.arange(0, 360, angular_step)
    
    # Pre-compute template FFT
    template_fft = fftn(template_norm)
    
    for rx in angles[::2]:
        for ry in angles[::2]:
            for rz in angles[::2]:
                # Create rotation matrix
                rx_rad, ry_rad, rz_rad = np.radians([rx, ry, rz])
                
                Rx = np.array([[1, 0, 0],
                              [0, np.cos(rx_rad), -np.sin(rx_rad)],
                              [0, np.sin(rx_rad), np.cos(rx_rad)]])
                
                Ry = np.array([[np.cos(ry_rad), 0, np.sin(ry_rad)],
                              [0, 1, 0],
                              [-np.sin(ry_rad), 0, np.cos(ry_rad)]])
                
                Rz = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0],
                              [np.sin(rz_rad), np.cos(rz_rad), 0],
                              [0, 0, 1]])
                
                R = Rz @ Ry @ Rx
                
                # Apply rotation
                rotated = apply_rotation_to_volume(target_norm, R)
                
                # Compute phase correlation
                rotated_fft = fftn(rotated)
                cross_power = template_fft * np.conj(rotated_fft)
                cross_power /= np.abs(cross_power) + 1e-10
                
                correlation = np.real(ifftn(cross_power))
                score = np.max(correlation)
                
                if score > best_score:
                    best_score = score
                    best_rotation = R
    
    aligned = apply_rotation_to_volume(target, best_rotation)
    # Recompute cross-correlation for consistency with other methods
    final_score = cross_correlation_3d(template, aligned)
    
    return aligned, final_score, best_rotation


def align_gradient_descent(
    template: np.ndarray,
    target: np.ndarray,
    max_iter: int = 50,
    initial_method: str = 'cross_correlation',
    initial_rotation: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Refine rotation using gradient descent optimization.
    Good for fine-tuning an initial estimate.
    
    Parameters
    ----------
    template : np.ndarray
        Template volume
    target : np.ndarray
        Target volume to align
    max_iter : int
        Maximum iterations for optimization
    initial_method : str
        Method to use for initial estimate ('cross_correlation')
    initial_rotation : np.ndarray, optional
        Initial rotation matrix (if None, computed using initial_method)
        
    Returns
    -------
    aligned_volume : np.ndarray
        Aligned target volume
    score : float
        Alignment quality score
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    """
    template_norm = normalize_volume(template)
    
    # Get initial rotation if not provided
    if initial_rotation is None:
        if initial_method == 'cross_correlation':
            _, _, initial_rotation = align_cross_correlation(template, target, angular_step=30.0)
        else:
            initial_rotation = np.eye(3)
    
    def objective(angles):
        rx, ry, rz = angles
        
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(rx), -np.sin(rx)],
                      [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                      [np.sin(rz), np.cos(rz), 0],
                      [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        rotated = apply_rotation_to_volume(target, R)
        
        # Return negative correlation (we minimize)
        return -cross_correlation_3d(template_norm, rotated)
    
    # Extract angles from initial rotation (approximate)
    ry = np.arcsin(-initial_rotation[2, 0])
    rx = np.arctan2(initial_rotation[2, 1], initial_rotation[2, 2])
    rz = np.arctan2(initial_rotation[1, 0], initial_rotation[0, 0])
    initial_angles = [rx, ry, rz]
    
    # Optimize
    result = minimize(objective, initial_angles, method='Powell', 
                     options={'maxiter': max_iter})
    
    # Get final rotation
    rx, ry, rz = result.x
    
    Rx = np.array([[1, 0, 0],
                  [0, np.cos(rx), -np.sin(rx)],
                  [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                  [np.sin(rz), np.cos(rz), 0],
                  [0, 0, 1]])
    
    best_rotation = Rz @ Ry @ Rx
    best_score = -result.fun
    
    aligned = apply_rotation_to_volume(target, best_rotation)
    return aligned, best_score, best_rotation


def align_ransac_icp(
    template: np.ndarray,
    target: np.ndarray,
    template_splats: Dict[str, np.ndarray],
    target_splats: Dict[str, np.ndarray],
    weight_percentile: float = 48.3,
    sphere_radius: float = 15.2,
    ransac_iterations: int = 252,
    ransac_inlier_threshold: float = 1.63,
    icp_iterations: int = 17,
    icp_threshold: float = 3.19,
    min_samples: int = 10,
    max_samples: int = 36,
    use_adaptive_threshold: bool = True,
    adaptive_threshold_percentile: float = 82,
    outlier_rejection_percentile: float = 99.6,
    icp_convergence_threshold: float = 0.000153,
    box_size: int = 48,
    volume_center: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    RANSAC + ICP alignment using Gaussian splats.
    Uses optimized parameters from Optuna tuning.
    
    Parameters
    ----------
    template : np.ndarray
        Template volume
    target : np.ndarray
        Target volume to align
    template_splats : Dict[str, np.ndarray]
        Template splat parameters with keys 'coordinates', 'weights', 'sigmas'
    target_splats : Dict[str, np.ndarray]
        Target splat parameters with keys 'coordinates', 'weights', 'sigmas'
    weight_percentile : float
        Percentile threshold for weight filtering
    sphere_radius : float
        Radius of sphere from center for filtering
    ransac_iterations : int
        Number of RANSAC iterations
    ransac_inlier_threshold : float
        Inlier threshold for RANSAC
    icp_iterations : int
        Number of ICP refinement iterations
    icp_threshold : float
        Distance threshold for ICP
    min_samples : int
        Minimum number of samples for alignment
    max_samples : int
        Maximum number of samples to use
    use_adaptive_threshold : bool
        Whether to use adaptive thresholding
    adaptive_threshold_percentile : float
        Percentile for adaptive threshold
    outlier_rejection_percentile : float
        Percentile for outlier rejection
    icp_convergence_threshold : float
        Convergence threshold for ICP
    box_size : int
        Size of the volume box
    volume_center : np.ndarray, optional
        Center of the volume (defaults to box_size/2)
        
    Returns
    -------
    aligned_volume : np.ndarray
        Aligned target volume
    score : float
        Alignment quality score (cross-correlation)
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    """
    if volume_center is None:
        volume_center = np.array([box_size / 2, box_size / 2, box_size / 2])
    
    def apply_filters(splat_dict):
        """Apply weight and sphere filters to splats."""
        weights = splat_dict['weights']
        coords = splat_dict['coordinates']
        
        # Weight filtering
        threshold = np.percentile(weights, weight_percentile)
        weight_mask = weights > threshold
        
        # Sphere filtering
        distances = np.linalg.norm(coords - volume_center, axis=1)
        sphere_mask = distances < sphere_radius
        
        # Outlier rejection
        outlier_threshold = np.percentile(weights, outlier_rejection_percentile)
        outlier_mask = weights <= outlier_threshold
        
        return weight_mask & sphere_mask & outlier_mask
    
    # Filter splats
    template_mask = apply_filters(template_splats)
    target_mask = apply_filters(target_splats)
    
    template_coords = template_splats['coordinates'][template_mask]
    template_weights = template_splats['weights'][template_mask]
    target_coords = target_splats['coordinates'][target_mask]
    target_weights = target_splats['weights'][target_mask]
    
    if len(target_coords) < min_samples or len(template_coords) < min_samples:
        logger.warning("Not enough splats after filtering, returning identity")
        return target, cross_correlation_3d(template, target), np.eye(3)
    
    # RANSAC
    best_R = np.eye(3)
    best_score = float('inf')
    
    # Adaptive threshold calculation
    if use_adaptive_threshold and len(target_coords) > 10:
        dists = cdist(target_coords[:10], target_coords[:10])
        np.fill_diagonal(dists, np.inf)
        nn_dist = np.min(dists, axis=1).mean()
        adaptive_threshold = max(ransac_inlier_threshold, nn_dist * 2)
    else:
        adaptive_threshold = ransac_inlier_threshold
    
    for iteration in range(ransac_iterations):
        # Sample
        n_sample = min(max_samples, max(min_samples, len(target_coords)//3, len(template_coords)//3))
        
        try:
            idx1 = np.random.choice(len(target_coords), n_sample, replace=False)
            idx2 = np.random.choice(len(template_coords), n_sample, replace=False)
            
            source = target_coords[idx1] - volume_center
            target_pts = template_coords[idx2] - volume_center
            
            H = source.T @ target_pts
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Apply rotation and compute score
            rotated = (R @ (target_coords - volume_center).T).T + volume_center
            
            distances = cdist(rotated, template_coords)
            min_dists = np.min(distances, axis=1)
            
            inliers = min_dists < adaptive_threshold
            n_inliers = np.sum(inliers)
            
            if n_inliers > 0:
                score = np.mean(min_dists[inliers])
                combined_score = score / (n_inliers + 1)
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_R = R
        except:
            continue
    
    # ICP refinement
    R = best_R.copy()
    prev_error = float('inf')
    
    for icp_iter in range(icp_iterations):
        source_rot = (R @ (target_coords - volume_center).T).T + volume_center
        
        distances = cdist(source_rot, template_coords)
        nearest_idx = np.argmin(distances, axis=1)
        min_dists = np.min(distances, axis=1)
        
        # Adaptive threshold for ICP
        if use_adaptive_threshold:
            threshold = np.percentile(min_dists, adaptive_threshold_percentile)
            threshold = min(threshold, icp_threshold)
        else:
            threshold = icp_threshold
        
        valid = min_dists < threshold
        if np.sum(valid) < min_samples:
            break
        
        source_valid = source_rot[valid] - volume_center
        target_matched = template_coords[nearest_idx[valid]] - volume_center
        
        H = source_valid.T @ target_matched
        U, S, Vt = np.linalg.svd(H)
        R_update = Vt.T @ U.T
        
        if np.linalg.det(R_update) < 0:
            Vt[-1, :] *= -1
            R_update = Vt.T @ U.T
        
        R = R_update @ R
        
        current_error = np.mean(min_dists[valid])
        if abs(prev_error - current_error) < icp_convergence_threshold:
            break
        prev_error = current_error
    
    # Apply final rotation to volume
    aligned = apply_rotation_to_volume(target, R)
    score = cross_correlation_3d(template, aligned)
    
    return aligned, score, R


def align_volumes(
    template: np.ndarray,
    target: np.ndarray,
    method: str = 'cross_correlation',
    splat_params: Optional[Tuple[Dict, Dict]] = None,
    **method_kwargs
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Unified interface for volume alignment using various methods.
    
    Parameters
    ----------
    template : np.ndarray
        Template volume to align to
    target : np.ndarray
        Target volume to align
    method : str
        Alignment method: 'cross_correlation', 'fourier', 'gradient_descent', 'ransac_icp'
    splat_params : Tuple[Dict, Dict], optional
        Tuple of (template_splats, target_splats) for splat-based methods
        Each dict should have keys: 'coordinates', 'weights', 'sigmas'
    **method_kwargs
        Additional method-specific parameters
        
    Returns
    -------
    aligned_volume : np.ndarray
        Aligned target volume
    score : float
        Alignment quality score (cross-correlation)
    rotation_matrix : np.ndarray
        3x3 rotation matrix used for alignment
        
    Raises
    ------
    ValueError
        If method is unknown or required parameters are missing
    """
    # Get default config for this method
    config = DEFAULT_CONFIGS.get(method, {}).copy()
    # Update with user-provided parameters
    config.update(method_kwargs)
    
    if method == 'cross_correlation':
        return align_cross_correlation(template, target, **config)
    
    elif method == 'fourier':
        return align_fourier(template, target, **config)
    
    elif method == 'gradient_descent':
        return align_gradient_descent(template, target, **config)
    
    elif method == 'ransac_icp':
        if splat_params is None:
            raise ValueError("ransac_icp method requires splat_params (template_splats, target_splats)")
        template_splats, target_splats = splat_params
        return align_ransac_icp(template, target, template_splats, target_splats, **config)
    
    else:
        raise ValueError(f"Unknown alignment method: {method}. "
                        f"Available methods: cross_correlation, fourier, gradient_descent, ransac_icp")
