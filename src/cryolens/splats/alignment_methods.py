"""
Unified alignment methods for reconstruction evaluation.

This module provides multiple alignment methods for aligning reconstructed
volumes, including both volume-based and splat-based approaches.

Available methods:
- volume: Volume-based cross-correlation alignment with coarse-to-fine search (default)
- pca: PCA-based splat alignment (requires splats)
- icp: ICP splat alignment (requires splats)
- ransac_icp: RANSAC+ICP on Gaussian splats with filtering (requires splats)
- fourier: Fourier-based phase correlation (fast but less accurate)
- gradient: Gradient descent refinement (most accurate for volumes, slowest)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from scipy.ndimage import affine_transform
from scipy.optimize import minimize
from scipy.fft import fftn, ifftn, fftshift
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import logging

logger = logging.getLogger(__name__)


# Default configurations for each method
DEFAULT_CONFIGS = {
    'volume': {
        'n_angles_coarse': 24,
        'n_angles_fine': 36,
        'refine': True
    },
    'pca': {
        'use_weights': True
    },
    'icp': {
        'n_iterations': 50,
        'n_init_attempts': 5,
        'convergence_threshold': 1e-6
    },
    'ransac_icp': {
        'weight_percentile': 25.0,
        'sphere_radius': 15.0,
        'ransac_iterations': 100,
        'ransac_inlier_threshold': 2.0,
        'icp_iterations': 10,
        'icp_threshold': 2.0,
        'min_samples': 3,
        'box_size': 48,
        'volume_center': None  # Will be computed as [box_size/2, box_size/2, box_size/2]
    },
    'fourier': {
        'n_angles': 18,
    },
    'gradient': {
        'max_iter': 50,
        'method': 'Powell',
        'initial_alignment_method': 'volume'
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


def cross_correlation_score(vol1: np.ndarray, vol2: np.ndarray) -> float:
    """
    Compute normalized cross-correlation between two volumes.
    Higher values indicate better alignment.
    """
    vol1_norm = normalize_volume(vol1)
    vol2_norm = normalize_volume(vol2)
    
    # Compute correlation
    correlation = np.sum(vol1_norm * vol2_norm) / vol1_norm.size
    return correlation


def chamfer_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Compute Chamfer distance between two point sets."""
    if len(A) == 0 or len(B) == 0:
        return float('inf')
    
    distances = cdist(A, B)
    forward = np.min(distances, axis=1).mean()
    backward = np.min(distances, axis=0).mean()
    return (forward + backward) / 2


def align_volume_based(
    template: np.ndarray,
    target: np.ndarray,
    n_angles_coarse: int = 24,
    n_angles_fine: int = 36,
    refine: bool = True
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Volume-based cross-correlation alignment with coarse-to-fine search.
    
    Parameters
    ----------
    template : np.ndarray
        Template volume
    target : np.ndarray
        Target volume to align
    n_angles_coarse : int
        Number of angles to sample in coarse search
    n_angles_fine : int
        Number of angles to sample in fine search
    refine : bool
        Whether to perform fine refinement
        
    Returns
    -------
    aligned_volume : np.ndarray
        Aligned target volume
    score : float
        Alignment quality score (cross-correlation)
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    """
    template_norm = normalize_volume(template)
    best_rotation = np.eye(3)
    best_score = -np.inf
    
    # Coarse search - sample rotation space
    angles_coarse = np.linspace(0, 2*np.pi, n_angles_coarse, endpoint=False)
    
    for rx in angles_coarse[::2]:  # Sample every other angle for speed
        for ry in angles_coarse[::2]:
            for rz in angles_coarse[::2]:
                # Build rotation matrix from Euler angles
                Rx = np.array([[1, 0, 0],
                              [0, np.cos(rx), -np.sin(rx)],
                              [0, np.sin(rx), np.cos(rx)]])
                
                Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                              [0, 1, 0],
                              [-np.sin(ry), 0, np.cos(ry)]])
                
                Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                              [np.sin(rz), np.cos(rz), 0],
                              [0, 0, 1]])
                
                R_mat = Rz @ Ry @ Rx
                
                # Apply rotation and compute score
                rotated = apply_rotation_to_volume(target, R_mat)
                score = cross_correlation_score(template_norm, rotated)
                
                if score > best_score:
                    best_score = score
                    best_rotation = R_mat
    
    # Fine refinement around best rotation
    if refine:
        # Extract approximate Euler angles from best rotation
        r = R.from_matrix(best_rotation)
        best_euler = r.as_euler('xyz')
        
        # Search in a narrow range around the best angles
        angle_range = 2*np.pi / n_angles_coarse
        angles_fine = np.linspace(-angle_range, angle_range, n_angles_fine)
        
        for drx in angles_fine:
            for dry in angles_fine:
                for drz in angles_fine:
                    rx = best_euler[0] + drx
                    ry = best_euler[1] + dry
                    rz = best_euler[2] + drz
                    
                    Rx = np.array([[1, 0, 0],
                                  [0, np.cos(rx), -np.sin(rx)],
                                  [0, np.sin(rx), np.cos(rx)]])
                    
                    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                                  [0, 1, 0],
                                  [-np.sin(ry), 0, np.cos(ry)]])
                    
                    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                                  [np.sin(rz), np.cos(rz), 0],
                                  [0, 0, 1]])
                    
                    R_mat = Rz @ Ry @ Rx
                    
                    rotated = apply_rotation_to_volume(target, R_mat)
                    score = cross_correlation_score(template_norm, rotated)
                    
                    if score > best_score:
                        best_score = score
                        best_rotation = R_mat
    
    aligned = apply_rotation_to_volume(target, best_rotation)
    return aligned, best_score, best_rotation


def align_pca_based(
    template: np.ndarray,
    target: np.ndarray,
    template_splats: Dict[str, np.ndarray],
    target_splats: Dict[str, np.ndarray],
    use_weights: bool = True
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    PCA-based alignment using Gaussian splat centroids.
    
    Parameters
    ----------
    template : np.ndarray
        Template volume
    target : np.ndarray
        Target volume to align
    template_splats : Dict[str, np.ndarray]
        Template splat parameters with keys 'centroids', 'weights', 'sigmas'
    target_splats : Dict[str, np.ndarray]
        Target splat parameters with keys 'centroids', 'weights', 'sigmas'
    use_weights : bool
        Whether to use splat weights in PCA computation
        
    Returns
    -------
    aligned_volume : np.ndarray
        Aligned target volume
    score : float
        Alignment quality score (cross-correlation)
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    """
    # Import the existing PCA alignment from alignment.py
    from .alignment_pca import weighted_pca_alignment
    
    # Extract centroids and weights
    template_centroids = template_splats['centroids']
    template_weights = template_splats['weights']
    target_centroids = target_splats['centroids']
    target_weights = target_splats['weights']
    
    # Perform weighted PCA alignment
    template_tuple = (template_centroids, template_splats['sigmas'], template_weights)
    target_tuple = (target_centroids, target_splats['sigmas'], target_weights)
    
    rotation_matrix, alignment_score = weighted_pca_alignment(
        template_tuple, target_tuple, use_weights=use_weights
    )
    
    # Apply rotation to volume
    aligned = apply_rotation_to_volume(target, rotation_matrix)
    
    # Compute cross-correlation for consistency
    score = cross_correlation_score(template, aligned)
    
    return aligned, score, rotation_matrix


def align_icp_based(
    template: np.ndarray,
    target: np.ndarray,
    template_splats: Dict[str, np.ndarray],
    target_splats: Dict[str, np.ndarray],
    n_iterations: int = 50,
    n_init_attempts: int = 5,
    convergence_threshold: float = 1e-6
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    ICP-based alignment using Gaussian splat centroids.
    
    Parameters
    ----------
    template : np.ndarray
        Template volume
    target : np.ndarray
        Target volume to align
    template_splats : Dict[str, np.ndarray]
        Template splat parameters with keys 'centroids', 'weights', 'sigmas'
    target_splats : Dict[str, np.ndarray]
        Target splat parameters with keys 'centroids', 'weights', 'sigmas'
    n_iterations : int
        Number of ICP iterations
    n_init_attempts : int
        Number of random initializations
    convergence_threshold : float
        Convergence threshold for early stopping
        
    Returns
    -------
    aligned_volume : np.ndarray
        Aligned target volume
    score : float
        Alignment quality score (cross-correlation)
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    """
    # Import the existing ICP alignment from alignment.py
    from .alignment import align_gaussian_splats_icp
    
    # Create splat tuples
    template_tuple = (template_splats['centroids'], template_splats['sigmas'], template_splats['weights'])
    target_tuple = (target_splats['centroids'], target_splats['sigmas'], target_splats['weights'])
    
    # Perform ICP alignment
    rotation_matrix, alignment_error = align_gaussian_splats_icp(
        template_tuple, target_tuple,
        n_iterations=n_iterations,
        n_init_attempts=n_init_attempts,
        convergence_threshold=convergence_threshold
    )
    
    # Apply rotation to volume
    aligned = apply_rotation_to_volume(target, rotation_matrix)
    
    # Compute cross-correlation for consistency
    score = cross_correlation_score(template, aligned)
    
    return aligned, score, rotation_matrix


def align_ransac_icp(
    template: np.ndarray,
    target: np.ndarray,
    template_splats: Dict[str, np.ndarray],
    target_splats: Dict[str, np.ndarray],
    weight_percentile: float = 25.0,
    sphere_radius: float = 15.0,
    ransac_iterations: int = 100,
    ransac_inlier_threshold: float = 2.0,
    icp_iterations: int = 10,
    icp_threshold: float = 2.0,
    min_samples: int = 3,
    box_size: int = 48,
    volume_center: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    RANSAC + ICP alignment using Gaussian splats with filtering.
    
    This is the method from cryolens-scripts/a_id_validation/ransac_icp_alignment_splats.py
    
    Parameters
    ----------
    template : np.ndarray
        Template volume
    target : np.ndarray
        Target volume to align
    template_splats : Dict[str, np.ndarray]
        Template splat parameters with keys 'centroids', 'weights', 'sigmas'
    target_splats : Dict[str, np.ndarray]
        Target splat parameters with keys 'centroids', 'weights', 'sigmas'
    weight_percentile : float
        Percentile threshold for weight filtering (25 = keep splats above 25th percentile)
    sphere_radius : float
        Radius of sphere from center for spatial filtering
    ransac_iterations : int
        Number of RANSAC iterations
    ransac_inlier_threshold : float
        Distance threshold for RANSAC inliers
    icp_iterations : int
        Number of ICP refinement iterations
    icp_threshold : float
        Distance threshold for ICP correspondences
    min_samples : int
        Minimum number of samples needed for alignment
    box_size : int
        Size of the volume box
    volume_center : np.ndarray, optional
        Center of the volume (defaults to [box_size/2, box_size/2, box_size/2])
        
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
    
    def apply_filters(splat_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply weight and sphere filters to splats."""
        weights = splat_dict['weights']
        coords = splat_dict['centroids']
        
        # Weight filtering - keep splats above percentile threshold
        threshold = np.percentile(weights, weight_percentile)
        weight_mask = weights > threshold
        
        # Sphere filtering - keep splats within radius from center
        distances = np.linalg.norm(coords - volume_center, axis=1)
        sphere_mask = distances < sphere_radius
        
        # Combine both filters
        return weight_mask & sphere_mask
    
    # Filter splats
    template_mask = apply_filters(template_splats)
    target_mask = apply_filters(target_splats)
    
    template_coords = template_splats['centroids'][template_mask]
    template_weights = template_splats['weights'][template_mask]
    target_coords = target_splats['centroids'][target_mask]
    target_weights = target_splats['weights'][target_mask]
    
    if len(target_coords) < min_samples or len(template_coords) < min_samples:
        logger.warning(f"Not enough splats after filtering (template: {len(template_coords)}, "
                      f"target: {len(target_coords)}), returning identity")
        return target, cross_correlation_score(template, target), np.eye(3)
    
    # RANSAC alignment
    best_R = np.eye(3)
    best_score = float('inf')
    best_inliers = 0
    
    # Normalize weights for probability sampling
    template_probs = template_weights / template_weights.sum() if template_weights.sum() > 0 else None
    target_probs = target_weights / target_weights.sum() if target_weights.sum() > 0 else None
    
    for iteration in range(ransac_iterations):
        # Sample points weighted by splat weights
        n_sample = min(30, len(target_coords)//3, len(template_coords)//3)
        n_sample = max(min_samples, n_sample)
        
        try:
            if target_probs is not None and template_probs is not None:
                target_idx = np.random.choice(len(target_coords), n_sample, p=target_probs, replace=False)
                template_idx = np.random.choice(len(template_coords), n_sample, p=template_probs, replace=False)
            else:
                target_idx = np.random.choice(len(target_coords), n_sample, replace=False)
                template_idx = np.random.choice(len(template_coords), n_sample, replace=False)
        except:
            continue
        
        # Center points around volume center
        target_sample = target_coords[target_idx] - volume_center
        template_sample = template_coords[template_idx] - volume_center
        
        # Compute rotation using SVD
        H = target_sample.T @ template_sample
        U, S, Vt = np.linalg.svd(H)
        R_candidate = Vt.T @ U.T
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_candidate) < 0:
            Vt[-1, :] *= -1
            R_candidate = Vt.T @ U.T
        
        # Apply rotation to all target points
        target_rot = (R_candidate @ (target_coords - volume_center).T).T + volume_center
        
        # Count inliers
        distances = cdist(target_rot, template_coords)
        min_dists = np.min(distances, axis=1)
        inliers = min_dists < ransac_inlier_threshold
        n_inliers = np.sum(inliers)
        
        # Score by weighted inlier distance
        if n_inliers > 0:
            inlier_weights = target_weights[inliers]
            if inlier_weights.sum() > 0:
                score = np.average(min_dists[inliers], weights=inlier_weights)
            else:
                score = np.mean(min_dists[inliers])
            
            if n_inliers > best_inliers or (n_inliers == best_inliers and score < best_score):
                best_R = R_candidate
                best_score = score
                best_inliers = n_inliers
    
    # ICP refinement
    R = best_R.copy()
    prev_error = float('inf')
    
    for icp_iter in range(icp_iterations):
        # Apply current rotation
        target_rot = (R @ (target_coords - volume_center).T).T + volume_center
        
        # Find nearest neighbors
        distances = cdist(target_rot, template_coords)
        nearest_idx = np.argmin(distances, axis=1)
        min_dists = np.min(distances, axis=1)
        
        # Only use close correspondences
        valid = min_dists < icp_threshold
        if np.sum(valid) < min_samples:
            break
        
        target_valid = target_rot[valid] - volume_center
        template_matched = template_coords[nearest_idx[valid]] - volume_center
        
        # Update rotation
        H = target_valid.T @ template_matched
        U, S, Vt = np.linalg.svd(H)
        R_update = Vt.T @ U.T
        
        if np.linalg.det(R_update) < 0:
            Vt[-1, :] *= -1
            R_update = Vt.T @ U.T
        
        # Accumulate rotation
        R = R_update @ R
        
        # Check convergence
        current_error = np.mean(min_dists[valid])
        if abs(prev_error - current_error) < 1e-6:
            break
        prev_error = current_error
    
    # Apply final rotation to volume
    aligned = apply_rotation_to_volume(target, R)
    score = cross_correlation_score(template, aligned)
    
    return aligned, score, R


def align_fourier_based(
    template: np.ndarray,
    target: np.ndarray,
    n_angles: int = 18
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fourier-based phase correlation alignment.
    Fast but less accurate than cross-correlation.
    
    Parameters
    ----------
    template : np.ndarray
        Template volume
    target : np.ndarray
        Target volume to align
    n_angles : int
        Number of angles to sample in rotation search
        
    Returns
    -------
    aligned_volume : np.ndarray
        Aligned target volume
    score : float
        Alignment quality score (cross-correlation for consistency)
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    """
    template_norm = normalize_volume(template)
    target_norm = normalize_volume(target)
    
    best_rotation = np.eye(3)
    best_score = -np.inf
    
    # Sample rotation space
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    
    # Pre-compute template FFT
    template_fft = fftn(template_norm)
    
    for rx in angles:
        for ry in angles:
            for rz in angles:
                # Build rotation matrix
                Rx = np.array([[1, 0, 0],
                              [0, np.cos(rx), -np.sin(rx)],
                              [0, np.sin(rx), np.cos(rx)]])
                
                Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                              [0, 1, 0],
                              [-np.sin(ry), 0, np.cos(ry)]])
                
                Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                              [np.sin(rz), np.cos(rz), 0],
                              [0, 0, 1]])
                
                R_mat = Rz @ Ry @ Rx
                
                # Apply rotation
                rotated = apply_rotation_to_volume(target_norm, R_mat)
                
                # Compute phase correlation
                rotated_fft = fftn(rotated)
                cross_power = template_fft * np.conj(rotated_fft)
                cross_power /= np.abs(cross_power) + 1e-10
                
                correlation = np.real(ifftn(cross_power))
                score = np.max(correlation)
                
                if score > best_score:
                    best_score = score
                    best_rotation = R_mat
    
    aligned = apply_rotation_to_volume(target, best_rotation)
    # Recompute cross-correlation for consistency with other methods
    final_score = cross_correlation_score(template, aligned)
    
    return aligned, final_score, best_rotation


def align_gradient_based(
    template: np.ndarray,
    target: np.ndarray,
    max_iter: int = 50,
    method: str = 'Powell',
    initial_alignment_method: str = 'volume'
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Gradient descent refinement of alignment.
    Most accurate for volumes but slowest. Best used with initial estimate.
    
    Parameters
    ----------
    template : np.ndarray
        Template volume
    target : np.ndarray
        Target volume to align
    max_iter : int
        Maximum iterations for optimization
    method : str
        Optimization method for scipy.optimize.minimize ('Powell', 'BFGS', etc.)
    initial_alignment_method : str
        Method to use for initial estimate ('volume', 'fourier', or None for identity)
        
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
    
    # Get initial rotation
    if initial_alignment_method == 'volume':
        _, _, initial_rotation = align_volume_based(template, target, n_angles_coarse=12, refine=False)
    elif initial_alignment_method == 'fourier':
        _, _, initial_rotation = align_fourier_based(template, target, n_angles=12)
    else:
        initial_rotation = np.eye(3)
    
    def objective(angles):
        """Objective function to minimize (negative correlation)."""
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
        
        R_mat = Rz @ Ry @ Rx
        rotated = apply_rotation_to_volume(target, R_mat)
        
        # Return negative correlation (we're minimizing)
        return -cross_correlation_score(template_norm, rotated)
    
    # Extract Euler angles from initial rotation
    r = R.from_matrix(initial_rotation)
    initial_angles = r.as_euler('xyz')
    
    # Optimize
    result = minimize(objective, initial_angles, method=method, 
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


def align_volumes(
    template: np.ndarray,
    target: np.ndarray,
    method: str = 'volume',
    splat_params: Optional[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = None,
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
        Alignment method: 'volume', 'pca', 'icp', 'ransac_icp', 'fourier', 'gradient'
    splat_params : Tuple[Dict, Dict], optional
        Tuple of (template_splats, target_splats) for splat-based methods.
        Each dict should have keys: 'centroids', 'weights', 'sigmas'
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
        
    Examples
    --------
    >>> # Volume-based alignment (default)
    >>> aligned, score, R = align_volumes(template, target)
    
    >>> # RANSAC+ICP with custom parameters
    >>> aligned, score, R = align_volumes(
    ...     template, target, 
    ...     method='ransac_icp',
    ...     splat_params=(template_splats, target_splats),
    ...     weight_percentile=25,
    ...     sphere_radius=15
    ... )
    
    >>> # Gradient descent with Fourier initialization
    >>> aligned, score, R = align_volumes(
    ...     template, target,
    ...     method='gradient',
    ...     initial_alignment_method='fourier'
    ... )
    """
    # Get default config for this method and update with user parameters
    config = DEFAULT_CONFIGS.get(method, {}).copy()
    config.update(method_kwargs)
    
    # Validate splat-based methods have splat params
    splat_methods = ['pca', 'icp', 'ransac_icp']
    if method in splat_methods and splat_params is None:
        raise ValueError(f"{method} method requires splat_params (template_splats, target_splats)")
    
    # Route to appropriate alignment function
    if method == 'volume':
        return align_volume_based(template, target, **config)
    
    elif method == 'pca':
        template_splats, target_splats = splat_params
        return align_pca_based(template, target, template_splats, target_splats, **config)
    
    elif method == 'icp':
        template_splats, target_splats = splat_params
        return align_icp_based(template, target, template_splats, target_splats, **config)
    
    elif method == 'ransac_icp':
        template_splats, target_splats = splat_params
        return align_ransac_icp(template, target, template_splats, target_splats, **config)
    
    elif method == 'fourier':
        return align_fourier_based(template, target, **config)
    
    elif method == 'gradient':
        return align_gradient_based(template, target, **config)
    
    else:
        available = list(DEFAULT_CONFIGS.keys())
        raise ValueError(f"Unknown alignment method: {method}. Available methods: {available}")


def align_multiple_volumes(
    template: np.ndarray,
    targets: List[np.ndarray],
    method: str = 'volume',
    splat_params_list: Optional[List[Tuple[Dict, Dict]]] = None,
    **method_kwargs
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    """
    Align multiple target volumes to a single template.
    
    Parameters
    ----------
    template : np.ndarray
        Template volume to align all targets to
    targets : List[np.ndarray]
        List of target volumes to align
    method : str
        Alignment method to use
    splat_params_list : List[Tuple[Dict, Dict]], optional
        List of (template_splats, target_splats) tuples for each target.
        Required for splat-based methods.
    **method_kwargs
        Additional method-specific parameters
        
    Returns
    -------
    aligned_volumes : List[np.ndarray]
        List of aligned target volumes
    scores : List[float]
        List of alignment scores
    rotation_matrices : List[np.ndarray]
        List of rotation matrices
    """
    aligned_volumes = []
    scores = []
    rotation_matrices = []
    
    for i, target in enumerate(targets):
        # Get splat params for this target if provided
        if splat_params_list is not None:
            splat_params = splat_params_list[i]
        else:
            splat_params = None
        
        # Align this target
        aligned, score, R = align_volumes(
            template, target,
            method=method,
            splat_params=splat_params,
            **method_kwargs
        )
        
        aligned_volumes.append(aligned)
        scores.append(score)
        rotation_matrices.append(R)
    
    return aligned_volumes, scores, rotation_matrices
