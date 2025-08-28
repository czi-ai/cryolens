"""
Gaussian splat alignment utilities.

This module provides ICP and PCA-based alignment algorithms for Gaussian splats.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional, Dict, Any, Union
import logging

# Import PCA-based alignment methods
from .alignment_pca import (
    align_gaussian_splats_pca,
    weighted_pca_alignment,
    compute_chamfer_distance,
    compute_alignment_score,
    compute_structural_metrics,
    apply_spherical_filter
)

logger = logging.getLogger(__name__)


def align_gaussian_splats_icp(
    template_splats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_splats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    n_iterations: int = 50,
    n_init_attempts: int = 5,
    convergence_threshold: float = 1e-6
) -> Tuple[np.ndarray, float]:
    """
    Align target Gaussian splats to template using ICP-like algorithm.
    
    Parameters
    ----------
    template_splats : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Template splats (centroids, sigmas, weights)
    target_splats : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Target splats to align (centroids, sigmas, weights)
    n_iterations : int
        Number of ICP iterations
    n_init_attempts : int
        Number of random initializations to try
    convergence_threshold : float
        Convergence threshold for early stopping
        
    Returns
    -------
    Tuple[np.ndarray, float]
        rotation_matrix: 3x3 rotation matrix
        alignment_error: Final alignment error
    """
    template_centroids, template_sigmas, template_weights = template_splats
    target_centroids, target_sigmas, target_weights = target_splats
    
    # Weight centroids by their weights for better alignment
    template_weighted = template_centroids * template_weights[:, np.newaxis]
    target_weighted = target_centroids * target_weights[:, np.newaxis]
    
    # Initialize with identity rotation
    best_rotation = np.eye(3)
    best_error = float('inf')
    
    # Try multiple initial random rotations to avoid local minima
    for init_attempt in range(n_init_attempts):
        if init_attempt == 0:
            # Start with identity
            current_rotation = np.eye(3)
        else:
            # Random initial rotation
            random_rot = R.random()
            current_rotation = random_rot.as_matrix()
        
        prev_error = float('inf')
        
        # ICP iterations
        for iteration in range(n_iterations):
            # Apply current rotation to target
            rotated_target = (current_rotation @ target_centroids.T).T
            rotated_weighted = (current_rotation @ target_weighted.T).T
            
            # Find nearest neighbors using weighted positions
            distances = cdist(template_weighted, rotated_weighted)
            correspondences = np.argmin(distances, axis=1)
            
            # Compute weighted centroids for alignment
            template_subset = template_centroids
            target_subset = target_centroids[correspondences]
            
            # Weight by both template and corresponding target weights
            weights_combined = template_weights * target_weights[correspondences]
            weights_combined = weights_combined / (weights_combined.sum() + 1e-8)
            
            # Compute weighted centers
            template_center = np.sum(template_subset * weights_combined[:, np.newaxis], axis=0)
            target_center = np.sum(target_subset * weights_combined[:, np.newaxis], axis=0)
            
            # Center the points
            template_centered = template_subset - template_center
            target_centered = target_subset - target_center
            
            # Apply weights to centered points
            template_weighted_centered = template_centered * np.sqrt(weights_combined[:, np.newaxis])
            target_weighted_centered = target_centered * np.sqrt(weights_combined[:, np.newaxis])
            
            # Compute rotation using SVD
            H = target_weighted_centered.T @ template_weighted_centered
            U, _, Vt = np.linalg.svd(H)
            R_update = Vt.T @ U.T
            
            # Ensure proper rotation (det = 1)
            if np.linalg.det(R_update) < 0:
                Vt[-1, :] *= -1
                R_update = Vt.T @ U.T
            
            # Update rotation
            current_rotation = R_update @ current_rotation
            
            # Compute error
            rotated_final = (current_rotation @ target_centroids.T).T
            error = np.mean(np.min(cdist(template_centroids, rotated_final), axis=1))
            
            # Check for convergence
            if abs(prev_error - error) < convergence_threshold:
                logger.debug(f"Converged at iteration {iteration} for attempt {init_attempt}")
                break
            prev_error = error
        
        # Keep best rotation
        if error < best_error:
            best_error = error
            best_rotation = current_rotation
            logger.debug(f"New best error: {best_error:.6f} from attempt {init_attempt}")
    
    return best_rotation, best_error


def align_gaussian_splats(
    template_splats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_splats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    method: str = 'pca',
    **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Align target Gaussian splats to template using specified method.
    
    Parameters
    ----------
    template_splats : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Template splats (centroids, sigmas, weights)
    target_splats : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Target splats to align (centroids, sigmas, weights)
    method : str
        Alignment method: 'pca' (default) or 'icp'
    **kwargs
        Additional arguments passed to the alignment function
        
    Returns
    -------
    Tuple[np.ndarray, float]
        rotation_matrix: 3x3 rotation matrix
        alignment_error: Final alignment error/score
    """
    if method == 'pca':
        # Use improved PCA-based alignment
        return align_gaussian_splats_pca(template_splats, target_splats, **kwargs)
    elif method == 'icp':
        # Use original ICP implementation
        return align_gaussian_splats_icp(template_splats, target_splats, **kwargs)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def compute_all_alignments(
    all_splats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    template_idx: int = 0,
    method: str = 'pca',
    **alignment_kwargs
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Compute alignments of all samples to a template.
    
    Parameters
    ----------
    all_splats : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (centroids, sigmas, weights) arrays, each of shape (N, num_splats, ...)
    template_idx : int
        Index of the template sample
    **alignment_kwargs
        Additional arguments passed to align_gaussian_splats
    
    Returns
    -------
    Tuple[List[np.ndarray], List[float]]
        rotation_matrices: List of 3x3 rotation matrices
        alignment_errors: List of alignment errors
    """
    centroids, sigmas, weights = all_splats
    n_samples = len(centroids)
    
    # Template splats
    template_splats = (centroids[template_idx], sigmas[template_idx], weights[template_idx])
    
    rotation_matrices = []
    alignment_errors = []
    
    logger.info(f"Aligning {n_samples} samples to template (index {template_idx})...")
    
    for i in range(n_samples):
        if i == template_idx:
            # Template aligns to itself with identity
            rotation_matrices.append(np.eye(3))
            alignment_errors.append(0.0)
            logger.debug(f"Sample {i+1}/{n_samples}: Template (identity)")
        else:
            target_splats = (centroids[i], sigmas[i], weights[i])
            rotation_matrix, error = align_gaussian_splats(
                template_splats, target_splats, method=method, **alignment_kwargs
            )
            rotation_matrices.append(rotation_matrix)
            alignment_errors.append(error)
            
            # Convert to axis-angle for display
            rotation_obj = R.from_matrix(rotation_matrix)
            axis_angle = rotation_obj.as_rotvec()
            angle = np.linalg.norm(axis_angle)
            logger.debug(f"Sample {i+1}/{n_samples}: Error={error:.4f}, Rotation angle={np.degrees(angle):.1f}°")
    
    return rotation_matrices, alignment_errors


def apply_rotation_to_volume(
    volume: np.ndarray,
    rotation_matrix: np.ndarray,
    order: int = 1
) -> np.ndarray:
    """
    Apply rotation to a 3D volume.
    
    Parameters
    ----------
    volume : np.ndarray
        3D volume to rotate
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    order : int
        Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns
    -------
    np.ndarray
        Rotated volume
    """
    from scipy.ndimage import affine_transform
    
    # Check if this is identity
    is_identity = np.allclose(rotation_matrix, np.eye(3))
    if is_identity:
        return volume.copy()
    
    # Create affine transformation matrix
    # Center the rotation around the volume center
    center = np.array(volume.shape) / 2.0
    
    # Create translation to origin
    T_origin = np.eye(4)
    T_origin[:3, 3] = -center
    
    # Create rotation matrix (4x4)
    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = rotation_matrix
    
    # Create translation back
    T_back = np.eye(4)
    T_back[:3, 3] = center
    
    # Combine transformations
    transform = T_back @ R_4x4 @ T_origin
    
    # scipy.ndimage.affine_transform expects the inverse transformation
    transform_inv = np.linalg.inv(transform)
    
    # Apply transformation
    rotated = affine_transform(
        volume,
        transform_inv[:3, :3],
        offset=transform_inv[:3, 3],
        order=order,
        mode='constant',
        cval=0.0
    )
    
    return rotated


def create_aligned_average(
    volumes: np.ndarray,
    rotation_matrices: List[np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Create average of aligned volumes.
    
    Parameters
    ----------
    volumes : np.ndarray
        Original volumes to align
    rotation_matrices : List[np.ndarray]
        Rotation matrices that align the volumes
        
    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        average: Average of aligned volumes
        aligned_volumes: List of aligned volumes
    """
    aligned_volumes = []
    
    for volume, rotation in zip(volumes, rotation_matrices):
        aligned = apply_rotation_to_volume(volume, rotation)
        aligned_volumes.append(aligned)
    
    # Average aligned volumes
    average = np.mean(aligned_volumes, axis=0)
    
    return average, aligned_volumes


def compute_alignment_statistics(
    rotation_matrices: List[np.ndarray],
    gt_poses: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute statistics about alignment results.
    
    Parameters
    ----------
    rotation_matrices : List[np.ndarray]
        List of rotation matrices from alignment
    gt_poses : np.ndarray, optional
        Ground truth poses for comparison
        
    Returns
    -------
    Dict[str, Any]
        Statistics about the alignments
    """
    # Convert to Euler angles for analysis
    euler_angles = []
    for R_mat in rotation_matrices:
        r = R.from_matrix(R_mat)
        euler = r.as_euler('xyz', degrees=True)
        euler_angles.append(euler)
    euler_angles = np.array(euler_angles)
    
    stats = {
        'n_samples': len(rotation_matrices),
        'euler_mean': np.mean(euler_angles, axis=0).tolist(),
        'euler_std': np.std(euler_angles, axis=0).tolist(),
        'euler_min': np.min(euler_angles, axis=0).tolist(),
        'euler_max': np.max(euler_angles, axis=0).tolist()
    }
    
    # Compare with ground truth if available
    if gt_poses is not None:
        angular_diffs = []
        for R_aligned, gt_pose in zip(rotation_matrices, gt_poses):
            # Convert GT to rotation matrix
            angle = gt_pose[0]
            axis = gt_pose[1:4]
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-10:
                axis = axis / axis_norm
                rotvec = axis * angle
                R_gt = R.from_rotvec(rotvec).as_matrix()
                # Compute relative rotation
                R_diff = R_aligned.T @ R_gt
                angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
                angular_diffs.append(np.degrees(angle_diff))
            else:
                angular_diffs.append(0.0)
        
        stats['angular_diff_mean'] = float(np.mean(angular_diffs))
        stats['angular_diff_std'] = float(np.std(angular_diffs))
        stats['angular_diff_min'] = float(np.min(angular_diffs))
        stats['angular_diff_max'] = float(np.max(angular_diffs))
    
    return stats
