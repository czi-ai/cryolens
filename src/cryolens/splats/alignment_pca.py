"""
PCA-based Gaussian splat alignment utilities.

This module provides improved PCA-based alignment algorithms for Gaussian splats
with spherical filtering and weighted covariance matrices.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def apply_spherical_filter(
    coords: np.ndarray,
    center: Optional[np.ndarray] = None,
    radius: float = 18.0
) -> np.ndarray:
    """
    Apply spherical filter to points, keeping only those within radius of center.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates of shape (N, 3)
    center : np.ndarray, optional
        Center point for sphere. If None, uses volume center (24, 24, 24)
    radius : float
        Radius of sphere for filtering
        
    Returns
    -------
    np.ndarray
        Boolean mask of points within sphere
    """
    if center is None:
        # Default to center of 48x48x48 volume
        center = np.array([24, 24, 24])
    
    distances = np.linalg.norm(coords - center, axis=1)
    return distances < radius


def weighted_pca_alignment(
    coords1: np.ndarray,
    coords2: np.ndarray,
    weights1: np.ndarray,
    weights2: np.ndarray,
    weight_threshold: float = 70,
    sphere_radius: float = 18,
    center: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA-based alignment using weighted covariance matrices with spherical filtering.
    
    Parameters
    ----------
    coords1 : np.ndarray
        Reference coordinates of shape (N1, 3)
    coords2 : np.ndarray
        Target coordinates to align of shape (N2, 3)
    weights1 : np.ndarray
        Weights for reference points of shape (N1,)
    weights2 : np.ndarray
        Weights for target points of shape (N2,)
    weight_threshold : float
        Percentile threshold for weight filtering (0-100)
    sphere_radius : float
        Radius for spherical filtering
    center : np.ndarray, optional
        Center point for spherical filter
        
    Returns
    -------
    Tuple containing:
        rotation_matrix : np.ndarray
            3x3 rotation matrix
        centroid1 : np.ndarray
            Weighted centroid of filtered reference points
        centroid2 : np.ndarray
            Weighted centroid of filtered target points
        coords2_aligned : np.ndarray
            Aligned target coordinates (all points)
        mask1 : np.ndarray
            Boolean mask for filtered reference points
        mask2 : np.ndarray
            Boolean mask for filtered target points
    """
    if center is None:
        center = np.array([24, 24, 24])
    
    # Apply spherical filter
    sphere_mask1 = apply_spherical_filter(coords1, center, sphere_radius)
    sphere_mask2 = apply_spherical_filter(coords2, center, sphere_radius)
    
    # Apply weight threshold filter
    threshold1 = np.percentile(weights1, weight_threshold)
    threshold2 = np.percentile(weights2, weight_threshold)
    weight_mask1 = weights1 > threshold1
    weight_mask2 = weights2 > threshold2
    
    # Combine masks
    mask1 = sphere_mask1 & weight_mask1
    mask2 = sphere_mask2 & weight_mask2
    
    # Check if we have enough points
    min_points = 10
    if np.sum(mask1) < min_points or np.sum(mask2) < min_points:
        logger.warning(f"Too few points after filtering: {np.sum(mask1)} ref, {np.sum(mask2)} target")
        # Fall back to using more points
        mask1 = sphere_mask1
        mask2 = sphere_mask2
    
    coords1_filtered = coords1[mask1]
    coords2_filtered = coords2[mask2]
    weights1_filtered = weights1[mask1]
    weights2_filtered = weights2[mask2]
    
    logger.debug(f"Filtered points: {len(coords1_filtered)} (ref) and {len(coords2_filtered)} (target)")
    
    # Compute weighted centroids
    centroid1 = np.average(coords1_filtered, axis=0, weights=weights1_filtered)
    centroid2 = np.average(coords2_filtered, axis=0, weights=weights2_filtered)
    
    # Center the points
    coords1_centered = coords1_filtered - centroid1
    coords2_centered = coords2_filtered - centroid2
    
    # Compute weighted covariance matrices
    cov1 = np.cov(coords1_centered.T, aweights=weights1_filtered)
    cov2 = np.cov(coords2_centered.T, aweights=weights2_filtered)
    
    # PCA to find principal axes
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    
    # Sort by eigenvalue (descending)
    idx1 = eigvals1.argsort()[::-1]
    idx2 = eigvals2.argsort()[::-1]
    eigvecs1 = eigvecs1[:, idx1]
    eigvecs2 = eigvecs2[:, idx2]
    
    # Ensure right-handed coordinate system
    if np.linalg.det(eigvecs1) < 0:
        eigvecs1[:, -1] *= -1
    if np.linalg.det(eigvecs2) < 0:
        eigvecs2[:, -1] *= -1
    
    # Compute rotation to align principal axes
    rotation_matrix = eigvecs1 @ eigvecs2.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(rotation_matrix) < 0:
        logger.warning("Improper rotation detected, correcting...")
        eigvecs1[:, -1] *= -1
        rotation_matrix = eigvecs1 @ eigvecs2.T
    
    # Apply transformation to all points
    coords2_aligned = (rotation_matrix @ (coords2 - centroid2).T).T + centroid1
    
    return rotation_matrix, centroid1, centroid2, coords2_aligned, mask1, mask2


def compute_chamfer_distance(
    coords1: np.ndarray,
    coords2: np.ndarray,
    weights1: Optional[np.ndarray] = None,
    weights2: Optional[np.ndarray] = None
) -> float:
    """
    Compute weighted Chamfer distance between two point sets.
    
    Parameters
    ----------
    coords1 : np.ndarray
        First set of coordinates of shape (N1, 3)
    coords2 : np.ndarray
        Second set of coordinates of shape (N2, 3)
    weights1 : np.ndarray, optional
        Weights for first set
    weights2 : np.ndarray, optional
        Weights for second set
        
    Returns
    -------
    float
        Weighted Chamfer distance
    """
    # Compute pairwise distances
    distances = cdist(coords1, coords2)
    
    # Find minimum distances
    min_distances_1to2 = np.min(distances, axis=1)
    min_distances_2to1 = np.min(distances, axis=0)
    
    # Apply weights if provided
    if weights1 is not None:
        weighted_dist_1to2 = np.average(min_distances_1to2, weights=weights1)
    else:
        weighted_dist_1to2 = np.mean(min_distances_1to2)
    
    if weights2 is not None:
        weighted_dist_2to1 = np.average(min_distances_2to1, weights=weights2)
    else:
        weighted_dist_2to1 = np.mean(min_distances_2to1)
    
    # Chamfer distance is the average of both directions
    chamfer_distance = (weighted_dist_1to2 + weighted_dist_2to1) / 2
    
    return chamfer_distance


def compute_alignment_score(
    coords1: np.ndarray,
    weights1: np.ndarray,
    coords2: np.ndarray,
    weights2: np.ndarray,
    sphere_radius: float = 18,
    weight_threshold: float = 70,
    center: Optional[np.ndarray] = None
) -> float:
    """
    Compute alignment score using Chamfer distance with filtering.
    
    Parameters
    ----------
    coords1 : np.ndarray
        Reference coordinates
    weights1 : np.ndarray
        Reference weights
    coords2 : np.ndarray
        Target coordinates
    weights2 : np.ndarray
        Target weights
    sphere_radius : float
        Radius for spherical filtering
    weight_threshold : float
        Percentile threshold for weight filtering
    center : np.ndarray, optional
        Center point for spherical filter
        
    Returns
    -------
    float
        Chamfer distance score
    """
    if center is None:
        center = np.array([24, 24, 24])
    
    # Apply filters
    sphere_mask1 = apply_spherical_filter(coords1, center, sphere_radius)
    sphere_mask2 = apply_spherical_filter(coords2, center, sphere_radius)
    
    threshold1 = np.percentile(weights1, weight_threshold)
    threshold2 = np.percentile(weights2, weight_threshold)
    weight_mask1 = weights1 > threshold1
    weight_mask2 = weights2 > threshold2
    
    mask1 = sphere_mask1 & weight_mask1
    mask2 = sphere_mask2 & weight_mask2
    
    # Check minimum points
    if np.sum(mask1) < 10 or np.sum(mask2) < 10:
        # Fall back to sphere mask only
        mask1 = sphere_mask1
        mask2 = sphere_mask2
    
    coords1_filtered = coords1[mask1]
    coords2_filtered = coords2[mask2]
    weights1_filtered = weights1[mask1]
    weights2_filtered = weights2[mask2]
    
    return compute_chamfer_distance(
        coords1_filtered, coords2_filtered,
        weights1_filtered, weights2_filtered
    )


def align_gaussian_splats_pca(
    template_splats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_splats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    sphere_radii: Union[float, List[float]] = [15, 18, 20],
    weight_threshold: float = 70,
    center: Optional[np.ndarray] = None,
    return_diagnostics: bool = False
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, Dict[str, Any]]]:
    """
    Align target Gaussian splats to template using PCA with spherical filtering.
    
    This method uses weighted PCA alignment with spherical filtering to focus
    on the core structure and ignore boundary artifacts. It tests multiple
    sphere radii and selects the one with the best alignment score.
    
    Parameters
    ----------
    template_splats : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Template splats (centroids, sigmas, weights)
    target_splats : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Target splats to align (centroids, sigmas, weights)
    sphere_radii : float or List[float]
        Sphere radii to test for filtering. If float, uses single radius.
    weight_threshold : float
        Percentile threshold for weight filtering (0-100)
    center : np.ndarray, optional
        Center point for spherical filter
    return_diagnostics : bool
        If True, returns additional diagnostic information
        
    Returns
    -------
    If return_diagnostics is False:
        Tuple[np.ndarray, float]
            rotation_matrix: 3x3 rotation matrix
            alignment_score: Chamfer distance score
    If return_diagnostics is True:
        Tuple[np.ndarray, float, Dict]
            rotation_matrix: 3x3 rotation matrix
            alignment_score: Chamfer distance score
            diagnostics: Dictionary with alignment details
    """
    template_centroids, template_sigmas, template_weights = template_splats
    target_centroids, target_sigmas, target_weights = target_splats
    
    # Ensure radii is a list
    if isinstance(sphere_radii, (int, float)):
        sphere_radii = [sphere_radii]
    
    # Try different sphere radii
    best_score = np.inf
    best_rotation = np.eye(3)
    best_radius = sphere_radii[0]
    best_masks = None
    best_centroids = None
    all_results = []
    
    for radius in sphere_radii:
        logger.debug(f"Trying sphere radius: {radius}")
        
        # Perform PCA alignment
        R_pca, c1, c2, coords_aligned, mask1, mask2 = weighted_pca_alignment(
            template_centroids, target_centroids,
            template_weights, target_weights,
            weight_threshold=weight_threshold,
            sphere_radius=radius,
            center=center
        )
        
        # Compute alignment score
        score = compute_alignment_score(
            template_centroids, template_weights,
            coords_aligned, target_weights,
            sphere_radius=radius,
            weight_threshold=weight_threshold,
            center=center
        )
        
        logger.debug(f"  Score: {score:.4f}")
        
        result = {
            'radius': radius,
            'score': score,
            'rotation': R_pca,
            'masks': (mask1, mask2),
            'centroids': (c1, c2),
            'n_points': (np.sum(mask1), np.sum(mask2))
        }
        all_results.append(result)
        
        if score < best_score:
            best_score = score
            best_rotation = R_pca
            best_radius = radius
            best_masks = (mask1, mask2)
            best_centroids = (c1, c2)
    
    # Log final result
    rotation_angle = np.degrees(np.arccos(np.clip((np.trace(best_rotation) - 1) / 2, -1, 1)))
    logger.info(f"Best alignment: radius={best_radius}, score={best_score:.4f}, angle={rotation_angle:.1f}°")
    
    if return_diagnostics:
        diagnostics = {
            'best_radius': best_radius,
            'best_masks': best_masks,
            'best_centroids': best_centroids,
            'rotation_angle': rotation_angle,
            'all_results': all_results
        }
        return best_rotation, best_score, diagnostics
    
    return best_rotation, best_score


def compute_structural_metrics(
    coords: np.ndarray,
    weights: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute structural metrics for a set of weighted points.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates of shape (N, 3)
    weights : np.ndarray
        Weights of shape (N,)
    mask : np.ndarray, optional
        Boolean mask to select subset of points
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing structural metrics
    """
    if mask is not None:
        coords = coords[mask]
        weights = weights[mask]
    
    # Weighted center of mass
    com = np.average(coords, axis=0, weights=weights)
    
    # Weighted radius of gyration
    distances_sq = np.sum((coords - com)**2, axis=1)
    rg = np.sqrt(np.average(distances_sq, weights=weights))
    
    # Weighted principal moments
    coords_centered = coords - com
    inertia_tensor = np.zeros((3, 3))
    for i in range(len(coords)):
        inertia_tensor += weights[i] * np.outer(coords_centered[i], coords_centered[i])
    inertia_tensor /= np.sum(weights)
    
    eigvals = np.linalg.eigvalsh(inertia_tensor)
    eigvals = np.sort(eigvals)[::-1]
    
    # Asphericity and acylindricity
    asphericity = eigvals[2] - 0.5 * (eigvals[0] + eigvals[1])
    acylindricity = eigvals[1] - eigvals[0]
    
    metrics = {
        'center_of_mass': com.tolist(),
        'radius_of_gyration': float(rg),
        'principal_moments': eigvals.tolist(),
        'asphericity': float(asphericity),
        'acylindricity': float(acylindricity),
        'n_points': len(coords),
        'total_weight': float(np.sum(weights))
    }
    
    return metrics


# Backward compatibility: provide a wrapper that matches the original interface
def align_gaussian_splats(
    template_splats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_splats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    n_iterations: int = 50,
    n_init_attempts: int = 5,
    convergence_threshold: float = 1e-6,
    use_pca: bool = True,
    **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Align target Gaussian splats to template.
    
    This is a wrapper function that provides backward compatibility while
    defaulting to the improved PCA-based method.
    
    Parameters
    ----------
    template_splats : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Template splats (centroids, sigmas, weights)
    target_splats : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Target splats to align (centroids, sigmas, weights)
    n_iterations : int
        Number of iterations (ignored if use_pca=True)
    n_init_attempts : int
        Number of initialization attempts (ignored if use_pca=True)
    convergence_threshold : float
        Convergence threshold (ignored if use_pca=True)
    use_pca : bool
        If True, use PCA-based alignment. If False, fall back to ICP.
    **kwargs
        Additional arguments passed to the alignment function
        
    Returns
    -------
    Tuple[np.ndarray, float]
        rotation_matrix: 3x3 rotation matrix
        alignment_error: Final alignment error/score
    """
    if use_pca:
        # Use improved PCA-based alignment
        return align_gaussian_splats_pca(template_splats, target_splats, **kwargs)
    else:
        # Fall back to original ICP implementation
        from .alignment import align_gaussian_splats as align_gaussian_splats_icp
        return align_gaussian_splats_icp(
            template_splats, target_splats,
            n_iterations=n_iterations,
            n_init_attempts=n_init_attempts,
            convergence_threshold=convergence_threshold
        )
