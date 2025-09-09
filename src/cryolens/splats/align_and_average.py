"""
Alignment and averaging utilities for volumes using Gaussian splats.

This module provides a simple interface for aligning multiple volumes
using their Gaussian splat representations and computing averages.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import logging

from .alignment import compute_all_alignments, apply_rotation_to_volume

logger = logging.getLogger(__name__)


def align_and_average_volumes(
    volumes: np.ndarray,
    splat_params_list: List[Dict],
    template_idx: int = 0,
    method: str = 'pca',
    coordinate_transform: bool = True,
    weight_threshold: Optional[float] = None
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Align volumes using their Gaussian splats and compute average.
    
    This function takes volumes and their corresponding Gaussian splat parameters,
    aligns them to a template using PCA or ICP, and returns both the individual
    aligned volumes and their average.
    
    Parameters
    ----------
    volumes : np.ndarray
        Volumes to align, shape (N, D, H, W) or list of volumes
    splat_params_list : List[Dict]
        List of splat parameters from InferencePipeline.process_volume()
        Each dict should contain 'centroids', 'sigmas', and 'weights'
    template_idx : int
        Index of template volume to align all others to
    method : str
        Alignment method: 'pca' or 'icp'
    coordinate_transform : bool
        Transform splat coordinates from normalized [-1,1] to voxel space [0, box_size]
    weight_threshold : float, optional
        Optional threshold to filter splats by weight before alignment
        
    Returns
    -------
    averaged_volume : np.ndarray
        Average of aligned volumes
    aligned_volumes : List[np.ndarray]
        Individual aligned volumes
    rotation_matrices : List[np.ndarray]
        Rotation matrices used for alignment
        
    Examples
    --------
    >>> from cryolens.inference import InferencePipeline
    >>> from cryolens.splats.align_and_average import align_and_average_volumes
    >>> 
    >>> # Process volumes and extract splats
    >>> pipeline = InferencePipeline(model, device)
    >>> results = []
    >>> for vol in volumes:
    ...     result = pipeline.process_volume(vol, return_splat_params=True)
    ...     results.append(result)
    >>> 
    >>> # Align and average
    >>> splat_params = [r['splat_params'] for r in results]
    >>> reconstructions = [r['reconstruction'] for r in results]
    >>> avg, aligned, rotations = align_and_average_volumes(
    ...     np.array(reconstructions), splat_params
    ... )
    """
    # Convert volumes to numpy array if needed
    if isinstance(volumes, list):
        volumes = np.array(volumes)
    
    # Extract splat components
    all_centroids = []
    all_sigmas = []
    all_weights = []
    
    for params in splat_params_list:
        centroids = params['centroids']
        
        # Transform coordinates if requested
        if coordinate_transform:
            # Assume box size from volume shape
            box_size = volumes.shape[-1] if volumes.ndim == 4 else 48
            # Transform from [-1, 1] to [0, box_size] for proper alignment
            centroids = (centroids + 1.0) * (box_size / 2.0)
        
        all_centroids.append(centroids)
        all_sigmas.append(params['sigmas'])
        all_weights.append(params['weights'])
    
    # Stack into arrays
    all_centroids = np.array(all_centroids)
    all_sigmas = np.array(all_sigmas)
    all_weights = np.array(all_weights)
    
    # Apply weight threshold if specified
    if weight_threshold is not None:
        logger.info(f"Filtering splats with weight threshold {weight_threshold}")
        for i in range(len(all_weights)):
            mask = all_weights[i] > weight_threshold
            n_before = len(all_weights[i])
            n_after = np.sum(mask)
            logger.debug(f"  Sample {i}: {n_after}/{n_before} splats kept")
    
    # Compute alignments
    splats = (all_centroids, all_sigmas, all_weights)
    
    logger.info(f"Computing {method} alignment to template {template_idx}")
    rotation_matrices, alignment_errors = compute_all_alignments(
        splats, 
        template_idx=template_idx,
        method=method
    )
    
    # Log alignment results
    mean_error = np.mean(alignment_errors[alignment_errors > 0])  # Exclude template
    logger.info(f"Mean alignment error: {mean_error:.4f}")
    
    # Apply rotations to volumes
    aligned_volumes = []
    for i, (vol, rot) in enumerate(zip(volumes, rotation_matrices)):
        if i == template_idx:
            # Template doesn't need rotation
            aligned_volumes.append(vol)
        else:
            aligned_vol = apply_rotation_to_volume(vol, rot)
            aligned_volumes.append(aligned_vol)
    
    # Compute average
    averaged_volume = np.mean(aligned_volumes, axis=0)
    
    logger.info(f"Alignment complete: {len(aligned_volumes)} volumes aligned and averaged")
    
    return averaged_volume, aligned_volumes, rotation_matrices


def align_to_ground_truth_poses(
    poses: np.ndarray,
    ground_truth_poses: np.ndarray,
    method: str = 'kabsch'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Align recovered poses to ground truth poses.
    
    Parameters
    ----------
    poses : np.ndarray
        Recovered poses as rotation matrices, shape (N, 3, 3)
    ground_truth_poses : np.ndarray
        Ground truth poses as rotation matrices, shape (N, 3, 3)
    method : str
        Alignment method: 'kabsch' or 'procrustes'
        
    Returns
    -------
    aligned_poses : np.ndarray
        Aligned poses
    global_rotation : np.ndarray
        Global rotation matrix used for alignment
    metrics : Dict
        Alignment metrics including mean angular error
    """
    from cryolens.utils.pose import align_rotation_sets
    
    aligned_poses, global_rotation, metrics = align_rotation_sets(
        poses, ground_truth_poses, method=method
    )
    
    return aligned_poses, global_rotation, metrics
