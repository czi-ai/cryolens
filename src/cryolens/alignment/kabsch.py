"""
Kabsch algorithm implementation for optimal rotation between point sets.

This module provides the Kabsch algorithm for finding the optimal rotation
matrix that aligns two sets of points, useful for pose recovery validation.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from scipy.spatial.transform import Rotation
import logging

logger = logging.getLogger(__name__)


class KabschAlignment:
    """
    Kabsch algorithm for optimal rotation alignment.
    
    This class implements the Kabsch algorithm (also known as the 
    Procrustes algorithm) for finding the optimal rotation and translation
    that aligns two sets of points.
    
    Attributes:
        center_points (bool): Whether to center points before alignment
        allow_reflection (bool): Whether to allow reflection in addition to rotation
    """
    
    def __init__(
        self,
        center_points: bool = True,
        allow_reflection: bool = False
    ):
        """
        Initialize Kabsch alignment.
        
        Args:
            center_points: Center point sets before alignment
            allow_reflection: Allow reflection in the transformation
        """
        self.center_points = center_points
        self.allow_reflection = allow_reflection
    
    def compute_optimal_rotation(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute optimal rotation matrix using Kabsch algorithm.
        
        Args:
            P: Source points (N x 3)
            Q: Target points (N x 3)
            weights: Optional point weights (N,)
            
        Returns:
            Tuple of (rotation_matrix, translation, rmsd)
        """
        if P.shape != Q.shape:
            raise ValueError(f"Point sets must have same shape, got {P.shape} and {Q.shape}")
        
        if P.shape[0] < 3:
            raise ValueError(f"Need at least 3 points, got {P.shape[0]}")
        
        # Apply weights if provided
        if weights is not None:
            if weights.shape[0] != P.shape[0]:
                raise ValueError(f"Weights shape {weights.shape} doesn't match points {P.shape}")
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(P.shape[0]) / P.shape[0]
        
        # Center points
        if self.center_points:
            P_centroid = np.sum(P * weights[:, np.newaxis], axis=0)
            Q_centroid = np.sum(Q * weights[:, np.newaxis], axis=0)
            P_centered = P - P_centroid
            Q_centered = Q - Q_centroid
        else:
            P_centered = P
            Q_centered = Q
            P_centroid = np.zeros(3)
            Q_centroid = np.zeros(3)
        
        # Compute covariance matrix
        H = np.zeros((3, 3))
        for i in range(P_centered.shape[0]):
            H += weights[i] * np.outer(P_centered[i], Q_centered[i])
        
        # SVD of covariance matrix
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = np.dot(Vt.T, U.T)
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            if self.allow_reflection:
                logger.warning("Reflection detected and allowed")
            else:
                # Flip the sign of the smallest singular value
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
        
        # Compute translation
        t = Q_centroid - np.dot(R, P_centroid)
        
        # Compute RMSD
        P_aligned = np.dot(P, R.T) + t
        diff = P_aligned - Q
        rmsd = np.sqrt(np.sum(weights * np.sum(diff**2, axis=1)))
        
        return R, t, rmsd
    
    def align_poses(
        self,
        source_rotations: np.ndarray,
        target_rotations: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Align sets of rotation matrices.
        
        Args:
            source_rotations: Source rotation matrices (N x 3 x 3)
            target_rotations: Target rotation matrices (N x 3 x 3)
            
        Returns:
            Tuple of (global_rotation, mean_angular_error)
        """
        if source_rotations.shape != target_rotations.shape:
            raise ValueError("Rotation sets must have same shape")
        
        n_poses = source_rotations.shape[0]
        
        # Convert to quaternions for easier manipulation
        source_quats = Rotation.from_matrix(source_rotations.reshape(-1, 3, 3)).as_quat()
        target_quats = Rotation.from_matrix(target_rotations.reshape(-1, 3, 3)).as_quat()
        
        # Find global rotation that best aligns all poses
        # This is done by finding R such that target ≈ R @ source
        
        # Stack all rotation matrices
        H = np.zeros((3, 3))
        for i in range(n_poses):
            H += np.dot(target_rotations[i], source_rotations[i].T)
        
        # SVD to find optimal global rotation
        U, S, Vt = np.linalg.svd(H)
        R_global = np.dot(U, Vt)
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_global) < 0:
            Vt[-1, :] *= -1
            R_global = np.dot(U, Vt)
        
        # Compute angular errors
        aligned_rotations = np.array([np.dot(R_global, R) for R in source_rotations])
        angular_errors = []
        
        for i in range(n_poses):
            # Compute relative rotation
            R_rel = np.dot(target_rotations[i], aligned_rotations[i].T)
            # Extract angle from rotation matrix
            trace = np.trace(R_rel)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            angular_errors.append(np.degrees(angle))
        
        mean_error = np.mean(angular_errors)
        
        return R_global, mean_error
    
    def compute_geodesic_distance(
        self,
        R1: np.ndarray,
        R2: np.ndarray
    ) -> float:
        """
        Compute geodesic distance between two rotations.
        
        Args:
            R1: First rotation matrix (3x3)
            R2: Second rotation matrix (3x3)
            
        Returns:
            Geodesic distance in radians
        """
        R_rel = np.dot(R1, R2.T)
        trace = np.trace(R_rel)
        # Clamp to avoid numerical issues
        cos_angle = np.clip((trace - 1) / 2, -1, 1)
        return np.arccos(cos_angle)
    
    def compute_pairwise_geodesic_correlation(
        self,
        rotations1: np.ndarray,
        rotations2: np.ndarray
    ) -> float:
        """
        Compute correlation between pairwise geodesic distances.
        
        This metric is invariant to global rotations and measures
        whether the relative orientations are preserved.
        
        Args:
            rotations1: First set of rotations (N x 3 x 3)
            rotations2: Second set of rotations (N x 3 x 3)
            
        Returns:
            Correlation coefficient
        """
        n = rotations1.shape[0]
        
        # Compute pairwise geodesic distances for both sets
        distances1 = np.zeros((n, n))
        distances2 = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                d1 = self.compute_geodesic_distance(rotations1[i], rotations1[j])
                d2 = self.compute_geodesic_distance(rotations2[i], rotations2[j])
                distances1[i, j] = distances1[j, i] = d1
                distances2[i, j] = distances2[j, i] = d2
        
        # Get upper triangular values (excluding diagonal)
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        d1_flat = distances1[mask]
        d2_flat = distances2[mask]
        
        # Compute correlation
        if len(d1_flat) > 1:
            correlation = np.corrcoef(d1_flat, d2_flat)[0, 1]
        else:
            correlation = 1.0
            
        return correlation


def kabsch_rotation(
    P: np.ndarray,
    Q: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convenience function for Kabsch algorithm.
    
    Args:
        P: Source points (N x 3)
        Q: Target points (N x 3) 
        weights: Optional weights (N,)
        
    Returns:
        Tuple of (rotation_matrix, translation, rmsd)
    """
    aligner = KabschAlignment()
    return aligner.compute_optimal_rotation(P, Q, weights)


def align_with_kabsch(
    source_poses: np.ndarray,
    target_poses: np.ndarray,
    compute_correlation: bool = True
) -> Dict[str, Any]:
    """
    Align pose sets using Kabsch algorithm.
    
    Args:
        source_poses: Source rotation matrices (N x 3 x 3)
        target_poses: Target rotation matrices (N x 3 x 3)
        compute_correlation: Whether to compute geodesic correlation
        
    Returns:
        Dictionary containing:
            - global_rotation: Optimal global rotation
            - mean_error: Mean angular error after alignment
            - geodesic_correlation: Correlation of pairwise distances (if requested)
    """
    aligner = KabschAlignment()
    
    global_rot, mean_error = aligner.align_poses(source_poses, target_poses)
    
    result = {
        'global_rotation': global_rot,
        'mean_error': mean_error
    }
    
    if compute_correlation:
        correlation = aligner.compute_pairwise_geodesic_correlation(
            source_poses, target_poses
        )
        result['geodesic_correlation'] = correlation
    
    return result
