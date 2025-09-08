"""
Pose analysis utilities for CryoLens.

This module provides utilities for working with 3D rotations and poses,
including alignment algorithms, distance metrics, and format conversions.
"""

from typing import Tuple, Optional, Union, List
import warnings

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation as R


def kabsch_alignment(
    P: np.ndarray,
    Q: np.ndarray,
    center: bool = True,
    return_transform: bool = False
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
    """
    Perform Kabsch alignment to find optimal rotation aligning P to Q.
    
    The Kabsch algorithm finds the optimal rotation matrix R that minimizes
    the RMSD between two sets of points: ||Q - PR||²
    
    Parameters
    ----------
    P : np.ndarray
        Source points, shape (N, 3)
    Q : np.ndarray
        Target points, shape (N, 3)
    center : bool
        Whether to center the point clouds before alignment
    return_transform : bool
        If True, also return the centroids used for centering
        
    Returns
    -------
    R : np.ndarray
        Optimal rotation matrix (3, 3)
    rmsd : float
        Root mean square deviation after alignment
    centroid_P : np.ndarray (optional)
        Centroid of P (only if return_transform=True)
    centroid_Q : np.ndarray (optional)
        Centroid of Q (only if return_transform=True)
        
    Examples
    --------
    >>> P = np.random.randn(100, 3)
    >>> R_true = Rotation.random().as_matrix()
    >>> Q = P @ R_true
    >>> R_opt, rmsd = kabsch_alignment(P, Q)
    >>> np.allclose(R_opt, R_true)
    True
    """
    P = np.asarray(P)
    Q = np.asarray(Q)
    
    if P.shape != Q.shape:
        raise ValueError(f"P and Q must have the same shape, got {P.shape} and {Q.shape}")
    
    if P.shape[1] != 3:
        raise ValueError(f"Points must be 3D, got shape {P.shape}")
    
    # Center the points if requested
    if center:
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
    else:
        centroid_P = np.zeros(3)
        centroid_Q = np.zeros(3)
        P_centered = P
        Q_centered = Q
    
    # Compute cross-covariance matrix
    H = P_centered.T @ Q_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T
    
    # Compute RMSD
    P_aligned = P_centered @ R_opt
    rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q_centered)**2, axis=1)))
    
    if return_transform:
        return R_opt, rmsd, centroid_P, centroid_Q
    else:
        return R_opt, rmsd


def compute_geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute geodesic distance between two rotation matrices on SO(3).
    
    The geodesic distance is the shortest path between two rotations
    on the rotation manifold, equal to the angle of the relative rotation.
    
    Parameters
    ----------
    R1 : np.ndarray
        First rotation matrix (3, 3)
    R2 : np.ndarray
        Second rotation matrix (3, 3)
        
    Returns
    -------
    float
        Geodesic distance in radians [0, π]
        
    Examples
    --------
    >>> R1 = np.eye(3)
    >>> R2 = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    >>> dist = compute_geodesic_distance(R1, R2)
    >>> np.isclose(dist, np.pi/2)
    True
    """
    R1 = np.asarray(R1)
    R2 = np.asarray(R2)
    
    if R1.shape != (3, 3) or R2.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) rotation matrices, got {R1.shape} and {R2.shape}")
    
    # Compute relative rotation
    R_diff = R1.T @ R2
    
    # Compute angle from trace
    # angle = arccos((tr(R) - 1) / 2)
    trace = np.trace(R_diff)
    # Clamp for numerical stability
    trace = np.clip(trace, -1.0, 3.0)
    angle = np.arccos((trace - 1.0) / 2.0)
    
    return angle


def align_rotation_sets(
    recovered: np.ndarray,
    ground_truth: np.ndarray,
    method: str = "kabsch"
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Find global rotation that best aligns a set of recovered poses to ground truth.
    
    This function finds a single global rotation R_global such that
    R_global @ recovered[i] ≈ ground_truth[i] for all i.
    
    Parameters
    ----------
    recovered : np.ndarray
        Recovered rotation matrices, shape (N, 3, 3)
    ground_truth : np.ndarray
        Ground truth rotation matrices, shape (N, 3, 3)
    method : str
        Alignment method: "kabsch" or "procrustes"
        
    Returns
    -------
    aligned : np.ndarray
        Aligned rotation matrices, shape (N, 3, 3)
    R_global : np.ndarray
        Global rotation used for alignment (3, 3)
    metrics : dict
        Dictionary containing alignment metrics:
        - 'rmsd': Root mean square deviation
        - 'mean_angular_error': Mean angular error in degrees
        - 'max_angular_error': Maximum angular error in degrees
        
    Examples
    --------
    >>> ground_truth = np.array([Rotation.random().as_matrix() for _ in range(10)])
    >>> R_global = Rotation.random().as_matrix()
    >>> recovered = np.array([R_global @ gt for gt in ground_truth])
    >>> aligned, R_est, metrics = align_rotation_sets(recovered, ground_truth)
    >>> np.allclose(R_est, R_global.T)
    True
    """
    recovered = np.asarray(recovered)
    ground_truth = np.asarray(ground_truth)
    
    if recovered.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: recovered {recovered.shape} vs ground_truth {ground_truth.shape}"
        )
    
    if recovered.ndim != 3 or recovered.shape[1:] != (3, 3):
        raise ValueError(f"Expected shape (N, 3, 3), got {recovered.shape}")
    
    n_poses = len(recovered)
    
    if method == "kabsch":
        # Convert rotation matrices to points for Kabsch alignment
        # Use the columns of rotation matrices as 3D points
        P = []
        Q = []
        for i in range(n_poses):
            # Use rotation matrix columns as feature points
            P.extend(recovered[i].T)
            Q.extend(ground_truth[i].T)
        
        P = np.array(P)
        Q = np.array(Q)
        
        # Find optimal global rotation
        R_global, rmsd = kabsch_alignment(P, Q, center=False)
        
    elif method == "procrustes":
        # Stack all rotation matrices
        P = recovered.reshape(-1, 3).T
        Q = ground_truth.reshape(-1, 3).T
        
        # Orthogonal Procrustes
        R_global, scale = orthogonal_procrustes(P.T, Q.T)
        
        # Compute RMSD
        P_aligned = P.T @ R_global
        rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q.T)**2, axis=1)))
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kabsch' or 'procrustes'")
    
    # Apply global rotation to all recovered poses
    aligned = np.array([R_global @ recovered[i] for i in range(n_poses)])
    
    # Compute metrics
    angular_errors = []
    for i in range(n_poses):
        angle = compute_geodesic_distance(aligned[i], ground_truth[i])
        angular_errors.append(np.degrees(angle))
    
    metrics = {
        'rmsd': float(rmsd),
        'mean_angular_error': float(np.mean(angular_errors)),
        'std_angular_error': float(np.std(angular_errors)),
        'max_angular_error': float(np.max(angular_errors)),
        'min_angular_error': float(np.min(angular_errors))
    }
    
    return aligned, R_global, metrics


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute distance between two quaternions.
    
    Parameters
    ----------
    q1 : np.ndarray
        First quaternion (w, x, y, z) or (x, y, z, w)
    q2 : np.ndarray
        Second quaternion (w, x, y, z) or (x, y, z, w)
        
    Returns
    -------
    float
        Distance between quaternions [0, 1]
        
    Notes
    -----
    The distance accounts for quaternion double cover (q and -q represent
    the same rotation) by choosing the closer representation.
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    
    if q1.shape != (4,) or q2.shape != (4,):
        raise ValueError(f"Expected shape (4,), got {q1.shape} and {q2.shape}")
    
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Account for double cover
    dot_product = np.dot(q1, q2)
    if dot_product < 0:
        q2 = -q2
        dot_product = -dot_product
    
    # Clamp for numerical stability
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Distance metric (0 when identical, 1 when perpendicular)
    distance = 1.0 - abs(dot_product)
    
    return distance


def rotation_matrix_to_euler(
    R: np.ndarray,
    convention: str = "XYZ",
    degrees: bool = False
) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles.
    
    Parameters
    ----------
    R : np.ndarray
        Rotation matrix (3, 3) or batch of matrices (N, 3, 3)
    convention : str
        Euler angle convention (e.g., "XYZ", "ZYX", "ZXZ")
    degrees : bool
        If True, return angles in degrees
        
    Returns
    -------
    np.ndarray
        Euler angles (3,) or (N, 3)
    """
    single_matrix = False
    if R.ndim == 2:
        R = R[np.newaxis, :]
        single_matrix = True
    
    if R.shape[1:] != (3, 3):
        raise ValueError(f"Expected shape (..., 3, 3), got {R.shape}")
    
    # Use scipy's Rotation for conversion
    rotations = R.from_matrix(R)
    euler = rotations.as_euler(convention, degrees=degrees)
    
    if single_matrix:
        euler = euler[0]
    
    return euler


def euler_to_rotation_matrix(
    euler: np.ndarray,
    convention: str = "XYZ",
    degrees: bool = False
) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.
    
    Parameters
    ----------
    euler : np.ndarray
        Euler angles (3,) or (N, 3)
    convention : str
        Euler angle convention (e.g., "XYZ", "ZYX", "ZXZ")
    degrees : bool
        If True, angles are in degrees
        
    Returns
    -------
    np.ndarray
        Rotation matrix (3, 3) or batch of matrices (N, 3, 3)
    """
    single_angle = False
    if euler.ndim == 1:
        euler = euler[np.newaxis, :]
        single_angle = True
    
    if euler.shape[1] != 3:
        raise ValueError(f"Expected shape (..., 3), got {euler.shape}")
    
    # Use scipy's Rotation for conversion
    rotations = R.from_euler(convention, euler, degrees=degrees)
    matrices = rotations.as_matrix()
    
    if single_angle:
        matrices = matrices[0]
    
    return matrices


def rotation_matrix_to_quaternion(
    R: np.ndarray,
    scalar_first: bool = True
) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.
    
    Parameters
    ----------
    R : np.ndarray
        Rotation matrix (3, 3) or batch of matrices (N, 3, 3)
    scalar_first : bool
        If True, return quaternion as (w, x, y, z), else (x, y, z, w)
        
    Returns
    -------
    np.ndarray
        Quaternion (4,) or batch of quaternions (N, 4)
    """
    single_matrix = False
    if R.ndim == 2:
        R = R[np.newaxis, :]
        single_matrix = True
    
    if R.shape[1:] != (3, 3):
        raise ValueError(f"Expected shape (..., 3, 3), got {R.shape}")
    
    # Use scipy's Rotation for conversion
    rotations = R.from_matrix(R)
    quaternions = rotations.as_quat()  # Returns (x, y, z, w)
    
    if scalar_first:
        # Convert to (w, x, y, z)
        quaternions = np.roll(quaternions, 1, axis=-1)
    
    if single_matrix:
        quaternions = quaternions[0]
    
    return quaternions


def quaternion_to_rotation_matrix(
    q: np.ndarray,
    scalar_first: bool = True
) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Parameters
    ----------
    q : np.ndarray
        Quaternion (4,) or batch of quaternions (N, 4)
    scalar_first : bool
        If True, quaternion is (w, x, y, z), else (x, y, z, w)
        
    Returns
    -------
    np.ndarray
        Rotation matrix (3, 3) or batch of matrices (N, 3, 3)
    """
    single_quat = False
    if q.ndim == 1:
        q = q[np.newaxis, :]
        single_quat = True
    
    if q.shape[1] != 4:
        raise ValueError(f"Expected shape (..., 4), got {q.shape}")
    
    if scalar_first:
        # Convert from (w, x, y, z) to (x, y, z, w)
        q = np.roll(q, -1, axis=-1)
    
    # Use scipy's Rotation for conversion
    rotations = R.from_quat(q)
    matrices = rotations.as_matrix()
    
    if single_quat:
        matrices = matrices[0]
    
    return matrices


def axis_angle_to_rotation_matrix(
    axis: np.ndarray,
    angle: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Convert axis-angle representation to rotation matrix.
    
    Parameters
    ----------
    axis : np.ndarray
        Rotation axis (3,) or (N, 3), will be normalized
    angle : float or np.ndarray
        Rotation angle in radians, scalar or (N,)
        
    Returns
    -------
    np.ndarray
        Rotation matrix (3, 3) or batch of matrices (N, 3, 3)
    """
    axis = np.asarray(axis)
    angle = np.asarray(angle)
    
    single_rotation = False
    if axis.ndim == 1:
        axis = axis[np.newaxis, :]
        angle = np.array([angle])
        single_rotation = True
    
    if axis.shape[1] != 3:
        raise ValueError(f"Expected axis shape (..., 3), got {axis.shape}")
    
    # Normalize axes
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = axis / (axis_norm + 1e-10)
    
    # Create rotation vectors
    rotvecs = axis * angle[:, np.newaxis]
    
    # Use scipy's Rotation for conversion
    rotations = R.from_rotvec(rotvecs)
    matrices = rotations.as_matrix()
    
    if single_rotation:
        matrices = matrices[0]
    
    return matrices


def rotation_matrix_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert rotation matrix to axis-angle representation.
    
    Parameters
    ----------
    R : np.ndarray
        Rotation matrix (3, 3) or batch of matrices (N, 3, 3)
        
    Returns
    -------
    axis : np.ndarray
        Rotation axis (3,) or (N, 3), unit normalized
    angle : np.ndarray
        Rotation angle in radians, scalar or (N,)
    """
    single_matrix = False
    if R.ndim == 2:
        R = R[np.newaxis, :]
        single_matrix = True
    
    if R.shape[1:] != (3, 3):
        raise ValueError(f"Expected shape (..., 3, 3), got {R.shape}")
    
    # Use scipy's Rotation for conversion
    rotations = R.from_matrix(R)
    rotvecs = rotations.as_rotvec()
    
    # Extract axis and angle
    angles = np.linalg.norm(rotvecs, axis=1)
    axes = np.zeros_like(rotvecs)
    
    # Handle zero rotation case
    non_zero = angles > 1e-10
    axes[non_zero] = rotvecs[non_zero] / angles[non_zero, np.newaxis]
    axes[~non_zero] = np.array([1, 0, 0])  # Default axis for zero rotation
    
    if single_matrix:
        axes = axes[0]
        angles = angles[0]
    
    return axes, angles


def random_rotation_matrix(
    n: int = 1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random rotation matrices uniformly sampled from SO(3).
    
    Parameters
    ----------
    n : int
        Number of rotation matrices to generate
    seed : Optional[int]
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Random rotation matrix (3, 3) if n=1, or (n, 3, 3) if n>1
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n == 1:
        return R.random().as_matrix()
    else:
        return R.random(n).as_matrix()


def compute_rotation_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    degrees: bool = True
) -> dict:
    """
    Compute comprehensive metrics for rotation predictions.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted rotation matrices (N, 3, 3)
    ground_truth : np.ndarray
        Ground truth rotation matrices (N, 3, 3)
    degrees : bool
        If True, return angular errors in degrees
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'mean_geodesic_error': Mean geodesic distance
        - 'std_geodesic_error': Standard deviation of geodesic distance
        - 'median_geodesic_error': Median geodesic distance
        - 'max_geodesic_error': Maximum geodesic distance
        - 'relative_pose_errors': Errors in relative poses
    """
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs ground_truth {ground_truth.shape}"
        )
    
    n = len(predictions)
    
    # Compute geodesic errors
    geodesic_errors = []
    for i in range(n):
        error = compute_geodesic_distance(predictions[i], ground_truth[i])
        if degrees:
            error = np.degrees(error)
        geodesic_errors.append(error)
    
    geodesic_errors = np.array(geodesic_errors)
    
    # Compute relative pose errors
    relative_errors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Ground truth relative pose
            rel_gt = ground_truth[i].T @ ground_truth[j]
            # Predicted relative pose
            rel_pred = predictions[i].T @ predictions[j]
            # Error
            error = compute_geodesic_distance(rel_pred, rel_gt)
            if degrees:
                error = np.degrees(error)
            relative_errors.append(error)
    
    metrics = {
        'mean_geodesic_error': float(np.mean(geodesic_errors)),
        'std_geodesic_error': float(np.std(geodesic_errors)),
        'median_geodesic_error': float(np.median(geodesic_errors)),
        'max_geodesic_error': float(np.max(geodesic_errors)),
        'min_geodesic_error': float(np.min(geodesic_errors)),
        'mean_relative_error': float(np.mean(relative_errors)) if relative_errors else 0.0,
        'std_relative_error': float(np.std(relative_errors)) if relative_errors else 0.0,
    }
    
    return metrics
