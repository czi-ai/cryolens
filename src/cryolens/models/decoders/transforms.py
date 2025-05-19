"""
Transform utilities for decoders.
"""

import torch
import numpy as np


def axis_angle_to_quaternion(axis_angle, normalize=True):
    """Convert axis-angle representation to quaternion.
    
    Parameters
    ----------
    axis_angle : torch.Tensor
        Axis-angle representation [batch_size, 4].
    normalize : bool
        Whether to normalize the quaternion.
    
    Returns
    -------
    torch.Tensor
        Quaternion representation [batch_size, 4].
    """
    angles = axis_angle[:, 0]
    axes = axis_angle[:, 1:]
    
    if normalize:
        axes = axes / (torch.norm(axes, dim=1, keepdim=True) + 1e-8)
    
    half_angles = angles * 0.5
    sin_half_angles = torch.sin(half_angles)
    
    quaternions = torch.cat([
        torch.cos(half_angles).unsqueeze(1),
        axes * sin_half_angles.unsqueeze(1)
    ], dim=1)
    
    return quaternions


def quaternion_to_rotation_matrix(quaternions):
    """Convert quaternion to rotation matrix.
    
    Parameters
    ----------
    quaternions : torch.Tensor
        Quaternion representation [batch_size, 4].
    
    Returns
    -------
    torch.Tensor
        Rotation matrix [batch_size, 3, 3].
    """
    batch_size = quaternions.shape[0]
    
    # Extract quaternion components
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Compute rotation matrix elements
    xx, xy, xz = x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    wx, wy, wz = w * x, w * y, w * z
    
    # Create rotation matrix
    rotation_matrix = torch.zeros(batch_size, 3, 3, device=quaternions.device)
    
    rotation_matrix[:, 0, 0] = 1 - 2 * (yy + zz)
    rotation_matrix[:, 0, 1] = 2 * (xy - wz)
    rotation_matrix[:, 0, 2] = 2 * (xz + wy)
    
    rotation_matrix[:, 1, 0] = 2 * (xy + wz)
    rotation_matrix[:, 1, 1] = 1 - 2 * (xx + zz)
    rotation_matrix[:, 1, 2] = 2 * (yz - wx)
    
    rotation_matrix[:, 2, 0] = 2 * (xz - wy)
    rotation_matrix[:, 2, 1] = 2 * (yz + wx)
    rotation_matrix[:, 2, 2] = 1 - 2 * (xx + yy)
    
    return rotation_matrix
