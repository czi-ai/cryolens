"""
Utility functions for CryoLens.
"""

from .checkpoint import load_checkpoint
from .version import get_git_commit_hash, get_version_info, log_version_info

# Import pose utilities
from .pose import (
    kabsch_alignment,
    compute_geodesic_distance,
    align_rotation_sets,
    quaternion_distance,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
    random_rotation_matrix,
    compute_rotation_metrics,
)

__all__ = [
    # Checkpoint utilities
    'load_checkpoint',
    # Version utilities
    'get_git_commit_hash',
    'get_version_info',
    'log_version_info',
    # Pose utilities
    'kabsch_alignment',
    'compute_geodesic_distance',
    'align_rotation_sets',
    'quaternion_distance',
    'rotation_matrix_to_euler',
    'euler_to_rotation_matrix',
    'rotation_matrix_to_quaternion',
    'quaternion_to_rotation_matrix',
    'axis_angle_to_rotation_matrix',
    'rotation_matrix_to_axis_angle',
    'random_rotation_matrix',
    'compute_rotation_metrics',
]
