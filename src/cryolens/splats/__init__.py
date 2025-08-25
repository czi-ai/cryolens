"""
Gaussian splat utilities for CryoLens.
"""

from .extraction import (
    extract_gaussian_splats,
    render_gaussian_splats,
    compare_splat_sets
)
from .alignment import (
    align_gaussian_splats,
    compute_all_alignments,
    apply_rotation_to_volume,
    create_aligned_average,
    compute_alignment_statistics
)

__all__ = [
    'extract_gaussian_splats',
    'render_gaussian_splats',
    'compare_splat_sets',
    'align_gaussian_splats',
    'compute_all_alignments',
    'apply_rotation_to_volume',
    'create_aligned_average',
    'compute_alignment_statistics'
]
