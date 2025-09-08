"""
Visualization module for CryoLens.

This module provides tools for visualizing reconstructions, embeddings,
and analysis results.
"""

from .orthoviews import (
    OrthoviewVisualizer,
    create_orthoviews,
    plot_orthoviews
)
from .projections import (
    ProjectionVisualizer,
    create_mip,
    create_projection_comparison,
    create_progressive_mip
)

__all__ = [
    'OrthoviewVisualizer',
    'create_orthoviews',
    'plot_orthoviews',
    'ProjectionVisualizer',
    'create_mip',
    'create_projection_comparison',
    'create_progressive_mip'
]
