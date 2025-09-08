"""
Projection visualization for CryoLens reconstructions.

This module provides tools for creating maximum intensity projections (MIP)
and other projection-based visualizations of 3D reconstructions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any, Union, Literal
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import logging

logger = logging.getLogger(__name__)


class ProjectionVisualizer:
    """
    Visualizer for projection-based views of 3D volumes.
    
    This class creates various projections (MIP, mean, sum) of 3D reconstructions
    for visualization and analysis.
    
    Attributes:
        figsize (tuple): Figure size for plots
        cmap (str): Colormap for visualization
        projection_type (str): Type of projection ('max', 'mean', 'sum', 'std')
        show_axes (bool): Whether to show axis labels
        show_colorbar (bool): Whether to show colorbar
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 4),
        cmap: str = 'hot',
        projection_type: str = 'max',
        show_axes: bool = False,
        show_colorbar: bool = True
    ):
        """
        Initialize projection visualizer.
        
        Args:
            figsize: Figure size (width, height)
            cmap: Matplotlib colormap
            projection_type: Type of projection to use
            show_axes: Show axis labels and ticks
            show_colorbar: Show colorbar
        """
        self.figsize = figsize
        self.cmap = cmap
        self.projection_type = projection_type
        self.show_axes = show_axes
        self.show_colorbar = show_colorbar
    
    def compute_projection(
        self,
        volume: np.ndarray,
        axis: int,
        projection_type: Optional[str] = None,
        slab_thickness: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute projection along specified axis.
        
        Args:
            volume: 3D volume
            axis: Axis for projection (0=Z, 1=Y, 2=X)
            projection_type: Type of projection (overrides instance default)
            slab_thickness: Optional thickness for slab projections
            
        Returns:
            2D projection
        """
        if projection_type is None:
            projection_type = self.projection_type
        
        # Handle slab projection if thickness specified
        if slab_thickness is not None and slab_thickness > 1:
            center = volume.shape[axis] // 2
            start = max(0, center - slab_thickness // 2)
            end = min(volume.shape[axis], center + slab_thickness // 2)
            
            # Extract slab
            if axis == 0:
                slab = volume[start:end, :, :]
            elif axis == 1:
                slab = volume[:, start:end, :]
            else:
                slab = volume[:, :, start:end]
        else:
            slab = volume
        
        # Compute projection
        if projection_type == 'max':
            projection = np.max(slab, axis=axis)
        elif projection_type == 'min':
            projection = np.min(slab, axis=axis)
        elif projection_type == 'mean':
            projection = np.mean(slab, axis=axis)
        elif projection_type == 'sum':
            projection = np.sum(slab, axis=axis)
        elif projection_type == 'std':
            projection = np.std(slab, axis=axis)
        elif projection_type == 'median':
            projection = np.median(slab, axis=axis)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
        
        return projection
    
    def create_projections(
        self,
        volume: np.ndarray,
        title: Optional[str] = None,
        projection_type: Optional[str] = None,
        slab_thickness: Optional[int] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> Figure:
        """
        Create three orthogonal projections.
        
        Args:
            volume: 3D volume to project
            title: Optional title for the figure
            projection_type: Type of projection
            slab_thickness: Thickness for slab projections
            vmin: Minimum value for color scaling
            vmax: Maximum value for color scaling
            
        Returns:
            Matplotlib figure
        """
        # Compute projections
        xy_proj = self.compute_projection(volume, 0, projection_type, slab_thickness)
        xz_proj = self.compute_projection(volume, 1, projection_type, slab_thickness)
        yz_proj = self.compute_projection(volume, 2, projection_type, slab_thickness)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        # Determine color scaling
        if vmin is None:
            vmin = min(xy_proj.min(), xz_proj.min(), yz_proj.min())
        if vmax is None:
            vmax = max(xy_proj.max(), xz_proj.max(), yz_proj.max())
        
        # Plot projections
        proj_type_str = projection_type or self.projection_type
        proj_type_str = proj_type_str.upper() if proj_type_str == 'mip' else proj_type_str.capitalize()
        
        im1 = axes[0].imshow(xy_proj, cmap=self.cmap, vmin=vmin, vmax=vmax,
                            aspect='equal', interpolation='bilinear')
        axes[0].set_title(f'XY {proj_type_str} Projection')
        
        im2 = axes[1].imshow(xz_proj, cmap=self.cmap, vmin=vmin, vmax=vmax,
                            aspect='equal', interpolation='bilinear')
        axes[1].set_title(f'XZ {proj_type_str} Projection')
        
        im3 = axes[2].imshow(yz_proj, cmap=self.cmap, vmin=vmin, vmax=vmax,
                            aspect='equal', interpolation='bilinear')
        axes[2].set_title(f'YZ {proj_type_str} Projection')
        
        # Configure axes
        for ax in axes:
            if not self.show_axes:
                ax.axis('off')
            else:
                ax.set_xlabel('Pixels')
                ax.set_ylabel('Pixels')
        
        # Add colorbar
        if self.show_colorbar:
            fig.colorbar(im1, ax=axes, orientation='horizontal',
                        fraction=0.046, pad=0.1, label='Intensity')
        
        # Add title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return fig
    
    def create_multi_projection_types(
        self,
        volume: np.ndarray,
        projection_types: List[str] = ['max', 'mean', 'std'],
        axis: int = 0,
        title: Optional[str] = None
    ) -> Figure:
        """
        Create multiple projection types for comparison.
        
        Args:
            volume: 3D volume
            projection_types: List of projection types to show
            axis: Axis for projection
            title: Optional title
            
        Returns:
            Matplotlib figure
        """
        n_types = len(projection_types)
        fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 4))
        
        if n_types == 1:
            axes = [axes]
        
        # Create each projection
        projections = []
        for proj_type in projection_types:
            proj = self.compute_projection(volume, axis, proj_type)
            projections.append(proj)
        
        # Determine color scaling
        vmin = min(p.min() for p in projections)
        vmax = max(p.max() for p in projections)
        
        # Plot each projection
        for ax, proj, proj_type in zip(axes, projections, projection_types):
            im = ax.imshow(proj, cmap=self.cmap, vmin=vmin, vmax=vmax,
                          aspect='equal', interpolation='bilinear')
            ax.set_title(f'{proj_type.capitalize()} Projection')
            
            if not self.show_axes:
                ax.axis('off')
        
        # Add colorbar
        if self.show_colorbar:
            fig.colorbar(im, ax=axes, orientation='horizontal',
                        fraction=0.046, pad=0.1, label='Intensity')
        
        # Add title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return fig
    
    def create_progressive_projections(
        self,
        volumes: List[np.ndarray],
        particle_counts: Optional[List[int]] = None,
        projection_type: str = 'max',
        axis: int = 0
    ) -> Figure:
        """
        Create projections showing progressive improvement.
        
        Args:
            volumes: List of volumes with increasing quality
            particle_counts: Number of particles for each volume
            projection_type: Type of projection
            axis: Axis for projection
            
        Returns:
            Matplotlib figure
        """
        n_volumes = len(volumes)
        
        if particle_counts is None:
            particle_counts = list(range(1, n_volumes + 1))
        
        # Create figure
        fig, axes = plt.subplots(1, n_volumes, figsize=(3 * n_volumes, 3))
        
        if n_volumes == 1:
            axes = [axes]
        
        # Compute all projections
        projections = []
        for vol in volumes:
            proj = self.compute_projection(vol, axis, projection_type)
            projections.append(proj)
        
        # Shared color scale
        vmin = min(p.min() for p in projections)
        vmax = max(p.max() for p in projections)
        
        # Plot each projection
        for ax, proj, n_particles in zip(axes, projections, particle_counts):
            im = ax.imshow(proj, cmap=self.cmap, vmin=vmin, vmax=vmax,
                          aspect='equal', interpolation='bilinear')
            ax.set_title(f'{n_particles} particles', fontsize=10)
            
            if not self.show_axes:
                ax.axis('off')
        
        # Add colorbar
        if self.show_colorbar:
            fig.colorbar(im, ax=axes, orientation='horizontal',
                        fraction=0.046, pad=0.15, label='Intensity')
        
        # Add main title
        axis_names = ['XY', 'XZ', 'YZ']
        proj_name = projection_type.upper() if projection_type == 'mip' else projection_type.capitalize()
        fig.suptitle(f'Progressive {axis_names[axis]} {proj_name} Projections',
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        return fig
    
    def create_colored_mip(
        self,
        volume: np.ndarray,
        axis: int = 0,
        colormap: str = 'hot',
        alpha_threshold: float = 0.1
    ) -> Figure:
        """
        Create colored maximum intensity projection with depth encoding.
        
        Args:
            volume: 3D volume
            axis: Projection axis
            colormap: Colormap for depth
            alpha_threshold: Threshold for transparency
            
        Returns:
            Matplotlib figure
        """
        # Get MIP and depth map
        mip = np.max(volume, axis=axis)
        depth_indices = np.argmax(volume, axis=axis)
        depth_normalized = depth_indices / volume.shape[axis]
        
        # Create RGBA image
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Standard MIP
        axes[0].imshow(mip, cmap='gray', interpolation='bilinear')
        axes[0].set_title('Standard MIP')
        axes[0].axis('off')
        
        # Colored MIP with depth
        im = axes[1].imshow(depth_normalized, cmap=colormap,
                           alpha=np.where(mip > alpha_threshold * mip.max(), 1.0, 0.0),
                           interpolation='bilinear')
        axes[1].set_title('Depth-Colored MIP')
        axes[1].axis('off')
        
        # Add colorbar for depth
        if self.show_colorbar:
            cbar = fig.colorbar(im, ax=axes[1], orientation='vertical',
                               fraction=0.046, pad=0.04)
            cbar.set_label('Relative Depth', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        return fig


def create_mip(
    volume: np.ndarray,
    title: Optional[str] = None,
    cmap: str = 'hot',
    figsize: Tuple[int, int] = (12, 4),
    show_colorbar: bool = True
) -> Figure:
    """
    Convenience function to create maximum intensity projections.
    
    Args:
        volume: 3D volume
        title: Optional title
        cmap: Colormap
        figsize: Figure size
        show_colorbar: Show colorbar
        
    Returns:
        Matplotlib figure
    """
    visualizer = ProjectionVisualizer(
        figsize=figsize,
        cmap=cmap,
        projection_type='max',
        show_colorbar=show_colorbar
    )
    
    return visualizer.create_projections(volume, title)


def create_projection_comparison(
    volume: np.ndarray,
    projection_types: List[str] = ['max', 'mean', 'std'],
    axis: int = 0,
    **kwargs
) -> Figure:
    """
    Create comparison of different projection types.
    
    Args:
        volume: 3D volume
        projection_types: Types of projections to compare
        axis: Projection axis
        **kwargs: Additional arguments for visualizer
        
    Returns:
        Matplotlib figure
    """
    visualizer = ProjectionVisualizer(**kwargs)
    
    return visualizer.create_multi_projection_types(
        volume, projection_types, axis
    )


def create_progressive_mip(
    volumes: List[np.ndarray],
    particle_counts: Optional[List[int]] = None,
    axis: int = 0,
    **kwargs
) -> Figure:
    """
    Create progressive MIP visualizations.
    
    Args:
        volumes: List of progressively improved volumes
        particle_counts: Particle counts for each volume
        axis: Projection axis
        **kwargs: Additional visualizer arguments
        
    Returns:
        Matplotlib figure
    """
    visualizer = ProjectionVisualizer(
        projection_type='max',
        **kwargs
    )
    
    return visualizer.create_progressive_projections(
        volumes, particle_counts, 'max', axis
    )
