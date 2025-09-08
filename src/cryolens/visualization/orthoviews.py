"""
Orthogonal view visualization for CryoLens reconstructions.

This module provides tools for creating orthogonal view visualizations
of 3D reconstructions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import logging

logger = logging.getLogger(__name__)


class OrthoviewVisualizer:
    """
    Visualizer for orthogonal views of 3D volumes.
    
    This class creates orthogonal slice views (XY, XZ, YZ) of 3D reconstructions
    for visualization and analysis.
    
    Attributes:
        figsize (tuple): Figure size for plots
        cmap (str): Colormap for visualization
        show_axes (bool): Whether to show axis labels
        show_colorbar (bool): Whether to show colorbar
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 4),
        cmap: str = 'gray',
        show_axes: bool = False,
        show_colorbar: bool = True
    ):
        """
        Initialize orthoview visualizer.
        
        Args:
            figsize: Figure size (width, height)
            cmap: Matplotlib colormap
            show_axes: Show axis labels and ticks
            show_colorbar: Show colorbar
        """
        self.figsize = figsize
        self.cmap = cmap
        self.show_axes = show_axes
        self.show_colorbar = show_colorbar
    
    def get_central_slices(
        self,
        volume: np.ndarray,
        slice_indices: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract central orthogonal slices from volume.
        
        Args:
            volume: 3D volume
            slice_indices: Optional specific slice indices (z, y, x)
            
        Returns:
            Tuple of (xy_slice, xz_slice, yz_slice)
        """
        if slice_indices is None:
            # Use central slices
            z_idx = volume.shape[0] // 2
            y_idx = volume.shape[1] // 2
            x_idx = volume.shape[2] // 2
        else:
            z_idx, y_idx, x_idx = slice_indices
        
        # Extract slices
        xy_slice = volume[z_idx, :, :]  # Axial
        xz_slice = volume[:, y_idx, :]  # Coronal
        yz_slice = volume[:, :, x_idx]  # Sagittal
        
        return xy_slice, xz_slice, yz_slice
    
    def create_orthoviews(
        self,
        volume: np.ndarray,
        title: Optional[str] = None,
        slice_indices: Optional[Tuple[int, int, int]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> Figure:
        """
        Create orthogonal view visualization.
        
        Args:
            volume: 3D volume to visualize
            title: Optional title for the figure
            slice_indices: Optional specific slice indices
            vmin: Minimum value for color scaling
            vmax: Maximum value for color scaling
            
        Returns:
            Matplotlib figure
        """
        # Get slices
        xy_slice, xz_slice, yz_slice = self.get_central_slices(volume, slice_indices)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        # Determine color scaling
        if vmin is None:
            vmin = min(xy_slice.min(), xz_slice.min(), yz_slice.min())
        if vmax is None:
            vmax = max(xy_slice.max(), xz_slice.max(), yz_slice.max())
        
        # Plot XY slice (axial)
        im1 = axes[0].imshow(xy_slice, cmap=self.cmap, vmin=vmin, vmax=vmax,
                             aspect='equal', interpolation='bilinear')
        axes[0].set_title('XY (Axial)')
        
        # Plot XZ slice (coronal)
        im2 = axes[1].imshow(xz_slice, cmap=self.cmap, vmin=vmin, vmax=vmax,
                             aspect='equal', interpolation='bilinear')
        axes[1].set_title('XZ (Coronal)')
        
        # Plot YZ slice (sagittal)
        im3 = axes[2].imshow(yz_slice, cmap=self.cmap, vmin=vmin, vmax=vmax,
                             aspect='equal', interpolation='bilinear')
        axes[2].set_title('YZ (Sagittal)')
        
        # Configure axes
        for ax in axes:
            if not self.show_axes:
                ax.axis('off')
            else:
                ax.set_xlabel('Voxels')
                ax.set_ylabel('Voxels')
        
        # Add colorbar if requested
        if self.show_colorbar:
            fig.colorbar(im1, ax=axes, orientation='horizontal', 
                        fraction=0.046, pad=0.1, label='Intensity')
        
        # Add title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return fig
    
    def create_multi_volume_orthoviews(
        self,
        volumes: List[np.ndarray],
        titles: Optional[List[str]] = None,
        slice_indices: Optional[Tuple[int, int, int]] = None,
        share_colorscale: bool = True
    ) -> Figure:
        """
        Create orthoviews for multiple volumes.
        
        Args:
            volumes: List of 3D volumes
            titles: Optional titles for each volume
            slice_indices: Slice indices to use
            share_colorscale: Use same color scale for all volumes
            
        Returns:
            Matplotlib figure
        """
        n_volumes = len(volumes)
        
        if titles is None:
            titles = [f'Volume {i+1}' for i in range(n_volumes)]
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_volumes, 3, 
                                 figsize=(self.figsize[0], self.figsize[1] * n_volumes))
        
        if n_volumes == 1:
            axes = axes.reshape(1, -1)
        
        # Determine color scaling
        if share_colorscale:
            vmin = min(v.min() for v in volumes)
            vmax = max(v.max() for v in volumes)
        else:
            vmin = vmax = None
        
        # Plot each volume
        for i, (volume, title) in enumerate(zip(volumes, titles)):
            xy_slice, xz_slice, yz_slice = self.get_central_slices(volume, slice_indices)
            
            if not share_colorscale:
                vmin = min(xy_slice.min(), xz_slice.min(), yz_slice.min())
                vmax = max(xy_slice.max(), xz_slice.max(), yz_slice.max())
            
            # XY slice
            im = axes[i, 0].imshow(xy_slice, cmap=self.cmap, vmin=vmin, vmax=vmax,
                                   aspect='equal', interpolation='bilinear')
            if i == 0:
                axes[i, 0].set_title('XY (Axial)')
            axes[i, 0].set_ylabel(title, fontsize=10)
            
            # XZ slice
            axes[i, 1].imshow(xz_slice, cmap=self.cmap, vmin=vmin, vmax=vmax,
                             aspect='equal', interpolation='bilinear')
            if i == 0:
                axes[i, 1].set_title('XZ (Coronal)')
            
            # YZ slice
            axes[i, 2].imshow(yz_slice, cmap=self.cmap, vmin=vmin, vmax=vmax,
                             aspect='equal', interpolation='bilinear')
            if i == 0:
                axes[i, 2].set_title('YZ (Sagittal)')
            
            # Configure axes
            for ax in axes[i]:
                if not self.show_axes:
                    ax.axis('off')
        
        # Add colorbar
        if self.show_colorbar:
            fig.colorbar(im, ax=axes, orientation='vertical',
                        fraction=0.046, pad=0.04, label='Intensity')
        
        plt.tight_layout()
        
        return fig
    
    def create_comparison_orthoviews(
        self,
        volume1: np.ndarray,
        volume2: np.ndarray,
        titles: Tuple[str, str] = ('Volume 1', 'Volume 2'),
        show_difference: bool = True
    ) -> Figure:
        """
        Create side-by-side comparison of two volumes.
        
        Args:
            volume1: First volume
            volume2: Second volume
            titles: Titles for the volumes
            show_difference: Show difference map
            
        Returns:
            Matplotlib figure
        """
        n_rows = 3 if show_difference else 2
        
        # Create figure
        fig, axes = plt.subplots(n_rows, 3, figsize=(self.figsize[0], self.figsize[1] * n_rows / 2))
        
        # Get slices for both volumes
        slices1 = self.get_central_slices(volume1)
        slices2 = self.get_central_slices(volume2)
        
        # Shared color scale
        vmin = min(volume1.min(), volume2.min())
        vmax = max(volume1.max(), volume2.max())
        
        # Plot volume 1
        for j, (slice_data, view_name) in enumerate(zip(slices1, ['XY', 'XZ', 'YZ'])):
            axes[0, j].imshow(slice_data, cmap=self.cmap, vmin=vmin, vmax=vmax,
                             aspect='equal', interpolation='bilinear')
            axes[0, j].set_title(view_name)
            if j == 0:
                axes[0, j].set_ylabel(titles[0])
            if not self.show_axes:
                axes[0, j].axis('off')
        
        # Plot volume 2
        for j, slice_data in enumerate(slices2):
            axes[1, j].imshow(slice_data, cmap=self.cmap, vmin=vmin, vmax=vmax,
                             aspect='equal', interpolation='bilinear')
            if j == 0:
                axes[1, j].set_ylabel(titles[1])
            if not self.show_axes:
                axes[1, j].axis('off')
        
        # Plot difference if requested
        if show_difference:
            diff_slices = [s1 - s2 for s1, s2 in zip(slices1, slices2)]
            diff_vmax = max(abs(d.min()) for d in diff_slices + [abs(d.max()) for d in diff_slices])
            
            for j, diff_slice in enumerate(diff_slices):
                im = axes[2, j].imshow(diff_slice, cmap='RdBu_r', 
                                      vmin=-diff_vmax, vmax=diff_vmax,
                                      aspect='equal', interpolation='bilinear')
                if j == 0:
                    axes[2, j].set_ylabel('Difference')
                if not self.show_axes:
                    axes[2, j].axis('off')
            
            # Add difference colorbar
            if self.show_colorbar:
                fig.colorbar(im, ax=axes[2, :], orientation='horizontal',
                            fraction=0.046, pad=0.1, label='Difference')
        
        plt.tight_layout()
        
        return fig


def create_orthoviews(
    volume: np.ndarray,
    title: Optional[str] = None,
    cmap: str = 'gray',
    figsize: Tuple[int, int] = (12, 4),
    show_colorbar: bool = True
) -> Figure:
    """
    Convenience function to create orthogonal views.
    
    Args:
        volume: 3D volume
        title: Optional title
        cmap: Colormap
        figsize: Figure size
        show_colorbar: Show colorbar
        
    Returns:
        Matplotlib figure
    """
    visualizer = OrthoviewVisualizer(
        figsize=figsize,
        cmap=cmap,
        show_colorbar=show_colorbar
    )
    
    return visualizer.create_orthoviews(volume, title)


def plot_orthoviews(
    volumes: Union[np.ndarray, List[np.ndarray]],
    titles: Optional[Union[str, List[str]]] = None,
    comparison: bool = False,
    **kwargs
) -> Figure:
    """
    Plot orthogonal views with flexible input.
    
    Args:
        volumes: Single volume or list of volumes
        titles: Optional titles
        comparison: Create comparison view for 2 volumes
        **kwargs: Additional arguments for visualizer
        
    Returns:
        Matplotlib figure
    """
    visualizer = OrthoviewVisualizer(**kwargs)
    
    # Handle single volume
    if isinstance(volumes, np.ndarray):
        return visualizer.create_orthoviews(volumes, titles)
    
    # Handle multiple volumes
    if len(volumes) == 2 and comparison:
        if titles is None:
            titles = ('Volume 1', 'Volume 2')
        return visualizer.create_comparison_orthoviews(
            volumes[0], volumes[1], titles
        )
    else:
        return visualizer.create_multi_volume_orthoviews(volumes, titles)
