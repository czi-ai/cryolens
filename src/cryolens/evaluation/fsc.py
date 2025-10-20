"""
Fourier Shell Correlation (FSC) metrics for reconstruction quality.

This module provides FSC computation for evaluating reconstruction quality
in cryo-ET data analysis.
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftshift
from typing import Tuple, Optional


def apply_soft_mask(
    volume: np.ndarray,
    radius: float = 20.0,
    soft_edge: float = 5.0
) -> np.ndarray:
    """
    Apply soft spherical mask with Gaussian edge.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume (D, H, W)
    radius : float
        Mask radius in voxels
    soft_edge : float
        Width of Gaussian soft edge in voxels
        
    Returns
    -------
    np.ndarray
        Masked volume
        
    Examples
    --------
    >>> volume = np.random.randn(48, 48, 48)
    >>> masked = apply_soft_mask(volume, radius=20, soft_edge=5)
    """
    shape = volume.shape
    center = np.array(shape) // 2
    
    nz, ny, nx = shape
    z, y, x = np.ogrid[:nz, :ny, :nx]
    r = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    
    mask = np.ones_like(r, dtype=float)
    transition_zone = (r > radius) & (r < radius + soft_edge)
    mask[transition_zone] = np.exp(-(r[transition_zone] - radius)**2 / (2 * (soft_edge/3)**2))
    mask[r >= radius + soft_edge] = 0
    
    return volume * mask


def compute_fsc_with_threshold(
    vol1: np.ndarray,
    vol2: np.ndarray,
    voxel_size: float = 10.0,
    threshold: float = 0.5,
    mask_radius: Optional[float] = None,
    soft_edge: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Fourier Shell Correlation and estimate resolution at threshold.
    
    This function uses the 0.5 threshold by default for map-to-model comparison
    (not half-map comparison where 0.143 would be appropriate).
    
    Parameters
    ----------
    vol1 : np.ndarray
        First volume (D, H, W)
    vol2 : np.ndarray
        Second volume (D, H, W)
    voxel_size : float
        Voxel size in Angstroms
    threshold : float
        FSC threshold for resolution estimation (default: 0.5)
    mask_radius : Optional[float]
        Radius for soft masking in voxels (default: None, no masking)
    soft_edge : float
        Width of mask soft edge
        
    Returns
    -------
    resolutions : np.ndarray
        Resolution at each frequency shell (Angstroms)
    fsc_values : np.ndarray
        FSC value at each shell
    resolution_at_threshold : float
        Estimated resolution using linear interpolation (Angstroms)
        
    Notes
    -----
    The 0.5 threshold is used because this performs map-to-model comparison
    with known ground truth structures, not independent half-map comparison
    where 0.143 threshold would be appropriate.
    
    Examples
    --------
    >>> vol1 = np.random.randn(48, 48, 48)
    >>> vol2 = vol1 + np.random.randn(48, 48, 48) * 0.1
    >>> res_array, fsc_array, res_at_half = compute_fsc_with_threshold(
    ...     vol1, vol2, voxel_size=10.0
    ... )
    >>> print(f"Resolution at FSC=0.5: {res_at_half:.1f}Å")
    """
    vol1 = np.asarray(vol1)
    vol2 = np.asarray(vol2)
    
    if vol1.shape != vol2.shape:
        raise ValueError(f"Shape mismatch: {vol1.shape} vs {vol2.shape}")
    
    if vol1.ndim != 3:
        raise ValueError(f"FSC requires 3D volumes, got shape {vol1.shape}")
    
    # Apply soft mask if requested
    if mask_radius is not None:
        vol1 = apply_soft_mask(vol1, radius=mask_radius, soft_edge=soft_edge)
        vol2 = apply_soft_mask(vol2, radius=mask_radius, soft_edge=soft_edge)
    
    # Compute FFTs
    fft1 = fftshift(fftn(vol1))
    fft2 = fftshift(fftn(vol2))
    
    # Get dimensions
    nz, ny, nx = vol1.shape
    center = np.array([nz//2, ny//2, nx//2])
    
    # Create distance array
    z, y, x = np.ogrid[:nz, :ny, :nx]
    r = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    
    # Compute FSC for each shell
    max_radius = min(center)
    n_shells = int(max_radius)
    
    fsc_values = []
    resolutions = []
    
    for i in range(1, n_shells):
        # Create shell mask
        mask = (r >= i - 0.5) & (r < i + 0.5)
        
        if np.sum(mask) < 10:  # Need minimum samples
            continue
        
        # Get FFT values in this shell
        fft1_shell = fft1[mask]
        fft2_shell = fft2[mask]
        
        # Compute correlation
        numerator = np.real(np.sum(fft1_shell * np.conj(fft2_shell)))
        denominator = np.sqrt(np.sum(np.abs(fft1_shell)**2) * np.sum(np.abs(fft2_shell)**2))
        
        if denominator > 0:
            fsc = numerator / denominator
        else:
            fsc = 0.0
        
        fsc_values.append(fsc)
        
        # Convert radius to resolution (Angstroms)
        resolution = (nx * voxel_size) / (2.0 * i)
        resolutions.append(resolution)
    
    fsc_array = np.array(fsc_values)
    res_array = np.array(resolutions)
    
    # Find resolution at threshold using linear interpolation
    resolution_at_threshold = res_array[0] if len(res_array) > 0 else 240.0  # Default
    
    if len(fsc_array) > 1:
        # Find indices bracketing threshold
        idx_above = np.where(fsc_array >= threshold)[0]
        idx_below = np.where(fsc_array < threshold)[0]
        
        if len(idx_above) > 0 and len(idx_below) > 0:
            # Get last point above threshold and first point below
            idx1 = idx_above[-1]
            idx2 = idx_below[0]
            
            if idx2 == idx1 + 1:  # Adjacent points bracket threshold
                # Linear interpolation
                fsc1, fsc2 = fsc_array[idx1], fsc_array[idx2]
                res1, res2 = res_array[idx1], res_array[idx2]
                
                # Interpolate in resolution space
                t = (threshold - fsc1) / (fsc2 - fsc1)
                resolution_at_threshold = res1 + t * (res2 - res1)
            else:
                # Use first point below threshold
                resolution_at_threshold = res_array[idx_below[0]]
        elif len(idx_below) > 0:
            # All points below threshold
            resolution_at_threshold = res_array[0]
        else:
            # All points above threshold
            resolution_at_threshold = res_array[-1] if len(res_array) > 0 else 240.0
    
    return res_array, fsc_array, resolution_at_threshold
