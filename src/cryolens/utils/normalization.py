"""
Data normalization utilities for CryoLens.

This module provides functions for normalizing and denormalizing 3D volumes
using various methods commonly used in cryo-EM data processing.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union


def normalize_volume(
    volume: np.ndarray, 
    method: str = "z-score",
    return_stats: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Normalize a 3D volume using the specified method.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume to normalize
    method : str
        Normalization method. Options:
        - "z-score": Zero mean, unit variance
        - "min-max": Scale to [0, 1] range
        - "percentile": Clip to percentile range and scale to [0, 1]
        - "none": No normalization
    return_stats : bool
        If True, return normalization statistics for denormalization
        
    Returns
    -------
    np.ndarray or tuple
        Normalized volume, optionally with stats dict for denormalization
    """
    if method == "none":
        if return_stats:
            return volume, {"method": "none"}
        return volume
    
    elif method == "z-score":
        mean_val = np.mean(volume)
        std_val = np.std(volume)
        normalized = (volume - mean_val) / (std_val + 1e-6)
        
        if return_stats:
            stats = {
                "method": "z-score",
                "mean": mean_val,
                "std": std_val
            }
            return normalized, stats
        return normalized
    
    elif method == "min-max":
        min_val = np.min(volume)
        max_val = np.max(volume)
        range_val = max_val - min_val
        normalized = (volume - min_val) / (range_val + 1e-6)
        
        if return_stats:
            stats = {
                "method": "min-max",
                "min": min_val,
                "max": max_val
            }
            return normalized, stats
        return normalized
    
    elif method == "percentile":
        # Use 1st and 99th percentiles to be robust to outliers
        p1 = np.percentile(volume, 1)
        p99 = np.percentile(volume, 99)
        range_val = p99 - p1
        normalized = np.clip((volume - p1) / (range_val + 1e-6), 0, 1)
        
        if return_stats:
            stats = {
                "method": "percentile",
                "p1": p1,
                "p99": p99
            }
            return normalized, stats
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize_volume(
    volume: np.ndarray, 
    stats: Dict,
    method: Optional[str] = None
) -> np.ndarray:
    """
    Denormalize a volume using stored statistics.
    
    Parameters
    ----------
    volume : np.ndarray
        Normalized volume to denormalize
    stats : dict
        Statistics from normalization, containing the method and parameters
    method : str, optional
        Override the method from stats (for compatibility)
        
    Returns
    -------
    np.ndarray
        Denormalized volume
    """
    norm_method = method if method is not None else stats.get("method", "none")
    
    if norm_method == "none":
        return volume
    
    elif norm_method == "z-score":
        mean_val = stats.get("mean", 0.0)
        std_val = stats.get("std", 1.0)
        return (volume * std_val) + mean_val
    
    elif norm_method == "min-max":
        min_val = stats.get("min", 0.0)
        max_val = stats.get("max", 1.0)
        range_val = max_val - min_val
        return (volume * range_val) + min_val
    
    elif norm_method == "percentile":
        p1 = stats.get("p1", 0.0)
        p99 = stats.get("p99", 1.0)
        range_val = p99 - p1
        return (volume * range_val) + p1
    
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")


def get_volume_statistics(volume: np.ndarray) -> Dict:
    """
    Compute various statistics for a volume.
    
    Parameters
    ----------
    volume : np.ndarray
        Input volume
        
    Returns
    -------
    dict
        Dictionary containing various statistics
    """
    return {
        "mean": float(np.mean(volume)),
        "std": float(np.std(volume)),
        "min": float(np.min(volume)),
        "max": float(np.max(volume)),
        "median": float(np.median(volume)),
        "p1": float(np.percentile(volume, 1)),
        "p5": float(np.percentile(volume, 5)),
        "p95": float(np.percentile(volume, 95)),
        "p99": float(np.percentile(volume, 99)),
        "shape": volume.shape,
        "dtype": str(volume.dtype)
    }
