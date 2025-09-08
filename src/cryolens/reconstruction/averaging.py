"""
Averaging methods for multiple reconstructions.

This module provides tools for averaging aligned reconstructions
with optional weighting schemes.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


class ReconstructionAverager:
    """
    Averager for multiple aligned reconstructions.
    
    This class provides methods for combining multiple reconstructions
    through various averaging schemes.
    
    Attributes:
        weighting_scheme (str): Type of weighting ('uniform', 'snr', 'variance')
        outlier_removal (bool): Whether to remove outliers before averaging
        smoothing_sigma (float): Gaussian smoothing after averaging
    """
    
    def __init__(
        self,
        weighting_scheme: str = 'uniform',
        outlier_removal: bool = False,
        smoothing_sigma: float = 0.0
    ):
        """
        Initialize reconstruction averager.
        
        Args:
            weighting_scheme: Weighting method for averaging
            outlier_removal: Remove outlier voxels
            smoothing_sigma: Post-averaging smoothing
        """
        self.weighting_scheme = weighting_scheme
        self.outlier_removal = outlier_removal
        self.smoothing_sigma = smoothing_sigma
    
    def compute_weights(
        self,
        reconstructions: List[np.ndarray],
        method: str = 'uniform'
    ) -> np.ndarray:
        """
        Compute weights for each reconstruction.
        
        Args:
            reconstructions: List of aligned reconstructions
            method: Weighting method
            
        Returns:
            Weight for each reconstruction
        """
        n_recons = len(reconstructions)
        
        if method == 'uniform':
            return np.ones(n_recons) / n_recons
        
        elif method == 'snr':
            # Weight by signal-to-noise ratio
            weights = np.zeros(n_recons)
            for i, recon in enumerate(reconstructions):
                signal = np.mean(np.abs(recon))
                noise = np.std(recon)
                if noise > 0:
                    weights[i] = signal / noise
                else:
                    weights[i] = 1.0
            
            # Normalize
            weights = weights / np.sum(weights)
            return weights
        
        elif method == 'variance':
            # Weight inversely by variance
            weights = np.zeros(n_recons)
            for i, recon in enumerate(reconstructions):
                var = np.var(recon)
                if var > 0:
                    weights[i] = 1.0 / var
                else:
                    weights[i] = 1.0
            
            # Normalize
            weights = weights / np.sum(weights)
            return weights
        
        elif method == 'contrast':
            # Weight by contrast
            weights = np.zeros(n_recons)
            for i, recon in enumerate(reconstructions):
                weights[i] = np.std(recon)
            
            # Normalize
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n_recons) / n_recons
            return weights
        
        else:
            raise ValueError(f"Unknown weighting method: {method}")
    
    def detect_outliers(
        self,
        reconstructions: List[np.ndarray],
        threshold: float = 3.0
    ) -> np.ndarray:
        """
        Detect outlier voxels across reconstructions.
        
        Args:
            reconstructions: List of reconstructions
            threshold: Z-score threshold for outliers
            
        Returns:
            Mask of outlier voxels
        """
        # Stack reconstructions
        stacked = np.stack(reconstructions, axis=0)
        
        # Compute mean and std across reconstructions
        mean_vol = np.mean(stacked, axis=0)
        std_vol = np.std(stacked, axis=0)
        
        # Initialize outlier mask
        outlier_mask = np.zeros_like(mean_vol, dtype=bool)
        
        # Check each reconstruction for outliers
        for recon in reconstructions:
            z_scores = np.abs((recon - mean_vol) / (std_vol + 1e-8))
            outlier_mask |= (z_scores > threshold)
        
        return outlier_mask
    
    def weighted_average(
        self,
        reconstructions: List[np.ndarray],
        weights: Optional[np.ndarray] = None,
        masks: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute weighted average of reconstructions.
        
        Args:
            reconstructions: List of aligned reconstructions
            weights: Weight for each reconstruction
            masks: Optional masks for each reconstruction
            
        Returns:
            Averaged reconstruction
        """
        if len(reconstructions) == 0:
            raise ValueError("No reconstructions to average")
        
        # Compute weights if not provided
        if weights is None:
            weights = self.compute_weights(reconstructions, self.weighting_scheme)
        
        # Initialize accumulator
        weighted_sum = np.zeros_like(reconstructions[0])
        weight_sum = np.zeros_like(reconstructions[0])
        
        # Accumulate weighted reconstructions
        for i, (recon, weight) in enumerate(zip(reconstructions, weights)):
            if masks is not None and masks[i] is not None:
                mask = masks[i]
                weighted_sum += recon * mask * weight
                weight_sum += mask * weight
            else:
                weighted_sum += recon * weight
                weight_sum += weight
        
        # Avoid division by zero
        mask = weight_sum > 0
        average = np.zeros_like(weighted_sum)
        average[mask] = weighted_sum[mask] / weight_sum[mask]
        
        # Apply smoothing if requested
        if self.smoothing_sigma > 0:
            average = gaussian_filter(average, sigma=self.smoothing_sigma)
        
        return average
    
    def robust_average(
        self,
        reconstructions: List[np.ndarray],
        outlier_threshold: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute robust average with outlier removal.
        
        Args:
            reconstructions: List of reconstructions
            outlier_threshold: Threshold for outlier detection
            
        Returns:
            Tuple of (averaged_reconstruction, outlier_mask)
        """
        # Detect outliers
        outlier_mask = self.detect_outliers(reconstructions, outlier_threshold)
        
        # Create masks excluding outliers
        masks = [~outlier_mask for _ in reconstructions]
        
        # Compute weighted average excluding outliers
        weights = self.compute_weights(reconstructions, self.weighting_scheme)
        average = self.weighted_average(reconstructions, weights, masks)
        
        return average, outlier_mask
    
    def compute_variance_map(
        self,
        reconstructions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute voxel-wise variance across reconstructions.
        
        Args:
            reconstructions: List of reconstructions
            
        Returns:
            Variance map
        """
        stacked = np.stack(reconstructions, axis=0)
        return np.var(stacked, axis=0)
    
    def compute_confidence_map(
        self,
        reconstructions: List[np.ndarray],
        method: str = 'consistency'
    ) -> np.ndarray:
        """
        Compute confidence map for averaged reconstruction.
        
        Args:
            reconstructions: List of reconstructions
            method: Method for confidence calculation
            
        Returns:
            Confidence map (0-1)
        """
        if method == 'consistency':
            # Based on variance - low variance = high confidence
            variance = self.compute_variance_map(reconstructions)
            max_var = np.max(variance)
            if max_var > 0:
                confidence = 1.0 - variance / max_var
            else:
                confidence = np.ones_like(variance)
            
        elif method == 'snr':
            # Based on local SNR
            stacked = np.stack(reconstructions, axis=0)
            mean_vol = np.mean(stacked, axis=0)
            std_vol = np.std(stacked, axis=0)
            
            # Local SNR
            confidence = np.abs(mean_vol) / (std_vol + 1e-8)
            # Normalize to 0-1
            confidence = np.tanh(confidence / 3.0)  # Sigmoid-like normalization
            
        elif method == 'occupancy':
            # Based on how many reconstructions have signal at each voxel
            threshold = 0.1  # Threshold for considering voxel occupied
            occupancy = np.zeros_like(reconstructions[0])
            
            for recon in reconstructions:
                occupancy += (np.abs(recon) > threshold).astype(float)
            
            confidence = occupancy / len(reconstructions)
            
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        
        return confidence
    
    def average_with_alignment(
        self,
        reconstructions: List[np.ndarray],
        alignment_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Average reconstructions with optional alignment information.
        
        Args:
            reconstructions: List of reconstructions (possibly unaligned)
            alignment_info: Optional alignment information
            
        Returns:
            Dictionary with average, variance, and confidence maps
        """
        # Apply alignment if info provided
        if alignment_info is not None and 'rotation_matrices' in alignment_info:
            from ..alignment.pca import PCAAlignment
            aligner = PCAAlignment()
            
            aligned_recons = []
            for recon, rot_mat in zip(reconstructions, alignment_info['rotation_matrices']):
                aligned = aligner.apply_rotation(recon, rot_mat)
                aligned_recons.append(aligned)
            
            reconstructions = aligned_recons
        
        # Compute average
        if self.outlier_removal:
            average, outlier_mask = self.robust_average(reconstructions)
        else:
            weights = self.compute_weights(reconstructions, self.weighting_scheme)
            average = self.weighted_average(reconstructions, weights)
            outlier_mask = None
        
        # Compute additional maps
        variance_map = self.compute_variance_map(reconstructions)
        confidence_map = self.compute_confidence_map(reconstructions)
        
        results = {
            'average': average,
            'variance': variance_map,
            'confidence': confidence_map
        }
        
        if outlier_mask is not None:
            results['outlier_mask'] = outlier_mask
        
        return results


def average_aligned_reconstructions(
    reconstructions: List[np.ndarray],
    weighting: str = 'uniform',
    remove_outliers: bool = False,
    smooth_sigma: float = 0.0
) -> np.ndarray:
    """
    Convenience function for averaging reconstructions.
    
    Args:
        reconstructions: List of aligned reconstructions
        weighting: Weighting scheme
        remove_outliers: Remove outlier voxels
        smooth_sigma: Smoothing after averaging
        
    Returns:
        Averaged reconstruction
    """
    averager = ReconstructionAverager(
        weighting_scheme=weighting,
        outlier_removal=remove_outliers,
        smoothing_sigma=smooth_sigma
    )
    
    if remove_outliers:
        average, _ = averager.robust_average(reconstructions)
    else:
        weights = averager.compute_weights(reconstructions, weighting)
        average = averager.weighted_average(reconstructions, weights)
    
    return average


def weighted_average(
    reconstructions: List[np.ndarray],
    weights: np.ndarray,
    masks: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """
    Simple weighted average of reconstructions.
    
    Args:
        reconstructions: List of reconstructions
        weights: Weight for each reconstruction
        masks: Optional masks
        
    Returns:
        Weighted average
    """
    averager = ReconstructionAverager()
    return averager.weighted_average(reconstructions, weights, masks)
