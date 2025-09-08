"""
Quality metrics for CryoLens reconstructions.

This module provides metrics for evaluating reconstruction quality,
including SNR, contrast, sharpness, and progressive improvement tracking.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import ndimage
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)


class ReconstructionMetrics:
    """
    Calculator for reconstruction quality metrics.
    
    This class provides comprehensive metrics for evaluating
    3D reconstruction quality and tracking progressive improvements.
    
    Attributes:
        compute_gradients (bool): Whether to compute gradient-based metrics
        compute_spectrum (bool): Whether to compute frequency spectrum metrics
    """
    
    def __init__(
        self,
        compute_gradients: bool = True,
        compute_spectrum: bool = True
    ):
        """
        Initialize metrics calculator.
        
        Args:
            compute_gradients: Compute gradient-based sharpness metrics
            compute_spectrum: Compute frequency spectrum analysis
        """
        self.compute_gradients = compute_gradients
        self.compute_spectrum = compute_spectrum
    
    def calculate_basic_stats(self, volume: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic statistical metrics.
        
        Args:
            volume: 3D reconstruction volume
            
        Returns:
            Dictionary of basic statistics
        """
        # Handle empty or invalid volumes
        if volume.size == 0:
            return {
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'min_intensity': 0.0,
                'max_intensity': 0.0,
                'dynamic_range': 0.0
            }
        
        return {
            'mean_intensity': float(np.mean(volume)),
            'std_intensity': float(np.std(volume)),
            'min_intensity': float(np.min(volume)),
            'max_intensity': float(np.max(volume)),
            'dynamic_range': float(np.max(volume) - np.min(volume))
        }
    
    def calculate_contrast_metrics(self, volume: np.ndarray) -> Dict[str, float]:
        """
        Calculate contrast-related metrics.
        
        Args:
            volume: 3D reconstruction volume
            
        Returns:
            Dictionary of contrast metrics
        """
        metrics = {}
        
        # Weber contrast
        mean_val = np.mean(np.abs(volume))
        if mean_val > 0:
            metrics['weber_contrast'] = float(np.std(volume) / mean_val)
        else:
            metrics['weber_contrast'] = 0.0
        
        # Michelson contrast  
        max_val = np.max(volume)
        min_val = np.min(volume)
        if (max_val + min_val) != 0:
            metrics['michelson_contrast'] = float((max_val - min_val) / (max_val + min_val))
        else:
            metrics['michelson_contrast'] = 0.0
        
        # RMS contrast
        metrics['rms_contrast'] = float(np.std(volume))
        
        return metrics
    
    def calculate_sharpness_metrics(self, volume: np.ndarray) -> Dict[str, float]:
        """
        Calculate sharpness/edge strength metrics.
        
        Args:
            volume: 3D reconstruction volume
            
        Returns:
            Dictionary of sharpness metrics
        """
        if not self.compute_gradients:
            return {}
        
        metrics = {}
        
        # Gradient magnitude (edge strength)
        grad_x = np.gradient(volume, axis=0)
        grad_y = np.gradient(volume, axis=1)
        grad_z = np.gradient(volume, axis=2)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        metrics['mean_edge_strength'] = float(np.mean(gradient_magnitude))
        metrics['max_edge_strength'] = float(np.max(gradient_magnitude))
        metrics['edge_density'] = float(np.sum(gradient_magnitude > np.mean(gradient_magnitude)) / volume.size)
        
        # Laplacian variance (focus measure)
        laplacian = ndimage.laplace(volume)
        metrics['laplacian_variance'] = float(np.var(laplacian))
        
        # Tenengrad metric
        sobel_x = ndimage.sobel(volume, axis=0)
        sobel_y = ndimage.sobel(volume, axis=1)
        sobel_z = ndimage.sobel(volume, axis=2)
        tenengrad = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)
        metrics['tenengrad'] = float(np.mean(tenengrad**2))
        
        return metrics
    
    def calculate_noise_metrics(self, volume: np.ndarray) -> Dict[str, float]:
        """
        Calculate noise-related metrics.
        
        Args:
            volume: 3D reconstruction volume
            
        Returns:
            Dictionary of noise metrics
        """
        metrics = {}
        
        # Signal-to-noise ratio (simplified)
        signal = np.mean(np.abs(volume))
        noise = np.std(volume)
        if noise > 0:
            metrics['snr'] = float(signal / noise)
            metrics['snr_db'] = float(20 * np.log10(signal / noise))
        else:
            metrics['snr'] = float('inf')
            metrics['snr_db'] = float('inf')
        
        # Peak SNR
        max_val = np.max(np.abs(volume))
        if noise > 0:
            metrics['psnr'] = float(20 * np.log10(max_val / noise))
        else:
            metrics['psnr'] = float('inf')
        
        # Entropy (information content)
        # Discretize for entropy calculation
        n_bins = 256
        hist, _ = np.histogram(volume, bins=n_bins)
        hist = hist / np.sum(hist)  # Normalize
        metrics['entropy'] = float(entropy(hist + 1e-10))  # Add small value to avoid log(0)
        
        return metrics
    
    def calculate_sparsity_metrics(self, volume: np.ndarray) -> Dict[str, float]:
        """
        Calculate sparsity-related metrics.
        
        Args:
            volume: 3D reconstruction volume
            
        Returns:
            Dictionary of sparsity metrics
        """
        metrics = {}
        
        # Threshold for considering voxel as "active"
        threshold = np.mean(np.abs(volume)) + np.std(volume)
        
        # Sparsity (fraction of significant voxels)
        metrics['sparsity'] = float(np.sum(np.abs(volume) > threshold) / volume.size)
        
        # L0 pseudo-norm (count of non-zero voxels)
        metrics['l0_norm'] = float(np.count_nonzero(volume))
        
        # Gini coefficient (inequality measure)
        sorted_vals = np.sort(np.abs(volume).flatten())
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n
        metrics['gini_coefficient'] = float(gini)
        
        return metrics
    
    def calculate_spectrum_metrics(self, volume: np.ndarray) -> Dict[str, float]:
        """
        Calculate frequency spectrum metrics.
        
        Args:
            volume: 3D reconstruction volume
            
        Returns:
            Dictionary of spectrum metrics
        """
        if not self.compute_spectrum:
            return {}
        
        metrics = {}
        
        # Compute 3D FFT
        fft = np.fft.fftn(volume)
        fft_abs = np.abs(fft)
        
        # Power spectrum
        power_spectrum = fft_abs**2
        
        # Total power
        metrics['total_power'] = float(np.sum(power_spectrum))
        
        # Low frequency power (center of FFT)
        center = [s // 2 for s in volume.shape]
        low_freq_radius = min(volume.shape) // 8
        
        # Create spherical mask for low frequencies
        coords = np.ogrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
        distance = np.sqrt(sum((c - cent)**2 for c, cent in zip(coords, center)))
        low_freq_mask = distance <= low_freq_radius
        
        low_freq_power = np.sum(power_spectrum[low_freq_mask])
        metrics['low_freq_power_ratio'] = float(low_freq_power / (metrics['total_power'] + 1e-10))
        
        # High frequency power
        high_freq_mask = distance > low_freq_radius * 2
        high_freq_power = np.sum(power_spectrum[high_freq_mask])
        metrics['high_freq_power_ratio'] = float(high_freq_power / (metrics['total_power'] + 1e-10))
        
        # Spectral entropy
        power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        metrics['spectral_entropy'] = float(-np.sum(power_norm * np.log(power_norm + 1e-10)))
        
        return metrics
    
    def calculate_all_metrics(self, volume: np.ndarray) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            volume: 3D reconstruction volume
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Combine all metric categories
        metrics.update(self.calculate_basic_stats(volume))
        metrics.update(self.calculate_contrast_metrics(volume))
        metrics.update(self.calculate_sharpness_metrics(volume))
        metrics.update(self.calculate_noise_metrics(volume))
        metrics.update(self.calculate_sparsity_metrics(volume))
        metrics.update(self.calculate_spectrum_metrics(volume))
        
        return metrics
    
    def calculate_progressive_metrics(
        self,
        reconstructions: List[np.ndarray],
        particle_counts: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Calculate metrics for progressive reconstructions.
        
        Args:
            reconstructions: List of reconstructions with increasing particles
            particle_counts: Number of particles for each reconstruction
            
        Returns:
            Dictionary with progressive metrics and trends
        """
        if particle_counts is None:
            particle_counts = list(range(1, len(reconstructions) + 1))
        
        # Calculate metrics for each reconstruction
        all_metrics = []
        for recon in reconstructions:
            metrics = self.calculate_all_metrics(recon)
            all_metrics.append(metrics)
        
        # Organize by metric type
        progressive_data = {
            'particle_counts': particle_counts,
            'metrics': all_metrics
        }
        
        # Calculate trends
        trends = {}
        if len(all_metrics) > 1:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                
                # Calculate trend (linear fit)
                if not any(np.isnan(values) or np.isinf(v) for v in values):
                    x = np.array(particle_counts)
                    y = np.array(values)
                    
                    # Fit linear trend
                    coeffs = np.polyfit(x, y, 1)
                    trends[key] = {
                        'slope': float(coeffs[0]),
                        'intercept': float(coeffs[1]),
                        'values': values
                    }
        
        progressive_data['trends'] = trends
        
        # Identify improving metrics
        improving_metrics = []
        degrading_metrics = []
        
        for metric, trend in trends.items():
            if abs(trend['slope']) > 1e-6:  # Significant change
                if 'contrast' in metric or 'edge' in metric or 'snr' in metric:
                    # These should increase
                    if trend['slope'] > 0:
                        improving_metrics.append(metric)
                    else:
                        degrading_metrics.append(metric)
                elif 'noise' in metric and 'snr' not in metric:
                    # Noise should decrease
                    if trend['slope'] < 0:
                        improving_metrics.append(metric)
                    else:
                        degrading_metrics.append(metric)
        
        progressive_data['improving_metrics'] = improving_metrics
        progressive_data['degrading_metrics'] = degrading_metrics
        
        return progressive_data


def calculate_quality_metrics(
    reconstruction: np.ndarray,
    reference: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Convenience function for calculating reconstruction quality metrics.
    
    Args:
        reconstruction: 3D reconstruction volume
        reference: Optional reference volume for comparison
        
    Returns:
        Dictionary of quality metrics
    """
    calculator = ReconstructionMetrics()
    metrics = calculator.calculate_all_metrics(reconstruction)
    
    # Add comparison metrics if reference provided
    if reference is not None:
        # MSE
        mse = np.mean((reconstruction - reference)**2)
        metrics['mse'] = float(mse)
        
        # RMSE
        metrics['rmse'] = float(np.sqrt(mse))
        
        # Normalized RMSE
        ref_range = np.max(reference) - np.min(reference)
        if ref_range > 0:
            metrics['nrmse'] = float(metrics['rmse'] / ref_range)
        
        # Correlation coefficient
        correlation = np.corrcoef(reconstruction.flatten(), reference.flatten())[0, 1]
        metrics['correlation'] = float(correlation)
        
        # Structural similarity (simplified)
        mean_recon = np.mean(reconstruction)
        mean_ref = np.mean(reference)
        std_recon = np.std(reconstruction)
        std_ref = np.std(reference)
        
        if std_recon > 0 and std_ref > 0:
            cov = np.mean((reconstruction - mean_recon) * (reference - mean_ref))
            c1 = 0.01**2  # Stability constant
            c2 = 0.03**2
            
            ssim = ((2 * mean_recon * mean_ref + c1) * (2 * cov + c2)) / \
                   ((mean_recon**2 + mean_ref**2 + c1) * (std_recon**2 + std_ref**2 + c2))
            metrics['ssim'] = float(ssim)
    
    return metrics


def calculate_progressive_metrics(
    reconstructions: List[np.ndarray],
    particle_counts: Optional[List[int]] = None,
    reference: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Convenience function for analyzing progressive reconstruction quality.
    
    Args:
        reconstructions: List of reconstructions
        particle_counts: Number of particles per reconstruction
        reference: Optional reference for comparison
        
    Returns:
        Dictionary with progressive analysis
    """
    calculator = ReconstructionMetrics()
    results = calculator.calculate_progressive_metrics(reconstructions, particle_counts)
    
    # Add comparison to reference if provided
    if reference is not None:
        reference_comparisons = []
        for recon in reconstructions:
            comp_metrics = calculate_quality_metrics(recon, reference)
            reference_comparisons.append({
                'mse': comp_metrics.get('mse', np.nan),
                'correlation': comp_metrics.get('correlation', np.nan),
                'ssim': comp_metrics.get('ssim', np.nan)
            })
        results['reference_comparisons'] = reference_comparisons
    
    return results
