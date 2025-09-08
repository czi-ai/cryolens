"""
Evaluation metrics for CryoLens.

This module provides comprehensive metrics for evaluating:
- Embedding space quality and clustering
- Reconstruction quality
- Pose recovery accuracy
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import chi2
from scipy.linalg import inv

try:
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn metrics not available. Install with: pip install scikit-learn",
        ImportWarning
    )


# ============================================================================
# Embedding Space Metrics
# ============================================================================

def compute_davies_bouldin_index(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Davies-Bouldin index for clustering quality.
    
    Lower values indicate better clustering. The index is the average similarity
    between each cluster and its most similar cluster, where similarity is the
    ratio of within-cluster to between-cluster distances.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding vectors, shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels for each sample, shape (n_samples,)
        
    Returns
    -------
    float
        Davies-Bouldin index (lower is better)
        
    Examples
    --------
    >>> embeddings = np.random.randn(100, 40)
    >>> labels = np.random.randint(0, 5, 100)
    >>> dbi = compute_davies_bouldin_index(embeddings, labels)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for Davies-Bouldin index. "
            "Install with: pip install scikit-learn"
        )
    
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    
    if len(embeddings) != len(labels):
        raise ValueError(f"Length mismatch: {len(embeddings)} embeddings vs {len(labels)} labels")
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        warnings.warn("Davies-Bouldin index requires at least 2 clusters")
        return 0.0
    
    return davies_bouldin_score(embeddings, labels)


def compute_silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """
    Compute silhouette score for clustering quality.
    
    The silhouette score is bounded between -1 and 1, where:
    - 1: Perfect clustering
    - 0: Overlapping clusters
    - -1: Misclassified samples
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding vectors, shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels for each sample, shape (n_samples,)
    metric : str
        Distance metric to use
        
    Returns
    -------
    float
        Silhouette score [-1, 1] (higher is better)
        
    Examples
    --------
    >>> embeddings = np.random.randn(100, 40)
    >>> labels = np.repeat([0, 1, 2, 3], 25)
    >>> score = compute_silhouette_score(embeddings, labels)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for silhouette score. "
            "Install with: pip install scikit-learn"
        )
    
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    
    if len(embeddings) != len(labels):
        raise ValueError(f"Length mismatch: {len(embeddings)} embeddings vs {len(labels)} labels")
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        warnings.warn("Silhouette score requires at least 2 clusters")
        return 0.0
    
    if len(unique_labels) == len(labels):
        warnings.warn("Each sample is its own cluster, silhouette score undefined")
        return 0.0
    
    return silhouette_score(embeddings, labels, metric=metric)


def compute_class_separation_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for class separation in embedding space.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding vectors, shape (n_samples, n_features)
    labels : np.ndarray
        Class labels for each sample, shape (n_samples,)
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'mean_within_class_distance': Average distance within classes
        - 'std_within_class_distance': Std of within-class distances
        - 'mean_between_class_distance': Average distance between classes
        - 'std_between_class_distance': Std of between-class distances
        - 'separation_ratio': Between/within class distance ratio
        - 'davies_bouldin_index': Davies-Bouldin clustering index
        - 'silhouette_score': Silhouette clustering score
        
    Examples
    --------
    >>> embeddings = np.random.randn(100, 40)
    >>> labels = np.repeat([0, 1, 2, 3], 25)
    >>> metrics = compute_class_separation_metrics(embeddings, labels)
    >>> print(f"Separation ratio: {metrics['separation_ratio']:.2f}")
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        warnings.warn("At least 2 classes required for separation metrics")
        return {
            'mean_within_class_distance': 0.0,
            'std_within_class_distance': 0.0,
            'mean_between_class_distance': 0.0,
            'std_between_class_distance': 0.0,
            'separation_ratio': 1.0,
            'davies_bouldin_index': 0.0,
            'silhouette_score': 0.0
        }
    
    within_distances = []
    between_distances = []
    
    for i, label_i in enumerate(unique_labels):
        mask_i = labels == label_i
        embeddings_i = embeddings[mask_i]
        
        # Within-class distances
        if len(embeddings_i) > 1:
            within_dists = pdist(embeddings_i)
            within_distances.extend(within_dists)
        
        # Between-class distances
        for j in range(i + 1, n_classes):
            label_j = unique_labels[j]
            mask_j = labels == label_j
            embeddings_j = embeddings[mask_j]
            
            between_dists = cdist(embeddings_i, embeddings_j).flatten()
            between_distances.extend(between_dists)
    
    metrics = {
        'mean_within_class_distance': float(np.mean(within_distances)) if within_distances else 0.0,
        'std_within_class_distance': float(np.std(within_distances)) if within_distances else 0.0,
        'mean_between_class_distance': float(np.mean(between_distances)) if between_distances else 0.0,
        'std_between_class_distance': float(np.std(between_distances)) if between_distances else 0.0,
    }
    
    # Separation ratio
    if metrics['mean_within_class_distance'] > 0:
        metrics['separation_ratio'] = metrics['mean_between_class_distance'] / metrics['mean_within_class_distance']
    else:
        metrics['separation_ratio'] = float('inf') if metrics['mean_between_class_distance'] > 0 else 1.0
    
    # Clustering metrics
    try:
        metrics['davies_bouldin_index'] = compute_davies_bouldin_index(embeddings, labels)
    except:
        metrics['davies_bouldin_index'] = -1.0
    
    try:
        metrics['silhouette_score'] = compute_silhouette_score(embeddings, labels)
    except:
        metrics['silhouette_score'] = 0.0
    
    return metrics


def compute_mahalanobis_overlap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    regularization: float = 1e-6
) -> Tuple[np.ndarray, List]:
    """
    Compute Mahalanobis distance-based overlap between classes.
    
    The overlap is computed as the probability that a sample from one class
    could belong to another class based on Mahalanobis distance.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding vectors, shape (n_samples, n_features)
    labels : np.ndarray
        Class labels for each sample, shape (n_samples,)
    regularization : float
        Regularization for covariance matrix inversion
        
    Returns
    -------
    overlap_matrix : np.ndarray
        Overlap probabilities between classes, shape (n_classes, n_classes)
    unique_labels : List
        List of unique class labels
        
    Examples
    --------
    >>> embeddings = np.random.randn(100, 40)
    >>> labels = np.repeat([0, 1, 2], [30, 30, 40])
    >>> overlap, class_names = compute_mahalanobis_overlap(embeddings, labels)
    >>> print(f"Overlap between class 0 and 1: {overlap[0, 1]:.3f}")
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    
    unique_labels = list(np.unique(labels))
    n_classes = len(unique_labels)
    n_features = embeddings.shape[1]
    
    overlap_matrix = np.zeros((n_classes, n_classes))
    
    # Compute mean and covariance for each class
    class_stats = {}
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_embeddings = embeddings[mask]
        
        if len(class_embeddings) < 2:
            warnings.warn(f"Class {label} has fewer than 2 samples, skipping")
            continue
        
        mean = np.mean(class_embeddings, axis=0)
        cov = np.cov(class_embeddings.T)
        
        # Regularize covariance matrix
        cov = cov + np.eye(n_features) * regularization
        
        try:
            inv_cov = inv(cov)
            class_stats[i] = {'mean': mean, 'cov': cov, 'inv_cov': inv_cov}
        except:
            warnings.warn(f"Could not invert covariance for class {label}")
            continue
    
    # Compute pairwise overlaps
    for i in range(n_classes):
        if i not in class_stats:
            continue
            
        for j in range(n_classes):
            if j not in class_stats:
                continue
                
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue
            
            # Mahalanobis distance between class means
            mean_diff = class_stats[j]['mean'] - class_stats[i]['mean']
            mahal_dist_sq = mean_diff @ class_stats[i]['inv_cov'] @ mean_diff
            
            # Convert to overlap probability using chi-squared distribution
            # Probability that a sample from class i could be as far as class j's mean
            overlap_prob = 1 - chi2.cdf(mahal_dist_sq, n_features)
            overlap_matrix[i, j] = overlap_prob
    
    return overlap_matrix, unique_labels


def compute_embedding_diversity(
    embeddings: np.ndarray,
    method: str = 'variance'
) -> float:
    """
    Compute diversity of embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding vectors, shape (n_samples, n_features)
    method : str
        Method to compute diversity:
        - 'variance': Average variance across dimensions
        - 'pairwise': Average pairwise distance
        - 'determinant': Log determinant of covariance
        
    Returns
    -------
    float
        Diversity score (higher means more diverse)
        
    Examples
    --------
    >>> embeddings = np.random.randn(100, 40)
    >>> diversity = compute_embedding_diversity(embeddings, method='variance')
    """
    embeddings = np.asarray(embeddings)
    
    if len(embeddings) < 2:
        return 0.0
    
    if method == 'variance':
        return float(np.mean(np.var(embeddings, axis=0)))
    
    elif method == 'pairwise':
        distances = pdist(embeddings)
        return float(np.mean(distances))
    
    elif method == 'determinant':
        cov = np.cov(embeddings.T)
        # Add small regularization for numerical stability
        cov = cov + np.eye(cov.shape[0]) * 1e-10
        sign, logdet = np.linalg.slogdet(cov)
        return float(logdet) if sign > 0 else -float('inf')
    
    else:
        raise ValueError(f"Unknown diversity method: {method}")


# ============================================================================
# Reconstruction Quality Metrics
# ============================================================================

def compute_reconstruction_metrics(
    reconstruction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive reconstruction quality metrics.
    
    Parameters
    ----------
    reconstruction : np.ndarray
        Reconstructed volume, shape (D, H, W) or (H, W)
    ground_truth : Optional[np.ndarray]
        Ground truth volume if available, same shape as reconstruction
    mask : Optional[np.ndarray]
        Binary mask for region of interest, same shape as reconstruction
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'mean_intensity': Mean intensity of reconstruction
        - 'std_intensity': Standard deviation of intensity
        - 'contrast': Contrast (std/mean ratio)
        - 'edge_strength': Mean gradient magnitude
        - 'dynamic_range': Max - min intensity
        - 'sparsity': Fraction of significant voxels
        - 'snr_estimate': Estimated signal-to-noise ratio
        - 'mse': Mean squared error (if ground_truth provided)
        - 'psnr': Peak signal-to-noise ratio (if ground_truth provided)
        - 'ssim': Structural similarity (if ground_truth provided)
        - 'correlation': Correlation coefficient (if ground_truth provided)
        
    Examples
    --------
    >>> recon = np.random.randn(48, 48, 48)
    >>> metrics = compute_reconstruction_metrics(recon)
    >>> print(f"Contrast: {metrics['contrast']:.2f}")
    """
    reconstruction = np.asarray(reconstruction)
    
    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != reconstruction.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match reconstruction {reconstruction.shape}")
        masked_recon = reconstruction[mask]
    else:
        masked_recon = reconstruction.flatten()
    
    # Basic statistics
    metrics = {
        'mean_intensity': float(np.mean(masked_recon)),
        'std_intensity': float(np.std(masked_recon)),
        'dynamic_range': float(np.max(masked_recon) - np.min(masked_recon)),
        'sparsity': float(np.sum(np.abs(masked_recon) > 0.1 * np.std(masked_recon)) / len(masked_recon))
    }
    
    # Contrast
    if abs(metrics['mean_intensity']) > 1e-10:
        metrics['contrast'] = metrics['std_intensity'] / abs(metrics['mean_intensity'])
    else:
        metrics['contrast'] = 0.0
    
    # Edge strength (gradient magnitude)
    if reconstruction.ndim == 3:
        grad_x = np.gradient(reconstruction, axis=0)
        grad_y = np.gradient(reconstruction, axis=1)
        grad_z = np.gradient(reconstruction, axis=2)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    elif reconstruction.ndim == 2:
        grad_x = np.gradient(reconstruction, axis=0)
        grad_y = np.gradient(reconstruction, axis=1)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    else:
        gradient_mag = np.abs(np.gradient(reconstruction))
    
    if mask is not None:
        gradient_mag = gradient_mag[mask]
    
    metrics['edge_strength'] = float(np.mean(gradient_mag))
    
    # SNR estimate (signal power / noise power)
    signal_power = np.mean(masked_recon**2)
    noise_estimate = np.std(masked_recon - np.mean(masked_recon))
    if noise_estimate > 0:
        metrics['snr_estimate'] = float(10 * np.log10(signal_power / (noise_estimate**2)))
    else:
        metrics['snr_estimate'] = float('inf')
    
    # Comparison metrics if ground truth is provided
    if ground_truth is not None:
        ground_truth = np.asarray(ground_truth)
        
        if ground_truth.shape != reconstruction.shape:
            raise ValueError(
                f"Shape mismatch: reconstruction {reconstruction.shape} vs ground_truth {ground_truth.shape}"
            )
        
        if mask is not None:
            masked_gt = ground_truth[mask]
        else:
            masked_gt = ground_truth.flatten()
        
        # MSE and PSNR
        mse = np.mean((masked_recon - masked_gt)**2)
        metrics['mse'] = float(mse)
        
        if mse > 0:
            max_val = max(np.max(np.abs(masked_gt)), np.max(np.abs(masked_recon)))
            metrics['psnr'] = float(20 * np.log10(max_val / np.sqrt(mse)))
        else:
            metrics['psnr'] = float('inf')
        
        # Correlation
        if len(masked_recon) > 1:
            correlation = np.corrcoef(masked_recon, masked_gt)[0, 1]
            metrics['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        else:
            metrics['correlation'] = 0.0
        
        # Structural Similarity (simplified version)
        metrics['ssim'] = compute_ssim(reconstruction, ground_truth, mask)
    
    return metrics


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: Optional[np.ndarray] = None,
    window_size: int = 7
) -> float:
    """
    Compute simplified Structural Similarity Index (SSIM).
    
    Parameters
    ----------
    img1 : np.ndarray
        First image/volume
    img2 : np.ndarray
        Second image/volume
    mask : Optional[np.ndarray]
        Binary mask for region of interest
    window_size : int
        Size of the sliding window
        
    Returns
    -------
    float
        SSIM score [0, 1] (higher is better)
    """
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    
    # Constants for SSIM
    C1 = 0.01**2
    C2 = 0.03**2
    
    # Compute local statistics
    mu1 = img1
    mu2 = img2
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - np.mean(img1)) * (img2 - np.mean(img2)))
    
    # SSIM formula
    numerator = (2 * mu1_mu2.mean() + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq.mean() + mu2_sq.mean() + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = numerator / denominator
    
    return float(np.clip(ssim, 0, 1))


def compute_fourier_shell_correlation(
    volume1: np.ndarray,
    volume2: np.ndarray,
    threshold: float = 0.143
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute Fourier Shell Correlation (FSC) between two volumes.
    
    Parameters
    ----------
    volume1 : np.ndarray
        First volume, shape (D, H, W)
    volume2 : np.ndarray
        Second volume, shape (D, H, W)
    threshold : float
        FSC threshold for resolution estimation
        
    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing:
        - 'resolution': Estimated resolution at threshold
        - 'fsc_curve': FSC values at different frequencies
        - 'frequencies': Frequency values
        
    Examples
    --------
    >>> vol1 = np.random.randn(64, 64, 64)
    >>> vol2 = vol1 + np.random.randn(64, 64, 64) * 0.1
    >>> fsc = compute_fourier_shell_correlation(vol1, vol2)
    """
    volume1 = np.asarray(volume1)
    volume2 = np.asarray(volume2)
    
    if volume1.shape != volume2.shape:
        raise ValueError(f"Shape mismatch: {volume1.shape} vs {volume2.shape}")
    
    if volume1.ndim != 3:
        raise ValueError(f"FSC requires 3D volumes, got shape {volume1.shape}")
    
    # Compute FFTs
    fft1 = np.fft.fftn(volume1)
    fft2 = np.fft.fftn(volume2)
    
    # Compute radial shells
    shape = volume1.shape
    center = [s // 2 for s in shape]
    
    # Create coordinate grids
    coords = np.ogrid[[slice(0, s) for s in shape]]
    
    # Compute distances from center
    distances = np.sqrt(sum((c - center[i])**2 for i, c in enumerate(coords)))
    
    # Define shells
    max_radius = min(center)
    n_shells = max_radius
    shell_boundaries = np.linspace(0, max_radius, n_shells + 1)
    
    fsc_curve = []
    frequencies = []
    
    for i in range(n_shells):
        # Get shell mask
        shell_mask = (distances >= shell_boundaries[i]) & (distances < shell_boundaries[i + 1])
        
        if np.sum(shell_mask) == 0:
            continue
        
        # Compute correlation in shell
        numerator = np.sum(fft1[shell_mask] * np.conj(fft2[shell_mask]))
        denominator = np.sqrt(
            np.sum(np.abs(fft1[shell_mask])**2) * 
            np.sum(np.abs(fft2[shell_mask])**2)
        )
        
        if denominator > 0:
            correlation = np.real(numerator / denominator)
        else:
            correlation = 0.0
        
        fsc_curve.append(correlation)
        frequencies.append((shell_boundaries[i] + shell_boundaries[i + 1]) / 2)
    
    fsc_curve = np.array(fsc_curve)
    frequencies = np.array(frequencies)
    
    # Estimate resolution
    resolution_idx = np.where(fsc_curve < threshold)[0]
    if len(resolution_idx) > 0:
        resolution = frequencies[resolution_idx[0]]
    else:
        resolution = frequencies[-1]
    
    return {
        'resolution': float(resolution),
        'fsc_curve': fsc_curve,
        'frequencies': frequencies
    }


# ============================================================================
# Pose Recovery Metrics (importing from pose module if available)
# ============================================================================

try:
    from cryolens.utils.pose import (
        compute_geodesic_distance,
        kabsch_alignment,
        quaternion_distance,
        compute_rotation_metrics
    )
    
    # Re-export pose metrics for convenience
    __all__ = [
        # Embedding metrics
        'compute_davies_bouldin_index',
        'compute_silhouette_score',
        'compute_class_separation_metrics',
        'compute_mahalanobis_overlap',
        'compute_embedding_diversity',
        # Reconstruction metrics
        'compute_reconstruction_metrics',
        'compute_ssim',
        'compute_fourier_shell_correlation',
        # Pose metrics (re-exported)
        'compute_geodesic_distance',
        'kabsch_alignment',
        'quaternion_distance',
        'compute_rotation_metrics',
    ]
    
except ImportError:
    warnings.warn(
        "Pose metrics not available. Install pose utilities module first.",
        ImportWarning
    )
    
    __all__ = [
        # Embedding metrics
        'compute_davies_bouldin_index',
        'compute_silhouette_score',
        'compute_class_separation_metrics',
        'compute_mahalanobis_overlap',
        'compute_embedding_diversity',
        # Reconstruction metrics
        'compute_reconstruction_metrics',
        'compute_ssim',
        'compute_fourier_shell_correlation',
    ]


# ============================================================================
# Combined Evaluation Pipeline
# ============================================================================

def evaluate_model_performance(
    embeddings: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    reconstructions: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    predicted_poses: Optional[np.ndarray] = None,
    true_poses: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation of model performance across all metrics.
    
    Parameters
    ----------
    embeddings : Optional[np.ndarray]
        Embedding vectors, shape (n_samples, n_features)
    labels : Optional[np.ndarray]
        Class labels for embeddings
    reconstructions : Optional[np.ndarray]
        Reconstructed volumes, shape (n_samples, D, H, W)
    ground_truth : Optional[np.ndarray]
        Ground truth volumes
    predicted_poses : Optional[np.ndarray]
        Predicted rotation matrices, shape (n_samples, 3, 3)
    true_poses : Optional[np.ndarray]
        True rotation matrices
    verbose : bool
        Whether to print results
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary with all computed metrics
        
    Examples
    --------
    >>> results = evaluate_model_performance(
    ...     embeddings=embeddings,
    ...     labels=labels,
    ...     reconstructions=recons,
    ...     ground_truth=gt_volumes
    ... )
    """
    results = {}
    
    # Evaluate embeddings
    if embeddings is not None and labels is not None:
        if verbose:
            print("Evaluating embedding space...")
        
        results['embedding_metrics'] = compute_class_separation_metrics(embeddings, labels)
        results['embedding_diversity'] = {
            'variance': compute_embedding_diversity(embeddings, 'variance'),
            'pairwise': compute_embedding_diversity(embeddings, 'pairwise'),
        }
        
        if verbose:
            print(f"  Separation ratio: {results['embedding_metrics']['separation_ratio']:.2f}")
            print(f"  Silhouette score: {results['embedding_metrics']['silhouette_score']:.3f}")
    
    # Evaluate reconstructions
    if reconstructions is not None:
        if verbose:
            print("Evaluating reconstructions...")
        
        results['reconstruction_metrics'] = []
        
        for i, recon in enumerate(reconstructions):
            gt = ground_truth[i] if ground_truth is not None else None
            metrics = compute_reconstruction_metrics(recon, gt)
            results['reconstruction_metrics'].append(metrics)
        
        # Aggregate metrics
        if results['reconstruction_metrics']:
            avg_metrics = {}
            for key in results['reconstruction_metrics'][0].keys():
                values = [m[key] for m in results['reconstruction_metrics'] if key in m]
                if values:
                    avg_metrics[f'mean_{key}'] = float(np.mean(values))
                    avg_metrics[f'std_{key}'] = float(np.std(values))
            
            results['reconstruction_summary'] = avg_metrics
            
            if verbose:
                print(f"  Mean contrast: {avg_metrics.get('mean_contrast', 0):.2f}")
                if 'mean_correlation' in avg_metrics:
                    print(f"  Mean correlation: {avg_metrics['mean_correlation']:.3f}")
    
    # Evaluate poses
    if predicted_poses is not None and true_poses is not None:
        if verbose:
            print("Evaluating pose recovery...")
        
        try:
            results['pose_metrics'] = compute_rotation_metrics(
                predicted_poses, true_poses, degrees=True
            )
            
            if verbose:
                print(f"  Mean angular error: {results['pose_metrics']['mean_geodesic_error']:.1f}°")
        except:
            warnings.warn("Could not compute pose metrics")
    
    return results
