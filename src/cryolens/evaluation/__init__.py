"""
CryoLens evaluation metrics module.
"""

from .metrics import (
    # Embedding space metrics
    compute_davies_bouldin_index,
    compute_silhouette_score,
    compute_class_separation_metrics,
    compute_mahalanobis_overlap,
    compute_embedding_diversity,
    # Reconstruction quality metrics
    compute_reconstruction_metrics,
    compute_ssim,
    compute_fourier_shell_correlation,
    # Integrated evaluation
    evaluate_model_performance,
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
    # Integrated evaluation
    'evaluate_model_performance',
]