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

from .fsc import (
    compute_fsc_with_threshold,
    apply_soft_mask,
)

from .ood_reconstruction import (
    evaluate_ood_structure,
    load_mrc_structure,
    align_volume,
    generate_resampled_reconstructions,
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
    # FSC metrics
    'compute_fsc_with_threshold',
    'apply_soft_mask',
    # OOD reconstruction
    'evaluate_ood_structure',
    'load_mrc_structure',
    'align_volume',
    'generate_resampled_reconstructions',
]
