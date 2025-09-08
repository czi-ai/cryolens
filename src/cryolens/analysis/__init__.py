"""
Analysis module for CryoLens.

This module provides tools for analyzing embeddings, reconstructions,
and model performance.
"""

from .diversity import (
    DiversityAnalyzer,
    compute_soap_similarity,
    compute_embedding_diversity
)
from .metrics import (
    ReconstructionMetrics,
    calculate_quality_metrics,
    calculate_progressive_metrics
)
from .classification import (
    ClassificationEvaluator,
    evaluate_embeddings,
    cross_validate_classifier
)

__all__ = [
    'DiversityAnalyzer',
    'compute_soap_similarity',
    'compute_embedding_diversity',
    'ReconstructionMetrics',
    'calculate_quality_metrics',
    'calculate_progressive_metrics',
    'ClassificationEvaluator',
    'evaluate_embeddings',
    'cross_validate_classifier'
]
