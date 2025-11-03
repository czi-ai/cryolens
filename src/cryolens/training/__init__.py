"""Training module for CryoLens."""

# Import losses with better error handling to prevent module caching issues
# This is particularly important when installing from zip files in environments
# like Google Colab where dependencies might not be fully initialized on first import
try:
    from .losses import (
        MissingWedgeLoss, 
        NormalizedMSELoss, 
        ContrastiveAffinityLoss, 
        AffinityCosineLoss,
    )
    
    __all__ = [
        'MissingWedgeLoss',
        'NormalizedMSELoss', 
        'ContrastiveAffinityLoss',
        'AffinityCosineLoss',
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import losses: {e}. Loss functions will not be available.")
    __all__ = []
