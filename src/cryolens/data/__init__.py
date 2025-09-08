"""
CryoLens data loading utilities.
"""

# Import Copick integration if available
try:
    from .copick import (
        CopickDataLoader,
        extract_particles_from_tomogram,
        load_ml_challenge_configs,
        COPICK_AVAILABLE
    )
    __all__ = [
        "CopickDataLoader",
        "extract_particles_from_tomogram", 
        "load_ml_challenge_configs",
        "COPICK_AVAILABLE"
    ]
except ImportError:
    COPICK_AVAILABLE = False
    __all__ = ["COPICK_AVAILABLE"]
