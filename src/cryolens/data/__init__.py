"""

# Import fetcher functions
from .fetchers import (
    fetch_checkpoint,
    get_copick_config,
    list_available_configs,
)
CryoLens data loading utilities.
"""

# Import fetcher functions
from .fetchers import (
    fetch_checkpoint,
    get_copick_config,
    list_available_configs,
)

# Import parquet loader functions
from .parquet_loader import (
    extract_volume_from_row,
    extract_pose_from_row,
    load_parquet_samples,
    get_available_structures,
    get_available_snrs,
    load_dataset_with_poses
)

# Import Copick integration if available
try:
    from .copick import (
        CopickDataLoader,
        extract_particles_from_tomogram,
        load_ml_challenge_configs,
        COPICK_AVAILABLE
    )
    __all__ = [
        # Fetcher functions
        "fetch_checkpoint",
        "get_copick_config",
        "list_available_configs",
        # Parquet loader functions
        "extract_volume_from_row",
        "extract_pose_from_row",
        "load_parquet_samples",
        "get_available_structures",
        "get_available_snrs",
        "load_dataset_with_poses",
        # Copick functions
        "CopickDataLoader",
        "extract_particles_from_tomogram", 
        "load_ml_challenge_configs",
        "COPICK_AVAILABLE"
    ]
except ImportError:
    COPICK_AVAILABLE = False
    __all__ = [
        # Fetcher functions
        "fetch_checkpoint",
        "get_copick_config",
        "list_available_configs",
        # Parquet loader functions
        "extract_volume_from_row",
        "extract_pose_from_row",
        "load_parquet_samples",
        "get_available_structures",
        "get_available_snrs",
        "load_dataset_with_poses",
        # Copick availability flag
        "COPICK_AVAILABLE"
    ]
