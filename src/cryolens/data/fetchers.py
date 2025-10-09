"""
Data and model fetching utilities using pooch.
"""
import pooch
from pathlib import Path

# Define the data registry
CRYOLENS_POOCH = pooch.create(
    path=pooch.os_cache("cryolens"),
    base_url="https://virtualcellmodels.cziscience.com/cryolens/",
    registry={
        "checkpoints/cryolens-v1-epoch2600.ckpt": "sha256:PLACEHOLDER_HASH",
    },
)


def fetch_checkpoint(version="v1", epoch=2600, progressbar=True):
    """
    Fetch pre-trained CryoLens checkpoint.
    
    Parameters
    ----------
    version : str
        Checkpoint version (default: "v1")
    epoch : int
        Training epoch of checkpoint (default: 2600)
    progressbar : bool
        Show download progress (default: True)
        
    Returns
    -------
    str
        Path to downloaded checkpoint file
        
    Examples
    --------
    >>> from cryolens.data import fetch_checkpoint
    >>> ckpt_path = fetch_checkpoint()
    >>> print(f"Checkpoint at: {ckpt_path}")
    
    Notes
    -----
    Checkpoints are cached in the pooch cache directory. Use
    ``pooch.os_cache("cryolens")`` to find the cache location.
    
    If the checkpoint file already exists in the cache, it will be used
    without verification (no hash checking during development).
    """
    filename = f"checkpoints/cryolens-{version}-epoch{epoch}.ckpt"
    cache_path = Path(pooch.os_cache("cryolens")) / filename
    
    # If file exists in cache, return it without hash checking
    if cache_path.exists():
        return str(cache_path)
    
    # Otherwise, try to fetch (will fail if not hosted yet)
    try:
        return CRYOLENS_POOCH.fetch(filename, progressbar=progressbar)
    except Exception as e:
        # Provide helpful error message
        raise FileNotFoundError(
            f"Checkpoint not found in cache or online.\n"
            f"Expected location: {cache_path}\n"
            f"To use a local checkpoint, copy it to the above location.\n"
            f"Original error: {e}"
        )


def get_copick_config(config_name):
    """
    Get path to embedded Copick configuration file.
    
    Parameters
    ----------
    config_name : str
        Name of the configuration:
        - "mlc_experimental_publictest": ML Challenge experimental test set
        - "training_synthetic": Synthetic training dataset
        
    Returns
    -------
    str
        Path to Copick configuration JSON file
        
    Examples
    --------
    >>> from cryolens.data import get_copick_config
    >>> config_path = get_copick_config("mlc_experimental_publictest")
    >>> import copick
    >>> root = copick.from_file(config_path)
    
    Notes
    -----
    These configurations connect to datasets on the CZ cryoET Data Portal.
    Data is fetched automatically when accessed through copick.
    """
    import importlib.resources
    
    # Map config names to files
    config_files = {
        "mlc_experimental_publictest": "mlc_experimental_publictest.json",
        "mlc_experimental_privatetest": "mlc_experimental_privatetest.json",
        "mlc_experimental_training": "mlc_experimental_training.json",
        "training_synthetic": "training_synthetic.json",
    }
    
    if config_name not in config_files:
        available = ", ".join(config_files.keys())
        raise ValueError(
            f"Unknown config name: {config_name}. "
            f"Available configs: {available}"
        )
    
    filename = config_files[config_name]
    
    # Use importlib.resources to get path to config file
    # This works whether installed or in development
    try:
        # Python 3.9+
        from importlib.resources import files
        config_path = files("cryolens.data.copick_configs").joinpath(filename)
        return str(config_path)
    except (ImportError, AttributeError):
        # Fallback for older Python
        import importlib.resources as resources
        with resources.path("cryolens.data.copick_configs", filename) as path:
            return str(path)


def list_available_configs():
    """
    List available embedded Copick configurations.
    
    Returns
    -------
    dict
        Dictionary mapping config names to descriptions
        
    Examples
    --------
    >>> from cryolens.data import list_available_configs
    >>> configs = list_available_configs()
    >>> for name, desc in configs.items():
    ...     print(f"{name}: {desc}")
    """
    return {
        "mlc_experimental_publictest": 
            "ML Challenge experimental public test set (6 structures, OOD)",
        "mlc_experimental_privatetest": 
            "ML Challenge experimental private test set (OOD)",
        "mlc_experimental_training": 
            "ML Challenge experimental training set",
        "training_synthetic": 
            "Synthetic training dataset (104 structures, in-distribution)",
    }
