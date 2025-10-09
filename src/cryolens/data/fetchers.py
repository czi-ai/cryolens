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
    """
    filename = f"checkpoints/cryolens-{version}-epoch{epoch}.ckpt"
    return CRYOLENS_POOCH.fetch(filename, progressbar=progressbar)


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
            "ML Challenge experimental test set (6 structures, OOD)",
        "training_synthetic": 
            "Synthetic training dataset (104 structures, in-distribution)",
    }
