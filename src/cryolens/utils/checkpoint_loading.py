"""
Enhanced utilities for checkpoint loading and model configuration.

This module provides comprehensive checkpoint loading functionality that handles
various checkpoint formats, infers configurations, and ensures compatibility.
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union

from cryolens.models.vae import AffinityVAE
from cryolens.models.encoders import Encoder3D
from cryolens.models.decoders import SegmentedGaussianSplatDecoder, GaussianSplatDecoder

# Configure logging
logger = logging.getLogger(__name__)


def create_dummy_classes():
    """Create dummy classes for handling checkpoints with missing dependencies."""
    class CurriculumScheduler:
        def __init__(self, *args, **kwargs):
            pass

    class TrainingConfig:
        def __init__(self, *args, **kwargs):
            pass

    class DummyTomotwin:
        def __init__(self):
            pass
        
        @property
        def TrainingConfig(self):
            return TrainingConfig
    
    class AdaptiveContrastiveAffinityLoss:
        """Dummy class for removed loss function."""
        def __init__(self, *args, **kwargs):
            pass
        
        def forward(self, *args, **kwargs):
            return 0.0

    # Add to multiple namespaces to ensure they're found
    import builtins
    import __main__
    import types
    
    # Add to builtins
    builtins.CurriculumScheduler = CurriculumScheduler
    builtins.TrainingConfig = TrainingConfig
    builtins.AdaptiveContrastiveAffinityLoss = AdaptiveContrastiveAffinityLoss
    
    # Add to __main__ module (where the script runs)
    setattr(__main__, 'CurriculumScheduler', CurriculumScheduler)
    setattr(__main__, 'TrainingConfig', TrainingConfig)
    setattr(__main__, 'tomotwin', DummyTomotwin())
    setattr(__main__, 'AdaptiveContrastiveAffinityLoss', AdaptiveContrastiveAffinityLoss)
    
    # Add to current module
    current_module = sys.modules[__name__]
    setattr(current_module, 'CurriculumScheduler', CurriculumScheduler)
    setattr(current_module, 'TrainingConfig', TrainingConfig)
    setattr(current_module, 'tomotwin', DummyTomotwin())
    setattr(current_module, 'AdaptiveContrastiveAffinityLoss', AdaptiveContrastiveAffinityLoss)
    
    # Also register as a module
    sys.modules['tomotwin'] = DummyTomotwin()
    sys.modules['CurriculumScheduler'] = CurriculumScheduler
    sys.modules['TrainingConfig'] = TrainingConfig
    
    # Create complete module structure for cryolens.training.losses and cryolens.training.distributed
    # This handles cases where the checkpoint expects these modules
    
    # Ensure cryolens exists
    if 'cryolens' not in sys.modules:
        cryolens_module = types.ModuleType('cryolens')
        sys.modules['cryolens'] = cryolens_module
    else:
        cryolens_module = sys.modules['cryolens']
    
    # Create cryolens.training as a package (module with submodules)
    if 'cryolens.training' not in sys.modules:
        training_module = types.ModuleType('cryolens.training')
        training_module.__path__ = []  # Make it a package
        sys.modules['cryolens.training'] = training_module
        setattr(cryolens_module, 'training', training_module)
    else:
        training_module = sys.modules['cryolens.training']
        if not hasattr(training_module, '__path__'):
            training_module.__path__ = []  # Make it a package if it isn't
    
    # Create cryolens.training.losses
    if 'cryolens.training.losses' not in sys.modules:
        losses_module = types.ModuleType('cryolens.training.losses')
        losses_module.__path__ = []  # Make it a package
        sys.modules['cryolens.training.losses'] = losses_module
        setattr(training_module, 'losses', losses_module)
    else:
        losses_module = sys.modules['cryolens.training.losses']
    
    # Add the loss class to the losses module
    setattr(losses_module, 'AdaptiveContrastiveAffinityLoss', AdaptiveContrastiveAffinityLoss)
    
    # Create cryolens.training.distributed
    if 'cryolens.training.distributed' not in sys.modules:
        distributed_module = types.ModuleType('cryolens.training.distributed')
        distributed_module.__path__ = []  # Make it a package
        sys.modules['cryolens.training.distributed'] = distributed_module
        setattr(training_module, 'distributed', distributed_module)
    else:
        distributed_module = sys.modules['cryolens.training.distributed']
        
    # Add common distributed training utilities as dummies
    class DummyDistributedTrainer:
        def __init__(self, *args, **kwargs):
            pass
    
    class DistributedConfig:
        """Dummy class for distributed configuration."""
        def __init__(self, *args, **kwargs):
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.backend = 'nccl'
            self.master_addr = 'localhost'
            self.master_port = '12355'
    
    setattr(distributed_module, 'DistributedTrainer', DummyDistributedTrainer)
    setattr(distributed_module, 'DistributedConfig', DistributedConfig)
    setattr(distributed_module, 'setup_distributed', lambda: None)
    setattr(distributed_module, 'cleanup_distributed', lambda: None)
    
    # Also add to __main__ and builtins for maximum compatibility
    setattr(__main__, 'DistributedConfig', DistributedConfig)
    builtins.DistributedConfig = DistributedConfig
    
    # Handle pathlib._local issue
    # Create a dummy pathlib._local module
    if 'pathlib._local' not in sys.modules:
        pathlib_local = types.ModuleType('pathlib._local')
        sys.modules['pathlib._local'] = pathlib_local
        
        # Add common pathlib classes in case they're needed
        from pathlib import Path, PosixPath, PurePath, PurePosixPath
        setattr(pathlib_local, 'Path', Path)
        setattr(pathlib_local, 'PosixPath', PosixPath)
        setattr(pathlib_local, 'PurePath', PurePath)
        setattr(pathlib_local, 'PurePosixPath', PurePosixPath)
    
    # Also ensure pathlib itself isn't being treated as a package
    import pathlib
    if not hasattr(pathlib, '_local'):
        pathlib._local = types.ModuleType('pathlib._local')
        from pathlib import Path, PosixPath, PurePath, PurePosixPath
        pathlib._local.Path = Path
        pathlib._local.PosixPath = PosixPath
        pathlib._local.PurePath = PurePath
        pathlib._local.PurePosixPath = PurePosixPath


def load_training_parameters(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load training parameters from the experiment directory.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Training parameters dictionary or None if not found
    """
    try:
        path_parts = Path(checkpoint_path).parts
        experiment_dir_idx = None
        
        # Look for experiment directory patterns
        for i, part in enumerate(path_parts):
            if part.startswith('abl-') or part.startswith('altcurr') or part.startswith('cryolens-') or part.startswith('experiment_'):
                experiment_dir_idx = i
                break
        
        if experiment_dir_idx is not None:
            experiment_dir = Path(*path_parts[:experiment_dir_idx + 1])
            
            # Try multiple possible parameter file names
            possible_files = [
                "training_params.json",
                "training_parameters.json", 
                "config.json",
                "experiment_config.json",
                "params.json"
            ]
            
            for filename in possible_files:
                params_file = experiment_dir / filename
                if params_file.exists():
                    with open(params_file, 'r') as f:
                        training_params = json.load(f)
                    logger.info(f"Loaded training parameters from: {params_file}")
                    return training_params
        
        logger.warning(f"Training parameters file not found for checkpoint: {checkpoint_path}")
        return None
        
    except Exception as e:
        logger.error(f"Error loading training parameters: {str(e)}")
        return None


def infer_config_from_checkpoint(state_dict: Dict[str, torch.Tensor], checkpoint_path: str = "") -> Dict[str, Any]:
    """
    Infer configuration from checkpoint state dictionary.
    
    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        Model state dictionary
    checkpoint_path : str
        Path to checkpoint (for context)
        
    Returns
    -------
    Dict[str, Any]
        Inferred configuration
    """
    config = {
        'box_size': 48,  # Default
        'latent_dims': 16,
        'num_splats': 768,
        'latent_ratio': 0.75,
        'splat_sigma_range': (0.005, 0.1),
        'pose_dims': 4,
        'use_rotated_affinity': False,
        'normalization': 'z-score'
    }
    
    # Determine latent dimensions
    if 'mu.bias' in state_dict:
        config['latent_dims'] = state_dict['mu.bias'].shape[0]
        logger.debug(f"Detected latent_dims = {config['latent_dims']}")
    
    # Determine splat configuration
    affinity_splats = None
    free_splats = None
    
    for key, value in state_dict.items():
        if 'decoder.affinity_weights.0.bias' in key:
            affinity_splats = value.shape[0]
        elif 'decoder.free_weights.0.bias' in key:
            free_splats = value.shape[0]
    
    if affinity_splats is not None and free_splats is not None:
        config['num_splats'] = affinity_splats + free_splats
        config['latent_ratio'] = affinity_splats / config['num_splats']
        logger.info(f"Inferred from checkpoint: num_splats={config['num_splats']}, latent_ratio={config['latent_ratio']:.3f}")
    
    return config


def load_vae_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    load_config: bool = True,
    strict_loading: bool = False
) -> Tuple[AffinityVAE, Dict[str, Any]]:
    """
    Load VAE model from checkpoint with automatic configuration detection.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file
    device : torch.device, optional
        Device to load model on
    load_config : bool
        Whether to attempt loading config from experiment directory
    strict_loading : bool
        Whether to use strict state dict loading
        
    Returns
    -------
    Tuple[AffinityVAE, Dict[str, Any]]
        Loaded VAE model and configuration
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy classes for compatibility
    create_dummy_classes()
    
    # Load checkpoint - use the approach that works in gaussian_splat_alignment.py
    checkpoint_state = None
    
    # First try with weights_only=True to avoid pickle issues
    try:
        checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        logger.debug("Loaded checkpoint with weights_only=True")
    except Exception as e1:
        logger.debug(f"weights_only=True failed: {e1}, trying weights_only=False")
        
        # Try with weights_only=False
        try:
            checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            logger.debug("Loaded checkpoint with weights_only=False")
        except Exception as e2:
            logger.debug(f"Standard loading failed: {e2}, trying custom unpickler")
            
            # Use the exact approach from gaussian_splat_alignment.py
            import pickle
            
            # Create a permissive unpickler by temporarily modifying pickle behavior
            old_find_class = None
            
            try:
                # Check if we can access Unpickler.find_class
                if hasattr(pickle.Unpickler, 'find_class'):
                    old_find_class = pickle.Unpickler.find_class
                    
                    def permissive_find_class(self, module, name):
                        try:
                            return old_find_class(self, module, name)
                        except (ImportError, AttributeError):
                            logger.warning(f"Creating dummy class for {module}.{name}")
                            return type(name, (), {})
                    
                    # This might fail on some Python versions, but worth trying
                    try:
                        pickle.Unpickler.find_class = permissive_find_class
                        checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
                        logger.debug("Loaded checkpoint with modified unpickler")
                    finally:
                        if old_find_class:
                            pickle.Unpickler.find_class = old_find_class
                else:
                    # If we can't modify Unpickler, just try loading anyway
                    checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    logger.debug("Loaded checkpoint with standard approach")
                    
            except Exception as e3:
                logger.error(f"All loading attempts failed. Last error: {e3}")
                # As a final fallback, try loading to CPU
                try:
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    logger.warning("Loaded checkpoint to CPU as last resort")
                except Exception as e4:
                    raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e4}")
    
    if checkpoint_state is None:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")
    
    # Extract state dict
    if 'state_dict' in checkpoint_state:
        state_dict = {k[6:] if k.startswith('model.') else k: v 
                      for k, v in checkpoint_state['state_dict'].items()}
    else:
        state_dict = checkpoint_state
    
    # Start with default configuration
    config = {
        'box_size': 48,  # Default
        'latent_dims': 16,
        'num_splats': 768,
        'latent_ratio': 0.75,
        'splat_sigma_range': (0.005, 0.1),
        'pose_dims': 4,
        'use_rotated_affinity': False,
        'normalization': 'z-score'
    }
    
    # Load or infer configuration
    if load_config:
        loaded_params = load_training_parameters(checkpoint_path)
        if loaded_params:
            # Update config with loaded parameters, keeping defaults for missing keys
            for key in config.keys():
                if key in loaded_params:
                    config[key] = loaded_params[key]
            # Also add any extra parameters from the loaded config
            for key, value in loaded_params.items():
                if key not in config:
                    config[key] = value
    
    # Now infer any missing values from the checkpoint
    inferred_config = infer_config_from_checkpoint(state_dict, checkpoint_path)
    
    # Update config with inferred values only if they weren't loaded from file
    for key, value in inferred_config.items():
        if key not in config or (loaded_params is None and key in ['latent_dims', 'num_splats', 'latent_ratio']):
            config[key] = value
    
    # Validate configuration against checkpoint
    if 'mu.bias' in state_dict:
        latent_dims_from_checkpoint = state_dict['mu.bias'].shape[0]
        if latent_dims_from_checkpoint != config['latent_dims']:
            logger.warning(f"Config latent_dims={config['latent_dims']} but checkpoint has {latent_dims_from_checkpoint}")
            config['latent_dims'] = latent_dims_from_checkpoint
    
    # Create model components
    encoder = Encoder3D(
        input_shape=(config['box_size'], config['box_size'], config['box_size']),
        layer_channels=(8, 16, 32, 64)
    )
    
    # Determine decoder type
    is_segmented_decoder = any('affinity_centroids' in k for k in state_dict.keys())
    logger.debug(f"Detected segmented decoder: {is_segmented_decoder}")
    
    if is_segmented_decoder:
        decoder = SegmentedGaussianSplatDecoder(
            (config['box_size'], config['box_size'], config['box_size']),
            latent_dims=config['latent_dims'],
            n_splats=config['num_splats'],
            output_channels=1,
            device=device,
            splat_sigma_range=config.get('splat_sigma_range', (0.005, 0.1)),
            padding=9,
            latent_ratio=config.get('latent_ratio', 0.75)
        )
    else:
        decoder = GaussianSplatDecoder(
            (config['box_size'], config['box_size'], config['box_size']),
            latent_dims=config['latent_dims'],
            n_splats=config['num_splats'],
            output_channels=1,
            device=device,
            splat_sigma_range=config.get('splat_sigma_range', (0.005, 0.1)),
            padding=9
        )
    
    # Create VAE
    vae = AffinityVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dims=config['latent_dims'],
        pose_channels=config.get('pose_dims', 4),
        # Note: use_rotated_affinity was removed from AffinityVAE
    )
    
    # Load weights
    missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=strict_loading)
    
    if missing_keys:
        logger.debug(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        logger.debug(f"Unexpected keys: {len(unexpected_keys)}")
    
    vae = vae.to(device)
    vae.eval()
    
    # Store normalization method
    vae._normalization_method = config.get('normalization', 'z-score')
    vae._config = config
    
    logger.info(f"Model loaded successfully from {checkpoint_path}")
    
    return vae, config




def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
    optimizer_state: Optional[Dict] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    save_training_objects: bool = False
) -> None:
    """
    Save a checkpoint with only essential state to avoid pickle issues.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to save
    checkpoint_path : str
        Path where to save the checkpoint
    config : Dict[str, Any], optional
        Model configuration
    optimizer_state : Dict, optional
        Optimizer state dict
    epoch : int, optional
        Current epoch
    loss : float, optional
        Current loss value
    metadata : Dict[str, Any], optional
        Additional metadata to save
    save_training_objects : bool
        Whether to save training objects (loss functions, schedulers, etc.)
        Default False to avoid pickle issues
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'checkpoint_version': '2.0',  # Version for tracking checkpoint format
    }
    
    # Add optional components
    if config is not None:
        checkpoint['config'] = config
    
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    if metadata is not None:
        # Filter out non-serializable objects from metadata
        safe_metadata = {}
        for key, value in metadata.items():
            # Only save basic types
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                safe_metadata[key] = value
            elif hasattr(value, '__dict__'):
                # Try to save class name for reference
                safe_metadata[f'{key}_class'] = value.__class__.__name__
        checkpoint['metadata'] = safe_metadata
    
    # Save training objects only if explicitly requested
    if save_training_objects and metadata:
        # Store class names instead of actual objects
        training_info = {}
        for key in ['loss_fn', 'scheduler', 'curriculum']:
            if key in metadata:
                obj = metadata[key]
                if hasattr(obj, '__class__'):
                    training_info[f'{key}_class'] = obj.__class__.__name__
                    training_info[f'{key}_module'] = obj.__class__.__module__
        if training_info:
            checkpoint['training_info'] = training_info
    
    # Save with weights_only=False to preserve all information
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint_safe(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    load_optimizer: bool = False
) -> Dict[str, Any]:
    """
    Safe checkpoint loading that handles various checkpoint formats.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint
    model : torch.nn.Module, optional
        Model to load state into
    device : torch.device, optional
        Device to load to
    load_optimizer : bool
        Whether to load optimizer state
        
    Returns
    -------
    Dict[str, Any]
        Checkpoint dictionary with loaded components
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure dummy classes exist
    create_dummy_classes()
    
    # Try loading with our safe approach
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        logger.info("Loaded checkpoint with weights_only=True")
    except:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            logger.info("Loaded checkpoint with weights_only=False")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    # Handle different checkpoint formats
    result = {}
    
    # Extract model state
    if 'model_state_dict' in checkpoint:
        result['model_state_dict'] = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        # Handle old format
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present
        result['model_state_dict'] = {k[6:] if k.startswith('model.') else k: v 
                                      for k, v in state_dict.items()}
    else:
        # Assume the checkpoint itself is the state dict
        result['model_state_dict'] = checkpoint
    
    # Load into model if provided
    if model is not None:
        model.load_state_dict(result['model_state_dict'], strict=False)
        model.to(device)
    
    # Extract other components
    for key in ['config', 'epoch', 'loss', 'metadata', 'training_info']:
        if key in checkpoint:
            result[key] = checkpoint[key]
    
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
    


# Keep the original load_checkpoint function for backward compatibility
def load_checkpoint(
    checkpoint_path: str, 
    config_path: Optional[str] = None, 
    device: Optional[torch.device] = None
) -> Tuple[AffinityVAE, Dict[str, Any]]:
    """
    Load model from a checkpoint with proper configuration.
    
    This is the original function kept for backward compatibility.
    Consider using load_vae_model() for more features.
    """
    # Use the new enhanced loading function
    if config_path:
        # Load config explicitly if provided
        with open(config_path, 'r') as f:
            config = json.load(f)
        vae, _ = load_vae_model(checkpoint_path, device, load_config=False)
        vae._config = config
        return vae, config
    else:
        return load_vae_model(checkpoint_path, device)
