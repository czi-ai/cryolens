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

    # Add to module namespace
    import builtins
    builtins.CurriculumScheduler = CurriculumScheduler
    builtins.TrainingConfig = TrainingConfig
    
    current_module = sys.modules[__name__]
    setattr(current_module, 'CurriculumScheduler', CurriculumScheduler)
    setattr(current_module, 'TrainingConfig', TrainingConfig)
    setattr(current_module, 'tomotwin', DummyTomotwin())
    sys.modules['tomotwin'] = DummyTomotwin()


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
            if part.startswith('abl-') or part.startswith('altcurr') or part.startswith('experiment_'):
                experiment_dir_idx = i
                break
        
        if experiment_dir_idx is not None:
            experiment_dir = Path(*path_parts[:experiment_dir_idx + 1])
            
            # Try multiple possible parameter file names
            possible_files = [
                "training_params.json",
                "training_parameters.json", 
                "config.json",
                "experiment_config.json"
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
    
    # Load checkpoint - try multiple strategies
    checkpoint_state = None
    
    # Strategy 1: Try standard PyTorch loading with weights_only=False (needed for some checkpoints)
    try:
        checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logger.debug("Loaded checkpoint with standard torch.load (weights_only=False)")
    except Exception as e1:
        logger.debug(f"Standard loading failed: {e1}")
        
        # Strategy 2: Try with CPU loading and weights_only=False
        try:
            checkpoint_state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            logger.debug("Loaded checkpoint to CPU (weights_only=False)")
        except Exception as e2:
            logger.debug(f"CPU loading with weights_only=False failed: {e2}")
            
            # Strategy 3: Custom loading with permissive unpickler
            import pickle
            import io
            
            # Create a custom unpickler class
            class PermissiveUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    try:
                        return super().find_class(module, name)
                    except (ImportError, AttributeError, ModuleNotFoundError):
                        logger.warning(f"Creating dummy class for {module}.{name}")
                        return type(name, (), {})
            
            # Custom torch.load function using our unpickler
            def custom_torch_load(path):
                # This is based on how torch.load works internally
                import torch._utils
                
                # Save original unpickler
                original_unpickler = pickle.Unpickler
                
                try:
                    # Monkey-patch pickle module
                    pickle.Unpickler = PermissiveUnpickler
                    
                    # Now use torch.load which will use our custom unpickler
                    result = torch.load(path, map_location='cpu', weights_only=False)
                    return result
                finally:
                    # Restore original unpickler
                    pickle.Unpickler = original_unpickler
            
            try:
                checkpoint_state = custom_torch_load(checkpoint_path)
                logger.debug("Loaded checkpoint with permissive unpickler")
            except Exception as e3:
                logger.debug(f"Custom loading failed: {e3}")
                
                # Strategy 4: Try with weights_only=True as last resort (may lose some data)
                try:
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                    logger.warning("Loaded checkpoint with weights_only=True - some metadata may be missing")
                except Exception as e4:
                    logger.error(f"All loading strategies failed. Last error: {e4}")
                    raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")
    
    if checkpoint_state is None:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")
    
    # Extract state dict
    if 'state_dict' in checkpoint_state:
        state_dict = {k[6:] if k.startswith('model.') else k: v 
                      for k, v in checkpoint_state['state_dict'].items()}
    else:
        state_dict = checkpoint_state
    
    # Load or infer configuration
    config = None
    if load_config:
        config = load_training_parameters(checkpoint_path)
    
    if config is None:
        config = infer_config_from_checkpoint(state_dict, checkpoint_path)
    
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
        use_rotated_affinity=config.get('use_rotated_affinity', False),
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
