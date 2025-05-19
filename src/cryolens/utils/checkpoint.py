"""
Utilities for working with checkpoints from the original cryolens repository.
"""

import os
import json
import torch
import logging
from typing import Dict, Tuple, Optional, Any, Union

from cryolens.models.vae import AffinityVAE
from cryolens.models.encoders import Encoder3D
from cryolens.models.decoders import SegmentedGaussianSplatDecoder, GaussianSplatDecoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_checkpoint(
    checkpoint_path: str, 
    config_path: Optional[str] = None, 
    device: Optional[torch.device] = None
) -> Tuple[AffinityVAE, Dict[str, Any]]:
    """
    Load model from a checkpoint with proper configuration to match the original cryolens.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file (.pt)
    config_path : str, optional
        Path to the configuration JSON file. If not provided, will attempt to infer parameters from checkpoint.
    device : torch.device, optional
        Device to load the model onto. If None, uses 'cuda' if available, else 'cpu'.
        
    Returns
    -------
    Tuple[AffinityVAE, Dict[str, Any]]
        The loaded VAE model and the configuration dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove module prefix if present (from DataParallel)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    # Load config if provided
    config = {}
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config file: {str(e)}. Will use checkpoint-based parameters.")
    
    # Determine model parameters from checkpoint
    # Latent dimensions from model's mu layer bias
    latent_dims = state_dict['mu.bias'].shape[0]
    logger.info(f"Detected latent dimensions: {latent_dims}")
    
    # Determine if this is a segmented decoder
    is_segmented_decoder = any('affinity_centroids' in k for k in state_dict.keys())
    logger.info(f"Detected segmented decoder: {is_segmented_decoder}")
    
    # Get volume size from config or use default
    volume_size = config.get('volume_size', config.get('box_size', 48))
    logger.info(f"Using volume size: {volume_size}")
    
    # Get pose dimensions or use default
    pose_dims = config.get('pose_dims', 4)
    logger.info(f"Using pose dimensions: {pose_dims}")
    
    # Create encoder with debugging enabled to check dimensions
    encoder = Encoder3D(
        input_shape=(volume_size, volume_size, volume_size),
        layer_channels=(8, 16, 32, 64),
        debug_output_shape=True
    ).to(device)
    
    # Check if we need to override the flat shape based on mu.weight dimensions
    if 'mu.weight' in state_dict:
        expected_flat_shape = state_dict['mu.weight'].shape[1]
        calculated_flat_shape = encoder.flat_shape
        
        if expected_flat_shape != calculated_flat_shape:
            logger.warning(f"Encoder output shape mismatch: expected {expected_flat_shape}, got {calculated_flat_shape}")
            logger.warning(f"Setting flat shape override to {expected_flat_shape} to match checkpoint")
            encoder.set_flat_shape_override(expected_flat_shape)
    
    # Determine decoder parameters
    if is_segmented_decoder:
        # Determine latent ratio (default to 0.9 for original cryolens compatibility)
        latent_ratio = config.get('latent_ratio', 0.9)
        
        # Try to determine number of splats from checkpoint
        affinity_splats = None
        free_splats = None
        
        for key, value in state_dict.items():
            if 'decoder.affinity_weights.0.bias' in key:
                affinity_splats = value.shape[0]
                logger.info(f"Found affinity_splats={affinity_splats} from checkpoint")
            elif 'decoder.free_weights.0.bias' in key:
                free_splats = value.shape[0]
                logger.info(f"Found free_splats={free_splats} from checkpoint")
        
        if affinity_splats is not None and free_splats is not None:
            n_splats = affinity_splats + free_splats
            logger.info(f"Using total n_splats={n_splats} from checkpoint")
        else:
            # Get from config or use default
            n_splats = config.get('num_splats', 768)
            logger.info(f"Using n_splats={n_splats} from config or default")
        
        # Create segmented decoder with parameters matching original cryolens
        decoder = SegmentedGaussianSplatDecoder(
            shape=(volume_size, volume_size, volume_size),
            latent_dims=latent_dims,
            n_splats=n_splats,
            output_channels=1,
            splat_sigma_range=(0.0005, 0.2),  # Match original server.py values
            padding=9,  # CRITICAL: Use padding=9 to match original!
            latent_ratio=latent_ratio,
            device=device
        ).to(device)
    else:
        # For standard GaussianSplatDecoder
        n_splats = config.get('num_splats', 768)
        for key, value in state_dict.items():
            if 'decoder.weights.0.bias' in key:
                n_splats = value.shape[0]
                logger.info(f"Found n_splats={n_splats} from checkpoint")
                break
        
        decoder = GaussianSplatDecoder(
            shape=(volume_size, volume_size, volume_size),
            latent_dims=latent_dims,
            n_splats=n_splats,
            output_channels=1,
            splat_sigma_range=(0.0005, 0.2),  # Match original server.py values
            padding=9,  # CRITICAL: Use padding=9 to match original!
            device=device
        ).to(device)
    
    # Create VAE model
    model = AffinityVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dims=latent_dims,
        pose_channels=pose_dims
    ).to(device)
    
    # Check for encoder structure mismatch, which is common in converted checkpoints
    # (encoder.0.weight vs encoder.model.0.weight)
    encoder_mismatch = any('encoder.0.' in k for k in state_dict.keys())
    if encoder_mismatch:
        # Remap encoder keys to match the expected structure
        encoder_keys_mapping = {}
        for k in list(state_dict.keys()):
            if k.startswith('encoder.') and not k.startswith('encoder.model.') and not k.startswith('encoder.flat'):
                parts = k.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    # Map encoder.0.weight to encoder.model.0.weight
                    new_key = f"encoder.model.{parts[1]}.{'.'.join(parts[2:])}".rstrip('.')
                    encoder_keys_mapping[k] = new_key
        
        # Apply the mapping
        for old_key, new_key in encoder_keys_mapping.items():
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
    
    # Load weights with lenient matching
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys during loading: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys during loading: {unexpected_keys}")
    
    model.eval()
    logger.info("Model loaded successfully and set to evaluation mode")
    
    # If config was not loaded from file, create a minimal config dict from detected parameters
    if not config:
        config = {
            'volume_size': volume_size,
            'box_size': volume_size,
            'latent_dims': latent_dims,
            'pose_dims': pose_dims,
            'num_splats': n_splats,
            'is_segmented_decoder': is_segmented_decoder
        }
        if is_segmented_decoder:
            config['latent_ratio'] = latent_ratio
    
    return model, config
