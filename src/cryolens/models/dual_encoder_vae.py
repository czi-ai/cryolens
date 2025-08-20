"""
Dual Encoder VAE implementation for CryoLens.

This module implements a VAE with separate encoders for content and pose,
allowing for better disentanglement of molecular identity and orientation.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import AffinityVAE, fallback_to_cpu


class DualEncoderAffinityVAE(AffinityVAE):
    """VAE with separate encoders for pose and content.
    
    This architecture uses two independent encoders:
    - Content encoder: Learns molecular identity features
    - Pose encoder: Learns rotation/orientation features
    
    The pose is deterministic while content remains variational.
    
    Parameters
    ----------
    content_encoder : torch.nn.Module
        Neural network to encode input data into content latent space.
    pose_encoder : torch.nn.Module
        Neural network to encode input data into pose representation.
    decoder : torch.nn.Module
        Neural network to decode latent space representation back to data space.
    latent_dims : int
        Dimension of the content latent space.
    pose_channels : int
        Number of pose channels (typically 4 for axis-angle representation).
    """
    
    def __init__(
        self,
        content_encoder: nn.Module,
        pose_encoder: nn.Module,
        decoder: nn.Module,
        *,
        latent_dims: int = 8,
        pose_channels: int = 4,
        use_rotated_affinity: bool = False,
        crossover_probability: float = 0.5,
    ):
        # Initialize without calling parent __init__ to avoid single encoder setup
        nn.Module.__init__(self)  # Call grandparent init directly
        
        # Store dual encoders
        self.content_encoder = content_encoder
        self.pose_encoder = pose_encoder
        self.decoder = decoder
        
        self.latent_dims = latent_dims
        self.pose_channels = pose_channels
        self.use_rotated_affinity = use_rotated_affinity
        self.crossover_probability = crossover_probability
        
        # Get flat shapes from encoders
        content_flat_shape = self.content_encoder.flat_shape
        pose_flat_shape = self.pose_encoder.flat_shape
        
        # Content projections (VAE - variational for content)
        self.mu = nn.Linear(content_flat_shape, latent_dims)
        self.log_var = nn.Linear(content_flat_shape, latent_dims)
        
        # Pose projections (deterministic)
        self.pose = nn.Linear(pose_flat_shape, pose_channels)
        self.global_weight = nn.Linear(pose_flat_shape, 1)
        
        # Initialize pose network close to identity rotation
        with torch.no_grad():
            nn.init.normal_(self.pose.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.pose.bias)
            nn.init.normal_(self.global_weight.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.global_weight.bias)
    
    def encode_content(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to content latent distribution.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Mean and log variance of content distribution.
        """
        content_features = self.content_encoder(x)
        mu = self.mu(content_features)
        log_var = self.log_var(content_features)
        return mu, log_var
    
    def encode_pose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to pose and global weight (deterministic).
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Pose and global weight tensors.
        """
        pose_features = self.pose_encoder(x)
        pose = self.pose(pose_features)
        
        # Normalize axis for axis-angle representation
        if self.pose_channels == 4:
            angle = pose[:, 0:1]
            axis = pose[:, 1:4]
            axis = F.normalize(axis, p=2, dim=1, eps=1e-6)
            pose = torch.cat([angle, axis], dim=1)
        
        global_weight = self.global_weight(pose_features)
        return pose, global_weight
    
    @fallback_to_cpu
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input data using both encoders.
        
        This method maintains compatibility with the parent class interface.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Mean, log variance, pose, and global_weight.
        """
        # Get content encoding
        mu, log_var = self.encode_content(x)
        
        # Get pose encoding
        pose, global_weight = self.encode_pose(x)
        
        return mu, log_var, pose, global_weight
    
    @fallback_to_cpu
    def forward(self, x: torch.Tensor, pose: Optional[torch.Tensor] = None, 
                global_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """Forward pass through the dual-encoder VAE.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
        pose : Optional[torch.Tensor]
            Optional explicit pose tensor. If provided, this will be used instead of
            the pose generated by the encoder.
        global_weight : Optional[torch.Tensor]
            Optional explicit global weight tensor.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Reconstructed data, latent representation, pose, global_weight, mean, and log variance
        """
        # Ensure all components are on the same device as input
        device = self._ensure_same_device(x, self.content_encoder, self.pose_encoder, 
                                         self.decoder, self.mu, self.log_var, 
                                         self.pose, self.global_weight)
        
        if not self.use_rotated_affinity:
            # Standard forward pass without rotation
            mu, log_var = self.encode_content(x)
            generated_pose, generated_global_weight = self.encode_pose(x)
            z = self.reparameterize(mu, log_var)
            
        else:
            # Enhanced forward pass with rotated affinity dimensions
            original_mu, original_log_var = self.encode_content(x)
            generated_pose, generated_global_weight = self.encode_pose(x)
            
            # Choose a random 90-degree rotation (excluding 0/360 degrees)
            with torch.no_grad():
                k = torch.randint(1, 4, (1,)).item()
                rotated_x = torch.rot90(x, k=k, dims=(3, 4))
            
            # Get content encoding for rotated input
            rotated_mu, rotated_log_var = self.encode_content(rotated_x)
            
            # Get the ratio of dimensions used for affinity
            latent_dims = original_mu.shape[1]
            affinity_ratio = getattr(self.decoder, 'latent_ratio', 0.75)
            
            # Calculate the number of affinity dimensions
            affinity_dims = int(latent_dims * affinity_ratio)
            
            # Create masks for uniform crossover
            crossover_mask = torch.rand(original_mu.shape[0], affinity_dims, device=device) < self.crossover_probability
            inverse_mask = ~crossover_mask
            
            # Initialize mu and log_var with the original values
            mu = original_mu.clone()
            log_var = original_log_var.clone()
            
            # Apply crossover mask to the affinity dimensions only
            mu[:, :affinity_dims] = crossover_mask * rotated_mu[:, :affinity_dims] + inverse_mask * original_mu[:, :affinity_dims]
            log_var[:, :affinity_dims] = crossover_mask * rotated_log_var[:, :affinity_dims] + inverse_mask * original_log_var[:, :affinity_dims]
            
            # Re-sample z using the crossover latent parameters
            z = self.reparameterize(mu, log_var)
        
        # Use provided pose if available, otherwise use the generated pose
        actual_pose = pose if pose is not None else generated_pose
        actual_global_weight = global_weight if global_weight is not None else generated_global_weight
        
        # Decode
        x_recon = self.decode(z, actual_pose, actual_global_weight)
        
        return x_recon, z, actual_pose, actual_global_weight, mu, log_var
    
    def _ensure_same_device(self, tensor, *modules):
        """Ensures that all modules are on the same device as the input tensor."""
        device = tensor.device
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Check if any module needs to be moved
        device_mismatch = False
        for module in modules:
            if module is None:
                continue
            if hasattr(module, 'parameters') and callable(module.parameters):
                if next(module.parameters(), None) is not None:
                    if next(module.parameters()).device != device:
                        device_mismatch = True
                        break
        
        # If there's a mismatch, move all modules to the tensor's device
        if device_mismatch:
            if rank == 0:
                print(f"Device mismatch detected. Moving components to {device}")
            for module in modules:
                if module is None:
                    continue
                if hasattr(module, 'parameters') and callable(module.parameters):
                    module_params = next(module.parameters(), None)
                    if module_params is not None:
                        module_device = module_params.device
                        if module_device != device:
                            module.to(device)
        
        return device