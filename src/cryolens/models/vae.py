"""
Variational Autoencoder (VAE) implementation for CryoLens.
"""

from typing import Tuple, Dict, Any, Optional, Union
import io
import sys
import pickle
import importlib
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


def fallback_to_cpu(method):
    """
    Decorator that handles CUDA errors by falling back to CPU execution.
    
    This decorator catches specific CUDA errors related to unsupported operations
    and re-executes the wrapped method on CPU before moving results back to the
    original device.
    
    Parameters
    ----------
    method : callable
        The method to wrap with CPU fallback logic
    
    Returns
    -------
    callable
        The wrapped method with error handling
    """
    def wrapper(self, *args, **kwargs):
        try:
            # Try normal execution
            return method(self, *args, **kwargs)
        except RuntimeError as e:
            # Check if it's a CUDA-specific error about slow_conv3d_forward
            if 'aten::slow_conv3d_forward' in str(e) and 'CUDA' in str(e):
                # Fall back to CPU
                import logging
                logger = logging.getLogger(__name__)
                method_name = method.__name__
                logger.warning(f"CUDA operation 'aten::slow_conv3d_forward' not supported in {method_name}, falling back to CPU")
                
                # Get original devices
                orig_devices = {}
                cpu_device = torch.device('cpu')
                
                # Find the device of the first tensor argument to use as target device
                original_device = None
                cpu_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        if original_device is None:
                            original_device = arg.device
                        # Create CPU copy of the tensor
                        cpu_args.append(arg.detach().to(cpu_device))
                    else:
                        cpu_args.append(arg)
                
                if not original_device:
                    # If no tensor was found, use the device of the first parameter
                    original_device = next(self.parameters()).device
                
                # Record original module devices and move to CPU
                for name, module in self.named_children():
                    if list(module.parameters()):  # Check if module has parameters
                        orig_devices[name] = next(module.parameters()).device
                        module.to(cpu_device)
                
                # Set model to eval mode to avoid dropout effects during fallback
                training_mode = self.training
                self.eval()
                
                # Run method on CPU
                with torch.no_grad():
                    outputs = method(self, *cpu_args, **kwargs)
                
                # Move outputs back to original device
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.to(original_device)
                elif isinstance(outputs, tuple) and all(isinstance(t, torch.Tensor) for t in outputs):
                    outputs = tuple(t.to(original_device) for t in outputs)
                
                # Move modules back to original devices
                for name, module in self.named_children():
                    if name in orig_devices:
                        module.to(orig_devices[name])
                
                # Restore training mode
                if training_mode:
                    self.train()
                
                return outputs
            else:
                # Re-raise other errors
                raise
    
    return wrapper


class AffinityCosineLoss(nn.Module):
    """Affinity loss based on pre-calculated shape similarity.

    Parameters
    ----------
    lookup : torch.Tensor (M, M)
        A square symmetric matrix where each column and row is the index of an
        object from the training set, consisting of M different objects. The
        value at (i, j) is a scalar value encoding the shape similarity between
        objects i and j, pre-calculated using some shape (or other) metric. The
        identity of the matrix should be 1 since these objects are the same
        shape. The affinity similarity should be normalized to the range
        (-1, 1).
    device : torch.device
        Device where computation will be performed.
    latent_ratio : float
        The ratio of latent dimensions to use for similarity calculation (0.0 to 1.0).
        Default is 0.75 to use 75% of the latent dimensions.
    """

    def __init__(self, lookup: torch.Tensor, device: torch.device, latent_ratio: float = 0.75):
        super().__init__()
        self.device = device
        self.register_buffer('lookup', torch.tensor(lookup, device=device))
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.l1loss = nn.L1Loss(reduction="none")  # Use "none" for per-sample loss
        self.latent_ratio = latent_ratio

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, per_sample: bool = False
    ) -> torch.Tensor:
        """Return the affinity loss.

        Parameters
        ----------
        y_true : torch.Tensor (N, )
            A vector of N objects in the mini-batch of the indices representing
            the identity of the object as an index. These indices should
            correspond to the rows and columns of the `lookup` table.
        y_pred : torch.Tensor (N, latent_dims)
            An array of latent encodings of the N objects.
        per_sample : bool, optional
            If True, return the per-sample loss for each pair in the batch. If False,
            return the mean loss for the entire batch.

        Returns
        -------
        loss : torch.Tensor
            The affinity loss. If `per_sample` is True, returns a tensor of
            per-sample losses for each pair. Otherwise, returns the mean loss.
        """
        # Make sure inputs are on the correct device
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        # Calculate number of dimensions to use
        n_dims = y_pred.shape[1]
        n_dims_to_use = int(n_dims * self.latent_ratio)
        
        # Use only a portion of the latent dimensions
        y_pred_partial = y_pred[:, :n_dims_to_use]

        # Calculate combinations on device
        z_id = torch.arange(y_pred_partial.shape[0], device=self.device)
        
        # Handle edge case with small batch size
        if z_id.shape[0] < 2:
            return torch.tensor(0.0, device=self.device)
            
        c = torch.combinations(y_true, r=2, with_replacement=False)
        c_latent = torch.combinations(z_id, r=2, with_replacement=False)

        # Get affinity from lookup table
        affinity = self.lookup[c[:, 0].long(), c[:, 1].long()]

        # Calculate latent similarity
        latent_similarity = self.cos(
            y_pred_partial[c_latent[:, 0], :], 
            y_pred_partial[c_latent[:, 1], :]
        )

        # Calculate L1 loss
        losses = self.l1loss(latent_similarity, affinity)

        if per_sample:
            return losses
        else:
            return torch.mean(losses)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class AffinityVAE(nn.Module):
    """Variational Autoencoder with affinity learning capability.
    
    Parameters
    ----------
    encoder : torch.nn.Module
        Neural network to encode input data into latent space.
    decoder : torch.nn.Module
        Neural network to decode latent space representation back to data space.
    latent_dims : int
        Dimension of the latent space.
    pose_channels : int
        Number of pose channels (typically 4 for axis-angle representation).
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        latent_dims: int = 8,
        pose_channels: int = 1,
        use_rotated_affinity: bool = False,
        crossover_probability: float = 0.5,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims
        self.pose_channels = pose_channels

        flat_shape = self.encoder.flat_shape

        self.mu = nn.Linear(flat_shape, latent_dims)
        self.log_var = nn.Linear(flat_shape, latent_dims)
        self.pose = nn.Linear(flat_shape, pose_channels)
        self.global_weight = nn.Linear(flat_shape, 1)
        self.use_rotated_affinity = use_rotated_affinity
        self.crossover_probability = crossover_probability

    def _ensure_same_device(self, tensor, *modules):
        """
        Ensures that all modules are on the same device as the input tensor.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Reference tensor whose device will be used
        *modules : torch.nn.Module
            Modules to move to the tensor's device if needed
        
        Returns
        -------
        torch.device
            The device that everything is now on
        """
        device = tensor.device
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Check if any module needs to be moved
        device_mismatch = False
        for module in modules:
            if next(module.parameters()).device != device:
                device_mismatch = True
                break
        
        # If there's a mismatch, move all modules to the tensor's device
        if device_mismatch:
            if rank == 0:
                print(f"Device mismatch detected. Moving components to {device}")
            for module in modules:
                module_device = next(module.parameters()).device
                if module_device != device:
                    module.to(device)
        
        return device

    @fallback_to_cpu
    def forward(self, x: torch.Tensor, pose: Optional[torch.Tensor] = None, global_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
        pose : Optional[torch.Tensor]
            Optional explicit pose tensor. If provided, this will be used instead of
            the pose generated by the encoder.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Reconstructed data, latent representation, pose, mean, and log variance
        """
        # Ensure all components are on the same device as input
        device = self._ensure_same_device(x, self.encoder, self.decoder, self.mu, self.log_var, self.pose)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        if rank == 0:
            print(f"Starting forward pass with input shape: {x.shape} on device {device}")
        
        start_time = time.time()
        
        if not self.use_rotated_affinity:
            # Standard forward pass without rotation
            mu, log_var, generated_pose, generated_global_weight = self.encode(x)
            encode_time = time.time() - start_time
            
            z_start = time.time()
            z = self.reparameterize(mu, log_var)
            reparameterize_time = time.time() - z_start
            
        else:
            # Enhanced forward pass with rotated affinity dimensions
            original_mu, original_log_var, generated_pose, generated_global_weight = self.encode(x)
            encode_time = time.time() - start_time
            
            # Choose a random 90-degree rotation (excluding 0/360 degrees)
            with torch.no_grad():
                k = torch.randint(1, 4, (1,)).item()  # Random integer from 1 to 3
                rotated_x = torch.rot90(x, k=k, dims=(3, 4))  
            
            # Get encoder outputs for rotated input
            rotated_encoded = self.encoder(rotated_x)
            rotated_mu = self.mu(rotated_encoded)
            rotated_log_var = self.log_var(rotated_encoded)
            
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
            z_start = time.time()
            z = self.reparameterize(mu, log_var)
            reparameterize_time = time.time() - z_start
        
        # Use provided pose if available, otherwise use the generated pose
        actual_pose = pose if pose is not None else generated_pose
        actual_global_weight = global_weight if global_weight is not None else generated_global_weight
        
        # If pose is provided, ensure it's on the correct device
        if pose is not None and pose.device != device:
            actual_pose = pose.to(device)
        if global_weight is not None and global_weight.device != device:
            actual_global_weight = global_weight.to(device)            
        
        decode_start = time.time()
        x_recon = self.decode(z, actual_pose, actual_global_weight)
        decode_time = time.time() - decode_start
        
        total_time = time.time() - start_time
        
        if rank == 0:
            print(f"Forward pass completed in {total_time:.3f}s (encode: {encode_time:.3f}s, reparam: {reparameterize_time:.3f}s, decode: {decode_time:.3f}s)")
        
        return x_recon, z, actual_pose, actual_global_weight, mu, log_var

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Apply reparameterization trick.
        
        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent Gaussian.
        log_var : torch.Tensor
            Log variance of the latent Gaussian.
            
        Returns
        -------
        torch.Tensor
            Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    @fallback_to_cpu
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input data to latent representation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Mean, log variance, and pose.
        """
        self._ensure_same_device(x, self.encoder, self.mu, self.log_var, self.pose, self.global_weight)
        
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        pose = self.pose(encoded)
        global_weight = self.global_weight(encoded)
        
        # CRITICAL FIX: Force L2 normalization during evaluation only
        # This ensures consistent similarity calculations during inference
        if not self.training:
            mu = F.normalize(mu, p=2, dim=1)
        
        return mu, log_var, pose, global_weight

    @fallback_to_cpu
    def decode(self, z: torch.Tensor, pose: torch.Tensor, global_weight: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to data space.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation.
        pose : torch.Tensor
            Pose tensor.
        global_weight : torch.Tensor            
            Splat weight
            
        Returns
        -------
        torch.Tensor
            Reconstructed data.
        """
        # Ensure decoder is on the same device as input
        device = self._ensure_same_device(z, self.decoder)
        
        # If pose is on a different device, move it
        if pose.device != device:
            pose = pose.to(device)
        if global_weight.device != device:
            global_weight = global_weight.to(device)
        
        return self.decoder(z, pose, global_weight=global_weight)