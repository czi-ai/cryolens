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
        self.use_rotated_affinity = use_rotated_affinity
        self.crossover_probability = crossover_probability

    def forward(self, x: torch.Tensor, pose: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
        pose : Optional[torch.Tensor]
            Optional explicit pose tensor. If provided, this will be used instead of
            the pose generated by the encoder. This is useful for controlling the
            orientation of the output or for fixing the pose during training.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Reconstructed data, latent representation, pose, mean, and log variance
            
        Notes
        -----
        If use_rotated_affinity is True, the method will compute the latent vectors for 
        both the original input and a randomly rotated version. It will then combine the 
        affinity dimensions using uniform crossover (randomly swapping values between the 
        two embeddings) to improve separation between affinity and background.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            print(f"Starting forward pass with input shape: {x.shape} on device {x.device}")
            print(f"Model components: encoder on {next(self.encoder.parameters()).device}, mu on {next(self.mu.parameters()).device}")
        
        try:
            # Ensure all model components are on the same device as input
            device = x.device
            self.encoder = self.encoder.to(device)
            self.decoder = self.decoder.to(device)
            self.mu = self.mu.to(device)
            self.log_var = self.log_var.to(device)
            self.pose = self.pose.to(device)
            
            # Try normal forward pass first
            start_time = time.time()
            
            if not self.use_rotated_affinity:
                # Standard forward pass without rotation
                mu, log_var, generated_pose = self.encode(x)
                encode_time = time.time() - start_time
                
                z_start = time.time()
                z = self.reparameterize(mu, log_var)
                reparameterize_time = time.time() - z_start
                
            else:
                # Enhanced forward pass with rotated affinity dimensions
                # First process the original input
                original_mu, original_log_var, generated_pose = self.encode(x)
                encode_time = time.time() - start_time
                
                # Choose a random 90-degree rotation (excluding 0/360 degrees)
                # k=1: 90 degrees, k=2: 180 degrees, k=3: 270 degrees
                with torch.no_grad():  # No need to track gradients for the rotation
                    # Randomly choose between 90, 180, and 270 degrees (k values of 1, 2, or 3)
                    k = torch.randint(1, 4, (1,)).item()  # Random integer from 1 to 3
                    # Rotate in the X-Y plane
                    rotated_x = torch.rot90(x, k=k, dims=(3, 4))  
                
                # Get encoder outputs for rotated input
                rotated_encoded = self.encoder(rotated_x)
                rotated_mu = self.mu(rotated_encoded)
                rotated_log_var = self.log_var(rotated_encoded)
                
                # Get the ratio of dimensions used for affinity (from decoder configuration)
                latent_dims = original_mu.shape[1]
                if hasattr(self.decoder, 'latent_ratio'):
                    affinity_ratio = self.decoder.latent_ratio
                else:
                    # Default to 0.75 if not specified
                    affinity_ratio = 0.75
                
                # Calculate the number of affinity dimensions
                affinity_dims = int(latent_dims * affinity_ratio)
                
                # Create masks for uniform crossover between original and rotated embeddings
                # For each affinity dimension, randomly decide which embedding to use
                # This is similar to uniform crossover in genetic algorithms
                crossover_mask = torch.rand(original_mu.shape[0], affinity_dims, device=device) < self.crossover_probability
                inverse_mask = ~crossover_mask
                
                # Initialize mu and log_var with the original values
                mu = original_mu.clone()
                log_var = original_log_var.clone()
                
                # Apply crossover mask to the affinity dimensions only
                # Where mask is True, use values from rotated embedding
                # Where mask is False, keep values from original embedding
                mu[:, :affinity_dims] = crossover_mask * rotated_mu[:, :affinity_dims] + inverse_mask * original_mu[:, :affinity_dims]
                log_var[:, :affinity_dims] = crossover_mask * rotated_log_var[:, :affinity_dims] + inverse_mask * original_log_var[:, :affinity_dims]
                
                # Re-sample z using the crossover latent parameters
                z_start = time.time()
                z = self.reparameterize(mu, log_var)
                reparameterize_time = time.time() - z_start
            
            # Use provided pose if available, otherwise use the generated pose
            actual_pose = pose if pose is not None else generated_pose
            
            # If pose is provided, ensure it's on the correct device
            if pose is not None and pose.device != device:
                actual_pose = pose.to(device)
            
            decode_start = time.time()
            x_recon = self.decode(z, actual_pose)
            decode_time = time.time() - decode_start
            
            total_time = time.time() - start_time
            
            if rank == 0:
                print(f"Forward pass completed in {total_time:.3f}s (encode: {encode_time:.3f}s, reparam: {reparameterize_time:.3f}s, decode: {decode_time:.3f}s)")
            
            return x_recon, z, pose, mu, log_var
        except RuntimeError as e:
            import traceback
            print(f"Error in forward pass on rank {rank}:")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            
            # Check if it's a CUDA-specific error about slow_conv3d_forward
            if 'aten::slow_conv3d_forward' in str(e) and 'CUDA' in str(e):
                # Fall back to CPU for the entire forward pass
                print(f"Rank {rank}: CUDA operation 'aten::slow_conv3d_forward' not supported, falling back to CPU for entire forward pass")
                
                # Get original device and set up CPU device
                original_device = x.device
                cpu_device = torch.device('cpu')
                
                # Store original model state
                original_model_device = next(self.parameters()).device
                
                # Set model to eval mode
                self.eval()
                
                # Move entire model and input to CPU
                self.to(cpu_device)
                x_cpu = x.detach().to(cpu_device)
                
                # Run forward pass on CPU
                with torch.no_grad():
                    mu, log_var, generated_pose = self.encode(x_cpu)
                    z = self.reparameterize(mu, log_var)
                    # Use provided pose if available, otherwise use the generated pose
                    actual_pose = pose.to(cpu_device) if pose is not None else generated_pose
                    
                    x_recon = self.decode(z, actual_pose)
                
                # Move outputs back to original device
                x_recon = x_recon.to(original_device)
                z = z.to(original_device)
                pose = pose.to(original_device)
                mu = mu.to(original_device)
                log_var = log_var.to(original_device)
                
                # Move model back to original device
                self.to(original_model_device)
                
                return x_recon, z, pose, mu, log_var
            else:
                # Re-raise other errors
                raise

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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input data to latent representation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Mean, log variance, and pose.
        """
        try:
            # Ensure components are on the same device as input
            device = x.device
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            
            # Check if all components are on the same device
            encoder_device = next(self.encoder.parameters()).device
            mu_device = next(self.mu.parameters()).device
            log_var_device = next(self.log_var.parameters()).device
            pose_device = next(self.pose.parameters()).device
            
            # If any mismatch, move all components to the input device
            if (encoder_device != device or mu_device != device or 
                log_var_device != device or pose_device != device):
                if rank == 0:
                    print(f"Device mismatch detected in encode: input on {device}, encoder on {encoder_device}, mu on {mu_device}")
                    print(f"Moving all encoder components to {device}")
                self.encoder = self.encoder.to(device)
                self.mu = self.mu.to(device)
                self.log_var = self.log_var.to(device)
                self.pose = self.pose.to(device)
            
            # Now process with all components on the same device
            encoded = self.encoder(x)
            mu = self.mu(encoded)
            log_var = self.log_var(encoded)
            pose = self.pose(encoded)
            return mu, log_var, pose
        except RuntimeError as e:
            # Check if it's a CUDA-specific error about slow_conv3d_forward
            if 'aten::slow_conv3d_forward' in str(e) and 'CUDA' in str(e):
                # Fall back to CPU
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("CUDA operation 'aten::slow_conv3d_forward' not supported in encoder, falling back to CPU")
                
                # Move model and inputs to CPU
                original_device = x.device
                cpu_device = torch.device('cpu')
                
                # Set model to eval mode to avoid dropout effects
                self.eval()
                
                # Create CPU copies of the tensors
                x_cpu = x.detach().to(cpu_device)
                
                # Store parameter devices
                encoder_device = next(self.encoder.parameters()).device
                mu_device = next(self.mu.parameters()).device
                log_var_device = next(self.log_var.parameters()).device
                pose_device = next(self.pose.parameters()).device
                
                # Temporarily move encoder components to CPU
                self.encoder.to(cpu_device)
                self.mu.to(cpu_device)
                self.log_var.to(cpu_device)
                self.pose.to(cpu_device)
                
                # Run encode on CPU
                with torch.no_grad():
                    encoded = self.encoder(x_cpu)
                    mu = self.mu(encoded)
                    log_var = self.log_var(encoded)
                    pose = self.pose(encoded)
                
                # Move outputs back to original device
                mu = mu.to(original_device)
                log_var = log_var.to(original_device)
                pose = pose.to(original_device)
                
                # Move components back to original devices
                self.encoder.to(encoder_device)
                self.mu.to(mu_device)
                self.log_var.to(log_var_device)
                self.pose.to(pose_device)
                
                return mu, log_var, pose
            else:
                # Re-raise other errors
                raise

    def decode(self, z: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to data space.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation.
        pose : torch.Tensor
            Pose tensor.
        skip_features : Optional[torch.Tensor]
            Features from the skip connection pathway directly from input.
            
        Returns
        -------
        torch.Tensor
            Reconstructed data.
        """
        try:
            # Ensure decoder is on the same device as input
            device = z.device
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            
            # Check if decoder is on the same device
            decoder_device = next(self.decoder.parameters()).device
            
            # If mismatch, move decoder to the input device
            if decoder_device != device:
                if rank == 0:
                    print(f"Device mismatch detected in decode: z on {device}, decoder on {decoder_device}")
                    print(f"Moving decoder to {device}")
                self.decoder = self.decoder.to(device)
            
            # If pose is on a different device, move it
            if pose.device != device:
                if rank == 0:
                    print(f"Device mismatch detected in decode: z on {device}, pose on {pose.device}")
                pose = pose.to(device)
            
            # Now decode with everything on the same device
            return self.decoder(z, pose)
        except RuntimeError as e:
            # Check if it's a CUDA-specific error about slow_conv3d_forward
            if 'aten::slow_conv3d_forward' in str(e) and 'CUDA' in str(e):
                # Fall back to CPU
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("CUDA operation 'aten::slow_conv3d_forward' not supported, falling back to CPU")
                
                # Move model and inputs to CPU
                original_device = z.device
                cpu_device = torch.device('cpu')
                
                # Set model to eval mode to avoid dropout effects
                self.eval()
                
                # Create CPU copies of the tensors
                z_cpu = z.detach().to(cpu_device)
                pose_cpu = pose.detach().to(cpu_device)
                
                # Store decoder device
                decoder_device = next(self.decoder.parameters()).device
                
                # Temporarily move decoder to CPU
                self.decoder.to(cpu_device)
                
                # Run decode on CPU
                with torch.no_grad():
                    output = self.decoder(z_cpu, pose_cpu)
                
                # Move output back to original device
                output = output.to(original_device)
                
                # Move decoder back to original device
                self.decoder.to(decoder_device)
                
                return output
            else:
                # Re-raise other errors
                raise
