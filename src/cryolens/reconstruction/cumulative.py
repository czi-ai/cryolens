"""
Cumulative reconstruction methods for CryoLens.

This module implements progressive reconstruction by accumulating
information from multiple particles.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class CumulativeReconstructor:
    """
    Cumulative reconstruction from multiple particles.
    
    This class implements the progressive reconstruction approach where
    reconstructions from individual particles are accumulated to improve
    quality.
    
    Attributes:
        use_identity_pose (bool): Use identity pose for accumulation
        normalize_before_accumulation (bool): Normalize each reconstruction
        accumulation_method (str): Method for accumulation ('mean', 'weighted', 'median')
    """
    
    def __init__(
        self,
        use_identity_pose: bool = True,
        normalize_before_accumulation: bool = False,
        accumulation_method: str = 'mean'
    ):
        """
        Initialize cumulative reconstructor.
        
        Args:
            use_identity_pose: Use identity pose transformation
            normalize_before_accumulation: Normalize reconstructions before combining
            accumulation_method: Method for combining reconstructions
        """
        self.use_identity_pose = use_identity_pose
        self.normalize_before_accumulation = normalize_before_accumulation
        self.accumulation_method = accumulation_method
    
    def prepare_particles(
        self,
        particles: Union[np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Prepare particles for reconstruction.
        
        Args:
            particles: Input particles (N x D x H x W) or (N x H x W x D)
            normalize: Whether to normalize particles
            
        Returns:
            Prepared particle tensor
        """
        # Convert to tensor if needed
        if isinstance(particles, np.ndarray):
            particles = torch.from_numpy(particles).float()
        
        # Ensure 4D tensor (batch, depth, height, width)
        if particles.dim() == 3:
            particles = particles.unsqueeze(0)
        
        # Normalize if requested
        if normalize:
            for i in range(particles.shape[0]):
                particle = particles[i]
                mean = particle.mean()
                std = particle.std()
                if std > 0:
                    particles[i] = (particle - mean) / std
        
        return particles
    
    def accumulate_splats(
        self,
        model,
        particles: Union[np.ndarray, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Accumulate Gaussian splats from multiple particles.
        
        Args:
            model: CryoLens VAE model
            particles: Input particles
            device: Device for computation
            
        Returns:
            Tuple of (accumulated_reconstruction, info_dict)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare particles
        particles = self.prepare_particles(particles)
        particles = particles.to(device)
        
        # Process each particle
        accumulated_splats = None
        accumulated_weights = None
        all_embeddings = []
        
        model.eval()
        with torch.no_grad():
            for i in range(particles.shape[0]):
                particle = particles[i:i+1]
                
                # Encode particle
                if hasattr(model, 'encode'):
                    mu, log_var = model.encode(particle)
                    z = model.reparameterize(mu, log_var)
                else:
                    # Assume forward pass returns necessary components
                    output = model(particle)
                    z = output[1] if isinstance(output, tuple) else output
                
                all_embeddings.append(z.cpu().numpy())
                
                # Generate pose (identity if specified)
                if self.use_identity_pose:
                    batch_size = z.shape[0]
                    pose = torch.zeros((batch_size, 4), device=device)
                    pose[:, 3] = 1.0  # Identity rotation
                else:
                    # Use model's pose generation
                    if hasattr(model, 'pose'):
                        pose = model.pose(z)
                    else:
                        pose = None
                
                # Decode to splats
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'decode_splats'):
                    splats, weights, sigmas = model.decoder.decode_splats(z, pose)
                    
                    # Render splats
                    if hasattr(model.decoder, '_splatter'):
                        rendered = model.decoder._splatter(
                            splats, weights, sigmas,
                            splat_sigma_range=getattr(model.decoder, '_splat_sigma_range', (0.005, 0.1))
                        )
                    else:
                        # Simple Gaussian rendering
                        rendered = self._render_gaussian_splats(splats, weights, sigmas)
                else:
                    # Fallback to full forward pass
                    rendered = model.decoder(z, pose)
                
                # Accumulate
                if accumulated_splats is None:
                    accumulated_splats = rendered
                    accumulated_weights = torch.ones_like(rendered)
                else:
                    if self.accumulation_method == 'mean':
                        accumulated_splats += rendered
                        accumulated_weights += 1
                    elif self.accumulation_method == 'weighted':
                        weight = weights.mean()  # Use average splat weight
                        accumulated_splats += rendered * weight
                        accumulated_weights += weight
                    elif self.accumulation_method == 'median':
                        # Store all for median calculation
                        if not hasattr(self, '_all_splats'):
                            self._all_splats = [accumulated_splats]
                        self._all_splats.append(rendered)
        
        # Final accumulation
        if self.accumulation_method == 'median' and hasattr(self, '_all_splats'):
            accumulated_splats = torch.median(torch.stack(self._all_splats), dim=0)[0]
            del self._all_splats
        elif self.accumulation_method in ['mean', 'weighted']:
            accumulated_splats = accumulated_splats / (accumulated_weights + 1e-8)
        
        # Apply final convolution if model has it
        if hasattr(model.decoder, 'final_convolution'):
            accumulated_splats = model.decoder.final_convolution(accumulated_splats)
        
        info = {
            'n_particles': len(particles),
            'embeddings': np.array(all_embeddings),
            'accumulation_method': self.accumulation_method,
            'used_identity_pose': self.use_identity_pose
        }
        
        return accumulated_splats, info
    
    def _render_gaussian_splats(
        self,
        centroids: torch.Tensor,
        weights: torch.Tensor,
        sigmas: torch.Tensor,
        volume_shape: Tuple[int, int, int] = (48, 48, 48)
    ) -> torch.Tensor:
        """
        Simple Gaussian splat rendering.
        
        Args:
            centroids: Splat positions (B x 3 x N)
            weights: Splat weights (B x N)
            sigmas: Splat scales (B x N)
            volume_shape: Output volume shape
            
        Returns:
            Rendered volume
        """
        batch_size = centroids.shape[0]
        device = centroids.device
        
        # Create coordinate grid
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, volume_shape[0], device=device),
            torch.linspace(-1, 1, volume_shape[1], device=device),
            torch.linspace(-1, 1, volume_shape[2], device=device),
            indexing='ij'
        ), dim=-1)  # Shape: (D, H, W, 3)
        
        # Initialize volume
        volume = torch.zeros((batch_size, 1) + volume_shape, device=device)
        
        # Render each splat
        for b in range(batch_size):
            for i in range(centroids.shape[2]):
                center = centroids[b, :, i]
                weight = weights[b, i]
                sigma = sigmas[b, i]
                
                # Compute Gaussian
                distances = torch.sum((coords - center)**2, dim=-1)
                gaussian = weight * torch.exp(-distances / (2 * sigma**2))
                
                # Add to volume
                volume[b, 0] += gaussian
        
        return volume
    
    def progressive_reconstruction(
        self,
        model,
        particles: Union[np.ndarray, torch.Tensor],
        particle_counts: Optional[List[int]] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Generate progressive reconstructions with increasing particle counts.
        
        Args:
            model: CryoLens VAE model
            particles: All available particles
            particle_counts: List of particle counts to use
            device: Device for computation
            
        Returns:
            Dictionary with progressive reconstructions and metrics
        """
        particles = self.prepare_particles(particles)
        n_total = particles.shape[0]
        
        if particle_counts is None:
            # Default progression
            particle_counts = [1, 5, 10, 20, 30, 40, 50]
            particle_counts = [n for n in particle_counts if n <= n_total]
        
        reconstructions = []
        embeddings_list = []
        info_list = []
        
        for n in particle_counts:
            # Select subset of particles
            indices = np.random.choice(n_total, n, replace=False)
            subset = particles[indices]
            
            # Generate reconstruction
            recon, info = self.accumulate_splats(model, subset, device)
            
            reconstructions.append(recon.cpu().numpy())
            embeddings_list.append(info['embeddings'])
            info_list.append(info)
        
        return {
            'particle_counts': particle_counts,
            'reconstructions': reconstructions,
            'embeddings': embeddings_list,
            'info': info_list
        }


def accumulate_reconstructions(
    model,
    particles: Union[np.ndarray, torch.Tensor],
    use_identity_pose: bool = True,
    method: str = 'mean',
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Convenience function for cumulative reconstruction.
    
    Args:
        model: CryoLens VAE model
        particles: Input particles
        use_identity_pose: Use identity pose
        method: Accumulation method
        device: Computation device
        
    Returns:
        Accumulated reconstruction
    """
    reconstructor = CumulativeReconstructor(
        use_identity_pose=use_identity_pose,
        accumulation_method=method
    )
    
    reconstruction, _ = reconstructor.accumulate_splats(model, particles, device)
    
    return reconstruction.cpu().numpy().squeeze()


def progressive_reconstruction(
    model,
    particles: Union[np.ndarray, torch.Tensor],
    particle_counts: Optional[List[int]] = None,
    return_metrics: bool = False
) -> Union[List[np.ndarray], Dict[str, Any]]:
    """
    Generate progressive reconstructions.
    
    Args:
        model: CryoLens VAE model
        particles: Input particles
        particle_counts: Particle counts for progression
        return_metrics: Return full metrics dictionary
        
    Returns:
        List of reconstructions or full results dictionary
    """
    reconstructor = CumulativeReconstructor()
    results = reconstructor.progressive_reconstruction(
        model, particles, particle_counts
    )
    
    if return_metrics:
        return results
    else:
        return results['reconstructions']
