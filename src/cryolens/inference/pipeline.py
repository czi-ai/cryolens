"""
Basic inference pipeline for CryoLens models.

This module provides a standardized inference pipeline for processing
volumes through CryoLens VAE models.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
import logging

from cryolens.utils.normalization import normalize_volume, denormalize_volume

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Standard inference pipeline for CryoLens models.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        normalization_method: str = "z-score"
    ):
        """
        Initialize inference pipeline.
        
        Parameters
        ----------
        model : torch.nn.Module
            CryoLens VAE model
        device : torch.device, optional
            Device for inference
        normalization_method : str
            Normalization method to use
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalization_method = normalization_method
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get normalization method from model if available
        if hasattr(self.model, '_normalization_method'):
            self.normalization_method = self.model._normalization_method
    
    def process_volume(
        self,
        volume: np.ndarray,
        return_embeddings: bool = True,
        return_reconstruction: bool = True,
        return_splat_params: bool = False,
        use_identity_pose: bool = True,
        splat_segment: str = 'affinity',
        use_final_convolution: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single volume through the model.
        
        Parameters
        ----------
        volume : np.ndarray
            Input volume of shape (D, H, W)
        return_embeddings : bool
            Whether to return latent embeddings
        return_reconstruction : bool
            Whether to return reconstruction
        return_splat_params : bool
            Whether to return Gaussian splat parameters
        use_identity_pose : bool
            Whether to use identity pose for reconstruction
        splat_segment : str
            'affinity' for structure splats only, 'all' for all splats
        use_final_convolution : bool
            Whether to apply the final convolutional layers in the decoder.
            Set to False to inspect raw Gaussian splat output without convolution.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing requested outputs
        """
        # Normalize volume
        normalized, norm_stats = normalize_volume(
            volume, 
            method=self.normalization_method,
            return_stats=True
        )
        
        # Convert to tensor and add batch/channel dimensions
        volume_tensor = torch.tensor(normalized, dtype=torch.float32)
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        results = {}
        
        with torch.no_grad():
            # Encode
            mu, log_var, pose, global_weight = self.model.encode(volume_tensor)
            
            if return_embeddings:
                results['embeddings'] = mu.cpu().numpy()[0]
                results['log_var'] = log_var.cpu().numpy()[0]
                results['pose'] = pose.cpu().numpy()[0] if pose is not None else None
                results['global_weight'] = global_weight.cpu().numpy()[0] if global_weight is not None else None
            
            if return_reconstruction:
                # Use identity pose if requested
                if use_identity_pose:
                    identity_pose = torch.zeros(1, 4, device=self.device)
                    identity_pose[:, 0] = 1.0  # Quaternion w component
                    identity_global_weight = torch.ones(1, 1, device=self.device)
                    reconstruction = self.model.decoder(mu, identity_pose, identity_global_weight, use_final_convolution=use_final_convolution)
                else:
                    reconstruction = self.model.decoder(mu, pose, global_weight, use_final_convolution=use_final_convolution)
                
                # Remove batch and channel dimensions
                reconstruction_np = reconstruction.cpu().numpy()[0, 0]
                
                # Handle potential size mismatch due to padding
                if reconstruction_np.shape != volume.shape:
                    reconstruction_np = self._crop_to_size(reconstruction_np, volume.shape)
                
                # Denormalize
                reconstruction_np = denormalize_volume(reconstruction_np, norm_stats)
                results['reconstruction'] = reconstruction_np
            
            if return_splat_params:
                # Extract Gaussian splat parameters
                splat_centroids, splat_sigmas, splat_weights = self._extract_splat_params(
                    mu, pose, global_weight, splat_segment
                )
                results['splat_params'] = {
                    'centroids': splat_centroids,
                    'sigmas': splat_sigmas,
                    'weights': splat_weights,
                    'segment': splat_segment
                }
        
        results['normalization_stats'] = norm_stats
        return results
    
    def process_batch(
        self,
        volumes: np.ndarray,
        batch_size: int = 8,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Process a batch of volumes.
        
        Parameters
        ----------
        volumes : np.ndarray
            Batch of volumes of shape (N, D, H, W)
        batch_size : int
            Batch size for processing
        **kwargs
            Additional arguments passed to process_volume
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing batched results
        """
        n_volumes = len(volumes)
        all_results = []
        
        for i in range(0, n_volumes, batch_size):
            batch = volumes[i:i+batch_size]
            batch_results = []
            
            for volume in batch:
                result = self.process_volume(volume, **kwargs)
                batch_results.append(result)
            
            all_results.extend(batch_results)
        
        # Combine results
        combined = {}
        if all_results:
            for key in all_results[0].keys():
                if key != 'normalization_stats':
                    values = [r[key] for r in all_results if key in r]
                    if values and values[0] is not None:
                        combined[key] = np.stack(values)
        
        return combined
    
    def encode_volumes(
        self,
        volumes: np.ndarray,
        batch_size: int = 8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode volumes to latent space.
        
        Parameters
        ----------
        volumes : np.ndarray
            Volumes of shape (N, D, H, W) or (D, H, W)
        batch_size : int
            Batch size for processing
            
        Returns
        -------
        Tuple containing:
            - embeddings: Latent embeddings (N, latent_dims)
            - log_vars: Log variances (N, latent_dims)
            - poses: Poses (N, pose_dims)
            - global_weights: Global weights (N, 1)
        """
        if volumes.ndim == 3:
            volumes = volumes[np.newaxis, ...]
        
        results = self.process_batch(
            volumes,
            batch_size=batch_size,
            return_embeddings=True,
            return_reconstruction=False
        )
        
        return (
            results.get('embeddings', None),
            results.get('log_var', None),
            results.get('pose', None),
            results.get('global_weight', None)
        )
    
    def reconstruct_from_embeddings(
        self,
        embeddings: np.ndarray,
        poses: Optional[np.ndarray] = None,
        global_weights: Optional[np.ndarray] = None,
        use_identity_pose: bool = True,
        reference_volume: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reconstruct volumes from embeddings.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Latent embeddings of shape (N, latent_dims) or (latent_dims,)
        poses : np.ndarray, optional
            Poses of shape (N, pose_dims) or (pose_dims,)
        global_weights : np.ndarray, optional
            Global weights of shape (N, 1) or (1,)
        use_identity_pose : bool
            Whether to use identity pose
        reference_volume : np.ndarray, optional
            Reference volume for denormalization
            
        Returns
        -------
        np.ndarray
            Reconstructed volumes
        """
        # Handle single embedding
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, ...]
            single_input = True
        else:
            single_input = False
        
        n_samples = len(embeddings)
        
        # Convert to tensors
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        # Handle poses
        if use_identity_pose or poses is None:
            poses_tensor = torch.zeros(n_samples, 4, device=self.device)
            poses_tensor[:, 0] = 1.0
        else:
            if poses.ndim == 1:
                poses = poses[np.newaxis, ...]
            poses_tensor = torch.tensor(poses, dtype=torch.float32).to(self.device)
        
        # Handle global weights
        if global_weights is None:
            global_weights_tensor = torch.ones(n_samples, 1, device=self.device)
        else:
            if global_weights.ndim == 1:
                global_weights = global_weights[np.newaxis, ...]
            global_weights_tensor = torch.tensor(global_weights, dtype=torch.float32).to(self.device)
        
        # Reconstruct
        with torch.no_grad():
            reconstructions = self.model.decoder(
                embeddings_tensor,
                poses_tensor,
                global_weights_tensor
            )
        
        # Convert to numpy
        reconstructions_np = reconstructions.cpu().numpy()[:, 0]  # Remove channel dimension
        
        # Denormalize if reference provided
        if reference_volume is not None:
            norm_stats = {
                'method': self.normalization_method,
                'mean': np.mean(reference_volume),
                'std': np.std(reference_volume),
                'min': np.min(reference_volume),
                'max': np.max(reference_volume)
            }
            
            denorm_reconstructions = []
            for recon in reconstructions_np:
                denorm = denormalize_volume(recon, norm_stats)
                denorm_reconstructions.append(denorm)
            reconstructions_np = np.stack(denorm_reconstructions)
        
        if single_input:
            return reconstructions_np[0]
        return reconstructions_np
    
    def _crop_to_size(self, volume: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Crop volume to target shape from center."""
        if volume.shape == target_shape:
            return volume
        
        # Calculate crop amounts
        crop_slices = []
        for actual, target in zip(volume.shape, target_shape):
            if actual > target:
                start = (actual - target) // 2
                end = start + target
                crop_slices.append(slice(start, end))
            else:
                crop_slices.append(slice(None))
        
        return volume[tuple(crop_slices)]
    
    def process_particle_with_splats(
        self,
        particle: np.ndarray,
        box_size: int = 48
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract Gaussian splat parameters from a particle.
        
        This is a simplified interface specifically for extracting splats
        from particles in voxel space coordinates.
        
        Parameters
        ----------
        particle : np.ndarray
            Input particle volume of shape (D, H, W)
        box_size : int
            Size of the box (default: 48)
            
        Returns
        -------
        Optional[Dict[str, np.ndarray]]
            Dictionary containing:
            - 'centroids': Splat positions in voxel space [0, box_size]
            - 'sigmas': Splat standard deviations
            - 'weights': Splat weights
            Or None if extraction fails
        """
        # Normalize volume
        normalized, _ = normalize_volume(
            particle,
            method=self.normalization_method,
            return_stats=True
        )
        
        # Convert to tensor and add batch/channel dimensions
        volume_tensor = torch.tensor(normalized, dtype=torch.float32)
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                # Encode
                mu, log_var, pose, global_weight = self.model.encode(volume_tensor)
                
                # Get decoder
                decoder = self.model.decoder
                
                # Check if decoder has decode_splats method (newer interface)
                if hasattr(decoder, 'decode_splats'):
                    splats, weights, sigmas = decoder.decode_splats(mu, pose)
                    weights = weights * torch.sigmoid(global_weight)
                    
                    # Convert to numpy and remove batch dimension
                    splats_np = splats.cpu().numpy()[0].T  # Transpose to (n_splats, 3)
                    weights_np = weights.cpu().numpy()[0]
                    sigmas_np = sigmas.cpu().numpy()[0]
                else:
                    # Use standard splat extraction
                    splats_np, sigmas_np, weights_np = self._extract_splat_params(
                        mu, pose, global_weight, splat_segment='affinity'
                    )
                
                # Transform from normalized [-1, 1] to voxel space [0, box_size]
                splats_voxel = (splats_np + 1.0) * (box_size / 2.0)
                
                # Remap axes for consistency with expected coordinate system
                # This maps (x, y, z) -> (z, x, y) which is common for cryo-EM data
                splats_mapped = splats_voxel[:, [2, 0, 1]]
                
                return {
                    'centroids': splats_mapped,
                    'sigmas': sigmas_np,
                    'weights': weights_np
                }
                
        except Exception as e:
            logger.warning(f"Failed to extract splats from particle: {e}")
            return None
    
    def _extract_splat_params(
        self,
        mu: torch.Tensor,
        pose: torch.Tensor,
        global_weight: torch.Tensor,
        splat_segment: str = 'affinity'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract Gaussian splat parameters from decoder.
        
        Parameters
        ----------
        mu : torch.Tensor
            Latent embeddings
        pose : torch.Tensor
            Pose parameters
        global_weight : torch.Tensor
            Global weight scaling
        splat_segment : str
            'affinity' for structure splats only, 'all' for all splats
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            centroids, sigmas, weights as numpy arrays
        """
        decoder = self.model.decoder
        
        with torch.no_grad():
            # Check if this is a segmented decoder
            if hasattr(decoder, 'affinity_centroids'):
                # SegmentedGaussianSplatDecoder
                if splat_segment == 'affinity':
                    # Only extract structure splats from affinity segment
                    affinity_size = decoder.affinity_segment_size
                    mu_affinity = mu[:, :affinity_size]
                    
                    # Get affinity splat parameters
                    centroids = decoder.affinity_centroids(mu_affinity)
                    sigmas = decoder.affinity_sigmas(mu_affinity)
                    weights = decoder.affinity_weights(mu_affinity)
                else:
                    # Extract all splats (both affinity and free segments)
                    affinity_size = decoder.affinity_segment_size
                    mu_affinity = mu[:, :affinity_size]
                    mu_free = mu[:, affinity_size:]
                    
                    # Get affinity splats
                    affinity_centroids = decoder.affinity_centroids(mu_affinity)
                    affinity_sigmas = decoder.affinity_sigmas(mu_affinity)
                    affinity_weights = decoder.affinity_weights(mu_affinity)
                    
                    # Get free splats
                    free_centroids = decoder.free_centroids(mu_free)
                    free_sigmas = decoder.free_sigmas(mu_free)
                    free_weights = decoder.free_weights(mu_free)
                    
                    # Concatenate
                    centroids = torch.cat([affinity_centroids, free_centroids], dim=1)
                    sigmas = torch.cat([affinity_sigmas, free_sigmas], dim=1)
                    weights = torch.cat([affinity_weights, free_weights], dim=1)
            else:
                # Standard GaussianSplatDecoder
                centroids = decoder.centroids(mu)
                sigmas = decoder.sigmas(mu)
                weights = decoder.weights(mu)
            
            # Apply global weight scaling
            weights = weights * global_weight
            
            # Reshape centroids to (batch, n_splats, 3)
            batch_size = centroids.shape[0]
            n_splats = sigmas.shape[1]
            centroids = centroids.reshape(batch_size, n_splats, 3)
            
            # Convert to numpy and remove batch dimension (single volume)
            centroids_np = centroids.cpu().numpy()[0]
            sigmas_np = sigmas.cpu().numpy()[0]
            weights_np = weights.cpu().numpy()[0]
            
        return centroids_np, sigmas_np, weights_np


def create_inference_pipeline(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> InferencePipeline:
    """
    Create an inference pipeline from a checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str, optional
        Path to checkpoint, version name (e.g., 'v001'), or URL.
        If None, uses default version.
    device : torch.device, optional
        Device for inference
        
    Returns
    -------
    InferencePipeline
        Configured inference pipeline
    
    Examples
    --------
    >>> # Use default weights
    >>> pipeline = create_inference_pipeline()
    
    >>> # Use specific version
    >>> pipeline = create_inference_pipeline('v001')
    
    >>> # Use local checkpoint
    >>> pipeline = create_inference_pipeline('/path/to/model.pt')
    
    >>> # Use URL
    >>> pipeline = create_inference_pipeline('https://example.com/model.pt')
    """
    from cryolens.utils.checkpoint_loading import load_vae_model
    
    model, config = load_vae_model(checkpoint_path, device)
    normalization = config.get('normalization', 'z-score')
    
    return InferencePipeline(model, device, normalization)
