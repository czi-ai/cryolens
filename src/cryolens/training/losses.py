"""
Loss functions for CryoLens model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveAffinityLoss(nn.Module):
    """
    Contrastive affinity loss that handles background samples.
    
    Uses a reformulated loss function based on Euclidean distances:
    L = (1/pairs) * sum_{i<j} [ S_{ij} * ||z'_i - z'_j||²₂ + (1 - S_{ij}) * max(0, m - ||z'_i - z'_j||₂)² ]
    
    Where:
    - S_{ij} is the similarity score between samples i and j
    - z'_i, z'_j are the latent representations (using first latent_ratio dimensions)
    - m is the margin parameter for dissimilar pairs
    
    Parameters
    ----------
    lookup : torch.Tensor
        Lookup table for molecule affinities. Already normalized by SimilarityCalculator.
    device : torch.device
        Device to run computation on.
    latent_ratio : float
        Ratio of latent dimensions to use (default: 0.75).
    margin : float
        Margin for dissimilar pairs in contrastive loss (default: 4.0).
    """

    def __init__(self, lookup: torch.Tensor, device: torch.device, latent_ratio: float = 0.75, margin: float = 2.0):
        super().__init__()
        # Register lookup buffer (similarity matrix)
        lookup = lookup.clone().detach().contiguous()
        self.register_buffer('lookup', lookup, persistent=False)
        self.device = device
        self.latent_ratio = latent_ratio
        self.margin = margin
        
        # Background similarity values
        self.background_sim = 0.2  # Low similarity between background samples
        self.background_other_sim = 0.01  # Very low similarity between background and objects

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
        """
        Compute contrastive affinity loss with background handling.
        
        Parameters
        ----------
        y_true : torch.Tensor
            True indices for lookup table or -1 for background.
        y_pred : torch.Tensor
            Predicted embeddings.
        per_sample : bool
            Whether to return per-sample losses.
            
        Returns
        -------
        torch.Tensor
            Contrastive affinity loss.
        """
        # Note: Similarity values from lookup table are already properly normalized
        # by SimilarityCalculator.load_matrix()
        try:
            # Ensure inputs are contiguous
            y_true = y_true.contiguous().to(self.device)
            y_pred = y_pred.contiguous().to(self.device)

            # Handle batch size < 2
            if y_true.shape[0] < 2:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # Handle all background case
            if torch.all(y_true == -1):
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # Get dimensions and use partial embedding based on latent_ratio
            n_dims = y_pred.shape[1]
            n_dims_to_use = int(n_dims * self.latent_ratio)
            y_pred_partial = y_pred[:, :n_dims_to_use].contiguous()

            # Get combinations of sample indices (all pairs)
            z_id = torch.arange(y_pred_partial.shape[0], device=self.device)
            c = torch.combinations(z_id, r=2, with_replacement=False)
            
            # Extract pairs of embeddings
            features1 = y_pred_partial[c[:, 0], :].contiguous()
            features2 = y_pred_partial[c[:, 1], :].contiguous()
            
            # Calculate Euclidean distances between embeddings
            # Normalize embeddings for more stable distance calculation
            features1_norm = F.normalize(features1, p=2, dim=1)
            features2_norm = F.normalize(features2, p=2, dim=1)
            distances = torch.norm(features1_norm - features2_norm, p=2, dim=1)
            
            # Handle background pairs
            is_background = (y_true == -1)
            pair1_is_bg = is_background[c[:, 0]]
            pair2_is_bg = is_background[c[:, 1]]
            
            # Initialize target similarities
            target_similarities = torch.zeros_like(distances, device=self.device)
            
            # Both background
            both_bg_mask = pair1_is_bg & pair2_is_bg
            target_similarities[both_bg_mask] = self.background_sim
            
            # One background, one object
            one_bg_mask = pair1_is_bg ^ pair2_is_bg
            target_similarities[one_bg_mask] = self.background_other_sim
            
            # Both objects - use lookup table
            both_obj_mask = ~(both_bg_mask | one_bg_mask)
            if torch.any(both_obj_mask):
                obj_pairs = c[both_obj_mask]
                obj_indices1 = y_true[obj_pairs[:, 0]]
                obj_indices2 = y_true[obj_pairs[:, 1]]
                
                # Validate indices before converting to long
                valid_indices = ((obj_indices1 >= 0) & (obj_indices2 >= 0) & 
                            (obj_indices1 < self.lookup.shape[0]) & 
                            (obj_indices2 < self.lookup.shape[0]))
                
                if torch.any(valid_indices):
                    # Only convert valid indices to long to avoid out-of-bounds indexing
                    valid_obj_indices1 = obj_indices1[valid_indices].long()
                    valid_obj_indices2 = obj_indices2[valid_indices].long()
                    
                    # Update only the valid pairs
                    valid_similarities = self.lookup[valid_obj_indices1, valid_obj_indices2]
                    target_similarities[both_obj_mask][valid_indices] = valid_similarities
            
            # Calculate contrastive loss components
            # Similar pairs: S_{ij} * distance²
            similar_term = target_similarities * (distances ** 2)
            
            # Dissimilar pairs: (1 - S_{ij}) * max(0, margin - distance)²
            margin_dist = torch.clamp(self.margin - distances, min=0)
            dissimilar_term = (1 - target_similarities) * (margin_dist ** 2)
            
            # Combined loss
            losses = similar_term + dissimilar_term
            
            # Return per-sample losses or mean loss
            if per_sample:
                return losses.contiguous()
            
            result = torch.mean(losses)
            
            # Handle NaN values while preserving gradients
            if torch.isnan(result):
                zero_tensor = torch.tensor(0.0, device=self.device)
                result = zero_tensor.requires_grad_() + result.detach() * 0
                
            return result

        except Exception as e:
            print(f"Error in contrastive affinity loss: {str(e)}")
            # Return gradient-maintaining zero tensor
            zero_tensor = torch.tensor(0.0, device=self.device)
            return zero_tensor.requires_grad_()


class NormalizedMSELoss(nn.Module):
    """MSE loss normalized by the size of the subvolume.
    
    This ensures that the loss magnitude is consistent regardless of volume size,
    making it comparable to other losses and preventing gradient explosion.
    The normalization divides by the total number of voxels (volume_size^3) to
    get the mean squared error per voxel, which is scale-invariant.
    """
    def __init__(self, volume_size: int):
        super().__init__()
        self.volume_size = volume_size
        self.normalization_factor = volume_size ** 3
        print(f"NormalizedMSELoss initialized with volume_size={volume_size}, normalization_factor={self.normalization_factor}")
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute normalized MSE loss.
        
        Args:
            pred: Predicted volume (B, 1, D, H, W)
            target: Target volume (B, 1, D, H, W)
            
        Returns:
            Normalized MSE loss
        """
        # Compute squared error
        squared_error = (pred - target) ** 2
        
        # Sum over spatial dimensions and normalize by volume size
        # This gives us the mean squared error per voxel
        loss = squared_error.sum() / (squared_error.shape[0] * self.normalization_factor)
        
        return loss


class MissingWedgeLoss(nn.Module):
    """
    Improved reconstruction loss that accounts for missing wedge artifacts in cryo-ET data.
    Uses a direct Fourier-space MSE calculation with appropriate weighting.
    """
    def __init__(
        self,
        volume_size: int,
        wedge_angle: float = 90.0,  # missing wedge angle from MLC data
        weight_factor: float = 0.3  # Weight factor for missing regions, default to 0.3 to match working version
    ):
        super().__init__()

        self.volume_size = volume_size
        self.wedge_angle = wedge_angle
        self.weight_factor = weight_factor
        
        # Print initialization parameters for debugging
        print(f"MissingWedgeLoss initialized with: volume_size={volume_size}, wedge_angle={wedge_angle}, weight_factor={weight_factor}")

        # Base MSE loss for real space comparison
        self.base_mse = nn.MSELoss(reduction="mean")

        # Pre-compute the wedge mask when initializing
        self.register_buffer('wedge_mask', self._compute_wedge_mask_rfft())
        self.register_buffer('wedge_volume', self._compute_wedge_mask_sum())

    def _compute_wedge_mask_rfft(self, apply_radial_weighting=True, apply_fftshift=False):
        """
        Compute a 3D weight mask for Fourier space that emphasizes visible regions
        and de-emphasizes missing wedge regions. Computed for rfft output by default.
        """
        # Create frequency space coordinates
        kyz = torch.fft.fftfreq(self.volume_size)
        kx = torch.fft.rfftfreq(self.volume_size)

        # Create 3D grid
        kz_grid, ky_grid, kx_grid = torch.meshgrid(kyz, kyz, kx, indexing='ij')

        # FFT shift for debug
        if apply_fftshift:
            kx_grid = torch.fft.ifftshift(kx_grid, (0,1))
            ky_grid = torch.fft.ifftshift(ky_grid, (0,1))
            kz_grid = torch.fft.ifftshift(kz_grid, (0,1))

        # Calculate angles in Fourier space (relative to Z axis)
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        angles = torch.abs(torch.atan2(torch.abs(kx_grid), torch.abs(kz_grid)))
        angles[kz_grid==0] = np.pi/2

        # Create wedge mask
        wedge_rad = (self.wedge_angle / 2) * np.pi / 180

        # Create a sharper transition at wedge boundaries
        steepness = 100  # Increased steepness for sharper transition
        mask = 1 - torch.sigmoid(steepness * (wedge_rad - angles))

        # Apply binary-like weighting to clearly distinguish regions
        # Visible regions get weight 1.0, missing wedge regions get weight_factor
        mask = self.weight_factor + (1.0 - self.weight_factor) * mask

        if apply_radial_weighting:
            # Apply mild radial weighting to emphasize low-frequency information
            rad = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
            rad_weight = torch.exp(-rad)  # Less aggressive radial weighting

            # Combine masks
            final_mask = mask * rad_weight
        else:
            final_mask = mask

        return final_mask

    def _compute_wedge_mask_sum(self):
        """
        Compute the sum of the wedge mask for normalization purposes.
        """
        return torch.sum(self.wedge_mask)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Apply missing-wedge-aware reconstruction loss:
        - Fourier-space MSE for frequency information

        Args:
            pred: Predicted volume (B, 1, D, H, W)
            target: Target volume (B, 1, D, H, W)

        Returns:
            Loss accounting for missing wedge
        """
        # Ensure inputs are on the same device as mask
        pred = pred.to(self.wedge_mask.device)
        target = target.to(self.wedge_mask.device)

        # Force tensors to float32 for FFT operations
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)

        # Fourier space loss (50% of total loss)
        # Convert to Fourier space
        pred_fft = torch.fft.rfftn(pred, dim=(-3, -2, -1))
        target_fft = torch.fft.rfftn(target, dim=(-3, -2, -1))

        # Calculate L2 error in Fourier space
        diff = torch.square(torch.abs(pred_fft - target_fft) * self.wedge_mask)
        total_loss = diff.sum() / (self.wedge_volume * self.volume_size**3)

        return total_loss


class AffinityCosineLoss(nn.Module):
    """Affinity loss based on cosine similarity that handles background samples.
    
    Parameters
    ----------
    lookup : torch.Tensor
        Lookup table for molecule affinities. Already normalized by SimilarityCalculator.
    device : torch.device
        Device to run computation on.
    latent_ratio : float
        Ratio of latent dimensions to use (default: 0.75).
    """

    def __init__(self, lookup: torch.Tensor, device: torch.device, latent_ratio: float = 0.75):
        super().__init__()
        # Register lookup buffer
        lookup = lookup.clone().detach().contiguous()
        self.register_buffer('lookup', lookup, persistent=False)
        self.device = device
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.l1loss = nn.L1Loss(reduction="none")  # Use "none" for per-sample loss
        self.latent_ratio = latent_ratio
        
        # Background similarity values
        self.background_sim = 0.2  # Low similarity between background samples
        self.background_other_sim = 0.01  # Very low similarity between background and objects

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
        """Compute affinity loss with background handling.
        
        Parameters
        ----------
        y_true : torch.Tensor
            True indices for lookup table or -1 for background.
        y_pred : torch.Tensor
            Predicted embeddings.
        per_sample : bool
            Whether to return per-sample losses.
            
        Returns
        -------
        torch.Tensor
            Affinity cosine loss.
        """
        # Note: Similarity values from lookup table are already properly normalized
        # by SimilarityCalculator.load_matrix()
        try:
            # Ensure inputs are contiguous
            y_true = y_true.contiguous().to(self.device)
            y_pred = y_pred.contiguous().to(self.device)

            # Handle batch size < 2
            if y_true.shape[0] < 2:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # Handle all background case
            if torch.all(y_true == -1):
                return torch.tensor(self.background_sim, device=self.device, requires_grad=True)

            # Get dimensions
            n_dims = y_pred.shape[1]
            n_dims_to_use = int(n_dims * self.latent_ratio)
            y_pred_partial = y_pred[:, :n_dims_to_use].contiguous()

            # Get combinations
            z_id = torch.arange(y_pred_partial.shape[0], device=self.device)
            c = torch.combinations(z_id, r=2, with_replacement=False)
            
            # Extract pairs
            features1 = y_pred_partial[c[:, 0], :].contiguous()
            features2 = y_pred_partial[c[:, 1], :].contiguous()
            
            # Normalize embeddings before calculating cosine similarity
            # This ensures the cosine similarity is properly bounded in [-1, 1]
            features1_norm = F.normalize(features1, p=2, dim=1)
            features2_norm = F.normalize(features2, p=2, dim=1)
            # Calculate cosine similarity between normalized features
            latent_similarity = self.cos(features1_norm, features2_norm)
            
            # Handle background pairs
            is_background = (y_true == -1)
            pair1_is_bg = is_background[c[:, 0]]
            pair2_is_bg = is_background[c[:, 1]]
            
            # Initialize target similarities
            target_similarities = torch.zeros_like(latent_similarity)
            
            # Both background
            both_bg_mask = pair1_is_bg & pair2_is_bg
            target_similarities[both_bg_mask] = self.background_sim
            
            # One background, one object
            one_bg_mask = pair1_is_bg ^ pair2_is_bg
            target_similarities[one_bg_mask] = self.background_other_sim
            
            # Both objects - use lookup table
            both_obj_mask = ~(both_bg_mask | one_bg_mask)
            if torch.any(both_obj_mask):
                obj_pairs = c[both_obj_mask]
                obj_indices1 = y_true[obj_pairs[:, 0]]
                obj_indices2 = y_true[obj_pairs[:, 1]]
                
                # Validate indices before converting to long
                valid_indices = ((obj_indices1 >= 0) & (obj_indices2 >= 0) & 
                            (obj_indices1 < self.lookup.shape[0]) & 
                            (obj_indices2 < self.lookup.shape[0]))
                
                if torch.any(valid_indices):
                    # Only convert valid indices to long to avoid out-of-bounds indexing
                    valid_obj_indices1 = obj_indices1[valid_indices].long()
                    valid_obj_indices2 = obj_indices2[valid_indices].long()
                    
                    # Update only the valid pairs
                    valid_similarities = self.lookup[valid_obj_indices1, valid_obj_indices2]
                    target_similarities[both_obj_mask][valid_indices] = valid_similarities
            
            # Calculate loss
            losses = self.l1loss(latent_similarity, target_similarities)
            
            # Return per-sample or mean loss
            if per_sample:
                return losses.contiguous()
            
            result = torch.mean(losses)
            
            # Handle NaN values while preserving gradients
            if torch.isnan(result):
                zero_tensor = torch.tensor(0.0, device=self.device)
                result = zero_tensor.requires_grad_() + result.detach() * 0
                
            return result

        except Exception as e:
            print(f"Error in affinity loss: {str(e)}")
            # Return gradient-maintaining zero tensor
            zero_tensor = torch.tensor(0.0, device=self.device)
            return zero_tensor.requires_grad_()
