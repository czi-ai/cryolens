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

    def __init__(self, lookup: torch.Tensor, device: torch.device, latent_ratio: float = 0.75, margin: float = 4.0):
        super().__init__()
        # Register lookup buffer (similarity matrix)
        lookup = lookup.clone().detach().contiguous()
        self.register_buffer('lookup', lookup, persistent=False)
        self.device = device
        self.latent_ratio = latent_ratio
        self.margin = margin
        
        # Background similarity values (in [0,1] range)
        self.background_sim = 0.2  # Low similarity between background samples
        self.background_other_sim = 0.01  # Very low similarity between background and objects
        
        print(f"AffinityCosineLoss initialized:")
        print(f"  Latent ratio: {self.latent_ratio}")
        print(f"  Background-background similarity: {self.background_sim}")
        print(f"  Background-object similarity: {self.background_other_sim}")

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
            # Option 1: Use raw embeddings without normalization for true Euclidean distance
            distances = torch.norm(features1 - features2, p=2, dim=1)
            
            # For numerical stability, we can optionally scale distances
            # This helps prevent gradient explosion/vanishing
            # Scale factor based on embedding dimension to keep distances in reasonable range
            scale_factor = 1.0 / np.sqrt(n_dims_to_use)
            distances = distances * scale_factor
            
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
            
            # Debug: Check target similarities range
            if torch.any(target_similarities < 0) or torch.any(target_similarities > 1):
                print(f"WARNING: Target similarities outside [0,1] range! Min: {target_similarities.min():.4f}, Max: {target_similarities.max():.4f}")
            
            # Calculate contrastive loss components
            # Similar pairs: S_{ij} * distance²
            similar_term = target_similarities * (distances ** 2)
            
            # Dissimilar pairs: (1 - S_{ij}) * max(0, margin - distance)²
            margin_dist = torch.clamp(self.margin - distances, min=0)
            dissimilar_term = (1 - target_similarities) * (margin_dist ** 2)
            
            # Combined loss
            losses = similar_term + dissimilar_term
            
            # Periodic debug output (every 100 steps)
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                
            if self._debug_counter % 100 == 0:
                print(f"\nContrastive Loss Debug (step {self._debug_counter}):")
                print(f"  Distances - Min: {distances.min():.4f}, Max: {distances.max():.4f}, Mean: {distances.mean():.4f}")
                print(f"  Target similarities - Min: {target_similarities.min():.4f}, Max: {target_similarities.max():.4f}, Mean: {target_similarities.mean():.4f}")
                print(f"  Similar term - Mean: {similar_term.mean():.4f}")
                print(f"  Dissimilar term - Mean: {dissimilar_term.mean():.4f}")
                print(f"  Total loss - Mean: {losses.mean():.4f}")
            
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


class GeodesicPoseLoss(nn.Module):
    """
    Compute geodesic distance between predicted and true rotations.
    
    For axis-angle representation: [angle, ax, ay, az]
    The loss is the geodesic distance on SO(3) manifold.
    
    Parameters
    ----------
    pose_channels : int
        Number of pose channels (1 for single rotation, 4 for axis-angle)
    reduction : str
        How to reduce the loss ('mean' or 'sum')
    """
    
    def __init__(self, pose_channels: int = 4, reduction: str = 'mean'):
        super().__init__()
        self.pose_channels = pose_channels
        self.reduction = reduction
        
    def axis_angle_to_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle representation to rotation matrix.
        
        Parameters
        ----------
        axis_angle : torch.Tensor
            Shape (B, 4) with [angle, ax, ay, az]
            
        Returns
        -------
        torch.Tensor
            Rotation matrices of shape (B, 3, 3)
        """
        batch_size = axis_angle.shape[0]
        angle = axis_angle[:, 0]  # (B,)
        axis = axis_angle[:, 1:4]  # (B, 3)
        
        # Normalize axis to unit vector
        axis = F.normalize(axis, p=2, dim=1, eps=1e-6)
        
        # Rodrigues' rotation formula
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Create skew-symmetric matrix K from axis
        K = torch.zeros((batch_size, 3, 3), device=axis_angle.device, dtype=axis_angle.dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        K_squared = torch.bmm(K, K)
        
        R = I + sin_angle.unsqueeze(-1).unsqueeze(-1) * K + \
            (1 - cos_angle).unsqueeze(-1).unsqueeze(-1) * K_squared
        
        return R
    
    def forward(self, pred_pose: torch.Tensor, true_pose: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic distance between predicted and true rotations.
        
        Parameters
        ----------
        pred_pose : torch.Tensor
            Predicted pose (B, pose_channels)
        true_pose : torch.Tensor
            True pose (B, pose_channels)
            
        Returns
        -------
        torch.Tensor
            Geodesic loss
        """
        # Ensure tensors are on same device and dtype
        if pred_pose.device != true_pose.device:
            true_pose = true_pose.to(pred_pose.device)
        if pred_pose.dtype != true_pose.dtype:
            true_pose = true_pose.to(pred_pose.dtype)
            
        if self.pose_channels == 4:
            # Check for NaN or invalid values in input
            if torch.isnan(pred_pose).any() or torch.isnan(true_pose).any():
                print(f"WARNING: NaN in pose inputs to GeodesicPoseLoss")
                return torch.tensor(0.0, device=pred_pose.device, dtype=pred_pose.dtype)
            
            # Convert to rotation matrices
            pred_rot = self.axis_angle_to_matrix(pred_pose)
            true_rot = self.axis_angle_to_matrix(true_pose)
            
            # Check for NaN in rotation matrices
            if torch.isnan(pred_rot).any() or torch.isnan(true_rot).any():
                print(f"WARNING: NaN in rotation matrices")
                return torch.tensor(0.0, device=pred_pose.device, dtype=pred_pose.dtype)
            
            # Compute relative rotation: R_rel = R_true @ R_pred^T
            R_rel = torch.bmm(true_rot, pred_rot.transpose(-2, -1))
            
            # Extract angle from relative rotation
            # angle = arccos((trace(R_rel) - 1) / 2)
            trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            cos_angle = (trace - 1) / 2
            
            # Clamp for numerical stability - use wider bounds
            cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
            
            # Check if clamping was needed (indicates potential numerical issues)
            if ((cos_angle < -0.999) | (cos_angle > 0.999)).any():
                print(f"WARNING: Cos angle near boundary: min={cos_angle.min()}, max={cos_angle.max()}")
            
            geodesic_dist = torch.acos(cos_angle)
            
            # Final NaN check
            if torch.isnan(geodesic_dist).any():
                print(f"WARNING: NaN in geodesic distance computation")
                print(f"  trace: {trace}")
                print(f"  cos_angle: {cos_angle}")
                return torch.tensor(0.0, device=pred_pose.device, dtype=pred_pose.dtype)
            
        elif self.pose_channels == 1:
            # For 1D rotation, use simple angular distance
            # Wrap angles to [-π, π]
            pred_wrapped = torch.remainder(pred_pose + np.pi, 2 * np.pi) - np.pi
            true_wrapped = torch.remainder(true_pose + np.pi, 2 * np.pi) - np.pi
            
            # Angular distance
            diff = pred_wrapped - true_wrapped
            geodesic_dist = torch.abs(torch.remainder(diff + np.pi, 2 * np.pi) - np.pi).squeeze(-1)
            
        else:
            # Fallback to L2 loss for other representations
            geodesic_dist = torch.norm(pred_pose - true_pose, p=2, dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return geodesic_dist.mean()
        elif self.reduction == 'sum':
            return geodesic_dist.sum()
        else:
            return geodesic_dist


class PoseWarmupScheduler:
    """
    Manages transition from supervised to unsupervised pose learning.
    
    Parameters
    ----------
    warmup_epochs : int
        Number of epochs for pose warmup
    cycle_with_curriculum : bool
        Whether to reset warmup at each curriculum phase
    epochs_per_phase : int
        Number of epochs per curriculum phase
    """
    
    def __init__(self, warmup_epochs: int, cycle_with_curriculum: bool = False, 
                 epochs_per_phase: int = 100):
        self.warmup_epochs = warmup_epochs
        self.cycle_with_curriculum = cycle_with_curriculum
        self.epochs_per_phase = epochs_per_phase
        
    def should_use_supervised(self, epoch: int) -> bool:
        """Determine if supervised pose should be used at this epoch."""
        if self.cycle_with_curriculum:
            # Reset warmup at the start of each curriculum phase
            phase_epoch = epoch % self.epochs_per_phase
            return phase_epoch < self.warmup_epochs
        else:
            # Global warmup only at the beginning
            return epoch < self.warmup_epochs
            
    def get_supervision_weight(self, epoch: int) -> float:
        """Get weight for supervised loss (can implement smooth transition)."""
        if self.should_use_supervised(epoch):
            if self.cycle_with_curriculum:
                phase_epoch = epoch % self.epochs_per_phase
                # Linear decay within warmup period
                return 1.0 - (phase_epoch / self.warmup_epochs) * 0.5
            else:
                return 1.0 - (epoch / self.warmup_epochs) * 0.5
        return 0.0


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


class AdaptiveContrastiveAffinityLoss(ContrastiveAffinityLoss):
    """
    Adaptive contrastive affinity loss that automatically emphasizes high-similarity pairs.
    
    This loss uses a non-parametric weighting scheme where the similarity value itself
    determines the importance of each pair. High-similarity pairs naturally get more
    weight, ensuring accuracy where it matters most.
    """
    
    def __init__(self, *args, weighting_power: float = 2.0, **kwargs):
        """
        Parameters
        ----------
        weighting_power : float
            Power to raise similarities to for importance weighting (default: 2.0).
            Higher values give more emphasis to high-similarity pairs.
        """
        super().__init__(*args, **kwargs)
        self.weighting_power = weighting_power
        print(f"AdaptiveContrastiveAffinityLoss initialized with weighting_power={weighting_power}")
    
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
        """Compute adaptive contrastive affinity loss."""
        try:
            # Get base loss computation up to the point where we have components
            y_true = y_true.contiguous().to(self.device)
            y_pred = y_pred.contiguous().to(self.device)

            if y_true.shape[0] < 2:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            if torch.all(y_true == -1):
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            n_dims = y_pred.shape[1]
            n_dims_to_use = int(n_dims * self.latent_ratio)
            y_pred_partial = y_pred[:, :n_dims_to_use].contiguous()

            z_id = torch.arange(y_pred_partial.shape[0], device=self.device)
            c = torch.combinations(z_id, r=2, with_replacement=False)
            
            features1 = y_pred_partial[c[:, 0], :].contiguous()
            features2 = y_pred_partial[c[:, 1], :].contiguous()
            
            # Calculate distances
            distances = torch.norm(features1 - features2, p=2, dim=1)
            scale_factor = 1.0 / np.sqrt(n_dims_to_use)
            distances = distances * scale_factor
            
            # Get target similarities (same logic as parent class)
            is_background = (y_true == -1)
            pair1_is_bg = is_background[c[:, 0]]
            pair2_is_bg = is_background[c[:, 1]]
            
            target_similarities = torch.zeros_like(distances, device=self.device)
            
            both_bg_mask = pair1_is_bg & pair2_is_bg
            target_similarities[both_bg_mask] = self.background_sim
            
            one_bg_mask = pair1_is_bg ^ pair2_is_bg
            target_similarities[one_bg_mask] = self.background_other_sim
            
            both_obj_mask = ~(both_bg_mask | one_bg_mask)
            if torch.any(both_obj_mask):
                obj_pairs = c[both_obj_mask]
                obj_indices1 = y_true[obj_pairs[:, 0]]
                obj_indices2 = y_true[obj_pairs[:, 1]]
                
                valid_indices = ((obj_indices1 >= 0) & (obj_indices2 >= 0) & 
                            (obj_indices1 < self.lookup.shape[0]) & 
                            (obj_indices2 < self.lookup.shape[0]))
                
                if torch.any(valid_indices):
                    valid_obj_indices1 = obj_indices1[valid_indices].long()
                    valid_obj_indices2 = obj_indices2[valid_indices].long()
                    valid_similarities = self.lookup[valid_obj_indices1, valid_obj_indices2]
                    target_similarities[both_obj_mask][valid_indices] = valid_similarities
            
            # Calculate loss components
            similar_term = target_similarities * (distances ** 2)
            margin_dist = torch.clamp(self.margin - distances, min=0)
            dissimilar_term = (1 - target_similarities) * (margin_dist ** 2)
            
            # Non-parametric adaptive weighting based on similarity
            # Use similarity^power as importance weight
            # This naturally emphasizes high-similarity pairs
            importance_weights = target_similarities ** self.weighting_power
            
            # Normalize weights to maintain gradient scale
            importance_weights = importance_weights / (importance_weights.mean() + 1e-8)
            
            # Apply adaptive weighting
            losses = importance_weights * (similar_term + dissimilar_term)
            
            # Debug output
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                
            if self._debug_counter % 100 == 0:
                print(f"\nAdaptive Contrastive Loss Debug (step {self._debug_counter}):")
                print(f"  Importance weights - Min: {importance_weights.min():.4f}, Max: {importance_weights.max():.4f}, Mean: {importance_weights.mean():.4f}")
                print(f"  High sim pairs (>0.9): {(target_similarities > 0.9).sum().item()} with avg weight: {importance_weights[target_similarities > 0.9].mean():.4f if (target_similarities > 0.9).any() else 0}")
                print(f"  Low sim pairs (<0.5): {(target_similarities < 0.5).sum().item()} with avg weight: {importance_weights[target_similarities < 0.5].mean():.4f if (target_similarities < 0.5).any() else 0}")
            
            if per_sample:
                return losses.contiguous()
            
            result = torch.mean(losses)
            
            if torch.isnan(result):
                zero_tensor = torch.tensor(0.0, device=self.device)
                result = zero_tensor.requires_grad_() + result.detach() * 0
                
            return result

        except Exception as e:
            print(f"Error in adaptive contrastive affinity loss: {str(e)}")
            zero_tensor = torch.tensor(0.0, device=self.device)
            return zero_tensor.requires_grad_()


class GeodesicPoseLoss(nn.Module):
    """
    Compute geodesic distance between predicted and true rotations.
    
    For axis-angle representation: [angle, ax, ay, az]
    The loss is the geodesic distance on SO(3) manifold.
    
    Parameters
    ----------
    pose_channels : int
        Number of pose channels (1 for single rotation, 4 for axis-angle)
    reduction : str
        How to reduce the loss ('mean' or 'sum')
    """
    
    def __init__(self, pose_channels: int = 4, reduction: str = 'mean'):
        super().__init__()
        self.pose_channels = pose_channels
        self.reduction = reduction
        
    def axis_angle_to_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle representation to rotation matrix.
        
        Parameters
        ----------
        axis_angle : torch.Tensor
            Shape (B, 4) with [angle, ax, ay, az]
            
        Returns
        -------
        torch.Tensor
            Rotation matrices of shape (B, 3, 3)
        """
        batch_size = axis_angle.shape[0]
        angle = axis_angle[:, 0]  # (B,)
        axis = axis_angle[:, 1:4]  # (B, 3)
        
        # Normalize axis to unit vector
        axis = F.normalize(axis, p=2, dim=1, eps=1e-6)
        
        # Rodrigues' rotation formula
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Create skew-symmetric matrix K from axis
        K = torch.zeros((batch_size, 3, 3), device=axis_angle.device, dtype=axis_angle.dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        K_squared = torch.bmm(K, K)
        
        R = I + sin_angle.unsqueeze(-1).unsqueeze(-1) * K + \
            (1 - cos_angle).unsqueeze(-1).unsqueeze(-1) * K_squared
        
        return R
    
    def forward(self, pred_pose: torch.Tensor, true_pose: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic distance between predicted and true rotations.
        
        Parameters
        ----------
        pred_pose : torch.Tensor
            Predicted pose (B, pose_channels)
        true_pose : torch.Tensor
            True pose (B, pose_channels)
            
        Returns
        -------
        torch.Tensor
            Geodesic loss
        """
        # Ensure tensors are on same device and dtype
        if pred_pose.device != true_pose.device:
            true_pose = true_pose.to(pred_pose.device)
        if pred_pose.dtype != true_pose.dtype:
            true_pose = true_pose.to(pred_pose.dtype)
            
        if self.pose_channels == 4:
            # Check for NaN or invalid values in input
            if torch.isnan(pred_pose).any() or torch.isnan(true_pose).any():
                print(f"WARNING: NaN in pose inputs to GeodesicPoseLoss")
                return torch.tensor(0.0, device=pred_pose.device, dtype=pred_pose.dtype)
            
            # Convert to rotation matrices
            pred_rot = self.axis_angle_to_matrix(pred_pose)
            true_rot = self.axis_angle_to_matrix(true_pose)
            
            # Check for NaN in rotation matrices
            if torch.isnan(pred_rot).any() or torch.isnan(true_rot).any():
                print(f"WARNING: NaN in rotation matrices")
                return torch.tensor(0.0, device=pred_pose.device, dtype=pred_pose.dtype)
            
            # Compute relative rotation: R_rel = R_true @ R_pred^T
            R_rel = torch.bmm(true_rot, pred_rot.transpose(-2, -1))
            
            # Extract angle from relative rotation
            # angle = arccos((trace(R_rel) - 1) / 2)
            trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            cos_angle = (trace - 1) / 2
            
            # Clamp for numerical stability - use wider bounds
            cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
            
            # Check if clamping was needed (indicates potential numerical issues)
            if ((cos_angle < -0.999) | (cos_angle > 0.999)).any():
                print(f"WARNING: Cos angle near boundary: min={cos_angle.min()}, max={cos_angle.max()}")
            
            geodesic_dist = torch.acos(cos_angle)
            
            # Final NaN check
            if torch.isnan(geodesic_dist).any():
                print(f"WARNING: NaN in geodesic distance computation")
                print(f"  trace: {trace}")
                print(f"  cos_angle: {cos_angle}")
                return torch.tensor(0.0, device=pred_pose.device, dtype=pred_pose.dtype)
            
        elif self.pose_channels == 1:
            # For 1D rotation, use simple angular distance
            # Wrap angles to [-π, π]
            pred_wrapped = torch.remainder(pred_pose + np.pi, 2 * np.pi) - np.pi
            true_wrapped = torch.remainder(true_pose + np.pi, 2 * np.pi) - np.pi
            
            # Angular distance
            diff = pred_wrapped - true_wrapped
            geodesic_dist = torch.abs(torch.remainder(diff + np.pi, 2 * np.pi) - np.pi).squeeze(-1)
            
        else:
            # Fallback to L2 loss for other representations
            geodesic_dist = torch.norm(pred_pose - true_pose, p=2, dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return geodesic_dist.mean()
        elif self.reduction == 'sum':
            return geodesic_dist.sum()
        else:
            return geodesic_dist


class PoseWarmupScheduler:
    """
    Manages transition from supervised to unsupervised pose learning.
    
    Parameters
    ----------
    warmup_epochs : int
        Number of epochs for pose warmup
    cycle_with_curriculum : bool
        Whether to reset warmup at each curriculum phase
    epochs_per_phase : int
        Number of epochs per curriculum phase
    """
    
    def __init__(self, warmup_epochs: int, cycle_with_curriculum: bool = False, 
                 epochs_per_phase: int = 100):
        self.warmup_epochs = warmup_epochs
        self.cycle_with_curriculum = cycle_with_curriculum
        self.epochs_per_phase = epochs_per_phase
        
    def should_use_supervised(self, epoch: int) -> bool:
        """Determine if supervised pose should be used at this epoch."""
        if self.cycle_with_curriculum:
            # Reset warmup at the start of each curriculum phase
            phase_epoch = epoch % self.epochs_per_phase
            return phase_epoch < self.warmup_epochs
        else:
            # Global warmup only at the beginning
            return epoch < self.warmup_epochs
            
    def get_supervision_weight(self, epoch: int) -> float:
        """Get weight for supervised loss (can implement smooth transition)."""
        if self.should_use_supervised(epoch):
            if self.cycle_with_curriculum:
                phase_epoch = epoch % self.epochs_per_phase
                # Linear decay within warmup period
                return 1.0 - (phase_epoch / self.warmup_epochs) * 0.5
            else:
                return 1.0 - (epoch / self.warmup_epochs) * 0.5
        return 0.0


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
        
        # Background similarity values (in [0,1] range)
        self.background_sim = 0.2  # Low similarity between background samples
        self.background_other_sim = 0.01  # Very low similarity between background and objects
        
        print(f"AffinityCosineLoss initialized:")
        print(f"  Latent ratio: {self.latent_ratio}")
        print(f"  Background-background similarity: {self.background_sim}")
        print(f"  Background-object similarity: {self.background_other_sim}")

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
            cosine_similarity = self.cos(features1_norm, features2_norm)
            
            # CRITICAL FIX: Map cosine similarity from [-1,+1] to [0,1] range
            # to match target similarities from lookup table which are in [0,1]
            latent_similarity = (cosine_similarity + 1.0) / 2.0
            
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


class GeodesicPoseLoss(nn.Module):
    """
    Compute geodesic distance between predicted and true rotations.
    
    For axis-angle representation: [angle, ax, ay, az]
    The loss is the geodesic distance on SO(3) manifold.
    
    Parameters
    ----------
    pose_channels : int
        Number of pose channels (1 for single rotation, 4 for axis-angle)
    reduction : str
        How to reduce the loss ('mean' or 'sum')
    """
    
    def __init__(self, pose_channels: int = 4, reduction: str = 'mean'):
        super().__init__()
        self.pose_channels = pose_channels
        self.reduction = reduction
        
    def axis_angle_to_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle representation to rotation matrix.
        
        Parameters
        ----------
        axis_angle : torch.Tensor
            Shape (B, 4) with [angle, ax, ay, az]
            
        Returns
        -------
        torch.Tensor
            Rotation matrices of shape (B, 3, 3)
        """
        batch_size = axis_angle.shape[0]
        angle = axis_angle[:, 0]  # (B,)
        axis = axis_angle[:, 1:4]  # (B, 3)
        
        # Normalize axis to unit vector
        axis = F.normalize(axis, p=2, dim=1, eps=1e-6)
        
        # Rodrigues' rotation formula
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Create skew-symmetric matrix K from axis
        K = torch.zeros((batch_size, 3, 3), device=axis_angle.device, dtype=axis_angle.dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        K_squared = torch.bmm(K, K)
        
        R = I + sin_angle.unsqueeze(-1).unsqueeze(-1) * K + \
            (1 - cos_angle).unsqueeze(-1).unsqueeze(-1) * K_squared
        
        return R
    
    def forward(self, pred_pose: torch.Tensor, true_pose: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic distance between predicted and true rotations.
        
        Parameters
        ----------
        pred_pose : torch.Tensor
            Predicted pose (B, pose_channels)
        true_pose : torch.Tensor
            True pose (B, pose_channels)
            
        Returns
        -------
        torch.Tensor
            Geodesic loss
        """
        # Ensure tensors are on same device and dtype
        if pred_pose.device != true_pose.device:
            true_pose = true_pose.to(pred_pose.device)
        if pred_pose.dtype != true_pose.dtype:
            true_pose = true_pose.to(pred_pose.dtype)
            
        if self.pose_channels == 4:
            # Check for NaN or invalid values in input
            if torch.isnan(pred_pose).any() or torch.isnan(true_pose).any():
                print(f"WARNING: NaN in pose inputs to GeodesicPoseLoss")
                return torch.tensor(0.0, device=pred_pose.device, dtype=pred_pose.dtype)
            
            # Convert to rotation matrices
            pred_rot = self.axis_angle_to_matrix(pred_pose)
            true_rot = self.axis_angle_to_matrix(true_pose)
            
            # Check for NaN in rotation matrices
            if torch.isnan(pred_rot).any() or torch.isnan(true_rot).any():
                print(f"WARNING: NaN in rotation matrices")
                return torch.tensor(0.0, device=pred_pose.device, dtype=pred_pose.dtype)
            
            # Compute relative rotation: R_rel = R_true @ R_pred^T
            R_rel = torch.bmm(true_rot, pred_rot.transpose(-2, -1))
            
            # Extract angle from relative rotation
            # angle = arccos((trace(R_rel) - 1) / 2)
            trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            cos_angle = (trace - 1) / 2
            
            # Clamp for numerical stability - use wider bounds
            cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
            
            # Check if clamping was needed (indicates potential numerical issues)
            if ((cos_angle < -0.999) | (cos_angle > 0.999)).any():
                print(f"WARNING: Cos angle near boundary: min={cos_angle.min()}, max={cos_angle.max()}")
            
            geodesic_dist = torch.acos(cos_angle)
            
            # Final NaN check
            if torch.isnan(geodesic_dist).any():
                print(f"WARNING: NaN in geodesic distance computation")
                print(f"  trace: {trace}")
                print(f"  cos_angle: {cos_angle}")
                return torch.tensor(0.0, device=pred_pose.device, dtype=pred_pose.dtype)
            
        elif self.pose_channels == 1:
            # For 1D rotation, use simple angular distance
            # Wrap angles to [-π, π]
            pred_wrapped = torch.remainder(pred_pose + np.pi, 2 * np.pi) - np.pi
            true_wrapped = torch.remainder(true_pose + np.pi, 2 * np.pi) - np.pi
            
            # Angular distance
            diff = pred_wrapped - true_wrapped
            geodesic_dist = torch.abs(torch.remainder(diff + np.pi, 2 * np.pi) - np.pi).squeeze(-1)
            
        else:
            # Fallback to L2 loss for other representations
            geodesic_dist = torch.norm(pred_pose - true_pose, p=2, dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return geodesic_dist.mean()
        elif self.reduction == 'sum':
            return geodesic_dist.sum()
        else:
            return geodesic_dist


class PoseWarmupScheduler:
    """
    Manages transition from supervised to unsupervised pose learning.
    
    Parameters
    ----------
    warmup_epochs : int
        Number of epochs for pose warmup
    cycle_with_curriculum : bool
        Whether to reset warmup at each curriculum phase
    epochs_per_phase : int
        Number of epochs per curriculum phase
    """
    
    def __init__(self, warmup_epochs: int, cycle_with_curriculum: bool = False, 
                 epochs_per_phase: int = 100):
        self.warmup_epochs = warmup_epochs
        self.cycle_with_curriculum = cycle_with_curriculum
        self.epochs_per_phase = epochs_per_phase
        
    def should_use_supervised(self, epoch: int) -> bool:
        """Determine if supervised pose should be used at this epoch."""
        if self.cycle_with_curriculum:
            # Reset warmup at the start of each curriculum phase
            phase_epoch = epoch % self.epochs_per_phase
            return phase_epoch < self.warmup_epochs
        else:
            # Global warmup only at the beginning
            return epoch < self.warmup_epochs
            
    def get_supervision_weight(self, epoch: int) -> float:
        """Get weight for supervised loss (can implement smooth transition)."""
        if self.should_use_supervised(epoch):
            if self.cycle_with_curriculum:
                phase_epoch = epoch % self.epochs_per_phase
                # Linear decay within warmup period
                return 1.0 - (phase_epoch / self.warmup_epochs) * 0.5
            else:
                return 1.0 - (epoch / self.warmup_epochs) * 0.5
        return 0.0
