"""
Gaussian Splat-based rendering for 3D density volumes.
"""

import torch
from typing import Tuple, Optional

from ..decoders.base import BaseDecoder


class GaussianSplatRenderer(BaseDecoder):
    """Perform gaussian splatting in 3D space.
    
    This renders Gaussian splats in 3D or 2D space by computing the distance
    between coordinates and splat centers, then applying a Gaussian function.
    
    Parameters
    ----------
    shape : tuple
        Shape of the output volume (depth, height, width) or (height, width).
    device : torch.device
        Device to use for computation.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        *,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self._shape = shape
        self._ndim = len(shape)

        if len(shape) not in (2, 3):
            raise ValueError("Only 2D or 3D rendering is currently supported")

        # Create coordinate grid and register as buffer for proper device management
        grids = torch.meshgrid(
            *[torch.linspace(-1, 1, sz) for sz in shape],
            indexing="ij",
        )

        # Add zeros for z if we have a 2d grid
        if len(shape) == 2:
            grids += (torch.zeros_like(grids[0]),)

        # Stack coordinates and reshape for efficient computation
        # Shape: [1, num_points, 3]
        # Stack in [X, Y, Z] order when using indexing="ij"
        coords = torch.stack([torch.ravel(grids[2]), torch.ravel(grids[1]), torch.ravel(grids[0])], axis=0).transpose(0, 1).unsqueeze(0)
        self.register_buffer('coords', coords)

    def forward(
        self,
        splats: torch.Tensor, 
        weights: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        splat_sigma_range: Tuple[float, float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """Render the Gaussian splats with correct tensor dimensions.
        
        Parameters
        ----------
        splats : torch.Tensor
            Tensor of shape [batch_size, 3, num_splats] containing the
            (x, y, z) coordinates of each splat.
        weights : torch.Tensor
            Tensor of shape [batch_size, num_splats] containing the
            amplitude/weight of each splat.
        sigmas : torch.Tensor
            Tensor of shape [batch_size, num_splats] containing the
            standard deviation of each splat.
        splat_sigma_range : tuple
            Min and max values to scale the sigmas to.
            
        Returns
        -------
        torch.Tensor
            Rendered volume of shape [batch_size, 1, *shape].
        """
        
        # Ensure all inputs are on the same device as coords
        device = self.coords.device
        splats = splats.to(device)
        weights = weights.to(device)
        sigmas = sigmas.to(device)
        
        # Clamp weights to prevent explosion
        weights = torch.clamp(weights, 0.0, 1.0)
        
        # Scale the sigma values with clamping
        min_sigma, max_sigma = splat_sigma_range
        sigmas = torch.clamp(
            sigmas * (max_sigma - min_sigma) + min_sigma,
            min=1e-6,
            max=1.0
        )

        # Transpose splats for efficient computation
        splats_t = splats.transpose(1, 2)  # [B, N, D]

        # Calculate squared distances efficiently
        coords_norm = torch.sum(self.coords ** 2, dim=-1, keepdim=True)  # [B, M, 1]
        splats_norm = torch.sum(splats_t ** 2, dim=-1)  # [B, N]
        
        # Compute cross term using matrix multiplication
        cross_term = torch.matmul(self.coords, splats)  # [B, M, N]
        
        # Combine terms for full distance calculation
        D_squared = coords_norm + splats_norm.unsqueeze(1) - 2 * cross_term
        D_squared = torch.clamp(D_squared, min=0.0)

        # Scale gaussians with numerical stability
        sigmas = 2.0 * sigmas.unsqueeze(1) ** 2  # [B, 1, N]
        
        # Calculate gaussian values with stability checks
        gaussian_values = weights.unsqueeze(1) * torch.exp(
            torch.clamp(-D_squared / sigmas, min=-88.0)
        )
        
        # Sum and normalize
        x = torch.sum(gaussian_values, dim=-1)
        x = torch.clamp(x, 0.0, 1.0)

        # Reshape to original volume dimensions
        return x.reshape((-1, *self._shape)).unsqueeze(1)
