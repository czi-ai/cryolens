"""
Base decoders and torch Modules
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Tuple, Optional, Union, List, Callable


class BaseDecoder(torch.nn.Module):
    """Base class for all decoders.
    
    This provides a common interface that all decoders should implement.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the decoder.
        
        Parameters
        ----------
        *args : Any
            Positional arguments to the decoder.
        **kwargs : Any
            Keyword arguments to the decoder.
            
        Returns
        -------
        torch.Tensor
            The decoded output.
        """
        raise NotImplementedError("Subclasses must implement forward")


class SoftStep(torch.nn.Module):
    """Soft (differentiable) step function in the range of 0-1.
    
    This is a sigmoid function with adjustable steepness.
    
    Parameters
    ----------
    k : float
        Controls the steepness of the sigmoid. Higher values make it more step-like.
    """

    def __init__(self, *, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the soft step function.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Output tensor with values sigmoidally mapped to the range [0, 1].
        """
        return 1.0 / (1.0 + torch.exp(-self.k * x))


class Negate(torch.nn.Module):
    """Simple layer that negates the input.
    
    This can be useful in sequential models where you need to flip the sign.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Negate the input tensor.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Negated tensor (-x).
        """
        return -x


class STEFunction(torch.autograd.Function):
    """Straight-Through Estimator autograd function.
    
    This implements a binary thresholding in the forward pass,
    but passes through gradients unmodified in the backward pass.
    """
    
    @staticmethod
    def forward(ctx, input):
        """Apply binary threshold in forward pass."""
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """Pass gradient through unmodified in backward pass."""
        return torch.nn.functional.hardtanh(grad_output)


class StraightThroughEstimator(torch.nn.Module):
    """Straight-through estimator for binary activation.
    
    During the forward pass, this applies a step function (binarization).
    During backprop, it acts as an identity function, allowing gradients to flow.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply straight-through estimation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Binarized tensor (0 or 1 values) in forward pass,
            but with unmodified gradients in backward pass.
        """
        return STEFunction.apply(x)


class GaussianModulationLayer(torch.nn.Module):
    """A layer that applies Gaussian modulation to feature maps.
    
    This can be used to introduce locality in feature maps.
    
    Parameters
    ----------
    channels : int
        Number of input channels.
    sigma : float
        Standard deviation of the Gaussian function.
    learnable : bool
        Whether the sigma parameter should be learnable.
    """
    
    def __init__(self, channels: int, sigma: float = 1.0, learnable: bool = False):
        super().__init__()
        self.channels = channels
        
        if learnable:
            self.sigma = nn.Parameter(torch.tensor(sigma))
        else:
            self.register_buffer('sigma', torch.tensor(sigma))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian modulation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, channels, *spatial_dims).
            
        Returns
        -------
        torch.Tensor
            Output tensor with Gaussian modulation applied.
        """
        # Get spatial dimensions
        spatial_dims = x.shape[2:]
        ndim = len(spatial_dims)
        
        # Create coordinates grid
        coords = [torch.linspace(-1, 1, d, device=x.device) for d in spatial_dims]
        
        # Create meshgrid with 'ij' indexing to match PyTorch convention
        if ndim == 3:
            X, Y, Z = torch.meshgrid(coords, indexing='ij')
            D_squared = X**2 + Y**2 + Z**2
        elif ndim == 2:
            X, Y = torch.meshgrid(coords, indexing='ij')
            D_squared = X**2 + Y**2
        else:
            raise ValueError(f"Only 2D and 3D inputs are supported, got {ndim}D input")
            
        # Apply Gaussian
        gaussian = torch.exp(-D_squared / (2 * self.sigma**2))
        
        # Reshape for broadcasting to match input tensor dimensions
        # For 3D: (1, 1, D, H, W)
        # For 2D: (1, 1, H, W)
        gaussian = gaussian.view(1, 1, *gaussian.shape)
        
        # Apply modulation
        return x * gaussian


class ConvDecoder3D(BaseDecoder):
    """Simple 3D convolutional decoder for volumetric data.
    
    Parameters
    ----------
    latent_dims : int
        Dimension of the latent space.
    output_shape : tuple
        Shape of the output volume (D, H, W).
    channels : list
        List of channel dimensions for each layer.
    output_channels : int
        Number of channels in the output volume (default 1).
    activation : callable
        Activation function to use after each layer except the last.
    """
    
    def __init__(
        self,
        latent_dims: int,
        output_shape: Tuple[int, int, int],
        channels: List[int] = [128, 64, 32, 16],
        output_channels: int = 1,
        activation: callable = torch.nn.ReLU,
    ):
        super().__init__()
        
        self.latent_dims = latent_dims
        self.output_shape = output_shape
        self.output_channels = output_channels
        
        # Calculate the initial feature map size
        # We will upsample log2(output_size) times
        depth, height, width = output_shape
        num_upsamples = len(channels)
        
        # Initial feature map size after linear projection
        init_d = depth // (2 ** num_upsamples)
        init_h = height // (2 ** num_upsamples)
        init_w = width // (2 ** num_upsamples)
        
        # Initial linear projection to 3D feature map
        self.linear = nn.Linear(latent_dims, channels[0] * init_d * init_h * init_w)
        self.init_shape = (channels[0], init_d, init_h, init_w)
        
        # Build transposed convolution layers
        layers = []
        
        for i in range(len(channels) - 1):
            layers.append(
                nn.ConvTranspose3d(
                    channels[i], 
                    channels[i + 1], 
                    kernel_size=4, 
                    stride=2, 
                    padding=1
                )
            )
            layers.append(activation())
        
        # Final layer to output_channels
        layers.append(
            nn.ConvTranspose3d(
                channels[-1], 
                output_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1
            )
        )
        
        # Sigmoid activation for final layer to get [0, 1] range
        layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, pose: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the decoder.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent vector with shape (batch_size, latent_dims).
        pose : Optional[torch.Tensor]
            Pose tensor (unused in this decoder, but included for API compatibility).
            
        Returns
        -------
        torch.Tensor
            Reconstructed 3D volume with shape 
            (batch_size, output_channels, depth, height, width).
        """
        # Initial linear projection
        x = self.linear(z)
        
        # Reshape to 3D feature map
        x = x.view(-1, *self.init_shape)
        
        # Apply transposed convolutions
        x = self.decoder(x)
        
        return x