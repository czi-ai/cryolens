"""
3D encoder models for cryoEM/cryoET data.
"""

import torch
import numpy as np
import logging
from typing import Tuple, List, Optional, Union

def dims_after_pooling(start: int, n_pools: int) -> int:
    """Calculate the size of a layer after n pooling ops.

    Parameters
    ----------
    start: int
        The size of the layer before pooling.
    n_pools: int
        The number of pooling operations.

    Returns
    -------
    dims: int
        The size of the layer after pooling.
    """
    return start // (2**n_pools)


class BaseEncoder(torch.nn.Module):
    """Base encoder class for implementing different encoder types."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        
    @property
    def flat_shape(self) -> Union[int, Tuple[int, ...]]:
        """Return the flattened output shape."""
        raise NotImplementedError("Subclasses must implement this property")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder."""
        raise NotImplementedError("Subclasses must implement forward")


class Encoder3D(BaseEncoder):
    """3D convolutional encoder for volumetric data.
    
    Parameters
    ----------
    input_shape : tuple or int
        Shape of the input data. If a single integer is provided, it's assumed to be a cube.
    layer_channels : tuple
        Number of channels for each convolutional layer.
    """

    def __init__(
        self,
        *,
        layer_channels: Tuple[int] = (8, 16, 32, 64),
        input_shape: Union[int, Tuple[int, ...]] = (32, 32, 32),
        debug_output_shape: bool = False  # Add debugging option
    ):
        super().__init__()
        
        # Handle single integer as cubic volume
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape, input_shape)
            
        self.input_shape = input_shape

        # Build the convolutional layers
        self.model = torch.nn.Sequential(
            torch.nn.Conv3d(1, layer_channels[0], 3, stride=2, padding=1),
            torch.nn.ReLU(),
        )

        # Add additional conv layers
        for in_channels, out_channels in zip(
            layer_channels, layer_channels[1:]
        ):
            self.model.append(
                torch.nn.Conv3d(
                    in_channels, out_channels, 3, stride=2, padding=1
                )
            )
            self.model.append(torch.nn.ReLU())

        # Add flattening layer
        self.model.append(torch.nn.Flatten())
        
        # Override flat_shape property to match expected dimensions from the checkpoint
        # This is needed because the original model's encoder output shape is different
        self._override_flat_shape = None
        
        # Calculate output dimensions after pooling
        xd, yd, zd = [
            dims_after_pooling(d, len(layer_channels)) for d in input_shape
        ]
        self.unflat_shape = (layer_channels[-1], xd, yd, zd)
        
        if debug_output_shape:
            logger = logging.getLogger(__name__)
            # Calculate expected output shape for input tensor of shape [batch, 1, *input_shape]
            # after each layer in the model
            logger.info(f"Encoder input shape: [batch, 1, {input_shape[0]}, {input_shape[1]}, {input_shape[2]}]")
            
            # First conv layer with stride 2, padding 1
            # Output shape: [batch, layer_channels[0], input_shape[0]//2, input_shape[1]//2, input_shape[2]//2]
            shape_after_first = [input_shape[i]//2 for i in range(3)]
            logger.info(f"After first conv: [batch, {layer_channels[0]}, {shape_after_first[0]}, {shape_after_first[1]}, {shape_after_first[2]}]")
            
            # Track shape through remaining layers
            curr_shape = shape_after_first
            curr_channels = layer_channels[0]
            
            for i, out_channels in enumerate(layer_channels[1:]):
                # Each layer reduces spatial dims by half
                curr_shape = [curr_shape[j]//2 for j in range(3)]
                curr_channels = out_channels
                logger.info(f"After conv {i+2}: [batch, {curr_channels}, {curr_shape[0]}, {curr_shape[1]}, {curr_shape[2]}]")
            
            # Final flattened shape
            flattened_size = curr_channels * curr_shape[0] * curr_shape[1] * curr_shape[2]
            logger.info(f"Final flattened shape: [batch, {flattened_size}]")
            
            # Set this value to be our unflat_shape
            self.unflat_shape = (curr_channels, curr_shape[0], curr_shape[1], curr_shape[2])
            logger.info(f"Unflat shape: {self.unflat_shape}")
            logger.info(f"Flat shape: {int(np.prod(self.unflat_shape))}")

    @property
    def flat_shape(self) -> int:
        """Return the flattened output shape."""
        if self._override_flat_shape is not None:
            return self._override_flat_shape
        return int(np.prod(self.unflat_shape))
        
    def set_flat_shape_override(self, shape: int) -> None:
        """Set an override value for the flat_shape property.
        
        This is used to make the encoder compatible with models trained
        with different architectures or dimensions.
        
        Parameters
        ----------
        shape : int
            The flat shape value to use.
        """
        self._override_flat_shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, *input_shape)
            
        Returns
        -------
        torch.Tensor
            The encoded features
        """
        # Get encoder output
        output = self.model(x)
        
        # Check if we need to reshape to match expected dimensions
        if self._override_flat_shape is not None and output.shape[1] != self._override_flat_shape:
            logger = logging.getLogger(__name__)
            logger.warning(f"Encoder output shape {output.shape} doesn't match expected shape [batch, {self._override_flat_shape}]")
            logger.warning(f"Reshaping output to match expected dimensions. This may affect model performance.")
            
            # Reshape to match expected dimensions
            batch_size = output.shape[0]
            output = output.reshape(batch_size, self._override_flat_shape)
            
        return output


class ResidualBlock3D(torch.nn.Module):
    """3D Residual block with skip connections.
    
    Parameters
    ----------
    channels : int
        Number of input and output channels.
    kernel_size : int
        Size of the convolutional kernel.
    """
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        
        self.conv1 = torch.nn.Conv3d(
            channels, channels, kernel_size, padding=kernel_size//2
        )
        self.bn1 = torch.nn.BatchNorm3d(channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv3d(
            channels, channels, kernel_size, padding=kernel_size//2
        )
        self.bn2 = torch.nn.BatchNorm3d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class DualStreamSeparator(torch.nn.Module):
    """Separates encoded features into content and pose streams.
    
    Following DINOv3 approach, this module splits the encoded features into
    two streams with different regularization strategies:
    - Content stream: more regularization for invariance
    - Pose stream: less regularization to preserve pose information
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features from encoder.
    content_dim : int
        Dimension of content features.
    pose_dim : int
        Dimension of pose features.
    dropout_rate : float
        Dropout rate for content stream (default: 0.2).
    """
    
    def __init__(
        self,
        input_dim: int,
        content_dim: int,
        pose_dim: int,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.content_dim = content_dim
        self.pose_dim = pose_dim
        
        # Content stream: more regularization for invariance
        self.content_stream = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // 2),
            torch.nn.BatchNorm1d(input_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),  # Higher dropout for invariance
            torch.nn.Linear(input_dim // 2, content_dim)
        )
        
        # Pose stream: less regularization to preserve pose info
        self.pose_stream = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // 2),
            torch.nn.BatchNorm1d(input_dim // 2),
            torch.nn.ReLU(),
            # No dropout to preserve pose information
            torch.nn.Linear(input_dim // 2, pose_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to separate features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features from encoder [batch_size, input_dim].
            
        Returns
        -------
        tuple
            (content_features, pose_features)
        """
        content_features = self.content_stream(x)
        pose_features = self.pose_stream(x)
        
        return content_features, pose_features


class ResNetEncoder3D(BaseEncoder):
    """3D ResNet-style encoder for volumetric data with residual connections.
    
    Parameters
    ----------
    input_shape : tuple or int
        Shape of the input data. If a single integer is provided, it's assumed to be a cube.
    layer_channels : tuple
        Number of channels for each layer group.
    blocks_per_layer : int or list
        Number of residual blocks in each layer group.
    """
    
    def __init__(
        self,
        *,
        layer_channels: Tuple[int] = (16, 32, 64, 128),
        input_shape: Union[int, Tuple[int, ...]] = (32, 32, 32),
        blocks_per_layer: Union[int, List[int]] = 2,
    ):
        super().__init__()
        
        # Handle single integer as cubic volume
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape, input_shape)
            
        self.input_shape = input_shape
        
        # Handle uniform blocks per layer
        if isinstance(blocks_per_layer, int):
            blocks_per_layer = [blocks_per_layer] * len(layer_channels)
            
        # Initial convolution
        self.conv1 = torch.nn.Conv3d(1, layer_channels[0], kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm3d(layer_channels[0])
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual layer groups
        self.layers = torch.nn.ModuleList()
        in_channels = layer_channels[0]
        
        for i, out_channels in enumerate(layer_channels[1:]):
            # Downsample at the beginning of each layer group
            downsample = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                torch.nn.BatchNorm3d(out_channels)
            )
            
            layer = torch.nn.Sequential()
            
            # First block with downsampling
            layer.append(
                self._make_downsample_block(in_channels, out_channels, downsample)
            )
            
            # Remaining blocks
            for _ in range(1, blocks_per_layer[i]):
                layer.append(ResidualBlock3D(out_channels))
                
            self.layers.append(layer)
            in_channels = out_channels
            
        # Global average pooling and flattening
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = torch.nn.Flatten()
        
        # Calculate output shape
        # First conv + maxpool = 2 down steps
        # Each layer group adds one down step
        total_down_steps = 2 + len(layer_channels) - 1
        
        # Calculate output dimensions
        xd, yd, zd = [
            dims_after_pooling(d, total_down_steps) for d in input_shape
        ]
        
        self.unflat_shape = (layer_channels[-1], xd, yd, zd)
        self.final_channels = layer_channels[-1]
        
    def _make_downsample_block(
        self, in_channels: int, out_channels: int, downsample: torch.nn.Module
    ) -> torch.nn.Module:
        """Create a residual block with downsampling."""
        
        class DownsampleBlock(torch.nn.Module):
            def __init__(self, in_ch, out_ch, downsample_layer):
                super().__init__()
                self.conv1 = torch.nn.Conv3d(
                    in_ch, out_ch, kernel_size=3, stride=2, padding=1
                )
                self.bn1 = torch.nn.BatchNorm3d(out_ch)
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv2 = torch.nn.Conv3d(
                    out_ch, out_ch, kernel_size=3, padding=1
                )
                self.bn2 = torch.nn.BatchNorm3d(out_ch)
                self.downsample = downsample_layer
                
            def forward(self, x):
                identity = self.downsample(x)
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                
                out += identity
                out = self.relu(out)
                
                return out
                
        return DownsampleBlock(in_channels, out_channels, downsample)
    
    @property
    def flat_shape(self) -> int:
        """Return the flattened output shape."""
        return self.final_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, *input_shape)
            
        Returns
        -------
        torch.Tensor
            The encoded features
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.avgpool(x)
        x = self.flatten(x)
        
        return x