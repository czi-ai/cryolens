"""
Utility functions and classes for decoders.
"""

import torch

class CartesianAxes:
    """Constants for Cartesian axes."""
    X = 0
    Y = 1
    Z = 2
    
    def __init__(self, axis):
        self.axis = axis
    
    def as_tensor(self, device=None):
        """Convert axis to one-hot tensor.
        
        Parameters
        ----------
        device : torch.device, optional
            Device to create tensor on.
            
        Returns
        -------
        torch.Tensor
            One-hot encoded tensor for the specified axis.
        """
        # Create one-hot tensor
        tensor = torch.zeros(3, device=device)
        tensor[self.axis] = 1.0
        return tensor
    
    @classmethod
    def as_tensor(cls, axis, device=None):
        """Convert axis to one-hot tensor (class method version).
        
        Parameters
        ----------
        axis : int or CartesianAxes
            Axis index (X=0, Y=1, Z=2) or CartesianAxes instance
        device : torch.device, optional
            Device to create tensor on.
            
        Returns
        -------
        torch.Tensor
            One-hot encoded tensor for the specified axis.
        """
        if isinstance(axis, int):
            axis_idx = axis
        elif hasattr(axis, 'axis'):
            axis_idx = axis.axis
        else:
            axis_idx = axis
                
        # Create one-hot tensor
        tensor = torch.zeros(3, device=device)
        tensor[axis_idx] = 1.0
        return tensor


class SpatialDims:
    """Constants for spatial dimensions."""
    TWO = 2
    THREE = 3
