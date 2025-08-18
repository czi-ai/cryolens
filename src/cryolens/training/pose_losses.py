"""
Pose-specific loss functions for CryoLens model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix.
    
    Args:
        axis_angle: [B, 4] tensor where each row is [angle, ax, ay, az]
                   with angle in radians and (ax, ay, az) as unit vector
    
    Returns:
        rotation_matrix: [B, 3, 3] rotation matrices
    """
    batch_size = axis_angle.shape[0]
    device = axis_angle.device
    
    # Check for NaN/Inf in input
    if torch.isnan(axis_angle).any() or torch.isinf(axis_angle).any():
        print(f"WARNING: NaN or Inf detected in axis_angle input: {axis_angle}")
        # Return identity matrices as fallback
        return torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Extract angle and axis
    angle = axis_angle[:, 0:1]  # [B, 1]
    axis = axis_angle[:, 1:4]   # [B, 3]
    
    # Normalize axis to unit vector (with small epsilon for stability)
    axis_norm = torch.norm(axis, p=2, dim=1, keepdim=True)
    # Handle zero-norm axis (return identity for no rotation)
    is_zero_axis = axis_norm < 1e-6
    axis = torch.where(is_zero_axis, torch.tensor([0., 0., 1.], device=device), axis)
    axis = F.normalize(axis, p=2, dim=1, eps=1e-6)
    
    # Compute components for Rodrigues' formula
    cos_angle = torch.cos(angle)  # [B, 1]
    sin_angle = torch.sin(angle)  # [B, 1]
    one_minus_cos = 1 - cos_angle  # [B, 1]
    
    # Extract axis components
    ax = axis[:, 0:1]  # [B, 1]
    ay = axis[:, 1:2]  # [B, 1]
    az = axis[:, 2:3]  # [B, 1]
    
    # Build the rotation matrix using Rodrigues' formula
    # R = I + sin(θ) * K + (1 - cos(θ)) * K^2
    # where K is the skew-symmetric matrix of the axis
    
    # Create identity matrix for each batch
    R = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Add sin(θ) * K term (skew-symmetric matrix)
    R[:, 0, 1] -= sin_angle.squeeze() * az.squeeze()
    R[:, 0, 2] += sin_angle.squeeze() * ay.squeeze()
    R[:, 1, 0] += sin_angle.squeeze() * az.squeeze()
    R[:, 1, 2] -= sin_angle.squeeze() * ax.squeeze()
    R[:, 2, 0] -= sin_angle.squeeze() * ay.squeeze()
    R[:, 2, 1] += sin_angle.squeeze() * ax.squeeze()
    
    # Add (1 - cos(θ)) * K^2 term
    # K^2 = aaT - I (for unit vector a)
    outer_product = axis.unsqueeze(2) @ axis.unsqueeze(1)  # [B, 3, 3]
    K_squared = outer_product - torch.eye(3, device=device).unsqueeze(0)
    R += one_minus_cos.unsqueeze(2) * K_squared
    
    return R


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quaternion: [B, 4] tensor of quaternions [x, y, z, w]
    
    Returns:
        rotation_matrix: [B, 3, 3] rotation matrices
    """
    # Normalize quaternion
    quaternion = F.normalize(quaternion, p=2, dim=1)
    
    x, y, z, w = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    # Compute rotation matrix elements
    batch_size = quaternion.shape[0]
    device = quaternion.device
    R = torch.zeros(batch_size, 3, 3, device=device)
    
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
    return R


class GeodesicPoseLoss(nn.Module):
    """
    Geodesic loss for rotation representations on SO(3) manifold.
    
    This loss computes the geodesic distance between predicted and ground truth
    rotations, which is the natural metric on the SO(3) manifold.
    
    Parameters
    ----------
    representation : str
        Type of rotation representation: 'axis_angle' or 'quaternion'
    reduction : str
        Reduction method: 'mean' or 'sum'
    """
    
    def __init__(self, representation: str = 'axis_angle', reduction: str = 'mean'):
        super().__init__()
        self.representation = representation
        self.reduction = reduction
        
    def forward(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic loss between predicted and ground truth poses.
        
        Args:
            pred_pose: Predicted pose representation
                      [B, 4] for axis_angle (angle, ax, ay, az) or quaternion (x, y, z, w)
            gt_pose: Ground truth pose representation (same format as pred_pose)
        
        Returns:
            loss: Geodesic distance loss
        """
        # Convert to rotation matrices based on representation
        if self.representation == 'axis_angle':
            pred_rot = axis_angle_to_rotation_matrix(pred_pose)
            gt_rot = axis_angle_to_rotation_matrix(gt_pose)
        elif self.representation == 'quaternion':
            pred_rot = quaternion_to_rotation_matrix(pred_pose)
            gt_rot = quaternion_to_rotation_matrix(gt_pose)
        else:
            raise ValueError(f"Unknown representation: {self.representation}")
        
        # Compute relative rotation: R_rel = R_gt^T @ R_pred
        R_rel = torch.bmm(gt_rot.transpose(1, 2), pred_rot)
        
        # Compute trace of relative rotation
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        
        # Geodesic distance: d(R1, R2) = arccos((tr(R1^T R2) - 1) / 2)
        # Clamp to avoid numerical issues with arccos
        cos_angle = (trace - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
        
        # Compute geodesic distance (in radians)
        geodesic_dist = torch.acos(cos_angle)
        
        # Check for NaN and replace with zero
        if torch.isnan(geodesic_dist).any():
            print(f"WARNING: NaN in geodesic distance, trace values: {trace}")
            geodesic_dist = torch.where(torch.isnan(geodesic_dist), 
                                       torch.zeros_like(geodesic_dist), 
                                       geodesic_dist)
        
        # Apply reduction
        if self.reduction == 'mean':
            return geodesic_dist.mean()
        elif self.reduction == 'sum':
            return geodesic_dist.sum()
        else:
            return geodesic_dist


class ChordPoseLoss(nn.Module):
    """
    Chord distance loss for rotation representations.
    
    This is a simpler alternative to geodesic distance that doesn't require
    arccos computation, making it more numerically stable.
    
    Chord distance: ||R1 - R2||_F where ||.||_F is Frobenius norm
    """
    
    def __init__(self, representation: str = 'axis_angle', reduction: str = 'mean'):
        super().__init__()
        self.representation = representation
        self.reduction = reduction
        
    def forward(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
        """
        Compute chord distance loss between predicted and ground truth poses.
        
        Args:
            pred_pose: Predicted pose representation
            gt_pose: Ground truth pose representation
        
        Returns:
            loss: Chord distance loss
        """
        # Convert to rotation matrices
        if self.representation == 'axis_angle':
            pred_rot = axis_angle_to_rotation_matrix(pred_pose)
            gt_rot = axis_angle_to_rotation_matrix(gt_pose)
        elif self.representation == 'quaternion':
            pred_rot = quaternion_to_rotation_matrix(pred_pose)
            gt_rot = quaternion_to_rotation_matrix(gt_pose)
        else:
            raise ValueError(f"Unknown representation: {self.representation}")
        
        # Compute Frobenius norm of difference
        diff = pred_rot - gt_rot
        chord_dist = torch.norm(diff, p='fro', dim=(1, 2))
        
        # Apply reduction
        if self.reduction == 'mean':
            return chord_dist.mean()
        elif self.reduction == 'sum':
            return chord_dist.sum()
        else:
            return chord_dist


class SupervisedPoseLoss(nn.Module):
    """
    Combined supervised pose loss with multiple components.
    
    This loss can combine geodesic distance with other regularization terms
    for more stable training.
    
    Parameters
    ----------
    loss_type : str
        Type of loss: 'geodesic', 'chord', or 'combined'
    representation : str
        Rotation representation: 'axis_angle' or 'quaternion'
    geodesic_weight : float
        Weight for geodesic loss component (if using combined)
    chord_weight : float
        Weight for chord loss component (if using combined)
    """
    
    def __init__(
        self,
        loss_type: str = 'geodesic',
        representation: str = 'axis_angle',
        geodesic_weight: float = 1.0,
        chord_weight: float = 0.1
    ):
        super().__init__()
        self.loss_type = loss_type
        self.representation = representation
        self.geodesic_weight = geodesic_weight
        self.chord_weight = chord_weight
        
        # Initialize loss components
        if loss_type in ['geodesic', 'combined']:
            self.geodesic_loss = GeodesicPoseLoss(representation)
        if loss_type in ['chord', 'combined']:
            self.chord_loss = ChordPoseLoss(representation)
            
    def forward(
        self,
        pred_pose: torch.Tensor,
        gt_pose: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute supervised pose loss.
        
        Args:
            pred_pose: Predicted pose
            gt_pose: Ground truth pose
            return_components: If True, return dict with individual loss components
        
        Returns:
            loss: Total loss or dict of components
        """
        # Check for NaN/Inf in inputs
        if torch.isnan(pred_pose).any() or torch.isinf(pred_pose).any():
            print(f"WARNING: NaN or Inf in predicted pose: {pred_pose}")
            # Return zero loss to avoid NaN propagation
            if return_components:
                return {'total': torch.tensor(0.0, device=pred_pose.device, requires_grad=True)}
            else:
                return torch.tensor(0.0, device=pred_pose.device, requires_grad=True)
                
        if torch.isnan(gt_pose).any() or torch.isinf(gt_pose).any():
            print(f"WARNING: NaN or Inf in ground truth pose: {gt_pose}")
            # Return zero loss to avoid NaN propagation
            if return_components:
                return {'total': torch.tensor(0.0, device=pred_pose.device, requires_grad=True)}
            else:
                return torch.tensor(0.0, device=pred_pose.device, requires_grad=True)
        
        losses = {}
        
        if self.loss_type == 'geodesic':
            loss = self.geodesic_loss(pred_pose, gt_pose)
            losses['geodesic'] = loss
            
        elif self.loss_type == 'chord':
            loss = self.chord_loss(pred_pose, gt_pose)
            losses['chord'] = loss
            
        elif self.loss_type == 'combined':
            geodesic = self.geodesic_loss(pred_pose, gt_pose)
            chord = self.chord_loss(pred_pose, gt_pose)
            
            losses['geodesic'] = geodesic
            losses['chord'] = chord
            
            loss = self.geodesic_weight * geodesic + self.chord_weight * chord
            losses['total'] = loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if return_components:
            return losses
        else:
            return loss if 'total' not in losses else losses['total']


def compute_angular_error(pred_pose: torch.Tensor, gt_pose: torch.Tensor, 
                          representation: str = 'axis_angle',
                          in_degrees: bool = True) -> torch.Tensor:
    """
    Compute angular error between predicted and ground truth poses.
    
    This is useful for monitoring and debugging.
    
    Args:
        pred_pose: Predicted pose
        gt_pose: Ground truth pose
        representation: Type of representation
        in_degrees: If True, return error in degrees; otherwise radians
    
    Returns:
        angular_error: Angular error for each sample in batch
    """
    # Check for NaN/Inf in inputs
    if torch.isnan(pred_pose).any() or torch.isinf(pred_pose).any():
        print(f"WARNING: NaN or Inf in predicted pose for angular error")
        return torch.zeros(pred_pose.shape[0], device=pred_pose.device)
        
    if torch.isnan(gt_pose).any() or torch.isinf(gt_pose).any():
        print(f"WARNING: NaN or Inf in ground truth pose for angular error")
        return torch.zeros(gt_pose.shape[0], device=gt_pose.device)
    
    # Convert to rotation matrices
    if representation == 'axis_angle':
        pred_rot = axis_angle_to_rotation_matrix(pred_pose)
        gt_rot = axis_angle_to_rotation_matrix(gt_pose)
    elif representation == 'quaternion':
        pred_rot = quaternion_to_rotation_matrix(pred_pose)
        gt_rot = quaternion_to_rotation_matrix(gt_pose)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    
    # Compute relative rotation
    R_rel = torch.bmm(gt_rot.transpose(1, 2), pred_rot)
    
    # Extract angle from relative rotation
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
    
    angle_error = torch.acos(cos_angle)
    
    if in_degrees:
        angle_error = angle_error * 180.0 / np.pi
    
    return angle_error
