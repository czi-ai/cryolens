"""
PCA-based alignment for CryoLens reconstructions.

This module implements PCA-based alignment methods for aligning multiple
3D reconstructions to a common reference frame.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.ndimage import affine_transform, center_of_mass
import logging

logger = logging.getLogger(__name__)


class PCAAlignment:
    """
    PCA-based alignment for 3D volumes.
    
    This class implements principal component analysis-based alignment
    for aligning multiple 3D reconstructions to a common reference frame.
    It's particularly useful for pose normalization in cumulative reconstruction.
    
    Attributes:
        use_weighted_pca (bool): Whether to use splat weights for PCA
        spherical_mask_radius (float): Radius for spherical masking (in Angstroms)
        center_volumes (bool): Whether to center volumes before alignment
    """
    
    def __init__(
        self,
        use_weighted_pca: bool = True,
        spherical_mask_radius: float = 18.0,
        center_volumes: bool = True
    ):
        """
        Initialize PCA alignment.
        
        Args:
            use_weighted_pca: Use weighted PCA with splat weights
            spherical_mask_radius: Radius for spherical mask in Angstroms
            center_volumes: Center volumes before alignment
        """
        self.use_weighted_pca = use_weighted_pca
        self.spherical_mask_radius = spherical_mask_radius
        self.center_volumes = center_volumes
        
    def create_spherical_mask(self, shape: Tuple[int, ...], radius: float) -> np.ndarray:
        """
        Create a spherical mask for the volume.
        
        Args:
            shape: Shape of the volume (Z, Y, X)
            radius: Radius of the sphere in voxels
            
        Returns:
            Binary mask with spherical region
        """
        center = np.array(shape) / 2.0
        coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
        distance = np.sqrt(sum((c - cent)**2 for c, cent in zip(coords, center)))
        mask = distance <= radius
        return mask
    
    def compute_weighted_covariance(
        self,
        volume: np.ndarray,
        weights: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute weighted covariance matrix.
        
        Args:
            volume: 3D volume
            weights: Optional weight map
            mask: Optional binary mask
            
        Returns:
            3x3 covariance matrix
        """
        # Apply mask if provided
        if mask is not None:
            volume = volume * mask
            if weights is not None:
                weights = weights * mask
        
        # Get coordinates of non-zero voxels
        coords = np.array(np.where(volume != 0)).T
        
        if len(coords) == 0:
            return np.eye(3)
        
        # Get values at coordinates
        values = volume[coords[:, 0], coords[:, 1], coords[:, 2]]
        
        # Apply weights if provided
        if weights is not None:
            weight_values = weights[coords[:, 0], coords[:, 1], coords[:, 2]]
            values = values * weight_values
        
        # Center coordinates
        if self.center_volumes:
            mean_coord = np.average(coords, weights=np.abs(values), axis=0)
            centered_coords = coords - mean_coord
        else:
            centered_coords = coords - np.array(volume.shape) / 2.0
        
        # Compute weighted covariance
        total_weight = np.sum(np.abs(values))
        if total_weight > 0:
            weighted_coords = centered_coords * np.abs(values)[:, np.newaxis]
            covariance = np.dot(weighted_coords.T, centered_coords) / total_weight
        else:
            covariance = np.eye(3)
            
        return covariance
    
    def compute_pca_axes(
        self,
        volume: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute principal axes of a volume.
        
        Args:
            volume: 3D volume
            weights: Optional weight map
            
        Returns:
            Tuple of (eigenvectors, eigenvalues)
        """
        # Create spherical mask
        mask = None
        if self.spherical_mask_radius > 0:
            radius_voxels = self.spherical_mask_radius  # Assumes 1Å per voxel
            mask = self.create_spherical_mask(volume.shape, radius_voxels)
        
        # Compute covariance matrix
        covariance = self.compute_weighted_covariance(volume, weights, mask)
        
        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # Sort by eigenvalue (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1
            
        return eigenvectors, eigenvalues
    
    def align_to_reference(
        self,
        volume: np.ndarray,
        reference_axes: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align a volume to reference axes.
        
        Args:
            volume: Volume to align
            reference_axes: Reference eigenvectors (3x3 matrix)
            weights: Optional weight map
            
        Returns:
            Tuple of (aligned_volume, rotation_matrix)
        """
        # Compute PCA axes of the volume
        volume_axes, _ = self.compute_pca_axes(volume, weights)
        
        # Compute rotation matrix
        rotation_matrix = np.dot(reference_axes, volume_axes.T)
        
        # Apply rotation
        aligned_volume = self.apply_rotation(volume, rotation_matrix)
        
        return aligned_volume, rotation_matrix
    
    def apply_rotation(
        self,
        volume: np.ndarray,
        rotation_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Apply rotation matrix to a volume.
        
        Args:
            volume: 3D volume
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Rotated volume
        """
        # Get volume center
        center = np.array(volume.shape) / 2.0
        
        # Create affine transformation matrix
        # Translation to origin, rotation, translation back
        offset = center - np.dot(rotation_matrix, center)
        
        # Apply transformation
        rotated = affine_transform(
            volume,
            rotation_matrix.T,  # Transpose for scipy convention
            offset=offset,
            order=1,  # Linear interpolation
            mode='constant',
            cval=0.0
        )
        
        return rotated
    
    def align_multiple(
        self,
        volumes: List[np.ndarray],
        weights: Optional[List[np.ndarray]] = None,
        reference_idx: int = 0
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
        """
        Align multiple volumes to a common reference frame.
        
        Args:
            volumes: List of 3D volumes to align
            weights: Optional list of weight maps
            reference_idx: Index of reference volume
            
        Returns:
            Tuple of (aligned_volumes, rotation_matrices, alignment_info)
        """
        if len(volumes) == 0:
            return [], [], {}
        
        if weights is None:
            weights = [None] * len(volumes)
            
        # Use first volume as reference
        reference_volume = volumes[reference_idx]
        reference_weight = weights[reference_idx]
        
        # Compute reference axes
        reference_axes, reference_eigenvalues = self.compute_pca_axes(
            reference_volume, reference_weight
        )
        
        # Align all volumes
        aligned_volumes = []
        rotation_matrices = []
        
        for i, (vol, weight) in enumerate(zip(volumes, weights)):
            if i == reference_idx:
                aligned_volumes.append(vol)
                rotation_matrices.append(np.eye(3))
            else:
                aligned_vol, rot_mat = self.align_to_reference(
                    vol, reference_axes, weight
                )
                aligned_volumes.append(aligned_vol)
                rotation_matrices.append(rot_mat)
        
        # Compute alignment quality metrics
        alignment_info = {
            'reference_idx': reference_idx,
            'reference_eigenvalues': reference_eigenvalues,
            'reference_axes': reference_axes,
            'n_volumes': len(volumes)
        }
        
        return aligned_volumes, rotation_matrices, alignment_info
    
    def compute_average(
        self,
        aligned_volumes: List[np.ndarray],
        weights: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute weighted average of aligned volumes.
        
        Args:
            aligned_volumes: List of aligned volumes
            weights: Optional list of weight maps
            
        Returns:
            Averaged volume
        """
        if len(aligned_volumes) == 0:
            raise ValueError("No volumes to average")
        
        if weights is None:
            # Simple average
            return np.mean(aligned_volumes, axis=0)
        
        # Weighted average
        weighted_sum = np.zeros_like(aligned_volumes[0])
        weight_sum = np.zeros_like(aligned_volumes[0])
        
        for vol, weight in zip(aligned_volumes, weights):
            weighted_sum += vol * weight
            weight_sum += weight
        
        # Avoid division by zero
        mask = weight_sum > 0
        average = np.zeros_like(weighted_sum)
        average[mask] = weighted_sum[mask] / weight_sum[mask]
        
        return average


def align_reconstructions_pca(
    reconstructions: List[np.ndarray],
    weights: Optional[List[np.ndarray]] = None,
    spherical_mask_radius: float = 18.0,
    return_average: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for PCA-based alignment of reconstructions.
    
    Args:
        reconstructions: List of 3D reconstructions
        weights: Optional list of weight maps
        spherical_mask_radius: Radius for spherical masking
        return_average: Whether to compute and return average
        
    Returns:
        Dictionary containing:
            - aligned: List of aligned reconstructions
            - rotations: List of rotation matrices
            - average: Average reconstruction (if requested)
            - info: Alignment information
    """
    aligner = PCAAlignment(
        use_weighted_pca=weights is not None,
        spherical_mask_radius=spherical_mask_radius
    )
    
    aligned, rotations, info = aligner.align_multiple(
        reconstructions, weights
    )
    
    result = {
        'aligned': aligned,
        'rotations': rotations,
        'info': info
    }
    
    if return_average:
        result['average'] = aligner.compute_average(aligned, weights)
    
    return result
