"""
Copick integration module for CryoLens.

This module provides utilities for loading and processing cryo-ET data from Copick projects,
including particle extraction from tomograms and integration with the CZ cryoET Data Portal.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

try:
    import copick
    import zarr
    COPICK_AVAILABLE = True
except ImportError:
    COPICK_AVAILABLE = False
    warnings.warn(
        "Copick is not installed. Install it with: pip install copick",
        ImportWarning
    )


class CopickDataLoader:
    """
    A data loader for Copick projects that extracts particles from tomograms.
    
    This class provides utilities for:
    - Loading Copick configurations
    - Extracting particles from tomograms at pick locations
    - Handling multiple voxel spacings
    - Processing orientation information
    
    Attributes
    ----------
    config_path : Path
        Path to the Copick configuration file
    root : copick.Root
        The loaded Copick project root
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the Copick data loader.
        
        Parameters
        ----------
        config_path : Union[str, Path]
            Path to the Copick configuration JSON file
            
        Raises
        ------
        ImportError
            If copick is not installed
        FileNotFoundError
            If the configuration file doesn't exist
        """
        if not COPICK_AVAILABLE:
            raise ImportError(
                "Copick is required for this functionality. "
                "Install it with: pip install copick"
            )
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.root = copick.from_file(str(self.config_path))
    
    def list_available_structures(self) -> List[str]:
        """
        List all available pickable object names in the project.
        
        Returns
        -------
        List[str]
            List of pickable object names
        """
        return [obj.name for obj in self.root.pickable_objects]
    
    def list_runs(self) -> List[str]:
        """
        List all available runs in the project.
        
        Returns
        -------
        List[str]
            List of run names
        """
        return [run.name for run in self.root.runs]
    
    def load_particles(
        self,
        structure_filter: Optional[List[str]] = None,
        max_particles_per_structure: int = 50,
        target_voxel_spacing: float = 10.0,
        voxel_spacing_tolerance: float = 1.0,
        box_size: int = 48,
        runs_to_process: Optional[int] = None,
        normalize: bool = True,
        verbose: bool = True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load particles from the Copick project.
        
        Parameters
        ----------
        structure_filter : Optional[List[str]]
            List of structure names to load. If None, loads all structures.
        max_particles_per_structure : int
            Maximum number of particles to load per structure
        target_voxel_spacing : float
            Target voxel spacing in Angstroms (e.g., 10.0)
        voxel_spacing_tolerance : float
            Tolerance for voxel spacing matching in Angstroms
        box_size : int
            Size of the particle box in voxels
        runs_to_process : Optional[int]
            Maximum number of runs to process. If None, processes all runs.
        normalize : bool
            Whether to normalize particles (z-score normalization)
        verbose : bool
            Whether to print progress information
            
        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Dictionary mapping structure names to dictionaries containing:
            - 'particles': np.ndarray of shape (n_particles, box_size, box_size, box_size)
            - 'orientations': np.ndarray of shape (n_particles, 3, 3) rotation matrices
            - 'positions': np.ndarray of shape (n_particles, 3) positions in Angstroms
            - 'metadata': Dict with additional information
        """
        results = {}
        
        # Initialize results for filtered structures
        if structure_filter:
            for structure in structure_filter:
                results[structure] = {
                    'particles': [],
                    'orientations': [],
                    'positions': [],
                    'metadata': {
                        'voxel_spacings': [],
                        'run_names': []
                    }
                }
        
        runs_processed = 0
        runs = list(self.root.runs)
        
        if runs_to_process is not None:
            runs = runs[:runs_to_process]
        
        for run in runs:
            if verbose:
                print(f"Processing run: {run.name}")
            
            # Check if we have enough particles for all structures
            if structure_filter and self._check_all_structures_complete(
                results, structure_filter, max_particles_per_structure
            ):
                if verbose:
                    print("All structures have enough particles, stopping")
                break
            
            # Find best matching voxel spacing
            best_voxel_spacing = self._find_best_voxel_spacing(
                run, target_voxel_spacing, voxel_spacing_tolerance
            )
            
            if best_voxel_spacing is None:
                if verbose:
                    available = [vs.voxel_size for vs in run.voxel_spacings]
                    print(f"  No tomogram near {target_voxel_spacing}Å (available: {available})")
                continue
            
            if verbose:
                print(f"  Using voxel spacing: {best_voxel_spacing.voxel_size}Å")
            
            # Get tomogram
            tomogram = self._get_tomogram(best_voxel_spacing)
            if tomogram is None:
                if verbose:
                    print("  No tomograms found")
                continue
            
            # Load tomogram data
            try:
                tomogram_data = self._load_tomogram_data(tomogram)
                if tomogram_data is None:
                    continue
                    
                if verbose:
                    print(f"  Loaded tomogram shape: {tomogram_data.shape}")
                
            except Exception as e:
                if verbose:
                    print(f"  Error loading tomogram: {e}")
                continue
            
            # Process picks for each structure
            for picks in run.get_picks(portal_meta_query={'ground_truth_status': True}):
                structure_name = picks.pickable_object_name
                
                # Apply structure filter
                if structure_filter and structure_name not in structure_filter:
                    continue
                
                # Initialize structure entry if needed
                if structure_name not in results:
                    results[structure_name] = {
                        'particles': [],
                        'orientations': [],
                        'positions': [],
                        'metadata': {
                            'voxel_spacings': [],
                            'run_names': []
                        }
                    }
                
                # Check if we have enough particles for this structure
                if len(results[structure_name]['particles']) >= max_particles_per_structure:
                    if verbose:
                        print(f"  {structure_name}: Already have {max_particles_per_structure} particles")
                    continue
                
                if verbose:
                    print(f"  Processing {structure_name}: {len(picks.points)} picks")
                
                # Extract particles from pick locations
                particles_extracted = 0
                for point in picks.points:
                    if len(results[structure_name]['particles']) >= max_particles_per_structure:
                        break
                    
                    # Extract particle
                    particle, position = self._extract_particle(
                        tomogram_data,
                        point,
                        best_voxel_spacing.voxel_size,
                        box_size,
                        normalize
                    )
                    
                    if particle is not None:
                        results[structure_name]['particles'].append(particle)
                        results[structure_name]['positions'].append(position)
                        
                        # Extract orientation
                        orientation = self._extract_orientation(point)
                        results[structure_name]['orientations'].append(orientation)
                        
                        # Store metadata
                        results[structure_name]['metadata']['voxel_spacings'].append(
                            best_voxel_spacing.voxel_size
                        )
                        results[structure_name]['metadata']['run_names'].append(run.name)
                        
                        particles_extracted += 1
                
                if verbose and particles_extracted > 0:
                    print(f"    Extracted {particles_extracted} particles")
            
            # Clean up tomogram data
            del tomogram_data
            runs_processed += 1
        
        # Convert lists to arrays
        for structure_name in results:
            if results[structure_name]['particles']:
                results[structure_name]['particles'] = np.array(
                    results[structure_name]['particles'], dtype=np.float32
                )
                results[structure_name]['orientations'] = np.array(
                    results[structure_name]['orientations'], dtype=np.float32
                )
                results[structure_name]['positions'] = np.array(
                    results[structure_name]['positions'], dtype=np.float32
                )
                
                if verbose:
                    n_particles = len(results[structure_name]['particles'])
                    print(f"Final: {structure_name} - {n_particles} particles")
            else:
                if verbose:
                    print(f"Final: {structure_name} - No particles extracted")
        
        return results
    
    def _check_all_structures_complete(
        self,
        results: Dict,
        structure_filter: List[str],
        max_particles: int
    ) -> bool:
        """Check if all filtered structures have enough particles."""
        for structure in structure_filter:
            if len(results[structure]['particles']) < max_particles:
                return False
        return True
    
    def _find_best_voxel_spacing(
        self,
        run,
        target_spacing: float,
        tolerance: float
    ):
        """Find the voxel spacing closest to the target."""
        best_spacing = None
        best_diff = float('inf')
        
        for vs in run.voxel_spacings:
            diff = abs(vs.voxel_size - target_spacing)
            if diff < best_diff and diff <= tolerance:
                best_diff = diff
                best_spacing = vs
        
        return best_spacing
    
    def _get_tomogram(self, voxel_spacing):
        """Get the first available tomogram, preferring denoised."""
        tomograms = list(voxel_spacing.tomograms)
        if not tomograms:
            return None
        
        # Prefer denoised tomograms
        for tomo in tomograms:
            if hasattr(tomo, 'tomo_type') and tomo.tomo_type == 'denoised':
                return tomo
        
        # Fall back to first available
        return tomograms[0]
    
    def _load_tomogram_data(self, tomogram) -> Optional[np.ndarray]:
        """Load tomogram data from zarr store."""
        try:
            tomo_zarr = zarr.open(tomogram.zarr(), mode='r')
            
            # Handle different zarr structures
            for key in ['0', 's0', 'data']:
                if key in tomo_zarr:
                    return np.array(tomo_zarr[key])
            
            # Try first available key
            keys = list(tomo_zarr.keys())
            if keys:
                return np.array(tomo_zarr[keys[0]])
            
            return None
            
        except Exception as e:
            warnings.warn(f"Error loading tomogram data: {e}")
            return None
    
    def _extract_particle(
        self,
        tomogram_data: np.ndarray,
        point,
        voxel_spacing: float,
        box_size: int,
        normalize: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract a particle subvolume from the tomogram."""
        # Get position in Angstroms
        location = point.location
        if hasattr(location, 'x'):
            position = np.array([location.x, location.y, location.z])
        elif isinstance(location, (list, tuple, np.ndarray)):
            position = np.array(location)
        else:
            return None, None
        
        # Convert to voxel coordinates
        voxel_pos = (position / voxel_spacing).astype(int)
        
        # Calculate bounds
        half_box = box_size // 2
        z_min = voxel_pos[2] - half_box
        z_max = voxel_pos[2] + half_box
        y_min = voxel_pos[1] - half_box
        y_max = voxel_pos[1] + half_box
        x_min = voxel_pos[0] - half_box
        x_max = voxel_pos[0] + half_box
        
        # Check bounds
        if (z_min < 0 or z_max > tomogram_data.shape[0] or
            y_min < 0 or y_max > tomogram_data.shape[1] or
            x_min < 0 or x_max > tomogram_data.shape[2]):
            return None, None
        
        # Extract subvolume (tomogram is in ZYX order)
        subvolume = tomogram_data[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Check if we got the right size
        if subvolume.shape != (box_size, box_size, box_size):
            return None, None
        
        # Normalize if requested
        if normalize:
            mean = np.mean(subvolume)
            std = np.std(subvolume)
            if std > 1e-6:
                subvolume = (subvolume - mean) / std
            else:
                subvolume = subvolume - mean
        
        return subvolume, position
    
    def _extract_orientation(self, point) -> np.ndarray:
        """Extract orientation/rotation matrix from a pick point."""
        try:
            if hasattr(point, 'transformation') and point.transformation is not None:
                transformation = point.transformation
                
                # Handle different transformation formats
                if hasattr(transformation, 'shape'):
                    if transformation.shape == (4, 4):
                        # 4x4 transformation matrix
                        return transformation[:3, :3]
                    elif transformation.shape == (3, 3):
                        # 3x3 rotation matrix
                        return transformation
                
                # Try to convert to array
                transformation = np.array(transformation)
                if transformation.shape == (4, 4):
                    return transformation[:3, :3]
                elif transformation.shape == (3, 3):
                    return transformation
        except Exception:
            pass
        
        # Return identity if no valid transformation
        return np.eye(3, dtype=np.float32)


def load_ml_challenge_configs(base_path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """
    Get the standard ML Challenge Copick configuration paths.
    
    Parameters
    ----------
    base_path : Optional[Union[str, Path]]
        Base path to the ML Challenge configs. If None, will look for
        configs in the current directory or environment variable.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping dataset names to configuration file paths
    """
    import os
    
    # Try to find base path from environment or parameter
    if base_path is None:
        base_path = os.environ.get("CRYOLENS_ML_CHALLENGE_PATH", "")
        if not base_path:
            warnings.warn(
                "ML Challenge config path not specified. "
                "Set CRYOLENS_ML_CHALLENGE_PATH environment variable or provide base_path parameter."
            )
            return {}
    
    base_path = Path(base_path)
    
    configs = {
        "synthetic": base_path / "ml_challenge_synthetic.json",
        "experimental_training": base_path / "ml_challenge_experimental_training.json",
        "experimental_public_test": base_path / "ml_challenge_experimental_publictest.json",
        "experimental_private_test": base_path / "ml_challenge_experimental_privatetest.json"
    }
    
    # Check which configs exist
    available_configs = {}
    for name, path in configs.items():
        if path.exists():
            available_configs[name] = str(path)
        else:
            warnings.warn(f"ML Challenge config not found: {name} at {path}")
    
    return available_configs


def extract_particles_from_tomogram(
    tomogram_data: np.ndarray,
    positions: Union[List[Tuple[float, float, float]], np.ndarray],
    voxel_spacing: float = 10.0,
    box_size: int = 48,
    normalize: bool = True
) -> List[np.ndarray]:
    """
    Extract multiple particle subvolumes from a tomogram.
    
    This is a standalone function for extracting particles when you already have
    tomogram data and pick positions.
    
    Parameters
    ----------
    tomogram_data : np.ndarray
        3D tomogram data array (Z, Y, X order)
    positions : Union[List[Tuple], np.ndarray]
        List of (x, y, z) positions in Angstroms
    voxel_spacing : float
        Voxel spacing in Angstroms
    box_size : int
        Size of the particle box in voxels
    normalize : bool
        Whether to apply z-score normalization
        
    Returns
    -------
    List[np.ndarray]
        List of extracted particle subvolumes
    """
    particles = []
    half_box = box_size // 2
    
    for position in positions:
        # Convert position to voxel coordinates
        voxel_pos = np.array(position) / voxel_spacing
        voxel_pos = voxel_pos.astype(int)
        
        # Calculate bounds
        z_min = voxel_pos[2] - half_box
        z_max = voxel_pos[2] + half_box
        y_min = voxel_pos[1] - half_box
        y_max = voxel_pos[1] + half_box
        x_min = voxel_pos[0] - half_box
        x_max = voxel_pos[0] + half_box
        
        # Check bounds
        if (z_min >= 0 and z_max <= tomogram_data.shape[0] and
            y_min >= 0 and y_max <= tomogram_data.shape[1] and
            x_min >= 0 and x_max <= tomogram_data.shape[2]):
            
            # Extract subvolume
            subvolume = tomogram_data[z_min:z_max, y_min:y_max, x_min:x_max]
            
            if subvolume.shape == (box_size, box_size, box_size):
                # Normalize if requested
                if normalize:
                    mean = np.mean(subvolume)
                    std = np.std(subvolume)
                    if std > 1e-6:
                        subvolume = (subvolume - mean) / std
                    else:
                        subvolume = subvolume - mean
                
                particles.append(subvolume)
    
    return particles
