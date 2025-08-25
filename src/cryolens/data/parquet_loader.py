"""
Utilities for loading data from Parquet files.

This module provides functions for loading cryo-EM subvolumes and metadata
from Parquet files organized by structure and SNR.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


def extract_volume_from_row(
    row: pd.Series,
    expected_shape: Tuple[int, int, int] = (48, 48, 48)
) -> Optional[np.ndarray]:
    """
    Extract volume data from a parquet row.
    
    Handles different storage formats including bytes and numpy arrays.
    
    Parameters
    ----------
    row : pd.Series
        Row from parquet dataframe
    expected_shape : Tuple[int, int, int]
        Expected shape of the volume
        
    Returns
    -------
    Optional[np.ndarray]
        Extracted volume or None if extraction fails
    """
    # Try different column names
    volume_data = None
    for col_name in ['subvolume', 'volume', 'data']:
        if col_name in row:
            volume_data = row[col_name]
            break
    
    if volume_data is None:
        return None
    
    # Handle different storage formats
    if isinstance(volume_data, bytes):
        try:
            # TomoTwin stores as float32 bytes
            volume = np.frombuffer(volume_data, dtype=np.float32)
            expected_size = np.prod(expected_shape)
            if len(volume) == expected_size:
                volume = volume.reshape(expected_shape)
                return volume
        except Exception as e:
            logger.warning(f"Could not decode bytes: {e}")
            return None
            
    elif isinstance(volume_data, np.ndarray):
        if volume_data.size == np.prod(expected_shape):
            return volume_data.reshape(expected_shape)
        else:
            logger.warning(f"Volume array has shape {volume_data.shape}, expected {expected_shape}")
            return None
    
    return None


def extract_pose_from_row(row: pd.Series) -> Optional[np.ndarray]:
    """
    Extract pose (orientation) data from a parquet row.
    
    Parameters
    ----------
    row : pd.Series
        Row from parquet dataframe
        
    Returns
    -------
    Optional[np.ndarray]
        Pose as axis-angle representation [angle, ax, ay, az] or None
    """
    if 'orientation_axis_angle' in row:
        try:
            aa_data = row['orientation_axis_angle']
            if aa_data is not None and not (isinstance(aa_data, float) and pd.isna(aa_data)):
                if isinstance(aa_data, bytes):
                    pose = np.frombuffer(aa_data, dtype=np.float64)
                    if len(pose) == 4:
                        return pose
                elif isinstance(aa_data, np.ndarray) and len(aa_data) == 4:
                    return aa_data
        except Exception:
            pass
    
    # Return identity pose if not found
    return np.array([0., 0., 0., 1.])


def load_parquet_samples(
    parquet_dir: str,
    structure_name: str,
    snr: float,
    num_samples: int,
    include_poses: bool = False,
    skip_metadata_files: bool = True
) -> Tuple[np.ndarray, List[Dict[str, Any]], Optional[np.ndarray]]:
    """
    Load samples from parquet files for a specific structure and SNR.
    
    Parameters
    ----------
    parquet_dir : str
        Base directory containing parquet files organized by structure/SNR
    structure_name : str
        PDB structure name (e.g., '1g3i')
    snr : float
        Signal-to-noise ratio
    num_samples : int
        Maximum number of samples to load
    include_poses : bool
        Whether to load pose/orientation data
    skip_metadata_files : bool
        Whether to skip files with 'metadata' in the name
        
    Returns
    -------
    Tuple containing:
        - volumes: np.ndarray of shape (N, 48, 48, 48)
        - metadata: List of metadata dictionaries
        - poses: Optional np.ndarray of shape (N, 4) if include_poses=True
    """
    base_path = Path(parquet_dir)
    structure_dir = base_path / structure_name / f"snr_{snr}"
    
    if not structure_dir.exists():
        # Try alternative directory structure
        structure_dir = base_path / f"{structure_name}_snr_{snr}"
        if not structure_dir.exists():
            raise ValueError(f"Directory not found: {base_path / structure_name / f'snr_{snr}'}")
    
    # Find all parquet files
    parquet_files = sorted(structure_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {structure_dir}")
    
    logger.info(f"Found {len(parquet_files)} parquet files in {structure_dir}")
    
    all_volumes = []
    all_metadata = []
    all_poses = [] if include_poses else None
    
    for parquet_file in parquet_files:
        if len(all_volumes) >= num_samples:
            break
        
        # Skip metadata files if requested
        if skip_metadata_files and 'metadata' in parquet_file.name.lower():
            continue
        
        logger.debug(f"Reading {parquet_file.name}...")
        
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            logger.warning(f"Could not read {parquet_file.name}: {e}")
            continue
        
        # Process each row
        for idx, row in df.iterrows():
            if len(all_volumes) >= num_samples:
                break
            
            # Extract volume
            volume = extract_volume_from_row(row)
            if volume is None:
                continue
            
            all_volumes.append(volume)
            
            # Extract pose if requested
            if include_poses:
                pose = extract_pose_from_row(row)
                all_poses.append(pose)
            
            # Extract metadata
            metadata = {
                'structure': structure_name,
                'snr': snr,
                'batch_file': parquet_file.name,
                'row_index': idx,
                'has_pose': 'orientation_axis_angle' in row
            }
            
            # Add additional metadata columns if present
            for col in ['mol_id', 'pdb_id', 'instance_id', 'batch_idx', 'shape']:
                if col in row:
                    metadata[col] = row[col]
            
            all_metadata.append(metadata)
    
    if not all_volumes:
        raise ValueError(f"No volumes could be loaded from parquet files")
    
    # Convert to numpy arrays
    volumes = np.stack(all_volumes[:num_samples])
    metadata = all_metadata[:num_samples]
    poses = np.stack(all_poses[:num_samples]) if include_poses else None
    
    logger.info(f"Loaded {len(volumes)} volumes from parquet files")
    if include_poses:
        logger.info(f"Poses available for {sum([m.get('has_pose', False) for m in metadata])} samples")
    
    return volumes, metadata, poses


def get_available_structures(parquet_dir: str) -> List[str]:
    """
    Get list of available structure names in the parquet directory.
    
    Parameters
    ----------
    parquet_dir : str
        Base directory containing parquet files
        
    Returns
    -------
    List[str]
        List of structure names found
    """
    base_path = Path(parquet_dir)
    structures = []
    
    # Look for directories that are structure names
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if it contains SNR subdirectories
            snr_dirs = list(item.glob("snr_*"))
            if snr_dirs:
                structures.append(item.name)
    
    return sorted(structures)


def get_available_snrs(parquet_dir: str, structure_name: str) -> List[float]:
    """
    Get list of available SNR values for a structure.
    
    Parameters
    ----------
    parquet_dir : str
        Base directory containing parquet files
    structure_name : str
        Structure name to query
        
    Returns
    -------
    List[float]
        List of available SNR values
    """
    base_path = Path(parquet_dir)
    structure_dir = base_path / structure_name
    
    if not structure_dir.exists():
        return []
    
    snr_values = []
    for snr_dir in structure_dir.glob("snr_*"):
        try:
            # Extract SNR value from directory name
            snr_str = snr_dir.name.replace("snr_", "")
            snr = float(snr_str)
            snr_values.append(snr)
        except ValueError:
            continue
    
    return sorted(snr_values)
