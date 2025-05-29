"""
Dataset implementation for TomotWin with structure-based random sampling.
"""

import os
import numpy as np
import torch
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from torch.utils.data import Dataset
import glob
import re

logger = logging.getLogger(__name__)

class StructureRandomParquetDataset(Dataset):
    """Dataset for loading particle data with one random SNR level per structure.
    
    This dataset randomly selects one parquet file per structure from a directory
    containing structure-organized parquet files (one structure per subdirectory).
    Each worker in a distributed setting will have a different random selection.
    
    Parameters
    ----------
    base_dir : str
        Path to the base directory containing structure-organized parquet files.
    name_to_pdb : dict, optional
        Mapping from molecule names to PDB IDs.
    box_size : int
        Size of the volume box.
    device : str or torch.device
        Device to load data to.
    augment : bool
        Whether to apply data augmentation.
    seed : int
        Random seed.
    rank : int, optional
        Process rank in distributed setup.
    world_size : int, optional
        Total number of processes in distributed setup.
    """
    
    def __init__(
        self,
        base_dir: str,
        name_to_pdb: Optional[Dict[str, str]] = None,
        box_size: int = 48,
        device: str = 'cpu',
        augment: bool = True,
        seed: int = 42,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        samples_per_epoch: int = 2000
    ):
        self.base_dir = Path(base_dir)
        self.name_to_pdb = name_to_pdb or {}
        self.box_size = box_size
        self.device = device
        self.augment = augment
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.samples_per_epoch = samples_per_epoch
        
        # Set different seed for each rank
        effective_seed = seed + (rank or 0)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)
        
        # Initialize dataset
        self._find_structure_parquets()
        self._initialize_molecule_mapping()
    
    def _find_structure_parquets(self):
        """Find parquet files for each structure and select one SNR level randomly per structure."""
        structure_dirs = [d for d in self.base_dir.glob("*") if d.is_dir()]
        
        if not structure_dirs:
            raise ValueError(f"No structure directories found in {self.base_dir}")
        
        self.datasets = []
        self.selected_parquets = []
        self.structure_names = []
        self.total_items = 0
        
        # Print summary of available structures
        if self.rank == 0 or self.rank is None:
            logger.info(f"Found {len(structure_dirs)} structure directories")
        
        for struct_dir in structure_dirs:
            structure_name = struct_dir.name
            
            # Find SNR subdirectories
            snr_dirs = [d for d in struct_dir.glob("snr_*") if d.is_dir()]
            
            if not snr_dirs:
                logger.warning(f"No SNR directories found for structure {structure_name}")
                continue
            
            # Sort SNR directories by SNR value
            def extract_snr_from_dir(dirname):
                match = re.search(r'snr[-_]?(\d+\.?\d*)', str(dirname.name).lower())
                if match:
                    return float(match.group(1))
                return float('inf')  # Default value if SNR not found
            
            snr_dirs.sort(key=extract_snr_from_dir)
            
            # Randomly select one SNR directory for this structure
            # Use a different random selection for each rank
            if self.rank is not None:
                # Deterministic selection based on rank
                rank_seed = self.seed + self.rank
                selection_idx = rank_seed % len(snr_dirs)
                selected_snr_dir = snr_dirs[selection_idx]
            else:
                # Random selection for single-process mode
                selected_snr_dir = np.random.choice(snr_dirs)
            
            # Get all batch parquet files from the selected SNR directory
            batch_files = list(selected_snr_dir.glob("batch_*.parquet"))
            
            if not batch_files:
                # Try to find any parquet files
                batch_files = list(selected_snr_dir.glob("*.parquet"))
                if not batch_files:
                    logger.warning(f"No batch files found in {selected_snr_dir}")
                    continue
            
            # Extract SNR value from directory name
            snr_value = 'unknown'
            snr_match = re.search(r'snr[-_]?(\d+\.?\d*)', selected_snr_dir.name.lower())
            if snr_match:
                snr_value = float(snr_match.group(1))
            
            # Randomly select just ONE batch file instead of loading all of them
            if self.rank is not None:
                # Deterministic selection based on rank
                rank_seed = (self.seed + self.rank) * 31  # Use a different seed than for SNR selection
                batch_idx = rank_seed % len(batch_files)
                selected_batch_file = batch_files[batch_idx]
            else:
                # Random selection for single-process mode
                selected_batch_file = np.random.choice(batch_files)
            
            # Skip metadata files
            if 'metadata' in selected_batch_file.name:
                # If we selected a metadata file, try to find a non-metadata file
                non_metadata_files = [f for f in batch_files if 'metadata' not in f.name]
                if non_metadata_files:
                    selected_batch_file = np.random.choice(non_metadata_files)
                else:
                    logger.warning(f"Only metadata files found in {selected_snr_dir}")
                    continue
            
            try:
                # Load only the selected batch file
                df = pd.read_parquet(selected_batch_file)
                
                if 'molecule_id' not in df.columns:
                    # Try to infer molecule ID from directory name
                    df['molecule_id'] = structure_name
                
                # Add SNR level if not present
                if 'snr_level' not in df.columns and 'snr' not in df.columns and snr_value != 'unknown':
                    df['snr_level'] = snr_value
                
                # Process subvolumes if in bytes format
                def process_subvolume(x):
                    if isinstance(x, bytes):
                        try:
                            shape = None
                            if 'shape' in df.columns:
                                shape_idx = df.columns.get_loc('shape')
                                shape = df.iloc[0, shape_idx]
                            
                            # If shape is a list or array, use it, otherwise use default box size
                            if isinstance(shape, (list, np.ndarray)) and len(shape) == 3:
                                return np.frombuffer(x, dtype=np.float32).reshape(shape)
                            else:
                                # Assume cubic box
                                return np.frombuffer(x, dtype=np.float32).reshape((self.box_size, self.box_size, self.box_size))
                        except:
                            return None
                    return x
                
                # Apply processing only if we have bytes data
                if 'subvolume' in df.columns and isinstance(df['subvolume'].iloc[0], bytes):
                    df['subvolume'] = df['subvolume'].apply(process_subvolume)
                    
                    # Filter out invalid rows
                    valid_mask = df['subvolume'].notna()
                    invalid_count = (~valid_mask).sum()
                    if invalid_count > 0:
                        logger.warning(f"Found {invalid_count} invalid subvolumes in {selected_snr_dir}")
                        df = df[valid_mask].reset_index(drop=True)
                
                # Get SNR level from the dataframe if available
                if 'snr_level' in df.columns:
                    snr_value = df['snr_level'].iloc[0]
                elif 'snr' in df.columns:
                    snr_value = df['snr'].iloc[0]
                
                if self.rank == 0 or self.rank is None:
                    logger.info(f"Structure {structure_name}: Selected SNR {snr_value}, batch file {selected_batch_file.name} with {len(df)} samples")
                
                self.datasets.append(df)
                self.selected_parquets.append(selected_batch_file)  # Store the specific batch file, not the SNR directory
                self.structure_names.append(structure_name)
                self.total_items += len(df)
                
            except Exception as e:
                logger.error(f"Error loading batch file {selected_batch_file}: {e}")
        
        if not self.datasets:
            raise ValueError("No valid parquet files could be loaded")
            
        if self.rank == 0 or self.rank is None:
            logger.info(f"Loaded {len(self.datasets)} datasets with {self.total_items} total samples")
            
        # Calculate weights proportional to dataset sizes
        dataset_sizes = [len(df) for df in self.datasets]
        self.weights = [size / sum(dataset_sizes) for size in dataset_sizes]
    
    def _initialize_molecule_mapping(self):
        """Create mapping between molecule IDs and indices using PDB IDs."""
        try:
            # Extract all unique molecule IDs from all datasets
            all_molecule_ids = set()
            for df in self.datasets:
                if 'molecule_id' in df.columns:
                    all_molecule_ids.update(df['molecule_id'].unique())
            
            # Filter out background, None, and NaN values
            non_background_ids = [
                mid for mid in all_molecule_ids 
                if mid not in ('background', None) and pd.notna(mid)
            ]
            
            if self.rank == 0 or self.rank is None:
                logger.info(f"Total unique molecules: {len(all_molecule_ids)}")
                logger.info(f"After filtering background/None: {len(non_background_ids)}")
            
            # Convert names to PDB IDs
            pdb_ids = []
            for name in non_background_ids:
                pdb_id = self.name_to_pdb.get(name)
                if pdb_id:
                    pdb_ids.append(pdb_id)
                else:
                    if self.rank == 0 or self.rank is None:
                        logger.warning(f"No PDB ID found for molecule {name}")
            
            if self.rank == 0 or self.rank is None:
                logger.info("Molecule mappings:")
                for i, (name, pdb) in enumerate(zip(non_background_ids, pdb_ids)):
                    if pdb:
                        logger.info(f"{i}: {name} -> {pdb}")
            
            # Create bidirectional mappings
            self.molecule_to_idx = {pdb: idx for idx, pdb in enumerate(sorted(pdb_ids)) if pdb}
            self.idx_to_molecule = {idx: pdb for pdb, idx in self.molecule_to_idx.items()}
            
            if self.rank == 0 or self.rank is None:
                logger.info(f"Lookup matrix will have shape: {len(self.molecule_to_idx)}x{len(self.molecule_to_idx)}")
        
        except Exception as e:
            logger.error(f"Error in _initialize_molecule_mapping: {str(e)}")
            self.molecule_to_idx = {}
            self.idx_to_molecule = {}
    
    def __len__(self):
        """Get dataset length."""
        if self.world_size:
            # Adjust samples per rank
            return self.samples_per_epoch // self.world_size
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        """Get random item based on dataset weights.
        
        Parameters
        ----------
        idx : int
            Index is ignored as we sample randomly.
            
        Returns
        -------
        tuple
            (volume, molecule_id)
        """
        try:
            # Choose dataset based on weights
            dataset_idx = np.random.choice(len(self.datasets), p=self.weights)
            df = self.datasets[dataset_idx]
            
            # Choose random item from the selected dataset
            item_idx = np.random.randint(len(df))
            item_data = df.iloc[item_idx]
            
            # Extract subvolume
            subvolume = item_data['subvolume']
            if isinstance(subvolume, bytes):
                shape = None
                if 'shape' in df.columns:
                    shape = item_data['shape']
                
                # If shape is a list or array, use it, otherwise use default box size
                if isinstance(shape, (list, np.ndarray)) and len(shape) == 3:
                    subvolume = np.frombuffer(subvolume, dtype=np.float32).reshape(shape)
                else:
                    # Assume cubic box
                    subvolume = np.frombuffer(subvolume, dtype=np.float32).reshape((self.box_size, self.box_size, self.box_size))
            
            # Apply augmentation if enabled
            if self.augment:
                subvolume = self._augment_volume(subvolume)
            
            # Add channel dimension and convert to tensor
            # Ensure array is contiguous before converting to tensor to avoid negative stride errors
            subvolume = np.array(subvolume, copy=True, dtype=np.float32)  # Ensure array is writable
            subvolume = np.expand_dims(subvolume, axis=0)
            subvolume = torch.from_numpy(subvolume).to(dtype=torch.float32)
            
            # Get molecule ID
            name = item_data['molecule_id']
            if name == 'background':
                molecule_idx = -1  # Use -1 to indicate background
            else:
                # Regular case - convert name to PDB ID then to index
                pdb_id = self.name_to_pdb.get(name)
                molecule_idx = self.molecule_to_idx.get(pdb_id, -1)
            
            return subvolume, torch.tensor(molecule_idx, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error in __getitem__: {e}")
            # Return empty tensor with correct shape in case of error
            default_shape = (1, self.box_size, self.box_size, self.box_size)
            return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long)
    
    def get_sample_by_index(self, idx_tuple):
        """
        Retrieve a sample by its tuple of (dataset_idx, sample_idx).
        Used primarily for visualization.
        
        Parameters
        ----------
        idx_tuple : tuple
            Tuple of (dataset_idx, sample_idx)
            
        Returns
        -------
        tuple
            Tuple of (volume tensor, molecule index tensor, source type)
        """
        try:
            dataset_idx, sample_idx = idx_tuple
            
            # Get the dataset
            if dataset_idx >= len(self.datasets):
                raise ValueError(f"Dataset index {dataset_idx} out of range (max: {len(self.datasets)-1})")
                
            df = self.datasets[dataset_idx]
            
            # Get data from the dataset's dataframe
            if sample_idx >= len(df):
                raise ValueError(f"Sample index {sample_idx} out of range (max: {len(df)-1})")
                
            item_data = df.iloc[sample_idx]
            
            # Extract subvolume
            subvolume = item_data['subvolume']
            if isinstance(subvolume, bytes):
                shape = None
                if 'shape' in df.columns:
                    shape = item_data['shape']
                
                # If shape is a list or array, use it, otherwise use default box size
                if isinstance(shape, (list, np.ndarray)) and len(shape) == 3:
                    subvolume = np.frombuffer(subvolume, dtype=np.float32).reshape(shape)
                else:
                    # Assume cubic box
                    subvolume = np.frombuffer(subvolume, dtype=np.float32).reshape((self.box_size, self.box_size, self.box_size))
            
            # No augmentation for visualization samples to ensure consistency
            # Add channel dimension and convert to tensor
            # Ensure array is contiguous before converting to tensor to avoid negative stride errors
            subvolume = np.array(subvolume, copy=True, dtype=np.float32)  # Ensure array is writable
            subvolume = np.expand_dims(subvolume, axis=0)
            subvolume = torch.from_numpy(subvolume).to(dtype=torch.float32)
            
            # Get molecule ID
            name = item_data['molecule_id']
            if name == 'background':
                molecule_idx = -1  # Use -1 to indicate background
            else:
                # Regular case - convert name to PDB ID then to index
                pdb_id = self.name_to_pdb.get(name)
                molecule_idx = self.molecule_to_idx.get(pdb_id, -1)
            
            # Use structure name as source type
            source_type = self.structure_names[dataset_idx] if dataset_idx < len(self.structure_names) else f"dataset_{dataset_idx}"
            
            return subvolume, torch.tensor(molecule_idx, dtype=torch.long), source_type
                
        except Exception as e:
            logger.error(f"Error loading item at index {idx_tuple}: {str(e)}")
            default_shape = (1, self.box_size, self.box_size, self.box_size)
            return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long), "unknown"
    
    def _augment_volume(self, volume):
        """Apply random augmentations to volume.
        
        Parameters
        ----------
        volume : np.ndarray
            Volume to augment.
            
        Returns
        -------
        np.ndarray
            Augmented volume with positive strides.
        """
        if not self.augment:
            return volume
        
        # Make a contiguous copy to avoid stride issues
        volume = np.ascontiguousarray(volume)
        
        # Random rotation
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            axes = tuple(np.random.choice([0, 1, 2], size=2, replace=False))
            volume = np.rot90(volume, k=k, axes=axes)
            # Make contiguous after rotation
            volume = np.ascontiguousarray(volume)
        
        # Random flips
        for axis in range(3):
            if np.random.random() > 0.5:
                volume = np.flip(volume, axis=axis)
                # Always make a copy after flip to ensure positive strides
                volume = np.ascontiguousarray(volume)
        
        return volume
        
    def get_visualization_samples(self, samples_per_mol_per_source=10):
        """
        Retrieve a fixed set of samples for visualization, organized by molecule ID and structure.
        The function memoizes results to ensure consistent visualization throughout training.
        
        Parameters
        ----------
        samples_per_mol_per_source : int
            Number of samples to collect per molecule per structure
            
        Returns
        -------
        dict
            Dict mapping (mol_id, source_type) tuples to lists of sample indices
        """
        # Check if we already have memoized samples
        if hasattr(self, '_visualization_samples') and self._visualization_samples is not None:
            return self._visualization_samples
        
        if self.rank == 0 or self.rank is None:
            logger.info(f"Generating memoized visualization samples ({samples_per_mol_per_source} per molecule per structure)")
        
        # Dictionary to hold visualization samples
        self._visualization_samples = {}
        
        # Collect samples from each dataset (each represents a structure)
        for dataset_idx, df in enumerate(self.datasets):
            # Skip empty datasets
            if len(df) == 0:
                continue
            
            # Extract unique molecule IDs (excluding background/None)
            unique_mols = df['molecule_id'].unique()
            valid_mols = [mol for mol in unique_mols if mol not in ('background', None) and pd.notna(mol)]
            
            # Use structure name or dataset index as source type
            source_type = self.structure_names[dataset_idx] if dataset_idx < len(self.structure_names) else f"dataset_{dataset_idx}"
            
            # Process each molecule
            for mol in valid_mols:
                # Get PDB ID for the molecule
                pdb_id = self.name_to_pdb.get(mol)
                if not pdb_id or pdb_id not in self.molecule_to_idx:
                    continue
                
                mol_idx = self.molecule_to_idx[pdb_id]
                
                # Find all samples matching this molecule
                mask = (df['molecule_id'] == mol)
                matching_indices = np.where(mask)[0]
                
                if len(matching_indices) == 0:
                    continue
                
                # Sample with replacement if needed
                num_samples = min(samples_per_mol_per_source, len(matching_indices))
                if num_samples < samples_per_mol_per_source:
                    sampled_indices = np.random.choice(
                        matching_indices, 
                        size=samples_per_mol_per_source,
                        replace=True
                    )
                else:
                    sampled_indices = np.random.choice(
                        matching_indices,
                        size=samples_per_mol_per_source,
                        replace=False
                    )
                
                # Store using a tuple key of (mol_idx, source_type)
                key = (mol_idx, source_type)
                
                # Store dataset index and sample index for each sampled item
                for idx in sampled_indices:
                    if key not in self._visualization_samples:
                        self._visualization_samples[key] = []
                    self._visualization_samples[key].append((dataset_idx, idx))
        
        if self.rank == 0 or self.rank is None:
            # Log statistics about the visualization samples
            total_samples = sum(len(indices) for indices in self._visualization_samples.values())
            num_mols = len({key[0] for key in self._visualization_samples.keys()})
            num_sources = len({key[1] for key in self._visualization_samples.keys()})
            
            logger.info(f"Memoized {total_samples} visualization samples across {num_mols} molecules and {num_sources} structures")
        
        return self._visualization_samples
