"""
Enhanced dataset implementations for TomoTwin data with better batch file utilization.

This module provides enhanced dataset classes that can better utilize all available
batch files for each structure, providing more diverse training data.
"""

import os
import torch
import logging
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from cryolens.data.datasets import CachedParquetDataset
from cryolens.data.tomotwin_loader import TomoTwinParquetDataset, StructureDataWrapper

# Try to import background generator if available
try:
    from cryolens.data.background_generator import BackgroundGenerator, OnlineBackgroundGenerator
    BACKGROUND_GENERATOR_AVAILABLE = True
except ImportError:
    BACKGROUND_GENERATOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class MultiBatchStructureDataset(Dataset):
    """Dataset that can utilize multiple batch files per structure for better data coverage.
    
    This dataset loads ALL available batch files for a structure and can cycle through
    them during training, providing much better utilization of available data.
    
    Parameters
    ----------
    structure_dir : Path
        Directory containing structure data (PDB ID directory)
    structure_name : str
        Name of the structure (PDB ID)
    name_to_pdb : dict
        Mapping from molecule names to PDB IDs
    box_size : int
        Size of the volume box
    snr_values : list of float or None
        List of SNR values to include
    device : str or torch.device
        Device to load data to
    augment : bool
        Whether to apply data augmentation
    seed : int
        Random seed
    rank : int, optional
        Process rank for distributed training
    world_size : int, optional
        World size for distributed training
    max_batches_per_structure : int, optional
        Maximum number of batch files to load per structure (None = load all)
    cycle_batches : bool
        Whether to cycle through all batches or stick to one per epoch
    """
    
    def __init__(
        self,
        structure_dir,
        structure_name,
        name_to_pdb=None,
        box_size=48,
        snr_values=None,
        device='cpu',
        augment=True,
        seed=171717,
        rank=None,
        world_size=None,
        max_batches_per_structure=None,
        cycle_batches=True
    ):
        self.structure_dir = structure_dir
        self.structure_name = structure_name
        self.name_to_pdb = name_to_pdb or {}
        self.box_size = box_size
        self.snr_values = snr_values
        self.device = device
        self.augment = augment
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.max_batches_per_structure = max_batches_per_structure
        self.cycle_batches = cycle_batches
        
        # Set random seed
        if self.seed is not None:
            effective_seed = self.seed
            if self.rank is not None:
                effective_seed += self.rank * 100
            
            random.seed(effective_seed)
            np.random.seed(effective_seed)
            
        self.batch_datasets = self._load_all_batches()
        self.current_batch_idx = 0
        
    def _load_all_batches(self):
        """Load multiple batch files for the given structure."""
        # Find all SNR directories for this structure
        snr_dirs = []
        for snr_dir in self.structure_dir.iterdir():
            if not snr_dir.is_dir() or not snr_dir.name.startswith('snr_'):
                continue
                
            # Extract SNR value
            try:
                snr_value = float(snr_dir.name.split('_')[1])
            except (IndexError, ValueError):
                continue
            
            # Filter by SNR values if specified
            if self.snr_values is not None and snr_value not in self.snr_values:
                continue
                
            snr_dirs.append(snr_dir)
        
        if not snr_dirs:
            logger.warning(f"No valid SNR directories found for structure {self.structure_name}")
            return []
        
        # Collect all batch files from all SNR directories
        all_batch_files = []
        for snr_dir in snr_dirs:
            batch_files = list(snr_dir.glob('batch_*.parquet'))
            all_batch_files.extend([(snr_dir, bf) for bf in batch_files])
        
        if not all_batch_files:
            logger.warning(f"No batch files found for structure {self.structure_name}")
            return []
        
        # Shuffle to get random order
        random.shuffle(all_batch_files)
        
        # Limit number of batches if specified
        if self.max_batches_per_structure is not None:
            all_batch_files = all_batch_files[:self.max_batches_per_structure]
        
        logger.debug(f"Found {len(all_batch_files)} batch files for structure {self.structure_name}")
        
        # Load all batch files as datasets
        batch_datasets = []
        for snr_dir, batch_file in all_batch_files:
            try:
                dataset = TomoTwinParquetDataset(
                    parquet_path=batch_file,
                    name_to_pdb=self.name_to_pdb,
                    box_size=self.box_size,
                    device=self.device,
                    augment=self.augment,
                    seed=self.seed,
                    rank=self.rank,
                    world_size=self.world_size,
                    structure_name=self.structure_name
                )
                
                if len(dataset) > 0:
                    batch_datasets.append({
                        'dataset': dataset,
                        'snr_dir': snr_dir.name,
                        'batch_file': batch_file.name,
                        'size': len(dataset)
                    })
                    logger.debug(f"Loaded batch {batch_file.name} with {len(dataset)} samples for {self.structure_name}")
                else:
                    logger.warning(f"Empty batch file {batch_file} for structure {self.structure_name}")
                    
            except Exception as e:
                logger.error(f"Error loading batch {batch_file} for structure {self.structure_name}: {str(e)}")
                continue
        
        if batch_datasets:
            total_samples = sum(bd['size'] for bd in batch_datasets)
            logger.info(f"Successfully loaded {len(batch_datasets)} batches with {total_samples} total samples for structure {self.structure_name}")
        
        return batch_datasets
    
    def __len__(self):
        """Get total dataset length across all batches."""
        if not self.batch_datasets:
            return 0
        return sum(bd['size'] for bd in self.batch_datasets)
    
    def __getitem__(self, idx):
        """Get dataset item, potentially from different batch files."""
        if not self.batch_datasets:
            # Return dummy data
            default_shape = (1, self.box_size, self.box_size, self.box_size)
            return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long)
        
        if self.cycle_batches:
            # Select batch in round-robin fashion
            batch_info = self.batch_datasets[self.current_batch_idx % len(self.batch_datasets)]
            self.current_batch_idx += 1
        else:
            # Randomly select batch for each sample
            batch_info = random.choice(self.batch_datasets)
        
        # Get random sample from selected batch
        dataset = batch_info['dataset']
        sample_idx = random.randrange(len(dataset))
        
        return dataset[sample_idx]
    
    def get_batch_info(self):
        """Get information about loaded batches."""
        if not self.batch_datasets:
            return "No batches loaded"
        
        info = f"Structure {self.structure_name}: {len(self.batch_datasets)} batches, {len(self)} total samples\n"
        for i, bd in enumerate(self.batch_datasets):
            info += f"  Batch {i+1}: {bd['snr_dir']}/{bd['batch_file']} ({bd['size']} samples)\n"
        
        return info.strip()


class EnhancedTomoTwinDataset(Dataset):
    """Enhanced TomoTwin dataset with better batch file utilization.
    
    This dataset can load multiple batch files per structure and provides
    better coverage of available training data.
    
    Parameters are the same as TomoTwinDataset, with additional options:
    
    max_batches_per_structure : int, optional
        Maximum number of batch files to load per structure (None = load all)
    cycle_batches : bool
        Whether to cycle through all batches or randomly sample
    batch_rotation_epochs : int
        How often to rotate to different batch files (0 = no rotation)
    """
    
    def __init__(
        self,
        base_dir,
        name_to_pdb=None,
        box_size=48,
        snr_values=None,
        device='cpu',
        augment=True,
        seed=171717,
        rank=None,
        world_size=None,
        samples_per_epoch=2000,
        normalization="z-score",
        max_structures=None,
        filtered_structure_ids=None,
        external_molecule_order=None,
        enable_background=False,
        background_ratio=0.2,
        background_params=None,
        max_batches_per_structure=None,
        cycle_batches=True,
        batch_rotation_epochs=0
    ):
        self.base_dir = Path(base_dir)
        self.name_to_pdb = name_to_pdb or {}
        self.box_size = box_size
        self.snr_values = snr_values
        self.device = device
        self.augment = augment
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.samples_per_epoch = samples_per_epoch
        self.normalization = normalization
        self.max_structures = max_structures
        self.filtered_structure_ids = filtered_structure_ids
        self.external_molecule_order = external_molecule_order
        self.enable_background = enable_background
        self.background_ratio = background_ratio
        self.background_params = background_params or {}
        self.max_batches_per_structure = max_batches_per_structure
        self.cycle_batches = cycle_batches
        self.batch_rotation_epochs = batch_rotation_epochs
        
        # Initialize background generator if enabled
        self.background_generator = None
        if self.enable_background and BACKGROUND_GENERATOR_AVAILABLE:
            self.background_generator = OnlineBackgroundGenerator(
                generator_params=self.background_params,
                cache_size=100
            )
            if rank == 0 or rank is None:
                logger.info("Background generator initialized with params: {}".format(self.background_params))
        elif self.enable_background and not BACKGROUND_GENERATOR_AVAILABLE:
            logger.warning("Background generation requested but generator not available")
            self.enable_background = False
        
        # Set random seed with distributed awareness
        self._set_random_seed()
        
        # Load all structure datasets
        self._load_all_structures()
        
    def _set_random_seed(self):
        """Set random seed with distributed awareness."""
        if self.seed is not None:
            effective_seed = self.seed
            if self.rank is not None:
                effective_seed += self.rank * 100
            
            np.random.seed(effective_seed)
            random.seed(effective_seed)
            torch.manual_seed(effective_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(effective_seed)