"""
Dataset implementation for randomly sampling one parquet file per structure.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import glob
from pathlib import Path
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
import random

from cryolens.data.datasets import CachedParquetDataset

logger = logging.getLogger(__name__)

class RandomSampledStructureParquetDataset(Dataset):
    """Dataset that randomly selects one parquet file per structure for each worker.
    
    This dataset is designed to work with a directory structure where each structure
    has multiple parquet files with different SNR levels.
    
    Parameters
    ----------
    parquet_dir : str
        Directory containing parquet files organized by structure.
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
    samples_per_epoch : int
        Number of samples per epoch.
    """
    
    def __init__(
        self,
        parquet_dir,
        name_to_pdb=None,
        box_size=48,
        device='cpu',
        augment=True,
        seed=42,
        rank=None,
        world_size=None,
        samples_per_epoch=1000
    ):
        self.parquet_dir = Path(parquet_dir)
        self.name_to_pdb = name_to_pdb or {}
        self.box_size = box_size
        self.device = device
        self.augment = augment
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.samples_per_epoch = samples_per_epoch
        
        # Set random seed with distributed awareness
        self._set_random_seed()
        
        # Discover structure directories and files
        self._discover_structures()
        
        # Randomly select one parquet file per structure
        self._select_parquet_files()
        
        # Load the selected parquet files into datasets
        self._load_datasets()
    
    def _set_random_seed(self):
        """Set random seed with distributed awareness."""
        if self.seed is not None:
            # Use different seed for each process
            effective_seed = self.seed
            if self.rank is not None:
                effective_seed += self.rank * 100
            
            np.random.seed(effective_seed)
            random.seed(effective_seed)
            torch.manual_seed(effective_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(effective_seed)
                
            logger.info(f"Set random seed to {effective_seed} for rank {self.rank}")
    
    def _discover_structures(self):
        """Discover structure directories and their parquet files."""
        self.structure_files = {}
        
        # Check if parquet_dir exists
        if not self.parquet_dir.exists():
            logger.error(f"Parquet directory does not exist: {self.parquet_dir}")
            return
        
        # Find all parquet files in the directory
        parquet_files = glob.glob(str(self.parquet_dir / "**/*.parquet"), recursive=True)
        
        if len(parquet_files) == 0:
            logger.error(f"No parquet files found in directory: {self.parquet_dir}")
            return
        
        # Group parquet files by structure name
        # Assuming file naming convention includes structure name
        for file_path in parquet_files:
            # Extract the structure name from the file path
            file_name = os.path.basename(file_path)
            
            # Get structure name from file name using regex pattern
            # The pattern needs to extract the structure name from the file name
            match = re.search(r'(.*?)_', file_name)
            if match:
                structure_name = match.group(1)
                if structure_name not in self.structure_files:
                    self.structure_files[structure_name] = []
                self.structure_files[structure_name].append(file_path)
            else:
                # If we can't extract structure name from the file name, use the directory name instead
                dir_name = os.path.basename(os.path.dirname(file_path))
                if dir_name not in self.structure_files:
                    self.structure_files[dir_name] = []
                self.structure_files[dir_name].append(file_path)
        
        # Log the found structures
        if self.rank == 0 or self.rank is None:
            logger.info(f"Found {len(self.structure_files)} structures with parquet files")
            for structure, files in self.structure_files.items():
                logger.info(f"Structure {structure}: {len(files)} parquet files")
    
    def _select_parquet_files(self):
        """Randomly select one parquet file per structure for this worker."""
        self.selected_files = {}
        
        for structure, files in self.structure_files.items():
            if len(files) == 0:
                continue
                
            # Randomly select one file
            selected_file = random.choice(files)
            self.selected_files[structure] = selected_file
        
        if self.rank == 0 or self.rank is None:
            logger.info(f"Selected {len(self.selected_files)} parquet files, one per structure")
            for structure, file_path in self.selected_files.items():
                logger.info(f"Structure {structure}: {os.path.basename(file_path)}")
    
    def _load_datasets(self):
        """Load the selected parquet files into CachedParquetDataset instances."""
        self.datasets = []
        self.structure_names = []
        
        for structure, file_path in self.selected_files.items():
            dataset = CachedParquetDataset(
                parquet_path=file_path,
                name_to_pdb=self.name_to_pdb,
                box_size=self.box_size,
                device=self.device,
                augment=self.augment,
                seed=self.seed,
                rank=self.rank,
                world_size=self.world_size
            )
            
            # Only add datasets that have valid data
            if len(dataset) > 0:
                self.datasets.append(dataset)
                self.structure_names.append(structure)
        
        # Calculate total number of items
        self.total_items = sum(len(dataset) for dataset in self.datasets)
        
        if self.rank == 0 or self.rank is None:
            logger.info(f"Loaded {len(self.datasets)} datasets with a total of {self.total_items} items")
            
        # Handle the case where no datasets were loaded
        if len(self.datasets) == 0:
            logger.error("No valid datasets were loaded!")
            # Create a dummy dataset to prevent errors
            self.total_items = 0
    
    def __len__(self):
        """Get dataset length."""
        if self.world_size:
            # Adjust samples per rank
            return self.samples_per_epoch // self.world_size
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        """Get dataset item with random sampling from available datasets.
        
        Parameters
        ----------
        idx : int
            Index of item to get.
            
        Returns
        -------
        tuple
            (volume, molecule_id)
        """
        if len(self.datasets) == 0:
            # Return dummy data if no datasets are available
            default_shape = (1, self.box_size, self.box_size, self.box_size)
            return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long)
        
        try:
            # Select a random dataset
            dataset_idx = random.randrange(len(self.datasets))
            dataset = self.datasets[dataset_idx]
            
            # Select a random item from the dataset
            item_idx = random.randrange(len(dataset))
            
            return dataset[item_idx]
            
        except Exception as e:
            logger.error(f"Error in RandomSampledStructureParquetDataset.__getitem__: {e}")
            # Return dummy data in case of any error
            default_shape = (1, self.box_size, self.box_size, self.box_size)
            return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long)
