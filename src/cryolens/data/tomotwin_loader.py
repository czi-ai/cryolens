"""
Dataset implementations for TomoTwin data.

This module provides specialized dataset classes for loading and processing
TomoTwin data with their specific directory structure.
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

logger = logging.getLogger(__name__)

class TomoTwinParquetDataset(CachedParquetDataset):
    """Dataset for loading TomoTwin parquet files with their specific column structure.
    
    This class extends CachedParquetDataset to handle the TomoTwin-specific file format
    which uses 'structure' instead of 'molecule_id' columns.
    
    Parameters
    ----------
    parquet_path : str
        Path to parquet file containing data
    name_to_pdb : dict, optional
        Mapping from molecule names to PDB IDs
    box_size : int
        Size of the volume box
    device : str or torch.device
        Device to load data to
    augment : bool
        Whether to apply data augmentation
    seed : int
        Random seed
    rank : int, optional
        Process rank in distributed setup
    world_size : int, optional
        Total number of processes in distributed setup
    structure_name : str, optional
        Name of the structure for this dataset (for logging)
    """
    
    def __init__(
        self,
        parquet_path,
        name_to_pdb=None,
        box_size=48,
        device='cpu',
        augment=True,
        seed=171717,
        rank=None,
        world_size=None,
        structure_name=None
    ):
        self.structure_name = structure_name
        super().__init__(
            parquet_path=parquet_path,
            name_to_pdb=name_to_pdb,
            box_size=box_size,
            device=device,
            augment=augment,
            seed=seed,
            rank=rank,
            world_size=world_size
        )
    
    def _load_data(self):
        """Load data from TomoTwin parquet file with appropriate column mapping."""
        rank_str = f"Rank {self.rank if self.rank is not None else 'None'}"
        struct_str = f" ({self.structure_name})" if self.structure_name else ""
        print(f"{rank_str}: DEADLOCK_DEBUG - Starting _load_data for {self.parquet_path}{struct_str}")
        
        try:
            # Load dataframe from parquet
            print(f"{rank_str}: DEADLOCK_DEBUG - Reading parquet file")
            self.df = pd.read_parquet(self.parquet_path)
            print(f"{rank_str}: DEADLOCK_DEBUG - Parquet file read successfully with {len(self.df)} rows")
            
            if self.rank == 0 or self.rank is None:
                logging.info(f"Loaded {len(self.df)} samples from {self.parquet_path}{struct_str}")
            
            # First check if this is a TomoTwin-formatted parquet file with 'structure' column
            if 'structure' in self.df.columns:
                print(f"{rank_str}: DEADLOCK_DEBUG - TomoTwin format detected with 'structure' column")
                # Create a molecule_id column from the structure column
                self.df['molecule_id'] = self.df['structure']
                print(f"{rank_str}: DEADLOCK_DEBUG - Created 'molecule_id' column from 'structure' column")
            
            # Now validate essential columns - must include 'subvolume', 'shape', and now 'molecule_id'
            missing_cols = set(['subvolume', 'shape', 'molecule_id']) - set(self.df.columns)
            if missing_cols:
                print(f"{rank_str}: DEADLOCK_DEBUG - Missing columns: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            print(f"{rank_str}: DEADLOCK_DEBUG - Processing subvolumes")
            # Process subvolumes if in bytes format
            def process_subvolume(x):
                if isinstance(x, bytes):
                    try:
                        return np.frombuffer(x, dtype=np.float32).copy()
                    except:
                        return None
                return x
                
            self.df['subvolume'] = self.df['subvolume'].apply(process_subvolume)
            print(f"{rank_str}: DEADLOCK_DEBUG - Subvolumes processed")
            
            # Filter out invalid rows
            valid_mask = self.df['subvolume'].notna()
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                print(f"{rank_str}: DEADLOCK_DEBUG - Filtering {invalid_count} invalid subvolumes")
                logging.warning(f"Found {invalid_count} invalid subvolumes")
                self.df = self.df[valid_mask].reset_index(drop=True)
                
            if len(self.df) == 0:
                print(f"{rank_str}: DEADLOCK_DEBUG - No valid data found after filtering")
                raise ValueError("No valid data found after filtering")
                
            print(f"{rank_str}: DEADLOCK_DEBUG - _load_data completed successfully with {len(self.df)} samples")
                
        except Exception as e:
            print(f"{rank_str}: DEADLOCK_DEBUG - Error in _load_data: {str(e)}")
            import traceback
            traceback.print_exc()
            
            logging.error(f"Error loading data from {self.parquet_path}: {e}")
            self.df = pd.DataFrame()
            # Create empty dataframe with required columns to prevent further errors
            self.df = pd.DataFrame(columns=['subvolume', 'shape', 'molecule_id'])
            print(f"{rank_str}: DEADLOCK_DEBUG - Created empty DataFrame as fallback")


class StructureDataWrapper(Dataset):
    """Wrapper for a structure dataset that provides a unified molecule_idx mapping.
    
    This wrapper uses a global molecule_to_idx mapping instead of the individual 
    dataset's mapping.
    
    Parameters
    ----------
    dataset : Dataset
        The original dataset to wrap
    structure_name : str
        Name of the structure (PDB ID)
    molecule_to_idx : dict
        Global mapping from molecule names to indices
    external_molecule_order : list, optional
        External ordering of molecules to use for consistent indexing
    """
    
    def __init__(self, dataset, structure_name, molecule_to_idx, external_molecule_order=None):
        self.dataset = dataset
        self.structure_name = structure_name
        self.molecule_to_idx = molecule_to_idx
        self.external_molecule_order = external_molecule_order
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get item from dataset with unified molecule index."""
        # Get the original item
        volume, _ = self.dataset[idx]
        
        # If external molecule order is provided, use it for indexing
        if self.external_molecule_order is not None:
            try:
                molecule_idx = self.external_molecule_order.index(self.structure_name)
            except ValueError:
                # Structure not in external order, use -1 (background)
                molecule_idx = -1
        else:
            # Use the local molecule_to_idx mapping for this structure
            molecule_idx = self.molecule_to_idx.get(self.structure_name, -1)
        
        return volume, torch.tensor(molecule_idx, dtype=torch.long)


class SingleStructureTomoTwinDataset(Dataset):
    """Dataset for a single structure's batch file from TomoTwin data.
    
    This dataset handles loading exactly one batch file for a given structure.
    
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
        world_size=None
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
        
        # Set random seed
        if self.seed is not None:
            effective_seed = self.seed
            if self.rank is not None:
                effective_seed += self.rank * 100
            
            random.seed(effective_seed)
            np.random.seed(effective_seed)
            
        self.dataset = self._load_single_batch()
        
    def _load_single_batch(self):
        """Load exactly one batch file for the given structure."""
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
            return None
            
        # Randomly select one SNR directory
        selected_snr = random.choice(snr_dirs)
        logger.debug(f"Selected SNR directory {selected_snr.name} for structure {self.structure_name}")
        
        # Find all batch files in this SNR directory
        batch_files = list(selected_snr.glob('batch_*.parquet'))
        if not batch_files:
            logger.warning(f"No batch files found in {selected_snr} for structure {self.structure_name}")
            return None
            
        # Randomly select one batch file
        selected_batch = random.choice(batch_files)
        logger.debug(f"Selected batch file {selected_batch.name} for structure {self.structure_name}")
        
        # Load the batch file as a dataset
        try:
            dataset = TomoTwinParquetDataset(
                parquet_path=selected_batch,
                name_to_pdb=self.name_to_pdb,
                box_size=self.box_size,
                device=self.device,
                augment=self.augment,
                seed=self.seed,
                rank=self.rank,
                world_size=self.world_size,
                structure_name=self.structure_name
            )
            
            if len(dataset) == 0:
                logger.warning(f"Empty dataset loaded from {selected_batch} for structure {self.structure_name}")
                return None
                
            logger.debug(f"Successfully loaded {len(dataset)} samples for structure {self.structure_name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset from {selected_batch} for structure {self.structure_name}: {str(e)}")
            return None
    
    def __len__(self):
        """Get dataset length."""
        return len(self.dataset) if self.dataset is not None else 0
    
    def __getitem__(self, idx):
        """Get dataset item."""
        if self.dataset is None:
            # Return dummy data
            default_shape = (1, self.box_size, self.box_size, self.box_size)
            return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long)
            
        return self.dataset[idx]


class TomoTwinDataset(Dataset):
    """Dataset for TomoTwin data with specific directory structure.
    
    This dataset is designed to work with directories organized as:
    base_dir/pdb_id/snr_value/batch_XXXX.parquet
    
    For each structure (PDB ID), it loads exactly one batch file from a randomly 
    selected SNR directory.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing TomoTwin data
    name_to_pdb : dict
        Mapping from molecule names to PDB IDs
    box_size : int
        Size of the volume box (default: 48)
    snr_values : list of float or None
        List of SNR values to include (default: None, includes all)
    device : str or torch.device
        Device to load data to (default: 'cpu')
    augment : bool
        Whether to apply data augmentation (default: True)
    seed : int
        Random seed (default: 171717)
    rank : int or None
        Process rank for distributed training
    world_size : int or None
        World size for distributed training
    samples_per_epoch : int
        Number of samples per epoch (default: 2000)
    normalization : str
        Type of normalization to apply (default: "z-score")
    max_structures : int or None
        Maximum number of structures to load per node (default: None, loads all available)
    filtered_structure_ids : list or None
        List of specific structure IDs to load (default: None, loads all available)
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
        filtered_structure_ids=None
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
        
        # Set random seed with distributed awareness
        self._set_random_seed()
        
        # Load all structure datasets
        self._load_all_structures()
        
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
    
    def _load_all_structures(self):
        """Load one random batch per structure from TomoTwin data."""
        # Get a list of all available structure directories
        structure_dirs = {}
        if not self.base_dir.exists():
            if self.rank == 0 or self.rank is None:
                logger.error(f"Base directory does not exist: {self.base_dir}")
            self.dataset = None
            self.total_items = 0
            return
        
        # If filtered_structure_ids is provided, only use those
        if self.filtered_structure_ids is not None:
            rank_str = f"Rank {self.rank if self.rank is not None else 'None'}"
            print(f"\n{'='*60}")
            print(f"{rank_str}: APPLYING PDB CODE FILTERING")
            print(f"{rank_str}: Filtered structure IDs: {self.filtered_structure_ids}")
            print(f"{'='*60}\n")
            
            for pdb_id in self.filtered_structure_ids:
                pdb_dir = self.base_dir / pdb_id
                if pdb_dir.exists() and pdb_dir.is_dir():
                    structure_dirs[pdb_id] = pdb_dir
                    print(f"{rank_str}: Found directory for PDB {pdb_id}: {pdb_dir}")
                else:
                    logger.warning(f"Structure directory {pdb_dir} not found for PDB ID {pdb_id}")
                    print(f"{rank_str}: WARNING - Missing directory for PDB {pdb_id}: {pdb_dir}")
                    
            print(f"\n{rank_str}: Final structure_dirs after filtering: {list(structure_dirs.keys())}\n")
        else:
            # Original logic: discover all structure directories
            for pdb_dir in self.base_dir.iterdir():
                if not pdb_dir.is_dir():
                    # Skip files like overall_metadata.parquet
                    continue
                    
                # Make sure the directory is a valid PDB directory (basic check)
                pdb_id = pdb_dir.name
                if len(pdb_id) != 4 or not pdb_id[0].isdigit():
                    continue
                    
                structure_dirs[pdb_id] = pdb_dir
        
        if not structure_dirs:
            if self.rank == 0 or self.rank is None:
                logger.error(f"No structure directories found in {self.base_dir}")
            self.dataset = None
            self.total_items = 0
            return
        
        # Limit the number of structures if max_structures is specified
        # This is particularly useful for large datasets and distributed training
        # BUT: If filtered_structure_ids was provided, don't apply max_structures again
        # since the filtering was already done explicitly
        if (self.max_structures is not None and self.max_structures > 0 and 
            self.max_structures < len(structure_dirs) and 
            self.filtered_structure_ids is None):  # Only apply if no explicit filtering was done
            # Get all structure IDs
            all_structure_ids = list(structure_dirs.keys())
            
            # Create a different subset for each rank in distributed training
            # This ensures diversity across nodes while limiting per-node memory usage
            rank_for_seed = 0 if self.rank is None else self.rank
            rank_seed = self.seed + 1000 * rank_for_seed
            random.seed(rank_seed)  # Use different seed per rank
            
            # Randomly select max_structures structures
            selected_ids = random.sample(all_structure_ids, self.max_structures)
            
            # Create a new structure_dirs with only the selected structures
            limited_dirs = {pdb_id: structure_dirs[pdb_id] for pdb_id in selected_ids}
            structure_dirs = limited_dirs
            
            # Log the limitation with details about the selection
            log_prefix = f"Rank {self.rank if self.rank is not None else 'None'}"
            logger.info(f"{log_prefix}: Limited structures from {len(all_structure_ids)} to {len(structure_dirs)} using max_structures={self.max_structures}")
            logger.info(f"{log_prefix}: Using seed {rank_seed} for structure selection")
            logger.info(f"{log_prefix}: Selected structures: {sorted(selected_ids)}")
            
            # Restore the random seed for other operations
            self._set_random_seed()
        elif self.filtered_structure_ids is not None:
            # If explicit filtering was provided, don't apply max_structures again
            log_prefix = f"Rank {self.rank if self.rank is not None else 'None'}"
            logger.info(f"{log_prefix}: Skipping max_structures limitation because explicit filtered_structure_ids was provided")
            logger.info(f"{log_prefix}: Using exactly {len(structure_dirs)} explicitly filtered structures")
            
        if self.rank == 0 or self.rank is None:
            logger.info(f"Using {len(structure_dirs)} structure directories")
            
        # Load one dataset per structure
        structure_datasets = []
        valid_structure_names = []
        
        rank_str = f"Rank {self.rank if self.rank is not None else 'None'}"
        print(f"{rank_str}: Loading one batch file per structure from {len(structure_dirs)} structures")
        
        # Log all structure directories we're going to attempt to load
        logger.info(f"{rank_str}: Attempting to load {len(structure_dirs)} structures: {sorted(structure_dirs.keys())}")
        
        for struct_name, struct_dir in structure_dirs.items():
            logger.info(f"{rank_str}: Loading structure {struct_name} from {struct_dir}")
            
            dataset = SingleStructureTomoTwinDataset(
                structure_dir=struct_dir,
                structure_name=struct_name,
                name_to_pdb=self.name_to_pdb,
                box_size=self.box_size,
                snr_values=self.snr_values,
                device=self.device,
                augment=self.augment,
                seed=self.seed,
                rank=self.rank,
                world_size=self.world_size
            )
            
            if dataset is not None and len(dataset) > 0:
                logger.info(f"{rank_str}: Successfully loaded structure {struct_name} with {len(dataset)} samples")
                valid_structure_names.append(struct_name)
                structure_datasets.append(dataset)
            else:
                logger.warning(f"{rank_str}: Failed to load structure {struct_name} - no valid data found")
        
        # Now create a global unified mapping from structure names to indices
        # This is used for the similarity matrix
        self.structure_names = valid_structure_names
        self.molecule_to_idx = {name: idx for idx, name in enumerate(sorted(valid_structure_names))}
        
        # Log all valid structure names being used on this worker
        log_prefix = f"Rank {self.rank if self.rank is not None else 'None'}"
        logger.info(f"{log_prefix}: Using the following {len(valid_structure_names)} structures:")
        logger.info(f"{log_prefix}: {sorted(valid_structure_names)}")
        
        # Wrap each dataset to use the unified mapping
        wrapped_datasets = []
        for i, dataset in enumerate(structure_datasets):
            struct_name = valid_structure_names[i]
            wrapped = StructureDataWrapper(dataset, struct_name, self.molecule_to_idx)
            wrapped_datasets.append(wrapped)
            
        # Combine datasets using ConcatDataset or create an empty dataset
        if wrapped_datasets:
            self.dataset = ConcatDataset(wrapped_datasets)
            self.total_items = len(self.dataset)
            
            # Log dataset creation with structure counts
            if self.rank == 0 or self.rank is None:
                logger.info(f"Successfully loaded {len(wrapped_datasets)} structure datasets with {self.total_items} total samples")
                logger.info(f"Average samples per structure: {self.total_items / len(wrapped_datasets):.1f}")
                logger.info(f"Global lookup matrix size: {len(self.molecule_to_idx)}x{len(self.molecule_to_idx)}")
                # Log a few example mappings
                sample_count = min(5, len(self.molecule_to_idx))
                for i, (name, idx) in enumerate(list(self.molecule_to_idx.items())[:sample_count]):
                    logger.info(f"  Structure mapping {i}: {name} -> index {idx}")
                if len(self.molecule_to_idx) > sample_count:
                    logger.info(f"  ... and {len(self.molecule_to_idx) - sample_count} more structures")
        else:
            self.dataset = None
            self.total_items = 0
            self.molecule_to_idx = {}
            
            if self.rank == 0 or self.rank is None:
                logger.error("No valid structure datasets could be loaded")
                
    def get_molecular_stats(self):
        """Get statistics about the loaded structures."""
        if not hasattr(self, 'structure_names') or not self.structure_names:
            return "No structures loaded"
            
        matrix_size = len(self.molecule_to_idx) if hasattr(self, 'molecule_to_idx') else 0
        return f"Loaded {len(self.structure_names)} structures with a total of {self.total_items} samples. Lookup matrix size: {matrix_size}x{matrix_size}"
    
    def __len__(self):
        """Return the dataset size."""
        if self.dataset is None or self.total_items == 0:
            return 0
        
        return self.total_items
    
    def __getitem__(self, idx):
        """Get dataset item by random sampling.
        
        Parameters
        ----------
        idx : int
            Index is ignored as we randomly sample.
            
        Returns
        -------
        tuple
            (volume, molecule_id)
        """
        try:
            if self.dataset is None or self.total_items == 0:
                default_shape = (1, self.box_size, self.box_size, self.box_size)
                return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long)
            
            # Randomly sample from the dataset
            real_idx = random.randrange(self.total_items)
            return self.dataset[real_idx]
            
        except Exception as e:
            logger.error(f"Error in __getitem__: {str(e)}")
            default_shape = (1, self.box_size, self.box_size, self.box_size)
            return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long)
            
    def get_visualization_samples(self, samples_per_source=5):
        """Get consistent samples for visualization.
        
        This method creates a dictionary mapping (mol_idx, source_type) to a list of indices
        that can be used for consistent visualization throughout training.
        
        Parameters
        ----------
        samples_per_source : int
            Number of samples to collect per molecule per source type
            
        Returns
        -------
        dict
            Dictionary mapping (mol_idx, source_type) to a list of indices
        """
        visualization_samples = {}
        
        if self.dataset is None or self.total_items == 0:
            # Return empty dict if no data available
            return visualization_samples
            
        # Set fixed seed for reproducible visualizations
        rng_state = random.getstate()
        random.seed(171717)
        
        try:
            # Collect samples for each structure in our dataset
            for struct_name, idx in self.molecule_to_idx.items():
                # Use structure name as source type for TomoTwin
                key = (idx, struct_name)
                
                # Collect sample indices for this structure
                indices = []
                
                # Try to collect the requested number of samples
                max_attempts = samples_per_source * 10
                for _ in range(max_attempts):
                    # If we've collected enough samples, break
                    if len(indices) >= samples_per_source:
                        break
                        
                    # Get a random index
                    try:
                        rand_idx = random.randrange(self.total_items)
                        volume, mol_id = self.dataset[rand_idx]
                        
                        # Check if this is the structure we want
                        if mol_id.item() == idx and rand_idx not in indices:
                            indices.append(rand_idx)
                    except Exception as e:
                        logger.warning(f"Error collecting visualization sample: {str(e)}")
                        continue
                
                # Add to our visualization samples dict
                if indices:
                    visualization_samples[key] = indices
        except Exception as e:
            logger.error(f"Error in get_visualization_samples: {str(e)}")
        
        # Restore random state
        random.setstate(rng_state)
        
        return visualization_samples
        
    def get_sample_by_index(self, idx):
        """Get a specific sample by its index.
        
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve
            
        Returns
        -------
        tuple
            (volume, molecule_id, source_type)
        """
        try:
            if self.dataset is None or idx >= self.total_items:
                default_shape = (1, self.box_size, self.box_size, self.box_size)
                return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long), "unknown"
            
            # Get the sample from the dataset
            volume, mol_id = self.dataset[idx]
            
            # Try to determine the source type/structure name from the molecule ID
            source_type = "unknown"
            for struct_name, idx in self.molecule_to_idx.items():
                if idx == mol_id.item():
                    source_type = struct_name
                    break
            
            return volume, mol_id, source_type
            
        except Exception as e:
            logger.error(f"Error in get_sample_by_index: {str(e)}")
            default_shape = (1, self.box_size, self.box_size, self.box_size)
            return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long), "unknown"


def create_tomotwin_dataloader(
    base_dir,
    name_to_pdb,
    config,
    dist_config=None,
    samples_per_epoch=2000,
    snr_values=None,
    max_structures=None,
    filtered_structure_ids=None
):
    """Create a DataLoader for TomoTwin data.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing TomoTwin data
    name_to_pdb : dict
        Mapping from molecule names to PDB IDs
    config : TrainingConfig
        Training configuration
    dist_config : DistributedConfig or None
        Distributed training configuration
    samples_per_epoch : int
        Number of samples per epoch
    snr_values : list of float or None
        List of SNR values to include
    max_structures : int or None
        Maximum number of structures to load per node (default: None, loads all available)
    filtered_structure_ids : list or None
        List of specific structure IDs to load (default: None, loads all available)
        
    Returns
    -------
    DataLoader
        DataLoader for TomoTwin data
    """
    # Get rank for logging
    rank = dist_config.node_rank if dist_config else 0
    
    # Create TomoTwin dataset
    dataset = TomoTwinDataset(
        base_dir=base_dir,
        name_to_pdb=name_to_pdb,
        box_size=config.box_size,
        snr_values=snr_values,
        device='cpu',
        augment=config.augment,
        seed=171717,
        rank=dist_config.node_rank if dist_config else None,
        world_size=dist_config.world_size if dist_config else None,
        samples_per_epoch=samples_per_epoch,
        normalization="z-score",
        max_structures=max_structures,
        filtered_structure_ids=filtered_structure_ids
    )
    
    # Print molecular stats
    if rank == 0:
        logger.info(dataset.get_molecular_stats())
    
    # Ensure minimum batch size per GPU
    min_batch_size = 8  # Minimum required for similarity calculation
    if dist_config:
        # Calculate per-GPU batch size, ensuring it's at least min_batch_size
        per_gpu_batch_size = max(min_batch_size, config.batch_size // dist_config.world_size)
        # Adjust global batch size to maintain at least min_batch_size per GPU
        config.batch_size = per_gpu_batch_size * dist_config.world_size
    else:
        per_gpu_batch_size = max(min_batch_size, config.batch_size)
    
    # Create distributed sampler if using DDP
    sampler = None
    if dist_config:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_config.world_size,
            rank=dist_config.node_rank * dist_config.devices_per_node + dist_config.local_rank,
            shuffle=True,
            seed=171717
        )
    
    # Use fewer workers to avoid potential deadlocks
    num_workers = min(16, max(1, os.cpu_count() // (torch.cuda.device_count() * 2 if torch.cuda.is_available() else 2)))
    
    if dist_config is None or dist_config.node_rank == 0:
        logger.info(f"Using {num_workers} workers for dataloader with batch size {per_gpu_batch_size}")
    
    # Create dataloader with distributed settings
    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=(sampler is None),  # Only shuffle if not using a sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        timeout=300  # Shorter timeout to detect hanging workers
    )
    
    if rank == 0:
        logger.info(f"Created TomoTwin DataLoader with {len(dataloader)} batches")
    
    return dataloader, dataset
