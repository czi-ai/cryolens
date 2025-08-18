"""
Dataset implementations for CryoLens.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class CachedParquetDataset(Dataset):
    """Dataset for loading pre-processed data from parquet files.
    
    Parameters
    ----------
    parquet_path : str
        Path to parquet file containing data.
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
    augment_config : dict, optional
        Configuration for augmentation parameters.
    """
    
    def __init__(
        self,
        parquet_path,
        name_to_pdb=None,
        box_size=48,
        device='cpu',
        augment=True,
        seed=42,
        rank=None,
        world_size=None,
        augment_config=None
    ):
        # Debug log
        rank_str = f"Rank {rank if rank is not None else 'None'}"
        print(f"{rank_str}: DEADLOCK_DEBUG - CachedParquetDataset.__init__ starting with path {parquet_path}")
        
        self.parquet_path = Path(parquet_path)
        self.name_to_pdb = name_to_pdb or {}
        self.box_size = box_size
        self.device = device
        self.augment = augment
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        
        # Set default augmentation configuration
        default_augment_config = {
            'rotation_prob': 0.5,
            'noise_prob': 0.7,
            'contrast_prob': 0.7,
            'noise_std_range': (0.05, 0.2),
            'contrast_range': (0.7, 1.3),
            'brightness_range': (-0.1, 0.1)
        }
        self.augment_config = {**default_augment_config, **(augment_config or {})}
        
        # Check if file exists
        print(f"{rank_str}: DEADLOCK_DEBUG - Checking if file exists: {self.parquet_path}")
        file_exists = self.parquet_path.exists()
        print(f"{rank_str}: DEADLOCK_DEBUG - File exists: {file_exists}")
        
        # Set random seed with distributed awareness
        print(f"{rank_str}: DEADLOCK_DEBUG - Setting random seed")
        self._set_random_seed()
        
        # Load data from parquet file
        print(f"{rank_str}: DEADLOCK_DEBUG - About to load data from parquet file")
        self._load_data()
        print(f"{rank_str}: DEADLOCK_DEBUG - Data loaded with {len(self.df)} samples")

        # Initialize molecule mapping
        print(f"{rank_str}: DEADLOCK_DEBUG - Initializing molecule mapping")
        self._initialize_molecule_mapping()
        print(f"{rank_str}: DEADLOCK_DEBUG - CachedParquetDataset initialization complete")
    
    def _set_random_seed(self):
        """Set random seed with distributed awareness."""
        if self.seed is not None:
            effective_seed = self.seed + (self.rank or 0)
            np.random.seed(effective_seed)
            torch.manual_seed(effective_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(effective_seed)
    
    def _load_data(self):
        """Load data from parquet file with validation."""
        rank_str = f"Rank {self.rank if self.rank is not None else 'None'}"
        print(f"{rank_str}: DEADLOCK_DEBUG - Starting _load_data for {self.parquet_path}")
        
        try:
            # Load dataframe from parquet
            print(f"{rank_str}: DEADLOCK_DEBUG - Reading parquet file")
            self.df = pd.read_parquet(self.parquet_path)
            print(f"{rank_str}: DEADLOCK_DEBUG - Parquet file read successfully with {len(self.df)} rows")
            
            if self.rank == 0 or self.rank is None:
                logging.info(f"Loaded {len(self.df)} samples from {self.parquet_path}")
            
            # Validate essential columns
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
    
    def _initialize_molecule_mapping(self):
        """Create mapping between molecule IDs and indices using PDB IDs."""
        try:
            # Load unique molecule IDs
            molecule_ids = self.df['molecule_id'].unique()
            
            # Filter out background, None, and NaN values
            non_background_ids = [
                mid for mid in molecule_ids 
                if mid not in ('background', None) and pd.notna(mid)
            ]
            
            if self.rank == 0 or self.rank is None:
                print("\nFiltering molecule IDs:")
                print(f"Total unique molecules: {len(molecule_ids)}")
                print(f"After filtering background/None: {len(non_background_ids)}")
            
            # Convert names to PDB IDs
            pdb_ids = []
            for name in non_background_ids:
                pdb_id = self.name_to_pdb.get(name)
                if pdb_id:
                    pdb_ids.append(pdb_id)
                else:
                    if self.rank == 0 or self.rank is None:
                        print(f"Warning: No PDB ID found for molecule {name}")
                    
            if self.rank == 0 or self.rank is None:
                print("\nMolecule mappings:")
                for i, (name, pdb) in enumerate(zip(non_background_ids, pdb_ids)):
                    if pdb:
                        print(f"{i}: {name} -> {pdb}")
                    
            # Create bidirectional mappings
            self.molecule_to_idx = {pdb: idx for idx, pdb in enumerate(sorted(pdb_ids)) if pdb}
            self.idx_to_molecule = {idx: pdb for pdb, idx in self.molecule_to_idx.items()}
            
            if self.rank == 0 or self.rank is None:
                print(f"\nLookup matrix will have shape: {len(self.molecule_to_idx)}x{len(self.molecule_to_idx)}")
                
        except Exception as e:
            if self.rank == 0 or self.rank is None:
                print(f"\nError in _initialize_molecule_mapping: {str(e)}")
            self.molecule_to_idx = {}
            self.idx_to_molecule = {}
    
    def __len__(self):
        """Get dataset length."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get dataset item.
        
        Parameters
        ----------
        idx : int
            Index of item to get.
            
        Returns
        -------
        tuple
            (volume, molecule_id) or (volume, molecule_id, pose) if return_poses=True
        """
        try:
            # Track first few items for debugging
            track_item = (idx < 5) and (self.rank is None or self.rank == 0)
            
            if track_item:
                print(f"ParquetDataset: Getting item {idx}, rank {self.rank}")
                
            if len(self.df) == 0:
                raise ValueError("Dataset is empty")
                
            snr_data = self.df.iloc[idx]
            subvolume = snr_data['subvolume']
            
            if track_item:
                print(f"Item {idx}: subvolume type: {type(subvolume)}")
            
            if isinstance(subvolume, (bytes, np.ndarray)):
                if isinstance(subvolume, bytes):
                    subvolume = np.frombuffer(subvolume, dtype=np.float32)
                    if track_item:
                        print(f"Item {idx}: converted bytes to array, shape: {subvolume.shape}")
                
                subvolume = subvolume.reshape(snr_data['shape'])
                if track_item:
                    print(f"Item {idx}: reshaped to {subvolume.shape}")
                
                # Normalize data using z-score normalization
                subvolume = self._normalize_volume(subvolume)
                if track_item:
                    print(f"Item {idx}: normalized volume")
                
                if self.augment:
                    subvolume = self._augment_volume(subvolume)
                    if track_item:
                        print(f"Item {idx}: augmented volume")
                    
                # Ensure contiguous memory layout before converting to tensor
                subvolume = np.ascontiguousarray(subvolume)
                # Ensure contiguous memory layout before converting to tensor
                subvolume = np.ascontiguousarray(subvolume)
                subvolume = np.expand_dims(subvolume, axis=0)
                subvolume = torch.from_numpy(subvolume).to(dtype=torch.float32)
                if track_item:
                    print(f"Item {idx}: final tensor shape: {subvolume.shape}")
            else:
                error_msg = f"Unexpected subvolume type: {type(subvolume)}"
                print(f"Error on item {idx}: {error_msg}")
                raise ValueError(error_msg)
            
            # Special case for background
            name = snr_data['molecule_id']
            if name == 'background':
                molecule_idx = -1  # Use -1 to indicate background
            else:
                # Regular case - convert name to PDB ID then to index
                pdb_id = self.name_to_pdb.get(name)
                molecule_idx = self.molecule_to_idx.get(pdb_id, -1)
            
            # Handle pose if available and requested
            if hasattr(self, 'return_poses') and self.return_poses and hasattr(self, 'pose_column'):
                pose_data = snr_data.get(self.pose_column, None)
                if pose_data is not None:
                    try:
                        if isinstance(pose_data, bytes):
                            pose = np.frombuffer(pose_data, dtype=np.float32)
                        else:
                            pose = np.array(pose_data, dtype=np.float32)
                        
                        # Ensure pose has correct shape (should be 4 values for axis-angle)
                        if pose.shape[0] != 4:
                            print(f"WARNING: Pose has shape {pose.shape}, expected 4 values")
                            pose = None
                        else:
                            # Convert to tensor
                            pose = torch.from_numpy(pose).to(dtype=torch.float32)
                            return subvolume, torch.tensor(molecule_idx, dtype=torch.long), pose
                    except Exception as e:
                        print(f"Error processing pose data: {e}")
                        # Fall through to return without pose
            
            return subvolume, torch.tensor(molecule_idx, dtype=torch.long)
                
        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            # Return empty tensor with correct shape
            default_shape = (1, self.box_size, self.box_size, self.box_size)
            return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long)
    
    def _normalize_volume(self, volume):
        """Normalize the volume using z-score normalization.
        
        Parameters
        ----------
        volume : np.ndarray
            Volume to normalize.
            
        Returns
        -------
        np.ndarray
            Normalized volume.
        """
        # Skip normalization if volume is empty or constant
        if volume.size == 0 or np.all(volume == volume.flat[0]):
            return volume
            
        # Z-score normalization (mean=0, std=1)
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 0:
            normalized = (volume - mean) / std
        else:
            normalized = volume - mean  # If std=0, just center the data
            
        return normalized
    
    def _augment_volume(self, volume):
        """Apply random augmentations to volume suitable for cryo-ET data.
        
        For cryo-ET subtomograms:
        - Rotations are only applied in the xy plane (around z-axis) to preserve
          the orientation relative to the ice surface
        - No flips are applied to preserve molecular handedness
        - Gaussian noise is added to simulate imaging noise
        - Contrast adjustments simulate varying imaging conditions
        
        Parameters
        ----------
        volume : np.ndarray
            Volume to augment.
            
        Returns
        -------
        np.ndarray
            Augmented volume.
        """
        if not self.augment:
            return volume
            
        # Make a contiguous copy to avoid stride issues
        volume = np.ascontiguousarray(volume)
        
        # Random rotation only in xy plane (around z-axis)
        # This preserves the orientation relative to the ice surface
        if np.random.random() < self.augment_config['rotation_prob']:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            volume = np.rot90(volume, k=k, axes=(0, 1))  # Only rotate in xy plane
            volume = np.ascontiguousarray(volume)
            
        # Add Gaussian noise to simulate imaging noise
        if np.random.random() < self.augment_config['noise_prob']:
            noise_std_min, noise_std_max = self.augment_config['noise_std_range']
            noise_std = np.random.uniform(noise_std_min, noise_std_max) * np.std(volume)
            noise = np.random.normal(0, noise_std, volume.shape)
            volume = volume + noise
            
        # Contrast adjustment to simulate varying imaging conditions
        if np.random.random() < self.augment_config['contrast_prob']:
            # Random contrast factor
            contrast_min, contrast_max = self.augment_config['contrast_range']
            contrast_factor = np.random.uniform(contrast_min, contrast_max)
            # Random brightness offset
            brightness_min, brightness_max = self.augment_config['brightness_range']
            brightness_offset = np.random.uniform(brightness_min, brightness_max) * np.std(volume)
            
            # Apply contrast and brightness
            mean_val = np.mean(volume)
            volume = contrast_factor * (volume - mean_val) + mean_val + brightness_offset
                
        return volume


class CurriculumParquetDataset(Dataset):
    """Dataset with curriculum learning capabilities.
    
    Parameters
    ----------
    parquet_paths : list
        List of paths to parquet files containing data.
    name_to_pdb : dict, optional
        Mapping from molecule names to PDB IDs.
    box_size : int
        Size of the volume box.
    curriculum_spec : list, optional
        Specification for curriculum learning.
    samples_per_epoch : int
        Number of samples per epoch.
    device : str or torch.device
        Device to load data to.
    augment : bool
        Whether to apply data augmentation.
    seed : int
        Random seed.
    rank : int, optional
        Process rank for distributed training.
    world_size : int, optional
        World size for distributed training.
    normalization: str, optional
        Normalization method to use (default: "z-score").
        Options: "z-score", "min-max", "percentile", "none"
    augment_config : dict, optional
        Configuration for augmentation parameters.
    """
    
    def __init__(
        self,
        parquet_paths,
        name_to_pdb=None,
        box_size=48,
        curriculum_spec=None,
        samples_per_epoch=1000,
        device='cpu',
        augment=True,
        seed=42,
        rank=None,
        world_size=None,
        normalization="z-score",
        augment_config=None
    ):
        logger.info(f"CurriculumParquetDataset.__init__ starting with {len(parquet_paths)} paths")
        
        self.parquet_paths = [Path(p) for p in parquet_paths]
        self.name_to_pdb = name_to_pdb or {}
        self.box_size = box_size
        self.curriculum_spec = curriculum_spec
        self.samples_per_epoch = samples_per_epoch
        self.device = device
        self.augment = augment
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.normalization = normalization
        
        logger.info(f"Setting up augmentation config...")
        # Set default augmentation configuration
        default_augment_config = {
            'rotation_prob': 0,
            'noise_prob': 0.7,
            'contrast_prob': 0.7,
            'noise_std_range': (0.05, 0.2),
            'contrast_range': (0.7, 1.3),
            'brightness_range': (-0.1, 0.1)
        }
        self.augment_config = {**default_augment_config, **(augment_config or {})}
        
        # Set current epoch and stage
        self.current_epoch = 0
        self.current_stage = 0
        
        logger.info(f"Setting random seed...")
        # Set random seed
        if self.rank is not None:
            # Use different seed for each process
            np.random.seed(seed + rank)
            torch.manual_seed(seed + rank)
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Load datasets
        self.datasets = []
        self.weights = []
        self.total_items = 0
        
        logger.info(f"About to load data from {len(self.parquet_paths)} parquet files...")
        # Combine data from all parquet files
        self._load_data()
        logger.info(f"Data loading completed. Total items: {self.total_items}")
        
        # Make sure we have at least one valid dataset
        if self.total_items == 0:
            logging.error("All datasets are empty! This will cause errors during sampling.")
            if rank == 0 or rank is None:
                print(f"Rank {rank if rank is not None else 'None'}: DEADLOCK_DEBUG - WARNING: All datasets are empty!")
                print("WARNING: All datasets are empty! This will cause errors during sampling.")
                # Try to provide more diagnostics information
                for i, path in enumerate(self.parquet_paths):
                    print(f"Dataset {i} path: {path}")
        
        # Update weights based on curriculum
        logger.info(f"Updating dataset weights...")
        self._update_weights()
        logger.info(f"CurriculumParquetDataset initialization complete")
    
    def _normalize_volume(self, volume):
        """Normalize the volume based on the selected normalization method.
        
        Parameters
        ----------
        volume : np.ndarray
            Volume to normalize.
            
        Returns
        -------
        np.ndarray
            Normalized volume.
        """
        # Skip normalization if volume is empty or contains all zeros
        if volume.size == 0 or np.all(volume == 0):
            return volume
            
        # Apply the selected normalization method
        if self.normalization == "z-score":
            # Z-score normalization (mean=0, std=1)
            mean = np.mean(volume)
            std = np.std(volume)
            if std > 0:
                normalized = (volume - mean) / std
            else:
                normalized = volume - mean  # If std=0, just center the data
                
        elif self.normalization == "min-max":
            # Min-max normalization to [0,1] range
            min_val = np.min(volume)
            max_val = np.max(volume)
            if max_val > min_val:
                normalized = (volume - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(volume)  # Handle constant volume
                
        elif self.normalization == "percentile":
            # Robust normalization using percentiles
            p_low, p_high = np.percentile(volume, [2, 98])
            if p_high > p_low:
                normalized = np.clip((volume - p_low) / (p_high - p_low), 0, 1)
            else:
                normalized = np.zeros_like(volume)
        else:
            # Default: no normalization
            normalized = volume
            
        return normalized
        
    def _load_data(self):
        """Load data from multiple parquet files with validation and apply normalization."""
        logger.info("Starting _load_data()...")
        try:
            # Load all dataframes
            dataframes = []
            
            for path_idx, path in enumerate(self.parquet_paths):
                logger.info(f"Loading parquet file {path_idx + 1}/{len(self.parquet_paths)}: {path}")
                try:
                    df = pd.read_parquet(path)
                    logger.info(f"Successfully loaded {len(df)} samples from {path}")
                    
                    # Add source identifier for curriculum sampling
                    source_name = f"source_{path_idx}"
                    df['dataset_source'] = source_name
                    
                    # Ensure 'source_type' column exists
                    if 'source_type' not in df.columns:
                        df['source_type'] = source_name
                    
                    # Validate data
                    missing_cols = set(['subvolume', 'shape', 'molecule_id']) - set(df.columns)
                    if missing_cols:
                        logging.warning(f"Missing required columns in {path}: {missing_cols}")
                        continue
                    
                    logger.info(f"Validating subvolumes for file {path_idx + 1}...")
                    # Validate subvolumes but defer normalization to __getitem__
                    def validate_subvolume(x):
                        if isinstance(x, bytes):
                            try:
                                # Just validate we can read it, don't process it yet
                                test_array = np.frombuffer(x, dtype=np.float32)
                                return x  # Keep original bytes
                            except Exception as e:
                                if self.rank == 0 or self.rank is None:
                                    logging.warning(f"Invalid subvolume bytes: {str(e)}")
                                return None
                        elif isinstance(x, np.ndarray):
                            return x  # Keep original array
                        return None
                    
                    logger.info(f"Applying validation to subvolumes...")
                    # Only validate, don't process all subvolumes at once
                    df['subvolume'] = df['subvolume'].apply(validate_subvolume)
                    logger.info(f"Validation completed for file {path_idx + 1}")
                    
                    # Filter out invalid rows
                    valid_mask = df['subvolume'].notna()
                    invalid_count = (~valid_mask).sum()
                    if invalid_count > 0:
                        logging.warning(f"Found {invalid_count} invalid subvolumes in {path}")
                        df = df[valid_mask].reset_index(drop=True)
                    
                    # Create sample identifiers for curriculum weighting
                    df['sample_id'] = df.apply(
                        lambda row: f"{row['source_type']}:{row['molecule_id']}", 
                        axis=1
                    )
                    
                    dataframes.append(df)
                    logger.info(f"File {path_idx + 1} processed successfully with {len(df)} valid samples")
                    
                except Exception as e:
                    logger.error(f"Error loading dataset from {path}: {str(e)}")
                    traceback.print_exc()
            
            logger.info(f"Combining {len(dataframes)} dataframes...")
            # Combine all dataframes
            if dataframes:
                self.df = pd.concat(dataframes, ignore_index=True)
                self.total_items = len(self.df)
                
                logger.info(f"Combined dataframe has {self.total_items} samples")
                
                # Initialize molecule mapping
                logger.info("Initializing molecule mapping...")
                self._initialize_molecule_mapping()
                
                # Log sample distribution
                if self.rank == 0 or self.rank is None:
                    sample_counts = self.df.groupby(['source_type', 'molecule_id']).size()
                    logging.info(f"Sample distribution: {sample_counts}")
                    logging.info(f"Dataset loaded with {len(self.df)} samples (normalization deferred)")
            else:
                # Create empty dataframe if no valid data was loaded
                self.df = pd.DataFrame(columns=['subvolume', 'shape', 'molecule_id', 'source_type', 'sample_id'])
                self.total_items = 0
                logging.warning("No valid data was loaded from any of the parquet files")
                
        except Exception as e:
            logging.error(f"Error loading data on rank {self.rank if self.rank is not None else 0}: {str(e)}")
            traceback.print_exc()
            # Create empty dataframe as fallback
            self.df = pd.DataFrame(columns=['subvolume', 'shape', 'molecule_id', 'source_type', 'sample_id'])
            self.total_items = 0
        
    def _initialize_molecule_mapping(self):
        """Create mapping between molecule IDs and indices using PDB IDs."""
        try:
            # Load unique molecule IDs
            molecule_ids = self.df['molecule_id'].unique()
            
            # Filter out background, None, and NaN values
            non_background_ids = [
                mid for mid in molecule_ids 
                if mid not in ('background', None) and pd.notna(mid)
            ]
            
            if self.rank == 0 or self.rank is None:
                print("\nFiltering molecule IDs:")
                print(f"Total unique molecules: {len(molecule_ids)}")
                print(f"After filtering background/None: {len(non_background_ids)}")
            
            # Convert names to PDB IDs
            pdb_ids = []
            for name in non_background_ids:
                pdb_id = self.name_to_pdb.get(name)
                if pdb_id:
                    pdb_ids.append(pdb_id)
                else:
                    if self.rank == 0 or self.rank is None:
                        print(f"Warning: No PDB ID found for molecule {name}")
                    
            if self.rank == 0 or self.rank is None:
                print("\nMolecule mappings:")
                for i, (name, pdb) in enumerate(zip(non_background_ids, pdb_ids)):
                    if pdb:
                        print(f"{i}: {name} -> {pdb}")
                    
            # Create bidirectional mappings
            self.molecule_to_idx = {pdb: idx for idx, pdb in enumerate(sorted(pdb_ids)) if pdb}
            self.idx_to_molecule = {idx: pdb for pdb, idx in self.molecule_to_idx.items()}
            
            if self.rank == 0 or self.rank is None:
                print(f"\nLookup matrix will have shape: {len(self.molecule_to_idx)}x{len(self.molecule_to_idx)}")
                
        except Exception as e:
            if self.rank == 0 or self.rank is None:
                print(f"\nError in _initialize_molecule_mapping: {str(e)}")
            self.molecule_to_idx = {}
            self.idx_to_molecule = {}
    
    def _update_weights(self):
        """Update dataset weights based on curriculum specification ensuring no empty datasets are selected."""
        # Debug info
        print(f"Rank {self.rank if self.rank is not None else 'None'}: DEADLOCK_DEBUG - Inside _update_weights")
        
        if not self.curriculum_spec:
            print(f"Rank {self.rank if self.rank is not None else 'None'}: DEADLOCK_DEBUG - No curriculum spec, using equal weights")
            # Equal weights for all samples if no curriculum
            self.sample_weights = np.ones(len(self.df)) / len(self.df) if len(self.df) > 0 else np.array([])
            self.class_weights = None
            return
        
        # Find current stage
        print(f"Rank {self.rank if self.rank is not None else 'None'}: DEADLOCK_DEBUG - Finding current curriculum stage for epoch {self.current_epoch}")
        self.current_stage = 0
        total_duration = 0
        
        for i, stage in enumerate(self.curriculum_spec):
            duration = stage.get('duration', float('inf'))
            if total_duration + duration > self.current_epoch:
                self.current_stage = i
                print(f"Rank {self.rank if self.rank is not None else 'None'}: DEADLOCK_DEBUG - Selected stage {i} (total_duration={total_duration}, stage_duration={duration})")
                break
            total_duration += duration
        
        # Get weights from current stage
        stage_weights = self.curriculum_spec[self.current_stage].get('weights', {})
        
        # Store class weights directly for generating epoch indices
        self.class_weights = stage_weights
        
        # Initialize weights for each sample
        self.sample_weights = np.zeros(len(self.df))
        
        # Group samples by class
        self.class_indices = {}
        
        if stage_weights:
            for sample_id, weight in stage_weights.items():
                mask = (self.df['sample_id'] == sample_id)
                indices = np.where(mask)[0]
                sample_count = len(indices)
                
                if sample_count > 0:
                    # Store indices for each class
                    self.class_indices[sample_id] = indices
                    
                    # Distribute weight evenly across all matching samples
                    self.sample_weights[mask] = weight / sample_count
        else:
            # Use uniform weights if no specific weights provided
            self.sample_weights = np.ones(len(self.df)) / len(self.df) if len(self.df) > 0 else np.array([])
        
        # Ensure no division by zero
        if np.sum(self.sample_weights) == 0 and len(self.df) > 0:
            self.sample_weights = np.ones(len(self.df)) / len(self.df)
            self.class_weights = None
            if self.rank == 0 or self.rank is None:
                logging.warning(f"No matching samples found for curriculum weights, using uniform sampling")
        elif len(self.df) > 0:
            # Normalize weights with stability check
            weight_sum = np.sum(self.sample_weights)
            if weight_sum > 0:
                self.sample_weights = self.sample_weights / weight_sum
            else:
                # Fallback to uniform sampling if weights sum to zero
                self.sample_weights = np.ones(len(self.df)) / len(self.df)
                if self.rank == 0 or self.rank is None:
                    logging.warning("Sample weights sum to zero, falling back to uniform sampling")
            
        print(f"Rank {self.rank if self.rank is not None else 'None'}: DEADLOCK_DEBUG - Final weights: {self.sample_weights.sum()}")
    
    def _generate_epoch_indices(self):
        """Generate fixed number of indices for the current epoch based on weights."""
        if len(self.df) == 0:
            self.epoch_indices = np.array([])
            return
            
        # Determine whether to use class-based sampling or direct sample weights
        if hasattr(self, 'class_weights') and self.class_weights and hasattr(self, 'class_indices'):
            try:
                # Calculate number of samples per class based on weights
                class_sample_counts = {}
                total_weight = sum(self.class_weights.values())
                samples_remaining = self.samples_per_epoch
                
                # First pass: calculate integer number of samples per class
                for class_id, weight in self.class_weights.items():
                    if class_id in self.class_indices:
                        proportion = weight / total_weight
                        count = int(proportion * self.samples_per_epoch)
                        class_sample_counts[class_id] = count
                        samples_remaining -= count
                
                # Second pass: distribute remaining samples
                classes_list = list(class_sample_counts.keys())
                if classes_list:
                    weights_list = [self.class_weights[c] for c in classes_list]
                    weights_array = np.array(weights_list)
                    prob = weights_array / np.sum(weights_array)
                    
                    # Distribute remaining samples based on weights
                    if samples_remaining > 0:
                        additional_counts = np.random.multinomial(samples_remaining, prob)
                        for i, class_id in enumerate(classes_list):
                            class_sample_counts[class_id] += additional_counts[i]
                
                # Now sample indices for each class
                self.epoch_indices = []
                
                for class_id, count in class_sample_counts.items():
                    if class_id in self.class_indices and count > 0:
                        class_indices = self.class_indices[class_id]
                        # Sample with replacement from each class
                        sampled_indices = np.random.choice(
                            class_indices, 
                            size=count,
                            replace=True
                        )
                        self.epoch_indices.extend(sampled_indices)
                
                # Shuffle the final indices
                if self.epoch_indices:
                    self.epoch_indices = np.array(self.epoch_indices)
                    np.random.shuffle(self.epoch_indices)
                else:
                    # Fallback if no indices were generated
                    self.epoch_indices = np.random.choice(
                        len(self.df), 
                        size=self.samples_per_epoch,
                        replace=True
                    )
                
            except Exception as e:
                logging.error(f"Error generating epoch indices: {str(e)}")
                # Fall back to uniform sampling
                self.epoch_indices = np.random.choice(
                    len(self.df), 
                    size=self.samples_per_epoch,
                    replace=True
                )
        else:
            # Use sample weights for direct sampling
            try:
                self.epoch_indices = np.random.choice(
                    len(self.df),
                    size=self.samples_per_epoch,
                    replace=True,
                    p=self.sample_weights if np.sum(self.sample_weights) > 0 else None
                )
            except Exception as e:
                logging.error(f"Error in weighted sampling: {str(e)}")
                # Fallback to uniform sampling
                self.epoch_indices = np.random.choice(
                    len(self.df), 
                    size=self.samples_per_epoch,
                    replace=True
                )

    def update_epoch(self, epoch):
        """Update current epoch and weights.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        """
        self.current_epoch = epoch
        
        # Check if we need to move to the next stage
        total_epochs = 0
        stage_changed = False
        
        for stage_idx, stage in enumerate(self.curriculum_spec) if self.curriculum_spec else []:
            total_epochs += stage.get("duration", float('inf'))
            if self.current_epoch < total_epochs:
                if stage_idx != self.current_stage:
                    self.current_stage = stage_idx
                    stage_changed = True
                    if self.rank == 0 or self.rank is None:
                        logging.info(f"Moving to curriculum stage {self.current_stage} at epoch {self.current_epoch}")
                break
        
        # Update sampling weights if stage changed
        if stage_changed:
            self._update_weights()
            
        # Set different random seed for each epoch to ensure variation
        np.random.seed(self.seed + self.current_epoch + (0 if self.rank is None else self.rank * 10000))
        
        # Generate new indices for this epoch
        self._generate_epoch_indices()
    
    def get_visualization_samples(self, samples_per_mol_per_source=10):
        """
        Retrieve a fixed set of samples for visualization, organized by molecule ID and source type.
        The function memoizes results to ensure consistent visualization throughout training.
        
        Parameters
        ----------
        samples_per_mol_per_source : int
            Number of samples to collect per molecule per source type
            
        Returns
        -------
        dict
            Dict mapping (mol_id, source_type) tuples to lists of sample indices
        """
        # Check if we already have memoized samples
        if hasattr(self, '_visualization_samples') and self._visualization_samples is not None:
            return self._visualization_samples
        
        if self.rank == 0 or self.rank is None:
            logging.info(f"Generating memoized visualization samples ({samples_per_mol_per_source} per mol per source)")
        
        # Dictionary to hold visualization samples
        self._visualization_samples = {}
        
        # Get unique molecule IDs (excluding background/None)
        unique_mols = self.df['molecule_id'].unique()
        valid_mols = [mol for mol in unique_mols if mol not in ('background', None) and pd.notna(mol)]
        
        # Get unique source types
        unique_sources = self.df['source_type'].unique()
        
        # For each combination of molecule and source, collect sample indices
        for mol in valid_mols:
            # Get PDB ID for the molecule
            pdb_id = self.name_to_pdb.get(mol)
            if not pdb_id or pdb_id not in self.molecule_to_idx:
                continue
                
            mol_idx = self.molecule_to_idx[pdb_id]
            
            for source in unique_sources:
                # Find all samples matching this molecule and source
                mask = (self.df['molecule_id'] == mol) & (self.df['source_type'] == source)
                matching_indices = np.where(mask)[0]
                
                if len(matching_indices) == 0:
                    continue
                    
                # Sample with replacement if we don't have enough samples
                num_samples = min(samples_per_mol_per_source, len(matching_indices))
                if num_samples < samples_per_mol_per_source:
                    # Sample with replacement
                    sampled_indices = np.random.choice(
                        matching_indices, 
                        size=samples_per_mol_per_source,
                        replace=True
                    )
                else:
                    # Sample without replacement to get variety
                    sampled_indices = np.random.choice(
                        matching_indices,
                        size=samples_per_mol_per_source,
                        replace=False
                    )
                
                # Store using a tuple key of (mol_idx, source)
                key = (mol_idx, source)
                self._visualization_samples[key] = sampled_indices.tolist()
        
        if self.rank == 0 or self.rank is None:
            # Log statistics about the visualization samples
            total_samples = sum(len(indices) for indices in self._visualization_samples.values())
            num_mols = len({key[0] for key in self._visualization_samples.keys()})
            num_sources = len({key[1] for key in self._visualization_samples.keys()})
            
            logging.info(f"Memoized {total_samples} visualization samples across {num_mols} molecules and {num_sources} source types")
        
        return self._visualization_samples

    def get_sample_by_index(self, idx):
        """
        Retrieve a sample by its direct index in the dataframe, bypassing epoch sampling.
        Used primarily for visualization.
        
        Parameters
        ----------
        idx : int
            Raw index into the dataframe
            
        Returns
        -------
        tuple
            Tuple of (volume tensor, molecule index tensor, source type)
        """
        try:
            snr_data = self.df.iloc[idx]
            subvolume = snr_data['subvolume']
            
            if isinstance(subvolume, (bytes, np.ndarray)):
                if isinstance(subvolume, bytes):
                    subvolume = np.frombuffer(subvolume, dtype=np.float32)
                
                subvolume = subvolume.reshape(snr_data['shape'])
                
                # Apply normalization at access time
                subvolume = self._normalize_volume(subvolume)
                
                # No augmentation for visualization samples to ensure consistency
                # Ensure contiguous memory layout before converting to tensor
                subvolume = np.ascontiguousarray(subvolume)
                subvolume = np.expand_dims(subvolume, axis=0)
                subvolume = torch.from_numpy(subvolume).to(dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected subvolume type: {type(subvolume)}")
            
            # Get molecule index
            name = snr_data['molecule_id']
            if name == 'background':
                molecule_idx = -1
            else:
                pdb_id = self.name_to_pdb.get(name)
                if pdb_id and pdb_id in self.molecule_to_idx:
                    molecule_idx = self.molecule_to_idx[pdb_id]
                else:
                    molecule_idx = -1
            
            # Get source type if available
            source_type = snr_data.get('source_type', snr_data.get('dataset_source', 'unknown'))
            
            # Return source_type as additional information
            return subvolume, torch.tensor(molecule_idx, dtype=torch.long), source_type
                    
        except Exception as e:
            logging.error(f"Error loading item at index {idx}: {str(e)}")
            default_shape = (1, self.box_size, self.box_size, self.box_size)
            return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long), "unknown"

    def __len__(self):
        """Return the fixed epoch size rather than the dataset size."""
        if self.world_size:
            # Adjust samples per rank in distributed training
            return self.samples_per_epoch // self.world_size
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """Get dataset item with epoch-based sampling.
        
        Parameters
        ----------
        idx : int
            Index of item to get.
            
        Returns
        -------
        tuple
            (volume, molecule_id)
        """
        try:
            # Handle empty dataset case
            if len(self.df) == 0:
                default_shape = (1, self.box_size, self.box_size, self.box_size)
                return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long)
            
            # Map the index to the pre-generated epoch indices
            if hasattr(self, 'epoch_indices') and len(self.epoch_indices) > 0:
                # Map idx to the correct index in our epoch indices
                epoch_idx = idx % len(self.epoch_indices)
                real_idx = self.epoch_indices[epoch_idx]
            else:
                # Fallback if epoch indices not available
                real_idx = idx % len(self.df)
            
            snr_data = self.df.iloc[real_idx]
            subvolume = snr_data['subvolume']
            
            if isinstance(subvolume, (bytes, np.ndarray)):
                if isinstance(subvolume, bytes):
                    subvolume = np.frombuffer(subvolume, dtype=np.float32)
                
                subvolume = subvolume.reshape(snr_data['shape'])
                
                # Apply normalization at access time
                subvolume = self._normalize_volume(subvolume)
                
                if self.augment:
                    subvolume = self._augment_volume(subvolume)
                    
                # Ensure contiguous memory layout before converting to tensor
                subvolume = np.ascontiguousarray(subvolume)
                subvolume = np.expand_dims(subvolume, axis=0)
                subvolume = torch.from_numpy(subvolume).to(dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected subvolume type: {type(subvolume)}")
            
            # Special case for background
            name = snr_data['molecule_id']
            if name == 'background':
                molecule_idx = -1  # Use -1 to indicate background
            else:
                # Regular case - convert name to PDB ID then to index
                pdb_id = self.name_to_pdb.get(name)
                if pdb_id and pdb_id in self.molecule_to_idx:
                    molecule_idx = self.molecule_to_idx[pdb_id]
                else:
                    molecule_idx = -1  # Use -1 for unknown molecules
            
            return subvolume, torch.tensor(molecule_idx, dtype=torch.long)
                
        except Exception as e:
            logging.error(f"Error loading item {idx}: {str(e)}")
            default_shape = (1, self.box_size, self.box_size, self.box_size)
            return torch.zeros(default_shape, dtype=torch.float32), torch.tensor(-1, dtype=torch.long)

    def _augment_volume(self, volume):
        """Apply data augmentation to 3D volume suitable for cryo-ET data.
        
        For cryo-ET subtomograms:
        - Rotations are only applied in the xy plane (around z-axis) to preserve
          the orientation relative to the ice surface
        - No flips are applied to preserve molecular handedness
        - Gaussian noise is added to simulate imaging noise
        - Contrast adjustments simulate varying imaging conditions
        
        Parameters
        ----------
        volume : np.ndarray
            Volume to augment.
            
        Returns
        -------
        np.ndarray
            Augmented volume.
        """
        if not self.augment:
            return volume
            
        # Make a contiguous copy to avoid stride issues
        volume = np.ascontiguousarray(volume)
        
        # Random rotation only in xy plane (around z-axis)
        # This preserves the orientation relative to the ice surface
        if np.random.random() < self.augment_config['rotation_prob']:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            volume = np.rot90(volume, k=k, axes=(0, 1))  # Only rotate in xy plane
            volume = np.ascontiguousarray(volume)
            
        # Add Gaussian noise to simulate imaging noise
        if np.random.random() < self.augment_config['noise_prob']:
            noise_std_min, noise_std_max = self.augment_config['noise_std_range']
            noise_std = np.random.uniform(noise_std_min, noise_std_max) * np.std(volume)
            noise = np.random.normal(0, noise_std, volume.shape)
            volume = volume + noise
            
        # Contrast adjustment to simulate varying imaging conditions
        if np.random.random() < self.augment_config['contrast_prob']:
            # Random contrast factor
            contrast_min, contrast_max = self.augment_config['contrast_range']
            contrast_factor = np.random.uniform(contrast_min, contrast_max)
            # Random brightness offset
            brightness_min, brightness_max = self.augment_config['brightness_range']
            brightness_offset = np.random.uniform(brightness_min, brightness_max) * np.std(volume)
            
            # Apply contrast and brightness
            mean_val = np.mean(volume)
            volume = contrast_factor * (volume - mean_val) + mean_val + brightness_offset
                
        return volume