"""
Structured batch sampler for ensuring k structures and m poses per batch.

This module provides specialized sampling strategies for training with
controlled batch composition, ensuring each batch contains a specific
number of structures with a specific number of poses per structure.
"""

import torch
import numpy as np
import random
import logging
from typing import List, Dict, Optional, Iterator, Tuple
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import math

logger = logging.getLogger(__name__)


class StructuredBatchSampler(Sampler):
    """
    A batch sampler that ensures each batch contains k structures with m poses per structure.
    
    This sampler is designed to work with datasets where samples are grouped by structure/molecule,
    and we want to ensure balanced representation in each batch.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from. Must have structure_names and molecule_to_idx attributes.
    structures_per_batch : int
        Number of different structures to include in each batch (k)
    poses_per_structure : int
        Number of poses/samples per structure in each batch (m)
    samples_per_epoch : int
        Total number of samples to generate per epoch
    include_background : bool
        Whether to include background samples in batches
    background_fraction : float
        Fraction of the batch to fill with background samples (if include_background=True)
    seed : int
        Random seed for reproducibility
    drop_last : bool
        Whether to drop the last incomplete batch
    """
    
    def __init__(
        self,
        dataset,
        structures_per_batch: int = 8,
        poses_per_structure: int = 4,
        samples_per_epoch: int = 1000,
        include_background: bool = True,
        background_fraction: float = 0.125,  # 1/8 for one structure slot
        seed: int = 0,
        drop_last: bool = True
    ):
        self.dataset = dataset
        self.structures_per_batch = structures_per_batch
        self.poses_per_structure = poses_per_structure
        self.samples_per_epoch = samples_per_epoch
        self.include_background = include_background
        self.background_fraction = background_fraction
        self.seed = seed
        self.drop_last = drop_last
        
        # Calculate batch size
        self.batch_size = structures_per_batch * poses_per_structure
        
        # Build index mappings
        self._build_structure_indices()
        
        # Set random seed
        self.epoch = 0
        self._set_random_seed()
        
    def _build_structure_indices(self):
        """Build mappings from structures to dataset indices."""
        self.structure_to_indices = {}
        self.background_indices = []
        
        # Scan through dataset to build index mappings
        logger.info("Building structure index mappings...")
        
        for idx in range(len(self.dataset)):
            try:
                # Get sample to determine its structure
                sample = self.dataset[idx]
                
                # Handle different return formats
                if len(sample) == 3:
                    _, mol_id, _ = sample  # With pose
                else:
                    _, mol_id = sample  # Without pose
                
                mol_id_val = mol_id.item() if torch.is_tensor(mol_id) else mol_id
                
                # Background samples have mol_id = -1
                if mol_id_val == -1:
                    self.background_indices.append(idx)
                else:
                    # Find structure name from mol_id
                    structure_name = None
                    if hasattr(self.dataset, 'molecule_to_idx'):
                        for name, s_idx in self.dataset.molecule_to_idx.items():
                            if s_idx == mol_id_val:
                                structure_name = name
                                break
                    
                    if structure_name:
                        if structure_name not in self.structure_to_indices:
                            self.structure_to_indices[structure_name] = []
                        self.structure_to_indices[structure_name].append(idx)
                        
            except Exception as e:
                logger.warning(f"Error processing index {idx}: {e}")
                continue
        
        # Get list of available structures (excluding background)
        self.available_structures = list(self.structure_to_indices.keys())
        
        # Validate we have enough structures
        if self.include_background:
            min_required = self.structures_per_batch - 1  # One slot for background
        else:
            min_required = self.structures_per_batch
            
        if len(self.available_structures) < min_required:
            raise ValueError(
                f"Not enough structures available. Need {min_required}, "
                f"but only have {len(self.available_structures)}"
            )
        
        logger.info(f"Found {len(self.available_structures)} structures with indices")
        logger.info(f"Found {len(self.background_indices)} background samples")
        
        # Log structure statistics
        for struct_name in sorted(self.available_structures)[:5]:
            logger.info(f"  {struct_name}: {len(self.structure_to_indices[struct_name])} samples")
        if len(self.available_structures) > 5:
            logger.info(f"  ... and {len(self.available_structures) - 5} more structures")
    
    def _set_random_seed(self):
        """Set random seed based on current epoch."""
        seed = self.seed + self.epoch
        random.seed(seed)
        np.random.seed(seed)
        
    def set_epoch(self, epoch: int):
        """Set the current epoch for shuffling."""
        self.epoch = epoch
        self._set_random_seed()
        
    def _generate_batch_indices(self) -> List[int]:
        """Generate indices for a single batch."""
        batch_indices = []
        
        if self.include_background and self.background_indices:
            # Calculate how many background samples to include
            num_background = max(1, int(self.batch_size * self.background_fraction))
            num_background = min(num_background, self.poses_per_structure)  # Cap at poses_per_structure
            
            # Calculate how many regular structures we need
            remaining_samples = self.batch_size - num_background
            num_structures = remaining_samples // self.poses_per_structure
            
            # Adjust if we don't have perfect division
            if remaining_samples % self.poses_per_structure != 0:
                # Either increase background or decrease to fit
                if num_structures > 0:
                    num_background = self.batch_size - (num_structures * self.poses_per_structure)
                else:
                    num_structures = self.structures_per_batch - 1
                    num_background = self.batch_size - (num_structures * self.poses_per_structure)
        else:
            num_structures = self.structures_per_batch
            num_background = 0
            
        # Select structures for this batch
        selected_structures = random.sample(self.available_structures, 
                                          min(num_structures, len(self.available_structures)))
        
        # Add poses from each selected structure
        for structure_name in selected_structures:
            structure_indices = self.structure_to_indices[structure_name]
            
            if len(structure_indices) >= self.poses_per_structure:
                # Sample without replacement if we have enough
                selected_poses = random.sample(structure_indices, self.poses_per_structure)
            else:
                # Sample with replacement if we don't have enough
                selected_poses = random.choices(structure_indices, k=self.poses_per_structure)
            
            batch_indices.extend(selected_poses)
        
        # Add background samples if needed
        if num_background > 0 and self.background_indices:
            if len(self.background_indices) >= num_background:
                background_samples = random.sample(self.background_indices, num_background)
            else:
                background_samples = random.choices(self.background_indices, k=num_background)
            batch_indices.extend(background_samples)
        
        # Ensure we have exactly batch_size samples
        if len(batch_indices) < self.batch_size:
            # Pad with random samples if needed
            padding_needed = self.batch_size - len(batch_indices)
            all_indices = list(range(len(self.dataset)))
            padding_indices = random.sample(all_indices, padding_needed)
            batch_indices.extend(padding_indices)
        elif len(batch_indices) > self.batch_size:
            # Truncate if we have too many
            batch_indices = batch_indices[:self.batch_size]
        
        # Shuffle the batch to mix structures
        random.shuffle(batch_indices)
        
        return batch_indices
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches."""
        # Calculate number of batches
        num_batches = self.samples_per_epoch // self.batch_size
        
        if not self.drop_last and (self.samples_per_epoch % self.batch_size != 0):
            num_batches += 1
        
        # Generate batches
        for batch_idx in range(num_batches):
            batch_indices = self._generate_batch_indices()
            yield batch_indices
    
    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return self.samples_per_epoch // self.batch_size
        else:
            return (self.samples_per_epoch + self.batch_size - 1) // self.batch_size


class DistributedStructuredBatchSampler(Sampler):
    """
    Distributed version of StructuredBatchSampler that works with DDP.
    
    This sampler ensures that each GPU gets complete structured batches
    while maintaining the k structures × m poses composition.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from
    structures_per_batch : int
        Number of different structures to include in each batch
    poses_per_structure : int
        Number of poses/samples per structure in each batch
    samples_per_epoch : int
        Total number of samples across all GPUs per epoch
    num_replicas : int
        Number of processes participating in distributed training
    rank : int
        Rank of the current process
    include_background : bool
        Whether to include background samples
    background_fraction : float
        Fraction of batch for background samples
    seed : int
        Random seed for reproducibility
    drop_last : bool
        Whether to drop the last incomplete batch
    """
    
    def __init__(
        self,
        dataset,
        structures_per_batch: int = 8,
        poses_per_structure: int = 4,
        samples_per_epoch: int = 1000,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        include_background: bool = True,
        background_fraction: float = 0.125,
        seed: int = 0,
        drop_last: bool = True
    ):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.structures_per_batch = structures_per_batch
        self.poses_per_structure = poses_per_structure
        self.batch_size = structures_per_batch * poses_per_structure
        self.samples_per_epoch = samples_per_epoch
        self.include_background = include_background
        self.background_fraction = background_fraction
        self.seed = seed
        self.drop_last = drop_last
        
        # Calculate per-replica samples
        self.samples_per_replica = samples_per_epoch // num_replicas
        if samples_per_epoch % num_replicas != 0 and not drop_last:
            # Add extra samples to some replicas to cover all data
            if rank < (samples_per_epoch % num_replicas):
                self.samples_per_replica += 1
        
        # Create underlying structured sampler for this replica
        self.structured_sampler = StructuredBatchSampler(
            dataset=dataset,
            structures_per_batch=structures_per_batch,
            poses_per_structure=poses_per_structure,
            samples_per_epoch=self.samples_per_replica,
            include_background=include_background,
            background_fraction=background_fraction,
            seed=seed + rank * 1000,  # Different seed per rank for diversity
            drop_last=drop_last
        )
        
        self.epoch = 0
        
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        self.epoch = epoch
        self.structured_sampler.set_epoch(epoch)
        
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches for this replica."""
        return iter(self.structured_sampler)
    
    def __len__(self) -> int:
        """Return number of batches for this replica."""
        return len(self.structured_sampler)


def create_structured_batch_sampler(
    dataset,
    structures_per_batch: int,
    poses_per_structure: int,
    samples_per_epoch: int,
    distributed: bool = False,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    include_background: bool = True,
    background_fraction: float = 0.125,
    seed: int = 0,
    drop_last: bool = True
) -> Sampler:
    """
    Factory function to create the appropriate structured batch sampler.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from
    structures_per_batch : int
        Number of structures per batch
    poses_per_structure : int
        Number of poses per structure
    samples_per_epoch : int
        Total samples per epoch
    distributed : bool
        Whether to use distributed sampler
    num_replicas : int, optional
        Number of distributed replicas
    rank : int, optional
        Current process rank
    include_background : bool
        Whether to include background samples
    background_fraction : float
        Fraction of batch for background
    seed : int
        Random seed
    drop_last : bool
        Whether to drop incomplete batches
        
    Returns
    -------
    Sampler
        Appropriate sampler instance
    """
    if distributed:
        return DistributedStructuredBatchSampler(
            dataset=dataset,
            structures_per_batch=structures_per_batch,
            poses_per_structure=poses_per_structure,
            samples_per_epoch=samples_per_epoch,
            num_replicas=num_replicas,
            rank=rank,
            include_background=include_background,
            background_fraction=background_fraction,
            seed=seed,
            drop_last=drop_last
        )
    else:
        return StructuredBatchSampler(
            dataset=dataset,
            structures_per_batch=structures_per_batch,
            poses_per_structure=poses_per_structure,
            samples_per_epoch=samples_per_epoch,
            include_background=include_background,
            background_fraction=background_fraction,
            seed=seed,
            drop_last=drop_last
        )
