"""
Structured batch sampler for ensuring k structures × m poses per batch.

This module provides samplers that guarantee each batch contains exactly
k structures with m poses each, while handling background samples specially.
"""

import torch
import random
import logging
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from typing import Iterator, Optional, List

logger = logging.getLogger(__name__)


class StructuredIndexSampler(Sampler):
    """
    Returns indices in a structured order ensuring k structures × m poses.
    Handles background samples specially since they don't have meaningful poses.
    """
    
    def __init__(self, dataset, structures_per_batch=8, poses_per_structure=4, 
                 background_fraction=0.125, shuffle=True, seed=0):
        """
        Initialize the structured sampler.
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to sample from
        structures_per_batch : int
            Number of structures per batch
        poses_per_structure : int
            Number of poses per structure
        background_fraction : float
            Fraction of batch dedicated to background samples
        shuffle : bool
            Whether to shuffle structures
        seed : int
            Random seed for reproducibility
        """
        self.dataset = dataset
        self.structures_per_batch = structures_per_batch
        self.poses_per_structure = poses_per_structure
        self.batch_size = structures_per_batch * poses_per_structure
        self.background_fraction = background_fraction
        self.shuffle = shuffle
        self.seed = seed
        
        # Build structure-to-indices mapping
        self.structure_indices = {}
        self.background_indices = []
        
        logger.info(f"Building structure index mapping for {len(dataset)} samples...")
        
        for idx in range(len(dataset)):
            # Get the molecule ID for this sample
            sample = dataset[idx]
            if len(sample) >= 2:
                mol_id = sample[1]
                mol_id = mol_id.item() if torch.is_tensor(mol_id) else mol_id
            else:
                logger.warning(f"Sample at index {idx} has unexpected format")
                continue
            
            if mol_id == -1:  # Background
                self.background_indices.append(idx)
            else:
                if mol_id not in self.structure_indices:
                    self.structure_indices[mol_id] = []
                self.structure_indices[mol_id].append(idx)
        
        logger.info(f"Found {len(self.structure_indices)} structures and {len(self.background_indices)} background samples")
        
        # Log structure distribution
        for mol_id, indices in list(self.structure_indices.items())[:5]:
            logger.debug(f"Structure {mol_id}: {len(indices)} samples")
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices ensuring each batch has proper composition."""
        indices = []
        structures = list(self.structure_indices.keys())
        
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(structures)
        
        # Calculate samples per batch
        num_background = int(self.batch_size * self.background_fraction)
        num_structures_needed = (self.batch_size - num_background) // self.poses_per_structure
        
        # Ensure we have at least one structure if no background
        if num_structures_needed == 0:
            num_structures_needed = 1
            num_background = self.batch_size - self.poses_per_structure
        
        logger.debug(f"Batch composition: {num_structures_needed} structures, {num_background} background samples")
        
        # Generate batches
        structure_idx = 0
        batch_count = 0
        
        while structure_idx + num_structures_needed <= len(structures):
            batch_indices = []
            
            # Add structured samples
            for i in range(num_structures_needed):
                struct_id = structures[structure_idx + i]
                struct_indices = self.structure_indices[struct_id]
                
                if len(struct_indices) >= self.poses_per_structure:
                    # Sample without replacement if we have enough
                    selected = random.sample(struct_indices, self.poses_per_structure)
                else:
                    # Sample with replacement if we don't have enough
                    selected = random.choices(struct_indices, k=self.poses_per_structure)
                batch_indices.extend(selected)
            
            structure_idx += num_structures_needed
            
            # Add background samples to fill the batch
            if num_background > 0 and self.background_indices:
                if len(self.background_indices) >= num_background:
                    bg_samples = random.sample(self.background_indices, num_background)
                else:
                    bg_samples = random.choices(self.background_indices, k=num_background)
                batch_indices.extend(bg_samples)
            
            # Ensure exactly batch_size indices
            while len(batch_indices) < self.batch_size:
                # Pad with additional samples if needed
                if self.background_indices:
                    batch_indices.append(random.choice(self.background_indices))
                elif indices:
                    # Use recently added samples if no background available
                    batch_indices.append(random.choice(indices[-self.poses_per_structure:]))
                else:
                    # Last resort: use any available sample
                    all_indices = []
                    for struct_indices in self.structure_indices.values():
                        all_indices.extend(struct_indices)
                    if all_indices:
                        batch_indices.append(random.choice(all_indices))
                    else:
                        break
            
            # Trim to exact batch size if we added too many
            batch_indices = batch_indices[:self.batch_size]
            
            # Shuffle within batch to mix structures and background
            if self.shuffle:
                random.shuffle(batch_indices)
            
            indices.extend(batch_indices)
            batch_count += 1
            
            # Log first few batches for debugging
            if batch_count <= 2:
                logger.debug(f"Batch {batch_count}: {len(batch_indices)} samples")
        
        logger.info(f"Generated {batch_count} batches with {len(indices)} total indices")
        return iter(indices)
    
    def __len__(self) -> int:
        """Return the total number of samples that will be yielded."""
        if not self.structure_indices:
            return 0
            
        # Calculate how many complete batches we can make
        num_structures = len(self.structure_indices)
        num_background = int(self.batch_size * self.background_fraction)
        num_structures_per_batch = (self.batch_size - num_background) // self.poses_per_structure
        
        if num_structures_per_batch == 0:
            return 0
            
        num_batches = num_structures // num_structures_per_batch
        return num_batches * self.batch_size


class DistributedStructuredSampler(DistributedSampler):
    """
    Distributed version of StructuredIndexSampler.
    
    This sampler ensures that each GPU gets its own set of complete batches
    with the proper k×m structure while coordinating across GPUs.
    """
    
    def __init__(self, dataset, structures_per_batch=8, poses_per_structure=4,
                 background_fraction=0.125, num_replicas=None, rank=None, 
                 shuffle=True, seed=0, drop_last=False, samples_per_epoch=None):
        """
        Initialize the distributed structured sampler.
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to sample from
        structures_per_batch : int
            Number of structures per batch
        poses_per_structure : int
            Number of poses per structure
        background_fraction : float
            Fraction of batch dedicated to background samples
        num_replicas : int, optional
            Number of processes participating in distributed training
        rank : int, optional
            Rank of the current process
        shuffle : bool
            Whether to shuffle the data
        seed : int
            Random seed
        drop_last : bool
            Whether to drop the last incomplete batch
        samples_per_epoch : int, optional
            Limit the number of samples per epoch
        """
        # Initialize parent DistributedSampler
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, 
                        shuffle=shuffle, seed=seed, drop_last=drop_last)
        
        self.structures_per_batch = structures_per_batch
        self.poses_per_structure = poses_per_structure
        self.batch_size = structures_per_batch * poses_per_structure
        self.background_fraction = background_fraction
        self.samples_per_epoch = samples_per_epoch
        
        # Create the base structured sampler
        self.base_sampler = StructuredIndexSampler(
            dataset=dataset,
            structures_per_batch=structures_per_batch,
            poses_per_structure=poses_per_structure,
            background_fraction=background_fraction,
            shuffle=False,  # We'll handle shuffling at the distributed level
            seed=seed
        )
        
        # Calculate samples per replica
        if samples_per_epoch is not None:
            # Ensure it's divisible by batch_size and num_replicas
            batches_per_replica = (samples_per_epoch // self.batch_size) // self.num_replicas
            self.num_samples = batches_per_replica * self.batch_size
        else:
            total_samples = len(self.base_sampler)
            self.num_samples = (total_samples + self.num_replicas - 1) // self.num_replicas
            # Round to nearest batch size
            self.num_samples = (self.num_samples // self.batch_size) * self.batch_size
        
        self.total_size = self.num_samples * self.num_replicas
        
        logger.info(f"Rank {self.rank}: Distributed structured sampler initialized")
        logger.info(f"  Structures per batch: {structures_per_batch}")
        logger.info(f"  Poses per structure: {poses_per_structure}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Samples per replica: {self.num_samples}")
        logger.info(f"  Total samples across all replicas: {self.total_size}")
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices for this rank ensuring proper batch structure."""
        # Set epoch-based seed for shuffling
        if self.shuffle:
            random.seed(self.seed + self.epoch)
        
        # Get all indices from the base sampler
        all_indices = list(self.base_sampler)
        
        # Ensure we have enough indices
        if len(all_indices) < self.total_size:
            # Pad with repeated indices if necessary
            num_extra = self.total_size - len(all_indices)
            extra_indices = []
            while len(extra_indices) < num_extra:
                extra_indices.extend(all_indices[:min(len(all_indices), num_extra - len(extra_indices))])
            all_indices.extend(extra_indices[:num_extra])
        
        # Shuffle complete batches if needed
        if self.shuffle:
            # Group indices into batches
            batches = [all_indices[i:i+self.batch_size] 
                      for i in range(0, len(all_indices), self.batch_size)]
            # Shuffle the batches (not within batches)
            random.shuffle(batches)
            # Flatten back to indices
            all_indices = [idx for batch in batches for idx in batch]
        
        # Select indices for this rank
        indices = all_indices[self.rank:self.total_size:self.num_replicas]
        
        # Ensure we return exactly num_samples
        if len(indices) > self.num_samples:
            indices = indices[:self.num_samples]
        elif len(indices) < self.num_samples:
            # Pad if necessary
            while len(indices) < self.num_samples:
                indices.append(indices[len(indices) % len(all_indices)])
        
        return iter(indices)
    
    def __len__(self) -> int:
        """Return the number of samples for this rank."""
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for shuffling."""
        super().set_epoch(epoch)
        # Also update the base sampler's seed
        self.base_sampler.seed = self.seed + epoch
