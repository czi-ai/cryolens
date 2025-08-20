"""
Structured batch sampler for ensuring K structures × M poses per batch.

This module implements a batch sampler that ensures each batch contains
samples from K different structures with M poses each, following the DINOv3
approach for better disentanglement of content and pose.
"""

import torch
import numpy as np
import random
import logging
from typing import List, Optional, Iterator
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class StructuredBatchSampler(Sampler):
    """Batch sampler that ensures K structures × M poses per batch.
    
    This sampler creates batches where each batch contains exactly K different
    structures with M samples (poses) from each structure. Background samples
    are handled specially as they don't have meaningful poses.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from. Must have a method to get structure ID.
    structures_per_batch : int
        Number of different structures per batch (K).
    poses_per_structure : int
        Number of poses per structure in each batch (M).
    drop_last : bool
        Whether to drop the last incomplete batch.
    shuffle : bool
        Whether to shuffle the structures before sampling.
    seed : int
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        dataset,
        structures_per_batch: int = 8,
        poses_per_structure: int = 4,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 171717
    ):
        self.dataset = dataset
        self.structures_per_batch = structures_per_batch
        self.poses_per_structure = poses_per_structure
        self.batch_size = structures_per_batch * poses_per_structure
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        # Set random seed
        self.rng = np.random.RandomState(seed)
        
        # Build structure index mapping
        self._build_structure_index()
        
    def _build_structure_index(self):
        """Build mapping from structure IDs to sample indices."""
        self.structure_to_indices = {}
        
        # Iterate through dataset to build index
        for idx in range(len(self.dataset)):
            # Get structure ID for this sample
            struct_id = self._get_structure_id(idx)
            
            # Skip background samples (ID = -1)
            if struct_id == -1:
                continue
                
            if struct_id not in self.structure_to_indices:
                self.structure_to_indices[struct_id] = []
            self.structure_to_indices[struct_id].append(idx)
        
        # Filter out structures with too few samples
        min_samples = self.poses_per_structure
        valid_structures = {}
        for struct_id, indices in self.structure_to_indices.items():
            if len(indices) >= min_samples:
                valid_structures[struct_id] = indices
            else:
                logger.warning(f"Structure {struct_id} has only {len(indices)} samples, "
                             f"need at least {min_samples}. Excluding from sampling.")
        
        self.structure_to_indices = valid_structures
        self.structure_ids = list(self.structure_to_indices.keys())
        
        logger.info(f"StructuredBatchSampler initialized with {len(self.structure_ids)} valid structures")
        
    def _get_structure_id(self, idx: int) -> int:
        """Get structure ID for a given dataset index.
        
        Parameters
        ----------
        idx : int
            Dataset index.
            
        Returns
        -------
        int
            Structure ID, or -1 for background samples.
        """
        # Try to use dataset's get_structure_id method if available
        if hasattr(self.dataset, 'get_structure_id'):
            return self.dataset.get_structure_id(idx)
        
        # Otherwise, load the sample and extract mol_id
        sample = self.dataset[idx]
        if len(sample) >= 2:
            mol_id = sample[1]
            if isinstance(mol_id, torch.Tensor):
                return mol_id.item()
            return mol_id
        
        return -1
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of indices.
        
        Yields
        ------
        List[int]
            Batch of dataset indices.
        """
        # Shuffle structures if requested
        if self.shuffle:
            structure_order = self.rng.permutation(self.structure_ids).tolist()
        else:
            structure_order = self.structure_ids.copy()
        
        # Generate batches
        batches = []
        for i in range(0, len(structure_order), self.structures_per_batch):
            # Get structures for this batch
            batch_structures = structure_order[i:i + self.structures_per_batch]
            
            # Skip incomplete batch if drop_last is True
            if len(batch_structures) < self.structures_per_batch and self.drop_last:
                break
            
            # Sample poses for each structure
            batch_indices = []
            for struct_id in batch_structures:
                indices = self.structure_to_indices[struct_id]
                
                # Sample with replacement if necessary
                if len(indices) >= self.poses_per_structure:
                    sampled = self.rng.choice(
                        indices, 
                        size=self.poses_per_structure, 
                        replace=False
                    ).tolist()
                else:
                    # Sample with replacement if not enough unique samples
                    sampled = self.rng.choice(
                        indices,
                        size=self.poses_per_structure,
                        replace=True
                    ).tolist()
                
                batch_indices.extend(sampled)
            
            # Shuffle within batch to mix structures
            if self.shuffle:
                self.rng.shuffle(batch_indices)
            
            batches.append(batch_indices)
        
        # Yield batches
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        """Get number of batches.
        
        Returns
        -------
        int
            Number of batches that will be generated.
        """
        n_batches = len(self.structure_ids) // self.structures_per_batch
        if not self.drop_last and len(self.structure_ids) % self.structures_per_batch > 0:
            n_batches += 1
        return n_batches


class DistributedStructuredSampler(DistributedSampler):
    """Distributed version of StructuredBatchSampler.
    
    This sampler ensures that structured batches work correctly in distributed
    training by dividing structures across workers.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from.
    num_replicas : int
        Number of processes in distributed training.
    rank : int
        Rank of the current process.
    structures_per_batch : int
        Number of different structures per batch.
    poses_per_structure : int
        Number of poses per structure in each batch.
    shuffle : bool
        Whether to shuffle structures.
    seed : int
        Random seed.
    drop_last : bool
        Whether to drop incomplete batches.
    """
    
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        structures_per_batch: int = 8,
        poses_per_structure: int = 4,
        shuffle: bool = True,
        seed: int = 171717,
        drop_last: bool = False
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        
        self.structures_per_batch = structures_per_batch
        self.poses_per_structure = poses_per_structure
        self.batch_size = structures_per_batch * poses_per_structure
        
        # Build structure index
        self._build_structure_index()
        
    def _build_structure_index(self):
        """Build mapping from structure IDs to sample indices."""
        self.structure_to_indices = {}
        
        # Iterate through dataset
        for idx in range(len(self.dataset)):
            struct_id = self._get_structure_id(idx)
            
            # Skip background
            if struct_id == -1:
                continue
            
            if struct_id not in self.structure_to_indices:
                self.structure_to_indices[struct_id] = []
            self.structure_to_indices[struct_id].append(idx)
        
        # Filter structures with enough samples
        min_samples = self.poses_per_structure
        valid_structures = {}
        for struct_id, indices in self.structure_to_indices.items():
            if len(indices) >= min_samples:
                valid_structures[struct_id] = indices
        
        self.structure_to_indices = valid_structures
        
        # Divide structures among workers
        all_structures = list(self.structure_to_indices.keys())
        
        # Ensure each worker gets equal number of structures
        structures_per_worker = len(all_structures) // self.num_replicas
        extra = len(all_structures) % self.num_replicas
        
        start_idx = self.rank * structures_per_worker + min(self.rank, extra)
        end_idx = start_idx + structures_per_worker + (1 if self.rank < extra else 0)
        
        self.structure_ids = all_structures[start_idx:end_idx]
        
        logger.info(f"Rank {self.rank}: Assigned {len(self.structure_ids)} structures "
                   f"out of {len(all_structures)} total")
    
    def _get_structure_id(self, idx: int) -> int:
        """Get structure ID for a dataset index."""
        if hasattr(self.dataset, 'get_structure_id'):
            return self.dataset.get_structure_id(idx)
        
        sample = self.dataset[idx]
        if len(sample) >= 2:
            mol_id = sample[1]
            if isinstance(mol_id, torch.Tensor):
                return mol_id.item()
            return mol_id
        
        return -1
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices for this worker.
        
        Yields
        ------
        int
            Dataset index.
        """
        # Set epoch seed
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            structure_order = rng.permutation(self.structure_ids).tolist()
        else:
            structure_order = self.structure_ids.copy()
        
        # Generate structured batches
        indices = []
        for i in range(0, len(structure_order), self.structures_per_batch):
            batch_structures = structure_order[i:i + self.structures_per_batch]
            
            if len(batch_structures) < self.structures_per_batch and self.drop_last:
                break
            
            # Sample poses for each structure
            for struct_id in batch_structures:
                struct_indices = self.structure_to_indices[struct_id]
                
                if self.shuffle:
                    if len(struct_indices) >= self.poses_per_structure:
                        sampled = rng.choice(
                            struct_indices,
                            size=self.poses_per_structure,
                            replace=False
                        )
                    else:
                        sampled = rng.choice(
                            struct_indices,
                            size=self.poses_per_structure,
                            replace=True
                        )
                else:
                    sampled = struct_indices[:self.poses_per_structure]
                
                indices.extend(sampled)
        
        # Shuffle all indices if requested
        if self.shuffle:
            rng.shuffle(indices)
        
        # Yield indices one by one
        for idx in indices:
            yield idx
    
    def __len__(self) -> int:
        """Get total number of samples for this worker."""
        n_batches = len(self.structure_ids) // self.structures_per_batch
        if not self.drop_last and len(self.structure_ids) % self.structures_per_batch > 0:
            n_batches += 1
        return n_batches * self.batch_size
