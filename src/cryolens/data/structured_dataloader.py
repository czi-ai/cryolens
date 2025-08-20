"""
Helper module to integrate structured sampler into alternating curriculum training.
"""

import torch
from torch.utils.data import DataLoader
from cryolens.data.structured_sampler import StructuredIndexSampler, DistributedStructuredSampler
import logging

logger = logging.getLogger(__name__)


def create_structured_dataloader(dataset, config, dist_config=None, 
                                structures_per_batch=8, poses_per_structure=4,
                                background_fraction=0.125, samples_per_epoch=None):
    """
    Create a DataLoader with structured batch sampler.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to create loader for
    config : TrainingConfig
        Training configuration
    dist_config : DistributedConfig, optional
        Distributed training configuration
    structures_per_batch : int
        Number of structures per batch
    poses_per_structure : int
        Number of poses per structure
    background_fraction : float
        Fraction of batch for background samples
    samples_per_epoch : int, optional
        Limit samples per epoch
        
    Returns
    -------
    DataLoader
        Configured dataloader with structured sampling
    """
    
    # Calculate batch size from structures and poses
    batch_size = structures_per_batch * poses_per_structure
    
    if dist_config:
        # Create distributed structured sampler
        sampler = DistributedStructuredSampler(
            dataset=dataset,
            structures_per_batch=structures_per_batch,
            poses_per_structure=poses_per_structure,
            background_fraction=background_fraction,
            num_replicas=dist_config.world_size,
            rank=dist_config.node_rank * dist_config.devices_per_node + dist_config.local_rank,
            shuffle=True,
            seed=171717,
            drop_last=True,
            samples_per_epoch=samples_per_epoch
        )
        logger.info(f"Created distributed structured sampler with {structures_per_batch}×{poses_per_structure} composition")
    else:
        # Create single-GPU structured sampler
        sampler = StructuredIndexSampler(
            dataset=dataset,
            structures_per_batch=structures_per_batch,
            poses_per_structure=poses_per_structure,
            background_fraction=background_fraction,
            shuffle=True,
            seed=171717
        )
        logger.info(f"Created structured sampler with {structures_per_batch}×{poses_per_structure} composition")
    
    # Create dataloader with the sampler
    # Note: batch_size must match the sampler's batch composition
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,  # Reduced for stability
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=1
    )
    
    logger.info(f"Created structured dataloader with {len(dataloader)} batches of size {batch_size}")
    
    return dataloader
