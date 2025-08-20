"""
Custom collate function for handling pose data in DataLoader.
"""

import torch
from torch.utils.data.dataloader import default_collate


def pose_aware_collate_fn(batch):
    """
    Custom collate function that properly handles pose data.
    
    When poses are included, ensures they are properly batched as [batch_size, 4]
    instead of being flattened to [batch_size * 4].
    
    Parameters
    ----------
    batch : list
        List of samples from dataset
        
    Returns
    -------
    tuple
        Batched data with properly shaped poses
    """
    if not batch:
        return batch
    
    # Check the format of the first sample
    first_sample = batch[0]
    
    if len(first_sample) == 2:
        # Standard format: (volume, mol_id)
        return default_collate(batch)
    
    elif len(first_sample) == 3:
        # Format with poses: (volume, mol_id, pose)
        volumes = []
        mol_ids = []
        poses = []
        
        for sample in batch:
            volume, mol_id, pose = sample
            volumes.append(volume)
            mol_ids.append(mol_id)
            
            # Ensure pose is a tensor
            if not isinstance(pose, torch.Tensor):
                pose = torch.tensor(pose, dtype=torch.float32)
            
            # Ensure pose has the right shape [4]
            if pose.numel() == 4:
                pose = pose.view(4)
            else:
                # If pose doesn't have 4 elements, create a default one
                print(f"WARNING: Pose has {pose.numel()} elements, expected 4. Using default.")
                pose = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)  # Default: no rotation
            
            poses.append(pose)
        
        # Stack into batches
        batched_volumes = torch.stack(volumes)
        batched_mol_ids = torch.stack(mol_ids)
        batched_poses = torch.stack(poses)  # This will create [batch_size, 4]
        
        return batched_volumes, batched_mol_ids, batched_poses
    
    elif len(first_sample) == 4:
        # Format with poses and source: (volume, mol_id, pose, source)
        volumes = []
        mol_ids = []
        poses = []
        sources = []
        
        for sample in batch:
            volume, mol_id, pose, source = sample
            volumes.append(volume)
            mol_ids.append(mol_id)
            sources.append(source)
            
            # Ensure pose is a tensor
            if not isinstance(pose, torch.Tensor):
                pose = torch.tensor(pose, dtype=torch.float32)
            
            # Ensure pose has the right shape [4]
            if pose.numel() == 4:
                pose = pose.view(4)
            else:
                print(f"WARNING: Pose has {pose.numel()} elements, expected 4. Using default.")
                pose = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
            
            poses.append(pose)
        
        # Stack into batches
        batched_volumes = torch.stack(volumes)
        batched_mol_ids = torch.stack(mol_ids)
        batched_poses = torch.stack(poses)
        
        # Sources are strings, so we return them as a list
        return batched_volumes, batched_mol_ids, batched_poses, sources
    
    else:
        # Unknown format, fall back to default
        print(f"WARNING: Unknown batch format with {len(first_sample)} elements, using default collate")
        return default_collate(batch)
