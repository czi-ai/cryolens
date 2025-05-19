"""
Training utilities for CryoLens.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def get_optimizer(config, model_parameters):
    """Get optimizer based on configuration.
    
    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    model_parameters : iterable
        Model parameters to optimize.
        
    Returns
    -------
    torch.optim.Optimizer
        Optimizer instance.
    """
    return optim.Adam(
        model_parameters, 
        lr=config.learning_rate,
        weight_decay=1e-5
    )


def create_experiment_dir(base_dir: str, experiment_id: Optional[str] = None) -> Path:
    """Create experiment directory with detailed logging and error handling.
    
    In distributed settings, only rank 0 creates the directory, and all ranks use
    the same experiment ID to ensure there's only one directory per run.
    
    Parameters
    ----------
    base_dir : str
        Base directory for experiments.
    experiment_id : str, optional
        Experiment identifier.
        
    Returns
    -------
    Path
        Path to experiment directory.
    """
    # Get rank for logging
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    is_rank0 = rank == 0
    is_distributed = torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if is_distributed else 1
    
    # Print directory creation intent
    print(f"Rank {rank}/{world_size-1}: {'Creating' if is_rank0 else 'Using'} experiment directory in {base_dir}")
    
    try:
        # Create a consistent experiment ID across all ranks
        if experiment_id is None:
            # Only rank 0 creates the timestamp for consistency
            timestamp = None
            if is_rank0 or not is_distributed:
                timestamp = int(time.time())
                experiment_id = f"exp_{timestamp}"
                print(f"Rank {rank}: Generated experiment ID: {experiment_id}")
            
            # For distributed training, broadcast experiment ID from rank 0 to all ranks
            if is_distributed:
                # Use filesystem-based coordination instead of explicit barriers
                # This helps avoid deadlocks in distributed training
                if is_rank0:
                    # Create a file in a known location with the timestamp
                    coordination_dir = Path(base_dir) / "coordination"
                    coordination_dir.mkdir(exist_ok=True, parents=True)
                    coord_file = coordination_dir / "latest_experiment_id.txt"
                    with open(coord_file, 'w') as f:
                        f.write(experiment_id)
                    print(f"Rank {rank}: Wrote experiment ID to coordination file")
                else:
                    # Wait for rank 0 to write the coordination file
                    # Use polling with a timeout to avoid hanging
                    coordination_dir = Path(base_dir) / "coordination"
                    coord_file = coordination_dir / "latest_experiment_id.txt"
                    
                    max_wait_time = 60  # Maximum seconds to wait
                    poll_interval = 0.5  # Check every half second
                    start_wait = time.time()
                    
                    print(f"Rank {rank}: Waiting for coordination file from rank 0")
                    while time.time() - start_wait < max_wait_time:
                        if coord_file.exists():
                            try:
                                with open(coord_file, 'r') as f:
                                    experiment_id = f.read().strip()
                                print(f"Rank {rank}: Read experiment ID from coordination file: {experiment_id}")
                                break
                            except Exception as e:
                                print(f"Rank {rank}: Error reading coordination file, retrying: {e}")
                        time.sleep(poll_interval)
                    
                    # If we failed to get the experiment ID from rank 0, generate one
                    # This is a fallback to avoid hanging the training process
                    if experiment_id is None:
                        print(f"Rank {rank}: Timed out waiting for experiment ID from rank 0, generating local ID")
                        timestamp = int(time.time())
                        experiment_id = f"exp_{timestamp}_rank{rank}"
        else:
            print(f"Rank {rank}: Using provided experiment ID: {experiment_id}")
        
        # Set up paths
        base_path = Path(base_dir)
        experiment_dir = base_path / experiment_id
        print(f"Rank {rank}: {'Creating' if is_rank0 else 'Using'} experiment directory: {experiment_dir}")
        
        # Only rank 0 creates directories
        if is_rank0:
            # Create base directory if it doesn't exist
            try:
                print(f"Rank {rank}: Creating base directory {base_path}")
                base_path.mkdir(parents=True, exist_ok=True)
                print(f"Rank {rank}: Base directory created or already exists")
                
                # Create experiment directory
                start_time = time.time()
                experiment_dir.mkdir(exist_ok=True)
                elapsed = time.time() - start_time
                print(f"Rank {rank}: Experiment directory created successfully in {elapsed:.2f} seconds")
                
                # Verify write access by writing a test file
                test_file = experiment_dir / f"test_write_{rank}_{time.time()}.txt"
                with open(test_file, 'w') as f:
                    f.write("Test write access")
                os.remove(test_file)  # Clean up
                print(f"Rank {rank}: Successfully verified write access to experiment directory")
            except Exception as e:
                print(f"Rank {rank}: Error creating directories: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
                
        # Non-rank 0 processes just need to know the path but don't create directories
        else:
            print(f"Rank {rank}: Using experiment directory without creating: {experiment_dir}")
        
        return experiment_dir
        
    except Exception as e:
        print(f"Rank {rank}: Critical error in create_experiment_dir: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to a local directory that should be writable
        fallback_dir = Path(f"/tmp/cryolens_fallback_{rank}_{int(time.time())}")
        print(f"Rank {rank}: Using fallback directory: {fallback_dir}")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir


def create_default_callbacks(
    checkpoint_dir: Path,
    monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 10,
    extra_callbacks: Optional[List] = None,
    rank: int = 0
) -> List[pl.Callback]:
    """Create default training callbacks.
    
    Parameters
    ----------
    checkpoint_dir : Path
        Directory for checkpoints.
    monitor : str
        Metric to monitor.
    mode : str
        Mode for monitoring ("min" or "max").
    patience : int
        Patience for early stopping.
    extra_callbacks : list, optional
        Additional callbacks.
    rank : int
        Process rank for distributed training.
        
    Returns
    -------
    list
        List of callbacks.
    """
    callbacks = []
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Add checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"model-{{epoch:02d}}-{{{monitor}:.4f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Add early stopping callback
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        verbose=rank == 0,
    )
    callbacks.append(early_stopping)
    
    # Add extra callbacks
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    
    return callbacks


def create_trainer(
    config,
    checkpoint_dir: Path,
    logger=None,
    dist_config=None,
    callbacks=None,
) -> pl.Trainer:
    """Create PyTorch Lightning trainer.
    
    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    checkpoint_dir : Path
        Directory for checkpoints.
    logger : Logger, optional
        Logger for training.
    dist_config : DistributedConfig, optional
        Configuration for distributed training.
    callbacks : list, optional
        Training callbacks.
        
    Returns
    -------
    pl.Trainer
        PyTorch Lightning trainer.
    """
    # Default callbacks if not provided
    if callbacks is None:
        callbacks = create_default_callbacks(checkpoint_dir)
    
    # Trainer arguments
    kwargs = {
        "max_epochs": config.epochs,
        "callbacks": callbacks,
        "logger": logger,
        "enable_checkpointing": True,
        "deterministic": True,
    }
    
    # Distributed training configuration
    if dist_config and dist_config.world_size > 1:
        kwargs.update({
            "accelerator": "gpu",
            "devices": dist_config.devices_per_node,
            "num_nodes": dist_config.num_nodes,
            "strategy": "ddp",
        })
    elif torch.cuda.is_available():
        kwargs["accelerator"] = "gpu"
        kwargs["devices"] = 1
    
    # Create trainer
    trainer = pl.Trainer(**kwargs)
    
    return trainer