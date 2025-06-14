"""
Trainer implementations for CryoLens.
"""

from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import Logger
from .loggers import SafeCSVLogger
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta
import traceback
from .environment import create_lightning_environment

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

from cryolens.training.utils import get_optimizer
from cryolens.visualization import VisualizationCallback

@dataclass
class DistributedConfig:
    """Configuration for distributed training environment."""
    world_size: int 
    num_nodes: int
    node_rank: int
    local_rank: int
    master_addr: str
    master_port: int
    devices_per_node: int

class MinimalCheckpoint(ModelCheckpoint):
    """Minimal checkpoint handler with Lightning-compatible saving."""
    
    def __init__(
        self,
        dirpath: str,
        monitor: str = "train_loss",
        save_top_k: int = 3,
        save_last: bool = True,
        every_n_epochs: int = 1,
        save_weights_only: bool = False,
        **kwargs
    ):
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            save_top_k=save_top_k,
            save_last=save_last,
            every_n_epochs=every_n_epochs,
            save_weights_only=save_weights_only,
            **kwargs
        )
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Save checkpoint based on configuration."""
        if not self._should_save_on_epoch(trainer):
            return
            
        # Only rank 0 saves checkpoints
        if self.rank != 0:
            return
            
        try:
            import os
            print(f"\nRank 0: Starting checkpoint save for epoch {trainer.current_epoch}...")
            
            # Create checkpoint dir if it doesn't exist
            os.makedirs(self.dirpath, exist_ok=True)
            
            # Create checkpoint dict with PyTorch Lightning metadata
            import pytorch_lightning as pl
            checkpoint = {
                'state_dict': pl_module.state_dict(),
                'epoch': trainer.current_epoch,
                'global_step': trainer.global_step,
                'pytorch-lightning_version': pl.__version__,
                'optimizer_states': [opt.state_dict() for opt in trainer.optimizers] if trainer.optimizers else [],
                'lr_schedulers': [scheduler['scheduler'].state_dict() for scheduler in trainer.lr_scheduler_configs] if trainer.lr_scheduler_configs else [],
                'hyper_parameters': pl_module.hparams if hasattr(pl_module, 'hparams') else {},
            }
            
            # Generate base filename without extension
            base_filename = f"model_epoch_{trainer.current_epoch:03d}"
            
            # Add current metric value if monitoring
            if self.monitor is not None:
                metrics = trainer.callback_metrics
                if self.monitor in metrics:
                    metric_val = metrics[self.monitor]
                    if isinstance(metric_val, torch.Tensor):
                        metric_val = metric_val.item()
                    base_filename += f"_{self.monitor}_{metric_val:.3f}"
                    checkpoint['monitor_val'] = metric_val
            
            # Save checkpoint
            filepath = os.path.join(self.dirpath, f"{base_filename}.pt")
            torch.save(checkpoint, filepath)
            print(f"Rank 0: Saved checkpoint to {filepath}")
            
            # Save last checkpoint if configured
            if self.save_last:
                last_filepath = os.path.join(self.dirpath, "last.pt")
                torch.save(checkpoint, last_filepath)
                print(f"Rank 0: Saved last checkpoint to {last_filepath}")
            
        except Exception as e:
            import traceback
            print(f"\nRank 0: Error during checkpoint save:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
    
    def _should_save_on_epoch(self, trainer) -> bool:
        """Determine if checkpoint should be saved this epoch."""
        return (
            trainer.current_epoch % self.every_n_epochs == 0 or
            trainer.current_epoch == trainer.max_epochs - 1
        )
        
    def _update_best_and_save(self, filepath: str, current_score: float):
        """Update best checkpoints list."""
        if len(self.best_k_models) < self.save_top_k:
            self.best_k_models[filepath] = current_score
        else:
            worst_score = min(self.best_k_models.values())
            if current_score > worst_score:
                worst_filepath = min(self.best_k_models.items(), key=lambda x: x[1])[0]
                if os.path.exists(worst_filepath):
                    os.remove(worst_filepath)
                del self.best_k_models[worst_filepath]
                self.best_k_models[filepath] = current_score


def create_default_callbacks(
    checkpoint_dir: Path,
    monitor: str = "train_loss",
    save_top_k: int = 3,
    save_last: bool = True,
    extra_callbacks: Optional[List[Callback]] = None,
    rank: int = 0
) -> List[Callback]:
    """Create standard set of callbacks with optional additions."""
    callbacks = []
    
    # Ensure checkpoint directory exists - only on rank 0
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks.append(MinimalCheckpoint(
        dirpath=checkpoint_dir,
        monitor=monitor,
        save_top_k=save_top_k,
        save_last=save_last,
        every_n_epochs=5,
        save_weights_only=True
    ))

    # Add visualization callback with rank info to ensure only rank 0 saves
    callbacks.append(VisualizationCallback(every_n_epochs=10, rank=rank))
    
    # Add extra callbacks for all ranks
    if extra_callbacks:
        filtered_extras = [cb for cb in extra_callbacks if cb is not None]
        callbacks.extend(filtered_extras)
    
    # Final safety check to remove any None values
    callbacks = [cb for cb in callbacks if cb is not None]
    
    return callbacks


class DensityVAE(pl.LightningModule):
    """Variational Autoencoder for density maps with distributed training support.
    
    Parameters
    ----------
    config
        Training configuration.
    encoder : nn.Module
        Encoder model.
    decoder : nn.Module
        Decoder model.
    lookup : torch.Tensor
        Similarity lookup matrix for molecules.
    """
    
    def __init__(
        self,
        config,
        encoder: nn.Module,
        decoder: nn.Module,
        lookup: torch.Tensor,
    ):
        super().__init__()
        
        # Save configuration
        self.config = config
        self.save_hyperparameters(ignore=['encoder', 'decoder', 'lookup'])
        
        # Initialize models
        self.encoder = encoder
        self.decoder = decoder
        
        # Register lookup tensor as buffer
        self.register_buffer('similarity_lookup', lookup, persistent=True)
        
        # Create VAE model
        self.model = self._create_model()
        
        # Initialize loss functions
        self.reconstruction_loss = None  # Will be initialized in setup()
        self.similarity_loss = None      # Will be initialized in setup()

    def _create_model(self) -> nn.Module:
        """Create VAE model.
        
        Returns
        -------
        nn.Module
            VAE model.
        """
        # Import locally to avoid circular imports
        from cryolens.models.vne.vae import AffinityVAE
        
        return AffinityVAE(
            encoder=self.encoder,
            decoder=self.decoder,
            latent_dims=self.config.latent_dims,
            pose_channels=self.config.pose_dims,
        )

    def setup(self, stage=None):
        """Initialize components that depend on the device."""
        # Import locally to avoid circular imports
        from cryolens.models.vne.vae import AffinityCosineLoss
        from cryolens.training.losses import ContrastiveAffinityLoss, MissingWedgeLoss
        
        # Initialize reconstruction loss
        if getattr(self.config, 'reconstruction_loss', 'mse') == "missingwedgeloss":
            self.reconstruction_loss = MissingWedgeLoss(
                volume_size=self.config.box_size,
                wedge_angle=90.0,
                weight_factor=self.config.wedge_weight_factor
            )
        else:
            self.reconstruction_loss = nn.MSELoss(reduction="mean")

        # Initialize similarity loss
        if getattr(self.config, 'affinity_loss', 'contrastive') == "cosine":
            self.similarity_loss = AffinityCosineLoss(
                lookup=self.similarity_lookup,
                device=self.device,
                latent_ratio=self.config.latent_ratio
            )
        else:
            self.similarity_loss = ContrastiveAffinityLoss(
                lookup=self.similarity_lookup,
                device=self.device
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (reconstructed output, latent vector, pose, mean, log variance)
        """
        # Ensure model is on correct device
        device = x.device
        if next(self.model.parameters()).device != device:
            self.model = self.model.to(device)
            
        # Ensure input is contiguous
        if not x.is_contiguous():
            x = x.contiguous()
            
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step.
        
        Parameters
        ----------
        batch : tuple
            (image, molecule_id)
        batch_idx : int
            Batch index.
            
        Returns
        -------
        torch.Tensor
            Loss value.
        """
        # Get input data
        img, mol_id = batch
        
        # Skip batches with invalid molecules
        if torch.any(mol_id == -1):
            return None
        
        # Forward pass
        output, z, pose, mu, log_var = self(img)
        
        # Calculate losses
        r_loss = self.reconstruction_loss(output, img)
        
        # KL divergence
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld = self.config.beta * kld
        
        # Similarity loss
        s_loss = self.config.gamma * self.similarity_loss(mol_id, mu)
        
        # Total loss
        loss = r_loss + s_loss + kld
        
        # Log losses
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("reconstruction_loss", r_loss, on_step=True, on_epoch=True, logger=True)
        self.log("kld_loss", kld, on_step=True, on_epoch=True, logger=True)
        self.log("similarity_loss", s_loss, on_step=True, on_epoch=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer.
        
        Returns
        -------
        torch.optim.Optimizer
            Optimizer instance.
        """
        return get_optimizer(self.config, self.parameters())

def create_trainer(
    checkpoint_dir: Path,
    logger: Logger,
    num_epochs: int = 1000,
    dist_config: Optional[DistributedConfig] = None,
    callbacks: Optional[List[Callback]] = None
) -> Trainer:
    """Create trainer with proper distributed setup."""
    
    rank = dist_config.node_rank if dist_config else 0
    local_rank = dist_config.local_rank if dist_config else 0
    print(f"Rank {rank}:{local_rank}: Creating Lightning Trainer")
    
    # Only rank 0 creates the checkpoint directory
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Rank {rank}:{local_rank}: Created checkpoint directory: {checkpoint_dir}")
    else:
        print(f"Rank {rank}:{local_rank}: Using checkpoint directory without creating: {checkpoint_dir}")
    
    # Use provided callbacks or create defaults
    if callbacks is None:
        callback_dir = checkpoint_dir / "checkpoints"
        callbacks = create_default_callbacks(
            callback_dir,
            rank=rank
        )
        print(f"Rank {rank}:{local_rank}: Created default callbacks")
    else:
        print(f"Rank {rank}:{local_rank}: Using {len(callbacks)} provided callbacks")
    
    # Validate device settings to prevent index out of range errors
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Rank {rank}:{local_rank}: Detected {device_count} CUDA devices")
        
        if local_rank >= device_count:
            print(f"WARNING: Rank {rank}:{local_rank}: local_rank is {local_rank} but only {device_count} devices available!")
            print(f"This will likely cause 'list index out of range' errors with PyTorch Lightning.")
    
    # Basic trainer configuration with improved settings for distributed stability
    trainer_kwargs = {
        "max_epochs": num_epochs,
        "callbacks": callbacks,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": "auto",
        "logger": logger if (not dist_config or dist_config.node_rank == 0) else None,
        "default_root_dir": checkpoint_dir,
        "precision": "16-mixed",
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "detect_anomaly": False,     # Disable anomaly detection which can cause hangs
        "reload_dataloaders_every_n_epochs": 0,  # Don't reload dataloaders which can cause hangs
        "num_sanity_val_steps": 0,   # Skip validation sanity checks
        "sync_batchnorm": False,     # Avoid using sync_batchnorm which can cause deadlocks
        "max_steps": -1,            # No step limit
        "gradient_clip_val": 1.0,    # Add gradient clipping for stability
        "benchmark": True,          # Enable cuDNN benchmarking for speed
        "deterministic": False,     # Disable deterministic mode for better performance
    }
    
    print(f"Rank {rank}:{local_rank}: Base trainer config: accelerator={trainer_kwargs['accelerator']}, precision={trainer_kwargs['precision']}")
    
    if dist_config:
        # Add verbose logging for distributed setup
        print(f"Rank {rank}:{local_rank}: Configuring distributed training with {dist_config.world_size} processes")
        print(f"Rank {rank}:{local_rank}: Distributed config - world_size: {dist_config.world_size}, num_nodes: {dist_config.num_nodes}, devices_per_node: {dist_config.devices_per_node}")
        
        # CRITICAL FIX: Properly configure devices and num_nodes for PyTorch Lightning
        if trainer_kwargs['accelerator'] == 'gpu':
            device_count = torch.cuda.device_count()
            
            # Set devices per node (this is what Lightning expects)
            trainer_kwargs['devices'] = dist_config.devices_per_node
            
            # CRITICAL: num_nodes must match the actual number of nodes
            trainer_kwargs['num_nodes'] = dist_config.num_nodes
            
            print(f"Rank {rank}:{local_rank}: Lightning config - devices per node: {trainer_kwargs['devices']}, num_nodes: {trainer_kwargs['num_nodes']}")
            print(f"Rank {rank}:{local_rank}: Expected total processes: {trainer_kwargs['devices'] * trainer_kwargs['num_nodes']} vs actual world_size: {dist_config.world_size}")
            
            # Validation check
            expected_processes = trainer_kwargs['devices'] * trainer_kwargs['num_nodes']
            if expected_processes != dist_config.world_size:
                print(f"ERROR: Lightning configuration mismatch!")
                print(f"  devices * num_nodes = {trainer_kwargs['devices']} * {trainer_kwargs['num_nodes']} = {expected_processes}")
                print(f"  but world_size = {dist_config.world_size}")
                
                # Try to fix the configuration automatically
                if dist_config.world_size % device_count == 0:
                    # Calculate correct number of nodes
                    correct_num_nodes = dist_config.world_size // device_count
                    print(f"Auto-correcting: setting num_nodes to {correct_num_nodes}")
                    trainer_kwargs['num_nodes'] = correct_num_nodes
                    trainer_kwargs['devices'] = device_count
                else:
                    print(f"Cannot auto-correct. World size {dist_config.world_size} is not evenly divisible by device count {device_count}")
                    # Fall back to single node mode
                    print(f"Falling back to single node mode with {dist_config.world_size} devices")
                    trainer_kwargs['devices'] = dist_config.world_size
                    trainer_kwargs['num_nodes'] = 1
        
        # Create a modified PyTorch DDP strategy optimized for reliable training
        print(f"Rank {rank}:{local_rank}: Creating DDP strategy")
                
        # Configure DDP strategy with improved settings to avoid deadlocks
        ddp_kwargs = {
            "find_unused_parameters": False,  # Set to False to avoid potential NCCL deadlocks
            "process_group_backend": "nccl",
            "timeout": timedelta(minutes=3),   # Shorter timeout to fail fast if there's a problem
            "static_graph": False,           # Disable static graph optimization which can cause issues
            "gradient_as_bucket_view": True,  # Memory optimization
        }
        
        # Create Lightning environment
        cluster_env = create_lightning_environment(dist_config)
        
        trainer_kwargs.update({
            "strategy": DDPStrategy(**ddp_kwargs),
            "plugins": [cluster_env]
        })
        
        print(f"Rank {rank}:{local_rank}: Final trainer config for distributed:")
        print(f"  devices: {trainer_kwargs['devices']}")
        print(f"  num_nodes: {trainer_kwargs['num_nodes']}")
        print(f"  strategy: DDP")
        print(f"  world_size: {dist_config.world_size}")
    else:
        print(f"Rank {rank}:{local_rank}: Running in single-GPU mode")
    
    trainer = Trainer(**trainer_kwargs)
    print(f"Rank {rank}:{local_rank}: Trainer created successfully")
    return trainer

def monitor_training(trainer, dataloader, model, max_epochs):
    """Monitor training progress with improved error handling and timeout detection
    
    This version includes more robust error detection and handling.
    """
    import threading
    import time
    
    # Setup a watchdog timer to detect hangs
    hang_detected = threading.Event()
    
    def watchdog():
        # Check every 10 minutes if training is making progress
        check_interval = 600  # 10 minutes
        last_epoch = -1
        last_step = -1
        inactive_count = 0
        
        while not hang_detected.is_set():
            time.sleep(check_interval)
            
            # Check if training is making progress
            current_epoch = trainer.current_epoch
            current_step = trainer.global_step
            
            if current_epoch == last_epoch and current_step == last_step:
                inactive_count += 1
                print(f"\nWARNING: Training appears stalled at epoch {current_epoch}, step {current_step} "
                      f"for {inactive_count * check_interval/60:.1f} minutes")
                
                # If no progress for 30 minutes, raise alarm
                if inactive_count >= 3:
                    print(f"\nCRITICAL: Training has been stalled for {inactive_count * check_interval/60:.1f} minutes. "
                          f"This may indicate a deadlock or communication issue.")
                    hang_detected.set()
            else:
                inactive_count = 0
            
            last_epoch = current_epoch
            last_step = current_step
    
    # Start watchdog thread if we're the master process
    if torch.distributed.get_rank() == 0:
        watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        watchdog_thread.start()
    
    try:
        # Simple training call with basic error handling
        trainer.fit(model, dataloader)
        print(f"Training completed successfully at epoch {trainer.current_epoch}, step {trainer.global_step}")
    except Exception as e:
        print(f"\nTraining error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Signal watchdog to stop
        hang_detected.set()
        
        # Check if it's a timeout or communication error
        error_str = str(e).lower()
        if "timeout" in error_str or "nccl" in error_str or "communication" in error_str:
            print("\nDistributed communication error detected. This may indicate network issues between nodes.")
            print("Recommended actions:")
            print("1. Check network connectivity between nodes")
            print("2. Verify that all processes are running correctly")
            print("3. Consider reducing the batch size or number of processes")
        
        raise
    finally:
        # Signal watchdog to stop on successful completion
        hang_detected.set()

# Remove the complex synchronize_checkpoint_path function
# It's not needed with our simplified approach