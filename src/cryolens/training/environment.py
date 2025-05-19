"""
Environment configuration for distributed training.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
import torch
from pytorch_lightning.plugins.environments import ClusterEnvironment, LightningEnvironment

from .distributed import DistributedConfig

logger = logging.getLogger(__name__)

class CryoLensClusterEnvironment(ClusterEnvironment):
    """Cluster environment for CryoLens."""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Detect environment
        self.num_nodes = int(os.environ.get("NUM_NODES", "1"))
        self.num_devices = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if torch.cuda.is_available():
            self.num_devices = torch.cuda.device_count()
        self._world_size = self.num_devices * self.num_nodes
            
        # Log environment details
        logger.info(f"Environment setup: {self.num_nodes} nodes, {self.num_devices} devices per node")
        logger.info(f"Total world size: {self._world_size}")
        
    @property
    def creates_processes_externally(self) -> bool:
        return True
        
    def world_size(self) -> int:
        return self._world_size
        
    def global_rank(self) -> int:
        return int(os.environ.get("RANK", "0"))
        
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", "0"))
        
    def node_rank(self) -> int:
        return int(os.environ.get("NODE_RANK", "0"))
        
    @property
    def main_address(self) -> str:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")
        
    @property
    def main_port(self) -> int:
        return int(os.environ.get("MASTER_PORT", "29500"))
        
    def setup_device(self) -> torch.device:
        """Set up and return appropriate device."""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.local_rank()}")
            torch.cuda.set_device(device)
            return device
        return torch.device("cpu")
    
    @property
    def distributed_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for distributed setup."""
        return {
            "world_size": self.world_size(),
            "rank": self.global_rank(),
            "local_rank": self.local_rank(),
            "node_rank": self.node_rank(),
            "num_nodes": self.num_nodes,
            "devices_per_node": self.num_devices
        }

def get_training_environment(distributed: bool = False) -> Tuple[CryoLensClusterEnvironment, torch.device]:
    """
    Create and configure training environment.
    
    Args:
        distributed: Whether to enable distributed training
    
    Returns:
        Tuple of (cluster environment, device)
    """
    if not distributed:
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["NODE_RANK"] = "0"
        os.environ["RANK"] = "0"
    
    env = CryoLensClusterEnvironment()
    device = env.setup_device()
    
    return env, device

def create_lightning_environment(dist_config: DistributedConfig) -> LightningEnvironment:
    """Create Lightning cluster environment from distributed config with improved reliability."""
    import socket
    
    # Create Lightning environment
    env = LightningEnvironment()
    
    # Define environment behavior
    env.world_size = lambda: dist_config.world_size
    env.global_rank = lambda: dist_config.node_rank * dist_config.devices_per_node + dist_config.local_rank
    env.local_rank = lambda: dist_config.local_rank
    env.node_rank = lambda: dist_config.node_rank
    
    # Get hostname for debugging
    hostname = socket.gethostname()
    rank = env.global_rank()
    
    # Override necessary environment variables
    os.environ["MASTER_ADDR"] = dist_config.master_addr
    os.environ["MASTER_PORT"] = str(dist_config.master_port)
    os.environ["WORLD_SIZE"] = str(dist_config.world_size)
    os.environ["LOCAL_RANK"] = str(dist_config.local_rank)
    os.environ["NODE_RANK"] = str(dist_config.node_rank)
    os.environ["RANK"] = str(env.global_rank())
    
    # Set PyTorch-specific environment variables
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    
    # Set CUDA-specific environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dist_config.local_rank % torch.cuda.device_count())
    
    # Optimize performance for distributed training
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Print environment variables for debugging
    if rank == 0 or rank % 8 == 0:  # Only print for rank 0 and first rank of each node
        print(f"[{hostname}:Rank {rank}] Lightning environment created with:")
        print(f"  MASTER_ADDR={os.environ['MASTER_ADDR']}")
        print(f"  MASTER_PORT={os.environ['MASTER_PORT']}")
        print(f"  WORLD_SIZE={os.environ['WORLD_SIZE']}")
        print(f"  NODE_RANK={os.environ['NODE_RANK']}")
        print(f"  LOCAL_RANK={os.environ['LOCAL_RANK']}")
        print(f"  RANK={os.environ['RANK']}")
    
    return env