import os
import time
import socket
import torch
import torch.distributed as dist
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class DistributedConfig:
    """Configuration for distributed training.
    
    Attributes:
        world_size: Total number of processes.
        node_rank: Rank of this node.
        local_rank: Local rank on this node.
        local_world_size: Number of processes on this node.
        master_addr: Address of the master node.
        master_port: Port for the master node.
        backend: PyTorch distributed backend.
        num_nodes: Total number of nodes.
        devices_per_node: Number of devices per node.
    """
    world_size: int
    node_rank: int
    local_rank: int
    local_world_size: int
    master_addr: str
    master_port: str
    backend: str
    num_nodes: int
    devices_per_node: int


def print_environment_info():
    """Print environment information useful for debugging distributed setup."""
    # Get hostname
    hostname = socket.gethostname()
    
    # Get environment variables
    env_vars = {
        'NODE_RANK': os.environ.get('NODE_RANK', 'Not set'),
        'WORLD_SIZE': os.environ.get('WORLD_SIZE', 'Not set'),
        'LOCAL_RANK': os.environ.get('LOCAL_RANK', 'Not set'),
        'RANK': os.environ.get('RANK', 'Not set'),
        'MASTER_ADDR': os.environ.get('MASTER_ADDR', 'Not set'),
        'MASTER_PORT': os.environ.get('MASTER_PORT', 'Not set'),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
    }
    
    # Check for torch.distributed.run environment variables
    torchrun_vars = {
        'LOCAL_RANK': os.environ.get('LOCAL_RANK', 'Not set'),
        'RANK': os.environ.get('RANK', 'Not set'),
        'WORLD_SIZE': os.environ.get('WORLD_SIZE', 'Not set'),
        'MASTER_ADDR': os.environ.get('MASTER_ADDR', 'Not set'),
        'MASTER_PORT': os.environ.get('MASTER_PORT', 'Not set'),
    }
    
    # Print environment info
    print(f"===== Distributed Environment Info on {hostname} =====")
    for var, value in env_vars.items():
        print(f"{var}: {value}")
    
    print("\nAdditional distributed environment variables:")
    for var, value in os.environ.items():
        if 'TORCHELASTIC' in var or 'PET_' in var or 'NCCL_' in var:
            print(f"{var}: {value}")
    
    # Print CUDA info
    if torch.cuda.is_available():
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available")
    
    # Print torch.distributed info
    if dist.is_available():
        print(f"torch.distributed is available")
        if dist.is_initialized():
            print(f"torch.distributed is initialized")
            print(f"world_size: {dist.get_world_size()}")
            print(f"rank: {dist.get_rank()}")
        else:
            print(f"torch.distributed is not initialized")
    else:
        print("torch.distributed is not available")


def setup_distributed_env(
    node_rank: Optional[int] = None,
    local_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    master_addr: Optional[str] = None,
    master_port: Optional[str] = None,
    backend: str = "nccl"
) -> Optional[DistributedConfig]:
    """
    Set up the distributed environment.
    
    Args:
        node_rank: The rank of the node in multi-node training
        local_rank: The local rank of the process on the node
        world_size: Total number of processes across all nodes
        master_addr: Address of the master node
        master_port: Port of the master node
        backend: PyTorch distributed backend
        
    Returns:
        DistributedConfig object with the setup information or None if setup failed
    """
    # Get hostname for logging
    hostname = socket.gethostname()
    
    # Check if we're running under torch.distributed.run (torchrun)
    is_torchrun = 'TORCHELASTIC_RUN_ID' in os.environ
    
    # When running under torchrun, we can get most parameters from the environment
    if is_torchrun:
        print(f"Detected torch.distributed.run (torchrun) environment")
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        global_rank = int(os.environ.get('RANK', '0'))
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        
        # In torchrun, we need to extract node_rank from global_rank and local_rank
        # if it's not explicitly provided
        if node_rank is None:
            if 'GROUP_RANK' in os.environ:
                node_rank = int(os.environ['GROUP_RANK'])
            else:
                # Estimate node_rank based on global_rank and local_rank
                # This assumes equal number of processes per node
                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    node_rank = global_rank // num_gpus
                else:
                    node_rank = 0
        
        # Set device based on local_rank
        if torch.cuda.is_available():
            if local_rank < torch.cuda.device_count():
                torch.cuda.set_device(local_rank)
                print(f"TORCHRUN: Setting CUDA device to {local_rank}")
            else:
                print(f"TORCHRUN: Warning - local_rank {local_rank} is >= available devices {torch.cuda.device_count()}")
                # Try to select a valid device
                torch.cuda.set_device(local_rank % torch.cuda.device_count())
        
        # Ensure world_size is set
        os.environ['WORLD_SIZE'] = str(world_size)
    else:
        # Standard RunAI environment setup
        # Use environment variables if parameters not provided
        if node_rank is None and 'NODE_RANK' in os.environ:
            node_rank = int(os.environ['NODE_RANK'])
        else:
            node_rank = node_rank or 0
            
        if local_rank is None and 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            local_rank = local_rank or 0
            
        if 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            # If WORLD_SIZE not set and not provided, assume it's the number of CUDA devices
            if world_size is None:
                if torch.cuda.is_available():
                    world_size = torch.cuda.device_count()
                else:
                    world_size = 1
        
        if 'MASTER_ADDR' in os.environ:
            master_addr = os.environ['MASTER_ADDR']
        else:
            master_addr = master_addr or 'localhost'
            
        if 'MASTER_PORT' in os.environ:
            master_port = os.environ['MASTER_PORT']
        else:
            master_port = master_port or '23456'
        
        # Calculate global rank based on node_rank and local_rank
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
        else:
            # Calculate global rank based on node_rank and local_rank
            local_world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
            rank = node_rank * local_world_size + local_rank
            os.environ['RANK'] = str(rank)
        
        # Set environment variables
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Add a delay for worker nodes to ensure master is ready first
        if node_rank > 0:
            delay = 10 + (node_rank * 2)  # Staggered delay based on node rank
            print(f"[{hostname}:Rank {rank}] Worker node waiting {delay}s for master to initialize...")
            time.sleep(delay)
        
        # Set device
        if torch.cuda.is_available():
            device = local_rank % torch.cuda.device_count()
            torch.cuda.set_device(device)
            print(f"[{hostname}:Rank {rank}] Using CUDA device {device}: {torch.cuda.get_device_name(device)}")
    
    # From this point on, the setup is the same for both torchrun and standard setup
    
    # Get current process rank
    rank = int(os.environ.get('RANK', '0'))
    
    # Initialize the process group
    print(f"[{hostname}:Rank {rank}] Initializing process group with backend={backend}, "
          f"rank={rank}, world_size={world_size}, master={master_addr}:{master_port}")
    
    # Detect number of nodes and devices per node
    if 'RUNAI_NUM_OF_GPUS' in os.environ:
        devices_per_node = int(os.environ.get('RUNAI_NUM_OF_GPUS', torch.cuda.device_count()))
    else:
        devices_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Estimate number of nodes based on world size and devices per node
    num_nodes = max(1, world_size // max(1, devices_per_node))
    local_world_size = devices_per_node
    
    try:
        if is_torchrun:
            # For torchrun, we don't need to manually initialize the process group
            # as it will be automatically done by torch.distributed.run
            if not dist.is_initialized():
                if torch.cuda.is_available():
                    dist.init_process_group(backend=backend)
                else:
                    dist.init_process_group(backend='gloo')
                
                print(f"[{hostname}:Rank {rank}] Initialized process group for torchrun")
        else:
            # Standard initialization
            if torch.cuda.is_available():
                dist.init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_addr}:{master_port}",
                    world_size=world_size,
                    rank=rank
                )
            else:
                print(f"[{hostname}:Rank {rank}] CUDA not available, using CPU!")
                dist.init_process_group(
                    backend='gloo',  # Fall back to gloo for CPU
                    init_method=f"tcp://{master_addr}:{master_port}",
                    world_size=world_size,
                    rank=rank
                )
        
        # Create distributed config
        config = DistributedConfig(
            world_size=world_size,
            node_rank=node_rank,
            local_rank=local_rank,
            local_world_size=local_world_size,
            master_addr=master_addr,
            master_port=master_port,
            backend=backend,
            num_nodes=num_nodes,
            devices_per_node=devices_per_node
        )
        
        # Sync at the end of initialization
        if dist.is_initialized():
            dist.barrier()
            print(f"[{hostname}:Rank {rank}] Distributed initialization completed successfully!")
        
        return config
    
    except Exception as e:
        print(f"[{hostname}:Rank {rank}] Error initializing distributed environment: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def cleanup():
    """
    Clean up the distributed environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
