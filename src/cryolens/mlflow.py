import os
import socket
import subprocess
import time
import atexit
import threading

from pathlib import Path
import json
import logging
import torch
from typing import Optional, Dict, Any
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from cryolens.training.utils import (
    create_experiment_dir
)

logger = logging.getLogger(__name__)

class MLflowConfig:
    """Handles MLflow configuration and initialization."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MLflow configuration.
        
        Args:
            config_path: Path to MLflow configuration JSON file. If None, uses default localhost configuration.
        """
        self.config = self._load_config(config_path)
        self.tracking_uri = self.config.get('tracking_uri', 'http://localhost:5000')
        self.artifacts_dir = self.config.get('artifacts_dir', 'mlflow-artifacts')
        
        # Set tracking URI for all processes
        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"Set MLflow tracking URI to: {self.tracking_uri}")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file with fallback to defaults."""
        default_config = {
            'tracking_uri': 'http://localhost:5000',
            'artifacts_dir': 'mlflow-artifacts'
        }
        
        if not config_path:
            logger.info("No MLflow config provided, using default localhost configuration")
            return default_config
            
        try:
            with open(config_path) as f:
                config = json.load(f)['mlflow_server']
            logger.info(f"Loaded MLflow configuration from: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Error loading MLflow config from {config_path}: {e}")
            logger.warning("Falling back to default localhost configuration")
            return default_config

    def create_logger(self, experiment_name: str) -> MLFlowLogger:
        """
        Create MLflow logger with proper configuration.
        
        Args:
            experiment_name: Name of the MLflow experiment
            
        Returns:
            MLFlowLogger: Configured PyTorch Lightning MLflow logger
        """
        try:
            # Create experiment (ignore if exists)
            mlflow.create_experiment(
                experiment_name,
                artifact_location=str(Path(self.artifacts_dir) / experiment_name)
            )
        except Exception as e:
            logger.debug(f"Note on experiment creation (usually ok if exists): {e}")
            
        return MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=self.tracking_uri
        )

    @property
    def is_configured(self) -> bool:
        """Check if MLflow is properly configured."""
        try:
            mlflow.get_tracking_uri()
            return True
        except Exception:
            return False

def verify_mlflow_connection(rank: int = 0, logger: Optional[MLFlowLogger] = None):
    """Verify MLflow connection and configuration.
    
    Args:
        rank: Process rank (default: 0)
        logger: Optional MLFlowLogger instance
    """
    if rank == 0:
        try:
            print("\nMLflow Configuration Check:")
            print(f"Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"Current run ID: {mlflow.active_run().info.run_id if mlflow.active_run() else 'No active run'}")
            
            # Print environment variables
            print("\nMLflow Environment Variables:")
            for key, value in os.environ.items():
                if 'MLFLOW' in key:
                    print(f"{key}: {value}")
                    
            # Test MLflow direct logging
            print("\nTesting direct MLflow logging...")
            mlflow.log_metric("mlflow_test_metric", 1.0)
            print("Successfully logged test metric directly via mlflow")
            
            # Test PyTorch Lightning MLflow logger if provided
            if logger:
                print("\nTesting PyTorch Lightning MLflow logger...")
                logger.log_metrics({"lightning_test_metric": 1.0}, step=0)
                print("Successfully logged test metric via Lightning logger")
                
            print("\nMLflow verification completed successfully")
            
        except Exception as e:
            print(f"\nError during MLflow verification:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            import traceback
            traceback.print_exc()
            raise

def verify_mlflow_experiment(experiment_name: str, rank: int = 0):
    """Verify MLflow experiment exists and is accessible.
    
    Args:
        experiment_name: Name of the MLflow experiment
        rank: Process rank (default: 0)
    """
    if rank == 0:
        try:
            print(f"\nVerifying MLflow experiment: {experiment_name}")
            
            # Get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                print(f"Found experiment:")
                print(f"  ID: {experiment.experiment_id}")
                print(f"  Name: {experiment.name}")
                print(f"  Artifact Location: {experiment.artifact_location}")
                print(f"  Tags: {experiment.tags}")
                print(f"  Lifecycle Stage: {experiment.lifecycle_stage}")
            else:
                print(f"Experiment '{experiment_name}' not found")
                
        except Exception as e:
            print(f"\nError verifying MLflow experiment:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise

def verify_mlflow_run(run_id: str, rank: int = 0):
    """Verify MLflow run exists and is accessible, or skip verification if it doesn't exist.
    
    Args:
        run_id: MLflow run ID
        rank: Process rank (default: 0)
    """
    if rank == 0:
        try:
            print(f"\nAttempting to verify MLflow run: {run_id}")
            
            # Try to get run info
            run = mlflow.get_run(run_id)
            print(f"Found run:")
            print(f"  ID: {run.info.run_id}")
            print(f"  Status: {run.info.status}")
            print(f"  Start Time: {run.info.start_time}")
            print(f"  Artifact URI: {run.info.artifact_uri}")
            print(f"  Tags: {run.data.tags}")
            
            return True
            
        except Exception as e:
            print(f"\nNote: Run ID {run_id} not found in the current MLflow database.")
            print(f"This is expected when using a new MLflow server instance.")
            print(f"A new run will be created instead.")
            
            # Don't treat this as an error
            return False
    
    return False

class MLflowMonitorCallback(pl.Callback):
    """Callback to monitor MLflow logging during training."""
    
    def __init__(self, rank: int = 0):
        super().__init__()
        self.rank = rank
        self.last_log_time = time.time()
        self.log_interval = 300  # 5 minutes
        
    def on_train_start(self, trainer, pl_module):
        if self.rank == 0:
            print(f"Rank {self.rank}: MLflow monitoring disabled - using file-based tracking instead")
            
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.rank == 0:
            current_time = time.time()
            if current_time - self.last_log_time > self.log_interval:
                # Skip MLflow logging - just update timestamp and log to console instead
                print(f"\nRank {self.rank}: Training heartbeat at step {trainer.global_step}")
                self.last_log_time = current_time

class MLflowLoggingCallback(pl.Callback):
    """Custom callback for file-based logging without MLflow dependency."""
    
    def __init__(self, rank: int = 0):
        super().__init__()
        self.rank = rank
        self.metrics_cache = {}
        self.metrics_file = None
        self.metrics_data = []
        self.file_lock = threading.Lock()  # Add lock for file access
        
    def on_train_start(self, trainer, pl_module):
        if self.rank == 0:
            # Create metrics file in the checkpoint directory
            checkpoint_dir = trainer.checkpoint_callback.dirpath
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Use rank-specific metrics file to avoid conflicts in distributed training
            metrics_filename = f"metrics_rank_{self.rank}.jsonl"
            self.metrics_file = os.path.join(checkpoint_dir, metrics_filename)
            print(f"Rank {self.rank}: Logging metrics to {self.metrics_file}")
            
            # Create an empty file to start fresh
            with open(self.metrics_file, "w") as f:
                pass
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.rank == 0:  # Only log on rank 0
            try:
                # Get all metrics from trainer
                metrics = trainer.callback_metrics
                
                # Log to file instead of MLflow
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    if key not in self.metrics_cache or self.metrics_cache[key] != value:
                        # Save to cache and periodic file logging
                        self.metrics_cache[key] = value
                        
                        # Log to file every 25 steps
                        if trainer.global_step % 25 == 0:
                            self._write_metrics_to_file(trainer.global_step)
                        
            except Exception as e:
                print(f"Metrics logging error in batch {batch_idx}: {str(e)}")
                
    def on_train_epoch_end(self, trainer, pl_module):
        if self.rank == 0:
            try:
                # Always write metrics at epoch end
                self._write_metrics_to_file(trainer.global_step)
                print(f"Epoch {trainer.current_epoch} metrics saved to {self.metrics_file}")
            except Exception as e:
                print(f"Metrics logging error at epoch end: {str(e)}")
                
    def on_exception(self, trainer, pl_module, exception):
        if self.rank == 0:
            try:
                # Log exception to file
                with open(os.path.join(os.path.dirname(self.metrics_file), "error_log.txt"), "a") as f:
                    f.write(f"\nError at step {trainer.global_step}:\n{str(exception)}\n")
            except:
                pass
                
    def _write_metrics_to_file(self, step):
        """Write current metrics to file with proper locking."""
        if not self.metrics_file:
            return
            
        metrics_entry = {
            "step": step,
            "metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) 
                       for k, v in self.metrics_cache.items()}
        }
        
        # Use a lock to ensure only one process writes at a time
        with self.file_lock:
            try:
                with open(self.metrics_file, "a") as f:
                    f.write(json.dumps(metrics_entry) + "\n")
                    f.flush()  # Ensure write completes
                    os.fsync(f.fileno())  # Force flush to disk
            except Exception as e:
                print(f"Error writing metrics to file: {str(e)}")
                # Create backup metrics file if writing fails
                backup_file = f"{self.metrics_file}.backup_{int(time.time())}"
                try:
                    with open(backup_file, "a") as bf:
                        bf.write(json.dumps(metrics_entry) + "\n")
                except:
                    pass

def start_mlflow_server(db_path, artifact_path, port=5000):
    """Start a local MLflow tracking server as a subprocess with separate DB and artifact paths.
    
    Args:
        db_path: Path to store MLflow SQLite database
        artifact_path: Path to store MLflow artifacts
        port: Port to run the MLflow server on
        
    Returns:
        tuple: (process, port) - The server process and the port it's running on
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(artifact_path, exist_ok=True)
    
    # Find an available port if the specified one is in use
    if not is_port_available(port):
        port = find_available_port(start_port=5000, end_port=5100)
    
    # Start the MLflow server process
    # IMPORTANT: Host on 0.0.0.0 to allow worker nodes to connect
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", f"sqlite:///{db_path}",
        "--default-artifact-root", artifact_path,
        "--host", "0.0.0.0",  # Listen on all interfaces
        "--port", str(port)
    ]
    
    print(f"Starting MLflow server with command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Register cleanup function to terminate the server when the script exits
    def cleanup_process(proc):
        try:
            proc.terminate()
            proc.wait(timeout=10)  # Wait for termination
        except:
            proc.kill()  # Force kill if terminate doesn't work
    
    atexit.register(cleanup_process, process)
    
    # Wait for the server to start
    time.sleep(5)
    
    # Verify server is running
    if process.poll() is not None:
        # Server failed to start
        stdout, stderr = process.communicate()
        raise RuntimeError(f"Failed to start MLflow server: {stderr}")
    
    print(f"MLflow server started on port {port}")
    print(f"MLflow database: {db_path}")
    print(f"MLflow artifacts: {artifact_path}")
    
    # Start a thread to monitor server output
    def monitor_output(process):
        for line in process.stderr:
            print(f"MLflow server: {line.strip()}")
    
    threading.Thread(target=monitor_output, args=(process,), daemon=True).start()
    
    return process, port

def is_port_available(port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def find_available_port(start_port=5000, end_port=6000):
    """Find an available port in the given range."""
    for port in range(start_port, end_port):
        if is_port_available(port):
            return port
    raise RuntimeError(f"No available ports in range {start_port}-{end_port}")


def setup_experiment_tracking(args, dist_config, rank=0, experiment_dir_base="./cryolens_experiments"):
    """Setup MLflow experiment tracking with simplified file-based approach to avoid hanging.
    
    Args:
        args: Command line arguments
        dist_config: Distributed training configuration
        rank: Process rank (default: 0)
        experiment_dir_base: Base directory for experiment output
        
    Returns:
        MLFlowLogger: Configured MLflow logger (only on rank 0)
    """
    logger = logging.getLogger(__name__)
    print(f"Rank {rank}: Starting MLflow setup with simplified approach")
    
    # Only set up MLflow on rank 0 - other nodes don't need MLflow
    if rank != 0:
        print(f"Rank {rank}: Worker node skipping MLflow setup")
        return None
    
    # Create timestamp for unique experiment naming
    timestamp = int(time.time())
    
    # Create experiment-specific directories
    if args.config_name:
        experiment_name = f"{args.config_name}_{timestamp}"
    else:
        experiment_name = f"mlc_experiment_{timestamp}"
    
    # Create experiment directory with the unique experiment ID
    experiment_id = args.unique_experiment_id if args.unique_experiment_id else f"exp_{timestamp}"
    experiment_dir = Path(experiment_dir_base) / experiment_id
    
    print(f"Rank {rank}: Creating MLflow experiment directory: {experiment_dir}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create MLflow database and artifact directories
    db_dir = experiment_dir / "mlflow_db"
    artifact_dir = experiment_dir / "mlflow_artifacts"
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Set up a simple file-based MLflow tracking
    db_uri = f"sqlite:///{db_dir}/mlflow.db"
    
    try:
        print(f"Rank {rank}: Setting up MLflow with file-based backend at {db_uri}")
        mlflow.set_tracking_uri(db_uri)
        
        # Create experiment
        print(f"Rank {rank}: Creating MLflow experiment: {experiment_name}")
        try:
            mlflow.create_experiment(
                experiment_name,
                artifact_location=str(artifact_dir)
            )
        except Exception as e:
            print(f"Rank {rank}: Note: Experiment creation returned: {e} (usually OK if it exists)")
        
        # Skip using MLFlowLogger which triggers the SQLAlchemy DB setup
        # Create a simple file handler to track experiment information
        exp_info_file = artifact_dir / "experiment_info.json"
        with open(exp_info_file, "w") as f:
            json.dump({
                "experiment_name": experiment_name,
                "timestamp": int(time.time()),
                "artifact_location": str(artifact_dir)
            }, f, indent=2)
        
        print(f"Rank {rank}: Created experiment tracking file at {exp_info_file}")
        print(f"Rank {rank}: Proceeding without formal MLflow logger")
        return None
    except Exception as e:
        print(f"Rank {rank}: Error setting up MLflow: {e}")
        import traceback
        traceback.print_exc()
        print(f"Rank {rank}: Continuing without MLflow")
        return None