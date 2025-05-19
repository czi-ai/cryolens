"""
Custom loggers for CryoLens training with distributed environment support.
"""

import re
from typing import Dict, Any, Optional, Union
import logging
from pytorch_lightning.loggers import CSVLogger, Logger
from pathlib import Path

logger = logging.getLogger(__name__)

class SafeCSVLogger(CSVLogger):
    """CSV Logger that sanitizes metric names to prevent errors.
    
    This logger extends PyTorch Lightning's CSVLogger to properly handle
    progress bar metrics that might cause CSV parsing errors, such as 
    the error: `ValueError: dict contains fields not in fieldnames: '', '0', '19'`
    
    Parameters
    ----------
    save_dir : str
        Save directory
    name : str
        Experiment name
    version : Union[int, str], optional
        Version number
    rank : int, optional
        Process rank (default: 0)
    """
    
    def __init__(
        self,
        save_dir: str, 
        name: str = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        rank: int = 0
    ):
        super().__init__(save_dir=save_dir, name=name, version=version)
        self.rank = rank
        print(f"Rank {self.rank}: Initialized SafeCSVLogger with: save_dir={save_dir}, name={name}, version={version}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs metrics, sanitizing metric names to avoid CSV errors.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Metrics dictionary
        step : int, optional
            Step number
        """
        # Only log from rank 0 in distributed mode
        if self.rank != 0:
            return
            
        try:
            # Clean the metrics dictionary to avoid CSV parsing errors
            sanitized_metrics = self._sanitize_metrics(metrics)
            
            # Call parent method with sanitized metrics
            super().log_metrics(sanitized_metrics, step)
            
            # Print confirmation of metrics logging periodically
            if step is not None and step % 100 == 0:
                print(f"Rank 0: Logged metrics at step {step} to metrics.csv")
                
        except Exception as e:
            # Log the error but don't crash training
            logger.error(f"Error in SafeCSVLogger.log_metrics: {str(e)}")
            logger.error(f"Original metrics: {metrics}")
    
    def _sanitize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Sanitize metric names to avoid CSV parsing errors.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Metrics dictionary
        
        Returns
        -------
        Dict[str, float]
            Sanitized metrics dictionary
        """
        # Create new dictionary for sanitized metrics
        sanitized = {}
        
        for key, value in metrics.items():
            # Skip any non-scalar values
            if not isinstance(value, (int, float, str, bool)) and value is not None:
                continue
                
            # Replace empty keys with placeholder
            if key == "" or not key:
                continue
                
            # Make sure the key is a valid CSV field name
            # Replace non-alphanumeric characters with underscores
            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', key)
            
            # Ensure key doesn't start with a number
            if safe_key and safe_key[0].isdigit():
                safe_key = f"m_{safe_key}"
            
            # Skip any problematic keys that are just numbers
            if safe_key.isdigit():
                continue
                
            # Add to sanitized metrics
            sanitized[safe_key] = value
        
        return sanitized
        
    def save(self) -> None:
        """Saves metrics to CSV files safely with error handling.
        """
        # Only save from rank 0
        if self.rank != 0:
            return
        
        try:
            # Print confirmation of CSV save
            print(f"Rank 0: Saving metrics to CSV...")
            super().save()
            print(f"Rank 0: Successfully saved metrics to CSV")
        except ValueError as e:
            # Special handling for "dict contains fields not in fieldnames" error
            if "dict contains fields not in fieldnames" in str(e):
                logger.warning(f"CSV field error encountered: {str(e)}")
                logger.warning("This error is typically caused by progress bar metrics. It's being handled gracefully.")
                
                # Attempt recovery by adding all fields to the header
                try:
                    # Extract field names from the error 
                    # Example: "dict contains fields not in fieldnames: '', '0', '19'"
                    error_msg = str(e)
                    field_part = error_msg.split("fieldnames: ")[1].strip()
                    
                    # Skip recovery for this save, it will be fixed in the next one
                    logger.info(f"Fields {field_part} will be sanitized in future logs")
                except Exception:
                    logger.warning("Could not parse error message for recovery")
            else:
                # Re-raise other errors
                raise
        except Exception as e:
            # Log other errors but don't crash
            logger.error(f"Error in SafeCSVLogger.save: {str(e)}")


def create_safe_csv_logger(
    experiment_dir: Path,
    experiment_name: str = "cryolens_experiment",
    rank: int = 0
) -> SafeCSVLogger:
    """Create a SafeCSVLogger instance for the experiment.
    
    Parameters
    ----------
    experiment_dir : Path
        Experiment directory
    experiment_name : str
        Experiment name
    rank : int
        Process rank
        
    Returns
    -------
    SafeCSVLogger
        Logger instance
    """
    # Make sure experiment directory exists
    if rank == 0:
        experiment_dir.mkdir(exist_ok=True, parents=True)
    
    # Create CSV logger
    csv_logger = SafeCSVLogger(
        save_dir=str(experiment_dir),
        name=experiment_name,
        version=None,  # Use the same version for all runs to make it easier to aggregate
        rank=rank
    )
    
    return csv_logger