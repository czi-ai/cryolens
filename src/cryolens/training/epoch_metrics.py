"""
Enhanced EpochMetricsCallback for clean metrics.csv output.

This module contains a standalone callback that ensures only one line per epoch
is written to metrics.csv with clean, well-formatted metrics.
"""

import os
import csv
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pytorch_lightning as pl

class EpochMetricsCallback(pl.Callback):
    """Callback to write one clean line per epoch to metrics.csv.
    
    This callback addresses issues with the default PyTorch Lightning 
    metrics.csv output, ensuring:
    
    1. Only one line is written per epoch
    2. Only rank 0 writes to the file
    3. All metrics are properly aggregated and formatted
    4. No duplicate entries or partial lines
    
    Parameters
    ----------
    metrics_dir : Path or str
        Directory where metrics.csv should be saved
    rank : int, optional
        Process rank (default: 0)
    """
    
    def __init__(self, metrics_dir: Path, rank: int = 0):
        super().__init__()
        self.metrics_dir = Path(metrics_dir)
        self.rank = rank
        self.metrics_file = self.metrics_dir / "metrics.csv"
        self.current_epoch_metrics = {}
        
        # Create metrics directory if it doesn't exist (rank 0 only)
        if self.rank == 0:
            self.metrics_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize the CSV file with headers if it doesn't exist
            if not self.metrics_file.exists():
                self._initialize_csv()
        
        # Log initialization
        print(f"Rank {self.rank}: Initialized EpochMetricsCallback with metrics_dir={metrics_dir}")
        
    def _initialize_csv(self):
        """Initialize the CSV file with column headers."""
        # Core metrics we always want to track
        core_headers = [
            'epoch', 
            'global_step', 
            'train_loss', 
            'reconstruction_loss', 
            'kld_loss', 
            'similarity_loss',
            'step_time',
            'steps_per_second',
            'curriculum_stage'
        ]
        
        try:
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(core_headers)
            print(f"Rank {self.rank}: Initialized metrics.csv with headers: {core_headers}")
        except Exception as e:
            print(f"Rank {self.rank}: Error initializing metrics.csv: {str(e)}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect metrics after each batch for later aggregation."""
        # Only collect on rank 0
        if self.rank != 0:
            return
            
        # Get all metrics from callback metrics
        for k, v in trainer.callback_metrics.items():
            # Skip metrics with _step or _epoch suffixes - we'll use the base metrics
            if k.endswith('_step') or k.endswith('_epoch'):
                continue
                
            # Convert tensor to float
            if isinstance(v, torch.Tensor):
                v = v.item()
                
            # Store metric
            self.current_epoch_metrics[k] = v
            
        # Always track epoch and global step
        self.current_epoch_metrics['epoch'] = trainer.current_epoch
        self.current_epoch_metrics['global_step'] = trainer.global_step
        
        # Track curriculum stage if available
        if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'dataset'):
            dataset = trainer.datamodule.dataset
            if hasattr(dataset, 'current_stage'):
                self.current_epoch_metrics['curriculum_stage'] = dataset.current_stage
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Write one clean line at the end of each epoch."""
        # Only proceed on rank 0
        if self.rank != 0:
            return
            
        try:
            # Write metrics to CSV
            self._write_epoch_metrics()
            
            # Clear metrics for next epoch
            self.current_epoch_metrics = {}
            
            # Log success
            print(f"\nRank {self.rank}: Successfully wrote metrics for epoch {trainer.current_epoch} to {self.metrics_file}")
        except Exception as e:
            print(f"\nRank {self.rank}: Error writing metrics for epoch {trainer.current_epoch}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _write_epoch_metrics(self):
        """Write current epoch metrics to CSV file."""
        # Check if we have metrics to write
        if not self.current_epoch_metrics:
            print(f"Rank {self.rank}: No metrics to write for this epoch")
            return
            
        # If file doesn't exist, initialize it
        if not self.metrics_file.exists():
            self._initialize_csv()
            
        # Read existing headers
        headers = []
        try:
            with open(self.metrics_file, 'r', newline='') as f:
                reader = csv.reader(f)
                headers = next(reader)
        except Exception as e:
            print(f"Rank {self.rank}: Error reading headers from metrics.csv: {str(e)}")
            # Fallback to core headers
            headers = ['epoch', 'global_step', 'train_loss', 'reconstruction_loss', 
                      'kld_loss', 'similarity_loss', 'step_time', 'steps_per_second',
                      'curriculum_stage']
        
        # Check for new metrics and update headers if needed
        new_headers = set(self.current_epoch_metrics.keys()) - set(headers)
        if new_headers:
            headers.extend(sorted(new_headers))
            
            # Rewrite the file with updated headers and existing data
            try:
                # Read existing data
                existing_data = []
                with open(self.metrics_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    existing_data = list(reader)
                
                # Write with new headers
                with open(self.metrics_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(existing_data)
                    
                print(f"Rank {self.rank}: Updated metrics.csv with new headers: {new_headers}")
            except Exception as e:
                print(f"Rank {self.rank}: Error updating headers in metrics.csv: {str(e)}")
                
        # Prepare row with current metrics
        row = []
        for header in headers:
            row.append(self.current_epoch_metrics.get(header, ''))
            
        # Append to file
        try:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"Rank {self.rank}: Error writing row to metrics.csv: {str(e)}")
            
    def on_exception(self, trainer, pl_module, exception):
        """Handle exceptions by writing final metrics."""
        if self.rank == 0:
            print(f"Rank {self.rank}: Training exception occurred. Writing final metrics...")
            try:
                self._write_epoch_metrics()
            except Exception as e:
                print(f"Rank {self.rank}: Error writing metrics during exception: {str(e)}")
