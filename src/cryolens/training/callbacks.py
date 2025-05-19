"""
Curriculum update callback for PyTorch Lightning.
"""

import logging
import pytorch_lightning as pl

from cryolens.logging.csv_helpers import safe_log_metrics

class CurriculumUpdateCallback(pl.Callback):
    """Callback to update curriculum stage based on current epoch."""
    
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.last_epoch = -1
        print(f"CurriculumUpdateCallback initialized with dataset: {dataset.__class__.__name__}")
        if hasattr(dataset, 'curriculum_spec') and dataset.curriculum_spec:
            print(f"Dataset has curriculum with {len(dataset.curriculum_spec)} stages")
            for i, stage in enumerate(dataset.curriculum_spec):
                print(f"Stage {i}: duration={stage.get('duration', 'inf')}, weights keys={list(stage.get('weights', {}).keys())}")
        else:
            print(f"Dataset does not have curriculum_spec")
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Update curriculum stage at the start of each epoch."""
        # Update curriculum epoch
        if hasattr(self.dataset, 'update_epoch'):
            current_epoch = trainer.current_epoch
            # Only print log messages when the epoch changes
            if current_epoch != self.last_epoch:
                self.last_epoch = current_epoch
                print(f"\nEpoch {current_epoch}: Updating curriculum stage")
                
            self.dataset.update_epoch(current_epoch)
            
            # Print current curriculum stage information
            if hasattr(self.dataset, 'current_stage'):
                current_stage = self.dataset.current_stage
                print(f"Epoch {current_epoch}: Current curriculum stage: {current_stage}")
                
                # Print weights if available
                if hasattr(self.dataset, 'current_weights') and self.dataset.current_weights:
                    print(f"Current weights: {self.dataset.current_weights}")
        else:
            print(f"Warning: Dataset does not support curriculum updating")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log curriculum stage information at the end of each epoch."""
        # Log current curriculum stage if it exists
        if hasattr(self.dataset, 'current_stage') and trainer.logger is not None:
            try:
                # Log to console
                print(f"Epoch {trainer.current_epoch} completed. Curriculum stage: {self.dataset.current_stage}")
                
                # Use safe logging function to avoid CSV key issues
                safe_log_metrics(
                    trainer.logger, 
                    {"curriculum_stage": self.dataset.current_stage}, 
                    step=trainer.global_step
                )
                
                # Log more detailed info if weights are available
                if hasattr(self.dataset, 'curriculum_spec') and self.dataset.curriculum_spec:
                    current_stage = self.dataset.current_stage
                    if current_stage < len(self.dataset.curriculum_spec):
                        stage_config = self.dataset.curriculum_spec[current_stage]
                        # Log total remaining epochs in this stage
                        total_epochs = 0
                        for i in range(current_stage):
                            if i < len(self.dataset.curriculum_spec):
                                total_epochs += self.dataset.curriculum_spec[i].get("duration", 0)
                        
                        remaining = stage_config.get("duration", float('inf'))
                        if remaining != float('inf'):
                            # Calculate how many epochs remain in this stage
                            epochs_in_stage = trainer.current_epoch - total_epochs
                            remaining -= epochs_in_stage
                            print(f"Remaining epochs in current stage: {remaining}")
                            
                            # Use safe logging function to avoid CSV key issues
                            safe_log_metrics(
                                trainer.logger,
                                {"curriculum_stage_remaining_epochs": remaining},
                                step=trainer.global_step
                            )
            except Exception as e:
                # Log errors instead of silently ignoring them
                print(f"Error in curriculum logging: {str(e)}")
                import traceback
                traceback.print_exc()
