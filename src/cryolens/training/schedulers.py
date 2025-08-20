"""
Progressive scheduling for disentanglement losses.

This module implements progressive scheduling strategies for applying
disentanglement losses during training, following the DINOv3 approach.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class DisentanglementScheduler:
    """Progressive scheduler for Gram disentanglement loss.
    
    This scheduler implements a three-stage progression:
    1. Warmup: No disentanglement, model learns basic features
    2. Mild disentanglement: Gradual introduction of Gram loss
    3. Full disentanglement: Full strength Gram loss
    
    Parameters
    ----------
    warmup_epochs : int
        Number of epochs for warmup (no disentanglement).
    mild_epochs : int
        Number of epochs for mild disentanglement phase.
    full_strength : float
        Maximum weight for Gram loss in full disentanglement.
    epochs_per_phase : int
        Number of epochs per curriculum phase (for alignment).
    """
    
    def __init__(
        self,
        warmup_epochs: int = 100,
        mild_epochs: int = 100,
        full_strength: float = 0.5,
        epochs_per_phase: int = 100
    ):
        self.warmup_epochs = warmup_epochs
        self.mild_epochs = mild_epochs
        self.full_strength = full_strength
        self.epochs_per_phase = epochs_per_phase
        
        # Track phase transitions
        self.current_phase = "warmup"
        self.phase_transitions_logged = set()
        
    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for current epoch.
        
        Parameters
        ----------
        epoch : int
            Current training epoch.
            
        Returns
        -------
        Dict[str, float]
            Dictionary with 'gram' weight and other relevant weights.
        """
        weights = {}
        
        # Determine current phase based on curriculum
        curriculum_phase = epoch // self.epochs_per_phase
        phase_epoch = epoch % self.epochs_per_phase
        
        # Three-stage progression aligned with curriculum
        if curriculum_phase == 0:
            # Phase 1: Warmup (first curriculum phase)
            weights['gram'] = 0.0
            phase = "warmup"
            
        elif curriculum_phase == 1:
            # Phase 2: Mild disentanglement (second curriculum phase)
            # Linear ramp from 0 to half strength
            progress = phase_epoch / self.epochs_per_phase
            weights['gram'] = (self.full_strength / 2) * progress
            phase = "mild"
            
        else:
            # Phase 3: Full disentanglement (third phase and beyond)
            # Ramp to full strength over first half of phase
            if phase_epoch < self.epochs_per_phase // 2:
                progress = phase_epoch / (self.epochs_per_phase // 2)
                weights['gram'] = (self.full_strength / 2) + (self.full_strength / 2) * progress
            else:
                weights['gram'] = self.full_strength
            phase = "full"
        
        # Log phase transitions
        if phase != self.current_phase and phase not in self.phase_transitions_logged:
            logger.info(f"\n{'='*60}")
            logger.info(f"Disentanglement Phase Transition: {self.current_phase} -> {phase}")
            logger.info(f"  Epoch: {epoch}")
            logger.info(f"  Curriculum phase: {curriculum_phase}")
            logger.info(f"  Gram weight: {weights['gram']:.4f}")
            logger.info(f"{'='*60}\n")
            
            self.current_phase = phase
            self.phase_transitions_logged.add(phase)
        
        # Add additional weights that might be adjusted
        weights['content_regularization'] = 1.0 if phase != "warmup" else 0.5
        weights['pose_regularization'] = 0.5 if phase == "full" else 0.2
        
        return weights
    
    def should_update_gram_teacher(self, epoch: int) -> bool:
        """Determine if Gram teacher should be updated.
        
        In DINOv3, the Gram teacher is periodically updated to the current
        model to adapt to learned features.
        
        Parameters
        ----------
        epoch : int
            Current training epoch.
            
        Returns
        -------
        bool
            Whether to update Gram teacher.
        """
        # Update every 10 epochs after mild phase starts
        if epoch < self.epochs_per_phase:
            return False
        
        # Update every 10 epochs
        return (epoch % 10) == 0
    
    def get_status(self, epoch: int) -> str:
        """Get human-readable status of scheduler.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
            
        Returns
        -------
        str
            Status string.
        """
        weights = self.get_weights(epoch)
        curriculum_phase = epoch // self.epochs_per_phase
        phase_epoch = epoch % self.epochs_per_phase
        
        status = f"Disentanglement Scheduler Status:\n"
        status += f"  Current phase: {self.current_phase}\n"
        status += f"  Curriculum phase: {curriculum_phase} (epoch {phase_epoch}/{self.epochs_per_phase})\n"
        status += f"  Gram weight: {weights['gram']:.4f}\n"
        status += f"  Content reg: {weights['content_regularization']:.4f}\n"
        status += f"  Pose reg: {weights['pose_regularization']:.4f}\n"
        
        return status
