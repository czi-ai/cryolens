"""
Memory Bank enhanced loss functions for CryoLens model training - Version 2.

Key improvements:
1. Uses training steps/epochs instead of wall clock time
2. Memory banks activate after a warmup period
3. Deterministic behavior across runs
4. Always uses memory bank embeddings for all structures once active
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class TrainingAwareMemoryBank:
    """Per-class memory bank that uses training time instead of wall clock time."""
    
    def __init__(self, num_classes: int, embedding_dim: int, device: torch.device, 
                 warmup_steps: int = 1000, activation_mode: str = "steps"):
        """
        Initialize training-aware memory bank.
        
        Parameters
        ----------
        num_classes : int
            Maximum number of classes (size of lookup table)
        embedding_dim : int
            Dimension of embeddings
        device : torch.device
            Device to store embeddings on
        warmup_steps : int
            Number of steps/epochs before memory bank becomes active
        activation_mode : str
            Either "steps" or "epochs" for when to activate
        """
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.device = device
        self.warmup_steps = warmup_steps
        self.activation_mode = activation_mode
        
        # Store one embedding per class
        self.embeddings = torch.zeros((num_classes, embedding_dim), device=device)
        self.update_steps = torch.zeros(num_classes, dtype=torch.long, device=device)  # Training step of last update
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)  # Track which classes have been seen
        
        # EMA momentum - starts low and increases over time
        self.base_momentum = 0.9
        self.momentum_warmup_steps = 5000
        
        # Track global training progress
        self.global_step = 0
        self.current_epoch = 0
        self.is_active = False
        
    def set_training_progress(self, global_step: int, current_epoch: int):
        """Update the current training progress."""
        self.global_step = global_step
        self.current_epoch = current_epoch
        
        # Check if memory bank should be active
        if self.activation_mode == "steps":
            self.is_active = global_step >= self.warmup_steps
        else:  # epochs
            self.is_active = current_epoch >= self.warmup_steps
            
    def get_momentum(self) -> float:
        """Get current momentum value based on training progress."""
        if self.global_step < self.momentum_warmup_steps:
            # Linear warmup from 0.5 to base_momentum
            progress = self.global_step / self.momentum_warmup_steps
            return 0.5 + (self.base_momentum - 0.5) * progress
        return self.base_momentum
        
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Update memory bank with new embeddings.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            New embeddings to add (batch_size, embedding_dim)
        labels : torch.Tensor
            Corresponding class labels (batch_size,)
        """
        if not self.is_active:
            return  # Don't update during warmup
            
        # Detach to prevent gradient flow
        embeddings = embeddings.detach()
        labels = labels.detach()
        
        # Get current momentum
        momentum = self.get_momentum()
        
        # Update each class
        for i in range(labels.shape[0]):
            label = labels[i].item()
            if label >= 0 and label < self.num_classes:  # Valid class (not background)
                # Normalize embedding before storing
                normalized_emb = F.normalize(embeddings[i], p=2, dim=0)
                
                # Update embedding with exponential moving average if already initialized
                if self.initialized[label]:
                    # EMA update: new = momentum * old + (1 - momentum) * new
                    self.embeddings[label] = momentum * self.embeddings[label] + (1 - momentum) * normalized_emb
                    # Re-normalize after EMA
                    self.embeddings[label] = F.normalize(self.embeddings[label], p=2, dim=0)
                else:
                    # First time seeing this class
                    self.embeddings[label] = normalized_emb
                    self.initialized[label] = True
                
                self.update_steps[label] = self.global_step
    
    def get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Get weights for given labels based on staleness.
        
        Parameters
        ----------
        labels : torch.Tensor
            Class labels to get weights for
            
        Returns
        -------
        torch.Tensor
            Weights based on recency (newer = higher weight)
        """
        if not self.is_active:
            return torch.zeros_like(labels, dtype=torch.float32)
            
        # Calculate staleness (steps since last update)
        staleness = self.global_step - self.update_steps[labels]
        staleness = staleness.float()
        
        # Convert to weights using exponential decay
        # Recent updates get weight close to 1, older updates decay
        decay_rate = 0.0001  # Slower decay based on steps
        weights = torch.exp(-decay_rate * staleness)
        
        # Set weight to 0 for uninitialized classes
        weights[~self.initialized[labels]] = 0.0
        
        return weights
    
    def get_all_initialized_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all initialized embeddings and their indices.
        
        Returns
        -------
        embeddings : torch.Tensor
            All initialized embeddings (num_initialized, embedding_dim)
        indices : torch.Tensor
            Indices of initialized classes
        """
        if not self.is_active:
            return torch.empty((0, self.embedding_dim), device=self.device), torch.empty(0, device=self.device, dtype=torch.long)
            
        indices = torch.where(self.initialized)[0]
        if len(indices) == 0:
            return torch.empty((0, self.embedding_dim), device=self.device), indices
            
        embeddings = self.embeddings[indices]
        return embeddings, indices


class ContrastiveAffinityLossWithMemoryV2(nn.Module):
    """
    Improved contrastive affinity loss with training-aware memory bank.
    """
    
    def __init__(
        self, 
        lookup: torch.Tensor, 
        device: torch.device, 
        latent_ratio: float = 0.75, 
        margin: float = 4.0,
        warmup_steps: int = 1000,
        activation_mode: str = "steps",
        memory_weight: float = 0.5
    ):
        """
        Initialize improved memory-enhanced contrastive loss.
        
        Parameters
        ----------
        lookup : torch.Tensor
            Lookup table for molecule affinities
        device : torch.device
            Device to run computation on
        latent_ratio : float
            Ratio of latent dimensions to use
        margin : float
            Margin for dissimilar pairs
        warmup_steps : int
            Steps/epochs before memory bank activates
        activation_mode : str
            "steps" or "epochs" for activation timing
        memory_weight : float
            Weight for memory bank loss vs batch loss
        """
        super().__init__()
        
        # Register lookup buffer
        lookup = lookup.clone().detach().contiguous()
        self.register_buffer('lookup', lookup, persistent=False)
        self.device = device
        self.latent_ratio = latent_ratio
        self.margin = margin
        self.memory_weight = memory_weight
        
        # Background similarity values
        self.background_sim = 0.2
        self.background_other_sim = 0.01
        
        # Memory bank will be initialized on first forward pass
        self.memory_bank = None
        self.num_classes = lookup.shape[0]
        self.warmup_steps = warmup_steps
        self.activation_mode = activation_mode
        
    def _init_memory_bank(self, embedding_dim: int):
        """Initialize memory bank with correct embedding dimension."""
        self.memory_bank = TrainingAwareMemoryBank(
            self.num_classes, 
            embedding_dim, 
            self.device,
            self.warmup_steps,
            self.activation_mode
        )
        
    def update_training_progress(self, global_step: int, current_epoch: int):
        """Update training progress for memory bank."""
        if self.memory_bank is not None:
            self.memory_bank.set_training_progress(global_step, current_epoch)
        
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
                global_step: Optional[int] = None, current_epoch: Optional[int] = None,
                per_sample: bool = False) -> torch.Tensor:
        """
        Compute contrastive affinity loss using memory bank.
        
        Parameters
        ----------
        y_true : torch.Tensor
            True class labels
        y_pred : torch.Tensor
            Predicted embeddings
        global_step : int, optional
            Current global training step
        current_epoch : int, optional
            Current training epoch
        per_sample : bool
            Whether to return per-sample loss
        """
        try:
            # Ensure inputs are contiguous
            y_true = y_true.contiguous().to(self.device)
            y_pred = y_pred.contiguous().to(self.device)

            # Handle batch size < 2
            if y_true.shape[0] < 2:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # Handle all background case
            if torch.all(y_true == -1):
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # Get dimensions and use partial embedding
            n_dims = y_pred.shape[1]
            n_dims_to_use = int(n_dims * self.latent_ratio)
            y_pred_partial = y_pred[:, :n_dims_to_use].contiguous()
            
            # Initialize memory bank if needed
            if self.memory_bank is None:
                self._init_memory_bank(n_dims_to_use)
            
            # Update training progress if provided
            if global_step is not None and current_epoch is not None:
                self.memory_bank.set_training_progress(global_step, current_epoch)
            
            # Update memory bank with current batch (only non-background)
            non_bg_mask = y_true != -1
            if torch.any(non_bg_mask):
                self.memory_bank.update(
                    y_pred_partial[non_bg_mask],
                    y_true[non_bg_mask]
                )
            
            # Standard within-batch loss calculation
            z_id = torch.arange(y_pred_partial.shape[0], device=self.device)
            c = torch.combinations(z_id, r=2, with_replacement=False)
            
            # Extract pairs of embeddings
            features1 = y_pred_partial[c[:, 0], :].contiguous()
            features2 = y_pred_partial[c[:, 1], :].contiguous()
            
            # Normalize embeddings
            features1_norm = F.normalize(features1, p=2, dim=1)
            features2_norm = F.normalize(features2, p=2, dim=1)
            distances = torch.norm(features1_norm - features2_norm, p=2, dim=1)
            
            # Get target similarities for batch pairs
            target_similarities = self._get_target_similarities(y_true, c)
            
            # Debug: Verify similarities are in [0,1] range
            if torch.any(target_similarities < 0) or torch.any(target_similarities > 1):
                print(f"WARNING in memory bank loss: Target similarities outside [0,1] range! Min: {target_similarities.min():.4f}, Max: {target_similarities.max():.4f}")
            
            # Calculate standard contrastive loss
            similar_term = target_similarities * (distances ** 2)
            margin_dist = torch.clamp(self.margin - distances, min=0)
            dissimilar_term = (1 - target_similarities) * (margin_dist ** 2)
            batch_losses = similar_term + dissimilar_term
            
            # Add memory bank loss if active
            if self.memory_bank.is_active:
                memory_losses = self._compute_memory_loss_v2(y_true, y_pred_partial)
                # Gradually increase memory weight over time
                memory_progress = min(1.0, (self.memory_bank.global_step - self.memory_bank.warmup_steps) / 5000)
                adjusted_memory_weight = self.memory_weight * memory_progress
                total_loss = (1 - adjusted_memory_weight) * torch.mean(batch_losses) + adjusted_memory_weight * memory_losses
            else:
                total_loss = torch.mean(batch_losses)
            
            # Return per-sample losses or mean loss
            if per_sample and not self.memory_bank.is_active:
                return batch_losses.contiguous()
            
            # Handle NaN values
            if torch.isnan(total_loss):
                zero_tensor = torch.tensor(0.0, device=self.device)
                total_loss = zero_tensor.requires_grad_() + total_loss.detach() * 0
                
            return total_loss

        except Exception as e:
            print(f"Error in memory contrastive loss v2: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return gradient-maintaining zero tensor
            zero_tensor = torch.tensor(0.0, device=self.device)
            return zero_tensor.requires_grad_()
    
    def _get_target_similarities(self, y_true: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """Get target similarities for pairs of samples."""
        # Handle background pairs
        is_background = (y_true == -1)
        pair1_is_bg = is_background[pairs[:, 0]]
        pair2_is_bg = is_background[pairs[:, 1]]
        
        # Initialize target similarities
        target_similarities = torch.zeros(pairs.shape[0], device=self.device)
        
        # Both background
        both_bg_mask = pair1_is_bg & pair2_is_bg
        target_similarities[both_bg_mask] = self.background_sim
        
        # One background, one object
        one_bg_mask = pair1_is_bg ^ pair2_is_bg
        target_similarities[one_bg_mask] = self.background_other_sim
        
        # Both objects - use lookup table
        both_obj_mask = ~(both_bg_mask | one_bg_mask)
        if torch.any(both_obj_mask):
            obj_pairs = pairs[both_obj_mask]
            obj_indices1 = y_true[obj_pairs[:, 0]]
            obj_indices2 = y_true[obj_pairs[:, 1]]
            
            # Validate indices
            valid_indices = ((obj_indices1 >= 0) & (obj_indices2 >= 0) & 
                        (obj_indices1 < self.lookup.shape[0]) & 
                        (obj_indices2 < self.lookup.shape[0]))
            
            if torch.any(valid_indices):
                valid_obj_indices1 = obj_indices1[valid_indices].long()
                valid_obj_indices2 = obj_indices2[valid_indices].long()
                valid_similarities = self.lookup[valid_obj_indices1, valid_obj_indices2]
                target_similarities[both_obj_mask][valid_indices] = valid_similarities
        
        return target_similarities
    
    def _compute_memory_loss_v2(self, y_true: torch.Tensor, y_pred_partial: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between current batch and ALL structures in memory bank.
        This ensures consistent training signal for all structures once memory is active.
        """
        # Get all initialized embeddings from memory
        memory_embs, memory_classes = self.memory_bank.get_all_initialized_embeddings()
        
        if memory_embs.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_loss = 0.0
        num_pairs = 0
        
        # For each sample in batch
        for i in range(y_true.shape[0]):
            if y_true[i] == -1:  # Skip background
                continue
                
            # Get current sample's embedding
            current_emb = F.normalize(y_pred_partial[i:i+1], p=2, dim=1)
            
            # Compute distances to ALL memory embeddings
            distances = torch.norm(current_emb - memory_embs, p=2, dim=1)
            
            # Get target similarities
            current_class = y_true[i].long()
            target_sims = self.lookup[current_class, memory_classes]
            
            # Get staleness weights
            weights = self.memory_bank.get_weights(memory_classes)
            
            # Weighted contrastive loss
            similar_term = target_sims * (distances ** 2) * weights
            margin_dist = torch.clamp(self.margin - distances, min=0)
            dissimilar_term = (1 - target_sims) * (margin_dist ** 2) * weights
            
            # Sum over all memory bank entries
            total_loss += torch.sum(similar_term + dissimilar_term)
            num_pairs += torch.sum(weights > 0.1)  # Count only reasonably fresh entries
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)


class AffinityCosineLossWithMemoryV2(nn.Module):
    """
    Improved cosine affinity loss with training-aware memory bank.
    """
    
    def __init__(
        self, 
        lookup: torch.Tensor, 
        device: torch.device, 
        latent_ratio: float = 0.75,
        warmup_steps: int = 1000,
        activation_mode: str = "steps",
        memory_weight: float = 0.5
    ):
        """
        Initialize improved memory-enhanced cosine loss.
        
        Parameters
        ----------
        lookup : torch.Tensor
            Lookup table for molecule affinities
        device : torch.device
            Device to run computation on
        latent_ratio : float
            Ratio of latent dimensions to use
        warmup_steps : int
            Steps/epochs before memory bank activates
        activation_mode : str
            "steps" or "epochs" for activation timing
        memory_weight : float
            Weight for memory bank loss vs batch loss
        """
        super().__init__()
        
        # Register lookup buffer
        lookup = lookup.clone().detach().contiguous()
        self.register_buffer('lookup', lookup, persistent=False)
        self.device = device
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.l1loss = nn.L1Loss(reduction="none")
        self.latent_ratio = latent_ratio
        self.memory_weight = memory_weight
        
        # Background similarity values
        self.background_sim = 0.2
        self.background_other_sim = 0.01
        
        # Memory bank
        self.memory_bank = None
        self.num_classes = lookup.shape[0]
        self.warmup_steps = warmup_steps
        self.activation_mode = activation_mode
        
    def _init_memory_bank(self, embedding_dim: int):
        """Initialize memory bank with correct embedding dimension."""
        self.memory_bank = TrainingAwareMemoryBank(
            self.num_classes, 
            embedding_dim, 
            self.device,
            self.warmup_steps,
            self.activation_mode
        )
        
    def update_training_progress(self, global_step: int, current_epoch: int):
        """Update training progress for memory bank."""
        if self.memory_bank is not None:
            self.memory_bank.set_training_progress(global_step, current_epoch)
        
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor,
                global_step: Optional[int] = None, current_epoch: Optional[int] = None,
                per_sample: bool = False) -> torch.Tensor:
        """
        Compute cosine affinity loss using memory bank.
        """
        try:
            # Ensure inputs are contiguous
            y_true = y_true.contiguous().to(self.device)
            y_pred = y_pred.contiguous().to(self.device)

            # Handle batch size < 2
            if y_true.shape[0] < 2:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # Handle all background case
            if torch.all(y_true == -1):
                return torch.tensor(self.background_sim, device=self.device, requires_grad=True)

            # Get dimensions
            n_dims = y_pred.shape[1]
            n_dims_to_use = int(n_dims * self.latent_ratio)
            y_pred_partial = y_pred[:, :n_dims_to_use].contiguous()
            
            # Initialize memory bank if needed
            if self.memory_bank is None:
                self._init_memory_bank(n_dims_to_use)
            
            # Update training progress if provided
            if global_step is not None and current_epoch is not None:
                self.memory_bank.set_training_progress(global_step, current_epoch)
            
            # Update memory bank
            non_bg_mask = y_true != -1
            if torch.any(non_bg_mask):
                self.memory_bank.update(
                    y_pred_partial[non_bg_mask],
                    y_true[non_bg_mask]
                )
            
            # Standard within-batch loss
            z_id = torch.arange(y_pred_partial.shape[0], device=self.device)
            c = torch.combinations(z_id, r=2, with_replacement=False)
            
            # Extract pairs
            features1 = y_pred_partial[c[:, 0], :].contiguous()
            features2 = y_pred_partial[c[:, 1], :].contiguous()
            
            # Normalize and compute cosine similarity
            features1_norm = F.normalize(features1, p=2, dim=1)
            features2_norm = F.normalize(features2, p=2, dim=1)
            latent_similarity = self.cos(features1_norm, features2_norm)
            
            # Get target similarities
            target_similarities = self._get_target_similarities(y_true, c)
            
            # Calculate loss
            batch_losses = self.l1loss(latent_similarity, target_similarities)
            
            # Add memory bank loss if active
            if self.memory_bank.is_active:
                memory_losses = self._compute_memory_loss_v2(y_true, y_pred_partial)
                # Gradually increase memory weight
                memory_progress = min(1.0, (self.memory_bank.global_step - self.memory_bank.warmup_steps) / 5000)
                adjusted_memory_weight = self.memory_weight * memory_progress
                total_loss = (1 - adjusted_memory_weight) * torch.mean(batch_losses) + adjusted_memory_weight * memory_losses
            else:
                total_loss = torch.mean(batch_losses)
            
            # Return per-sample or mean loss
            if per_sample and not self.memory_bank.is_active:
                return batch_losses.contiguous()
            
            # Handle NaN values
            if torch.isnan(total_loss):
                zero_tensor = torch.tensor(0.0, device=self.device)
                total_loss = zero_tensor.requires_grad_() + total_loss.detach() * 0
                
            return total_loss

        except Exception as e:
            print(f"Error in memory cosine loss v2: {str(e)}")
            import traceback
            traceback.print_exc()
            zero_tensor = torch.tensor(0.0, device=self.device)
            return zero_tensor.requires_grad_()
    
    def _get_target_similarities(self, y_true: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """Get target similarities for pairs of samples."""
        # Same implementation as in ContrastiveAffinityLossWithMemory
        is_background = (y_true == -1)
        pair1_is_bg = is_background[pairs[:, 0]]
        pair2_is_bg = is_background[pairs[:, 1]]
        
        target_similarities = torch.zeros(pairs.shape[0], device=self.device)
        
        both_bg_mask = pair1_is_bg & pair2_is_bg
        target_similarities[both_bg_mask] = self.background_sim
        
        one_bg_mask = pair1_is_bg ^ pair2_is_bg
        target_similarities[one_bg_mask] = self.background_other_sim
        
        both_obj_mask = ~(both_bg_mask | one_bg_mask)
        if torch.any(both_obj_mask):
            obj_pairs = pairs[both_obj_mask]
            obj_indices1 = y_true[obj_pairs[:, 0]]
            obj_indices2 = y_true[obj_pairs[:, 1]]
            
            valid_indices = ((obj_indices1 >= 0) & (obj_indices2 >= 0) & 
                        (obj_indices1 < self.lookup.shape[0]) & 
                        (obj_indices2 < self.lookup.shape[0]))
            
            if torch.any(valid_indices):
                valid_obj_indices1 = obj_indices1[valid_indices].long()
                valid_obj_indices2 = obj_indices2[valid_indices].long()
                valid_similarities = self.lookup[valid_obj_indices1, valid_obj_indices2]
                target_similarities[both_obj_mask][valid_indices] = valid_similarities
        
        return target_similarities
    
    def _compute_memory_loss_v2(self, y_true: torch.Tensor, y_pred_partial: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between current batch and ALL structures in memory bank.
        """
        # Get all initialized embeddings from memory
        memory_embs, memory_classes = self.memory_bank.get_all_initialized_embeddings()
        
        if memory_embs.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(y_true.shape[0]):
            if y_true[i] == -1:  # Skip background
                continue
                
            current_emb = F.normalize(y_pred_partial[i:i+1], p=2, dim=1)
            
            # Compute cosine similarities to ALL memory embeddings
            similarities = torch.mm(current_emb, memory_embs.t()).squeeze(0)
            
            # Get target similarities
            current_class = y_true[i].long()
            target_sims = self.lookup[current_class, memory_classes]
            
            # Get staleness weights
            weights = self.memory_bank.get_weights(memory_classes)
            
            # Weighted L1 loss
            weighted_loss = self.l1loss(similarities, target_sims) * weights
            
            total_loss += torch.sum(weighted_loss)
            num_pairs += torch.sum(weights > 0.1)  # Count only reasonably fresh entries
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
