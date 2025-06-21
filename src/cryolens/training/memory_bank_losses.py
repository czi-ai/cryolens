"""
Memory Bank enhanced loss functions for CryoLens model training.

These losses maintain a per-class memory bank of embeddings with timestamps
to provide better class separation during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Optional


class ClassMemoryBank:
    """Per-class memory bank that stores the most recent embedding for each class."""
    
    def __init__(self, num_classes: int, embedding_dim: int, device: torch.device):
        """
        Initialize per-class memory bank.
        
        Parameters
        ----------
        num_classes : int
            Maximum number of classes (size of lookup table)
        embedding_dim : int
            Dimension of embeddings
        device : torch.device
            Device to store embeddings on
        """
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Store one embedding per class
        self.embeddings = torch.zeros((num_classes, embedding_dim), device=device)
        self.timestamps = torch.zeros(num_classes, device=device)  # Time of last update
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)  # Track which classes have been seen
        
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
        current_time = time.time()
        
        # Detach to prevent gradient flow
        embeddings = embeddings.detach()
        labels = labels.detach()
        
        # Update each class
        for i in range(labels.shape[0]):
            label = labels[i].item()
            if label >= 0 and label < self.num_classes:  # Valid class (not background)
                # Update embedding with exponential moving average if already initialized
                if self.initialized[label]:
                    # EMA update: new = alpha * new + (1 - alpha) * old
                    alpha = 0.7  # Weight for new embedding
                    self.embeddings[label] = alpha * embeddings[i] + (1 - alpha) * self.embeddings[label]
                else:
                    # First time seeing this class
                    self.embeddings[label] = embeddings[i]
                    self.initialized[label] = True
                
                self.timestamps[label] = current_time
    
    def get_weights(self, labels: torch.Tensor, current_time: Optional[float] = None) -> torch.Tensor:
        """
        Get time-based weights for given labels.
        
        Parameters
        ----------
        labels : torch.Tensor
            Class labels to get weights for
        current_time : float, optional
            Current time (defaults to time.time())
            
        Returns
        -------
        torch.Tensor
            Weights based on recency (newer = higher weight)
        """
        if current_time is None:
            current_time = time.time()
            
        # Calculate time differences
        time_diffs = current_time - self.timestamps[labels]
        
        # Convert to weights using exponential decay
        # Recent updates get weight close to 1, older updates decay
        decay_rate = 0.01  # Controls how fast weights decay
        weights = torch.exp(-decay_rate * time_diffs)
        
        # Set weight to 0 for uninitialized classes
        weights[~self.initialized[labels]] = 0.0
        
        return weights


class ContrastiveAffinityLossWithMemory(nn.Module):
    """
    Contrastive affinity loss that uses a per-class memory bank.
    This is a minimal modification of the original ContrastiveAffinityLoss.
    """
    
    def __init__(
        self, 
        lookup: torch.Tensor, 
        device: torch.device, 
        latent_ratio: float = 0.75, 
        margin: float = 4.0,
        use_memory_bank: bool = True
    ):
        """
        Initialize memory-enhanced contrastive loss.
        
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
        use_memory_bank : bool
            Whether to use memory bank (default: True)
        """
        super().__init__()
        
        # Register lookup buffer
        lookup = lookup.clone().detach().contiguous()
        self.register_buffer('lookup', lookup, persistent=False)
        self.device = device
        self.latent_ratio = latent_ratio
        self.margin = margin
        self.use_memory_bank = use_memory_bank
        
        # Background similarity values
        self.background_sim = 0.2
        self.background_other_sim = 0.01
        
        # Memory bank will be initialized on first forward pass
        self.memory_bank = None
        self.num_classes = lookup.shape[0]
        
    def _init_memory_bank(self, embedding_dim: int):
        """Initialize memory bank with correct embedding dimension."""
        if self.use_memory_bank:
            self.memory_bank = ClassMemoryBank(self.num_classes, embedding_dim, self.device)
        
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
        """
        Compute contrastive affinity loss using memory bank.
        
        This is mostly the same as the original ContrastiveAffinityLoss,
        but adds memory bank updates and uses memory embeddings in loss calculation.
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
            if self.memory_bank is None and self.use_memory_bank:
                self._init_memory_bank(n_dims_to_use)
            
            # Update memory bank with current batch (only non-background)
            if self.use_memory_bank and self.memory_bank is not None:
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
            
            # Calculate standard contrastive loss
            similar_term = target_similarities * (distances ** 2)
            margin_dist = torch.clamp(self.margin - distances, min=0)
            dissimilar_term = (1 - target_similarities) * (margin_dist ** 2)
            batch_losses = similar_term + dissimilar_term
            
            # Add memory bank loss if enabled
            if self.use_memory_bank and self.memory_bank is not None:
                memory_losses = self._compute_memory_loss(y_true, y_pred_partial)
                # Combine batch and memory losses
                total_loss = 0.7 * torch.mean(batch_losses) + 0.3 * memory_losses
            else:
                total_loss = torch.mean(batch_losses)
            
            # Return per-sample losses or mean loss
            if per_sample and not self.use_memory_bank:
                return batch_losses.contiguous()
            
            # Handle NaN values
            if torch.isnan(total_loss):
                zero_tensor = torch.tensor(0.0, device=self.device)
                total_loss = zero_tensor.requires_grad_() + total_loss.detach() * 0
                
            return total_loss

        except Exception as e:
            print(f"Error in memory contrastive loss: {str(e)}")
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
    
    def _compute_memory_loss(self, y_true: torch.Tensor, y_pred_partial: torch.Tensor) -> torch.Tensor:
        """Compute loss between current batch and memory bank."""
        current_time = time.time()
        total_loss = 0.0
        num_pairs = 0
        
        # For each sample in batch
        for i in range(y_true.shape[0]):
            if y_true[i] == -1:  # Skip background
                continue
                
            # Get current sample's embedding
            current_emb = F.normalize(y_pred_partial[i:i+1], p=2, dim=1)
            
            # Compare with all initialized classes in memory
            initialized_classes = torch.where(self.memory_bank.initialized)[0]
            if len(initialized_classes) == 0:
                continue
                
            # Get memory embeddings
            memory_embs = F.normalize(self.memory_bank.embeddings[initialized_classes], p=2, dim=1)
            
            # Compute distances
            distances = torch.norm(current_emb - memory_embs, p=2, dim=1)
            
            # Get target similarities
            current_class = y_true[i].long()
            
            # Check if current_class is within bounds
            if current_class < 0 or current_class >= self.lookup.shape[0]:
                continue
                
            target_sims = self.lookup[current_class, initialized_classes]
            
            # Get time-based weights
            weights = self.memory_bank.get_weights(initialized_classes, current_time)
            
            # Weighted contrastive loss
            similar_term = target_sims * (distances ** 2) * weights
            margin_dist = torch.clamp(self.margin - distances, min=0)
            dissimilar_term = (1 - target_sims) * (margin_dist ** 2) * weights
            
            total_loss += torch.sum(similar_term + dissimilar_term)
            num_pairs += len(initialized_classes)
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)


class AffinityCosineLossWithMemory(nn.Module):
    """
    Cosine affinity loss that uses a per-class memory bank.
    This is a minimal modification of the original AffinityCosineLoss.
    """
    
    def __init__(
        self, 
        lookup: torch.Tensor, 
        device: torch.device, 
        latent_ratio: float = 0.75,
        use_memory_bank: bool = True
    ):
        """
        Initialize memory-enhanced cosine loss.
        
        Parameters
        ----------
        lookup : torch.Tensor
            Lookup table for molecule affinities
        device : torch.device
            Device to run computation on
        latent_ratio : float
            Ratio of latent dimensions to use
        use_memory_bank : bool
            Whether to use memory bank (default: True)
        """
        super().__init__()
        
        # Register lookup buffer
        lookup = lookup.clone().detach().contiguous()
        self.register_buffer('lookup', lookup, persistent=False)
        self.device = device
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.l1loss = nn.L1Loss(reduction="none")
        self.latent_ratio = latent_ratio
        self.use_memory_bank = use_memory_bank
        
        # Background similarity values
        self.background_sim = 0.2
        self.background_other_sim = 0.01
        
        # Memory bank
        self.memory_bank = None
        self.num_classes = lookup.shape[0]
        
    def _init_memory_bank(self, embedding_dim: int):
        """Initialize memory bank with correct embedding dimension."""
        if self.use_memory_bank:
            self.memory_bank = ClassMemoryBank(self.num_classes, embedding_dim, self.device)
        
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
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
            if self.memory_bank is None and self.use_memory_bank:
                self._init_memory_bank(n_dims_to_use)
            
            # Update memory bank
            if self.use_memory_bank and self.memory_bank is not None:
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
            
            # Add memory bank loss if enabled
            if self.use_memory_bank and self.memory_bank is not None:
                memory_losses = self._compute_memory_loss(y_true, y_pred_partial)
                total_loss = 0.7 * torch.mean(batch_losses) + 0.3 * memory_losses
            else:
                total_loss = torch.mean(batch_losses)
            
            # Return per-sample or mean loss
            if per_sample and not self.use_memory_bank:
                return batch_losses.contiguous()
            
            # Handle NaN values
            if torch.isnan(total_loss):
                zero_tensor = torch.tensor(0.0, device=self.device)
                total_loss = zero_tensor.requires_grad_() + total_loss.detach() * 0
                
            return total_loss

        except Exception as e:
            print(f"Error in memory cosine loss: {str(e)}")
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
    
    def _compute_memory_loss(self, y_true: torch.Tensor, y_pred_partial: torch.Tensor) -> torch.Tensor:
        """Compute loss between current batch and memory bank."""
        current_time = time.time()
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(y_true.shape[0]):
            if y_true[i] == -1:  # Skip background
                continue
                
            current_emb = F.normalize(y_pred_partial[i:i+1], p=2, dim=1)
            
            initialized_classes = torch.where(self.memory_bank.initialized)[0]
            if len(initialized_classes) == 0:
                continue
                
            memory_embs = F.normalize(self.memory_bank.embeddings[initialized_classes], p=2, dim=1)
            
            # Compute cosine similarities
            similarities = torch.mm(current_emb, memory_embs.t()).squeeze(0)
            
            # Get target similarities
            current_class = y_true[i].long()
            
            # Check if current_class is within bounds
            if current_class < 0 or current_class >= self.lookup.shape[0]:
                continue
                
            target_sims = self.lookup[current_class, initialized_classes]
            
            # Get time-based weights
            weights = self.memory_bank.get_weights(initialized_classes, current_time)
            
            # Weighted L1 loss
            weighted_loss = self.l1loss(similarities, target_sims) * weights
            
            total_loss += torch.sum(weighted_loss)
            num_pairs += len(initialized_classes)
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
