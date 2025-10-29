"""
Cross-validation with attention fusion support.

Extends stratified_cross_validation to handle attention fusion by training
the fusion model inside each CV fold to prevent test/train leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import average_precision_score, accuracy_score
import h5py


class SimpleAttentionFusion(nn.Module):
    """Simple attention-based fusion with 32D output."""
    def __init__(self, tt_dim: int = 32, cl_dim: int = 32, output_dim: int = 32):
        super().__init__()
        
        self.tt_attention = nn.Sequential(
            nn.Linear(tt_dim, 1),
            nn.Sigmoid()
        )
        
        self.cl_attention = nn.Sequential(
            nn.Linear(cl_dim, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(tt_dim + cl_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, tt_emb, cl_emb):
        att_tt = self.tt_attention(tt_emb)
        att_cl = self.cl_attention(cl_emb)
        
        tt_weighted = tt_emb * att_tt
        cl_weighted = cl_emb * att_cl
        
        combined = torch.cat([tt_weighted, cl_weighted], dim=1)
        output = self.fusion(combined)
        
        return output


def train_attention_fusion_fold(
    tt_train: np.ndarray,
    cl_train: np.ndarray,
    y_train: np.ndarray,
    tt_test: np.ndarray,
    cl_test: np.ndarray,
    n_classes: int,
    n_epochs: int = 10,
    device: str = 'cpu',
    random_seed: int = 42
) -> np.ndarray:
    """
    Train attention fusion on train fold and return test embeddings.
    
    This ensures no test/train leakage by training only on the train fold.
    
    Returns:
        Fused embeddings for test fold only (n_test, 32)
    """
    # Standardize
    scaler_tt = StandardScaler()
    scaler_cl = StandardScaler()
    
    tt_train_scaled = scaler_tt.fit_transform(tt_train)
    cl_train_scaled = scaler_cl.fit_transform(cl_train)
    tt_test_scaled = scaler_tt.transform(tt_test)
    cl_test_scaled = scaler_cl.transform(cl_test)
    
    # Create model
    model = SimpleAttentionFusion().to(device)
    classifier = nn.Linear(32, n_classes).to(device)
    
    # Setup training
    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()
    
    # Create data loaders
    train_data = TensorDataset(
        torch.FloatTensor(tt_train_scaled),
        torch.FloatTensor(cl_train_scaled),
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    # Train
    model.train()
    classifier.train()
    
    for epoch in range(n_epochs):
        for tt_batch, cl_batch, y_batch in train_loader:
            tt_batch = tt_batch.to(device)
            cl_batch = cl_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            fused = model(tt_batch, cl_batch)
            logits = classifier(fused)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
    
    # Get test embeddings
    model.eval()
    with torch.no_grad():
        fused_test = model(
            torch.FloatTensor(tt_test_scaled).to(device),
            torch.FloatTensor(cl_test_scaled).to(device)
        ).cpu().numpy()
    
    return fused_test


def stratified_cross_validation_with_attention(
    tt_embeddings: np.ndarray,
    cl_embeddings: np.ndarray,
    labels: List[str],
    n_folds: int = 10,
    n_epochs: int = 10,
    random_seed: int = 171717,
    device: str = 'cpu',
    return_predictions: bool = False
) -> Dict:
    """
    Stratified CV with attention fusion trained inside each fold (no leakage).
    
    Args:
        tt_embeddings: TomoTwin embeddings (N, 32)
        cl_embeddings: CryoLens embeddings (N, 32)
        labels: Class labels
        n_folds: Number of CV folds
        n_epochs: Epochs to train fusion per fold
        random_seed: Random seed
        device: Device for training
        return_predictions: Whether to return predictions
        
    Returns:
        Same format as stratified_cross_validation()
    """
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    n_classes = len(le.classes_)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    map_scores = []
    accuracy_scores = []
    predictions = [] if return_predictions else None
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(tt_embeddings, labels_encoded)):
        print(f"    Fold {fold_idx+1}/{n_folds}: Training attention fusion...")
        
        # Split data
        tt_train, tt_test = tt_embeddings[train_idx], tt_embeddings[test_idx]
        cl_train, cl_test = cl_embeddings[train_idx], cl_embeddings[test_idx]
        y_train, y_test = labels_encoded[train_idx], labels_encoded[test_idx]
        
        # Train fusion and get test embeddings (NO LEAKAGE)
        fused_test = train_attention_fusion_fold(
            tt_train, cl_train, y_train,
            tt_test, cl_test,
            n_classes, n_epochs, device, random_seed
        )
        
        # Train classifier on fused test embeddings
        # (We can't train on train fold since we'd need to fuse that too)
        # Instead, use test embeddings directly for evaluation
        
        # Actually, we need to fuse train fold too for classifier training
        # Let me fix this...
        fused_train = train_attention_fusion_fold(
            tt_train[:len(tt_train)//2], cl_train[:len(cl_train)//2], y_train[:len(y_train)//2],
            tt_train[len(tt_train)//2:], cl_train[len(cl_train)//2:],
            n_classes, n_epochs, device, random_seed
        )
        
        # Wait, this is getting complicated. Let me think...
        # We need fused embeddings for BOTH train and test
        # But we can only train the fusion on train data
        
        # Solution: Train fusion on train fold, apply to both train and test
        model = SimpleAttentionFusion().to(device)
        
        # Standardize
        scaler_tt = StandardScaler()
        scaler_cl = StandardScaler()
        
        tt_train_scaled = scaler_tt.fit_transform(tt_train)
        cl_train_scaled = scaler_cl.fit_transform(cl_train)
        tt_test_scaled = scaler_tt.transform(tt_test)
        cl_test_scaled = scaler_cl.transform(cl_test)
        
        # Train fusion model
        classifier_head = nn.Linear(32, n_classes).to(device)
        optimizer = optim.Adam(
            list(model.parameters()) + list(classifier_head.parameters()),
            lr=0.001
        )
        criterion = nn.CrossEntropyLoss()
        
        train_data = TensorDataset(
            torch.FloatTensor(tt_train_scaled),
            torch.FloatTensor(cl_train_scaled),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        
        model.train()
        classifier_head.train()
        
        for epoch in range(n_epochs):
            for tt_batch, cl_batch, y_batch in train_loader:
                tt_batch = tt_batch.to(device)
                cl_batch = cl_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                fused = model(tt_batch, cl_batch)
                logits = classifier_head(fused)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
        
        # Get fused embeddings for train and test
        model.eval()
        with torch.no_grad():
            fused_train = model(
                torch.FloatTensor(tt_train_scaled).to(device),
                torch.FloatTensor(cl_train_scaled).to(device)
            ).cpu().numpy()
            
            fused_test = model(
                torch.FloatTensor(tt_test_scaled).to(device),
                torch.FloatTensor(cl_test_scaled).to(device)
            ).cpu().numpy()
        
        # Now train logistic regression classifier on fused train embeddings
        clf = LogisticRegression(max_iter=1000, random_state=random_seed)
        clf.fit(fused_train, y_train)
        
        # Predict on test
        y_pred = clf.predict(fused_test)
        y_scores = clf.predict_proba(fused_test)
        
        # Store predictions if requested
        if return_predictions:
            y_test_labels = [labels[i] for i in test_idx]
            y_pred_labels = le.inverse_transform(y_pred)
            clf_classes_str = le.inverse_transform(clf.classes_)
            predictions.append((y_test_labels, y_pred_labels, y_scores, clf_classes_str))
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # MAP for multi-class
        map_scores_class = []
        unique_classes = np.unique(y_train)
        for class_label in unique_classes:
            y_true_binary = (y_test == class_label).astype(int)
            class_idx = np.where(clf.classes_ == class_label)[0]
            if len(class_idx) > 0 and np.sum(y_true_binary) > 0:
                y_score_binary = y_scores[:, class_idx[0]]
                map_score = average_precision_score(y_true_binary, y_score_binary)
                map_scores_class.append(map_score)
        
        if map_scores_class:
            map_scores.append(np.mean(map_scores_class))
        else:
            map_scores.append(0.0)
        
        accuracy_scores.append(accuracy)
    
    result = {
        'map_per_fold': np.array(map_scores),
        'accuracy_per_fold': np.array(accuracy_scores),
        'mean_map': float(np.mean(map_scores)),
        'std_map': float(np.std(map_scores)),
        'mean_accuracy': float(np.mean(accuracy_scores)),
        'std_accuracy': float(np.std(accuracy_scores))
    }
    
    if return_predictions:
        result['predictions'] = predictions
        result['label_encoder'] = le
    
    return result


def train_final_fusion_and_save(
    tt_embeddings: np.ndarray,
    cl_embeddings: np.ndarray,
    labels: List[str],
    output_h5_path: Path,
    sample_metadata: Optional[List[Dict]] = None,
    n_epochs: int = 20,
    random_seed: int = 171717,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Train final attention fusion model on full dataset and save fused embeddings.
    
    This function is called after cross-validation to train a single fusion model
    on all available data. The resulting fused embeddings can be reused for
    downstream tasks without needing to retrain the fusion model.
    
    Args:
        tt_embeddings: TomoTwin embeddings (N, 32)
        cl_embeddings: CryoLens embeddings (N, 32)
        labels: Class labels for all samples
        output_h5_path: Path to save fused embeddings HDF5 file
        sample_metadata: Optional list of metadata dicts (from CryoLens) containing run_id, coordinates, etc.
        n_epochs: Number of training epochs
        random_seed: Random seed for reproducibility
        device: Device for training ('cpu', 'cuda', 'mps')
        verbose: Whether to print progress
        
    Returns:
        Tuple of (fused_embeddings array, metadata dict)
    """
    if verbose:
        print("\nTraining final fusion model on full dataset...")
        print(f"  Samples: {len(labels)}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Device: {device}")
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    n_classes = len(le.classes_)
    
    # Standardize
    scaler_tt = StandardScaler()
    scaler_cl = StandardScaler()
    
    tt_scaled = scaler_tt.fit_transform(tt_embeddings)
    cl_scaled = scaler_cl.fit_transform(cl_embeddings)
    
    # Create model
    model = SimpleAttentionFusion().to(device)
    classifier_head = nn.Linear(32, n_classes).to(device)
    
    # Setup training
    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier_head.parameters()),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()
    
    # Create data loader
    train_data = TensorDataset(
        torch.FloatTensor(tt_scaled),
        torch.FloatTensor(cl_scaled),
        torch.LongTensor(labels_encoded)
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    # Train
    model.train()
    classifier_head.train()
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        
        for tt_batch, cl_batch, y_batch in train_loader:
            tt_batch = tt_batch.to(device)
            cl_batch = cl_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            fused = model(tt_batch, cl_batch)
            logits = classifier_head(fused)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if verbose and (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")
    
    # Generate fused embeddings for all samples
    if verbose:
        print("\nGenerating fused embeddings...")
    
    model.eval()
    with torch.no_grad():
        fused_embeddings = model(
            torch.FloatTensor(tt_scaled).to(device),
            torch.FloatTensor(cl_scaled).to(device)
        ).cpu().numpy()
    
    # Save to HDF5 in same format as CryoLens embeddings
    if verbose:
        print(f"\nSaving fused embeddings to {output_h5_path}...")
    
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_h5_path, 'w') as f:
        # Create embeddings group
        embeddings_group = f.create_group('embeddings')
        
        # Save each sample with metadata
        for i, (embedding, label) in enumerate(zip(fused_embeddings, labels)):
            # Get metadata for this sample if available
            if sample_metadata is not None and i < len(sample_metadata):
                meta = sample_metadata[i]
                sample_id = meta.get('sample_id', f"sample_{i}")
            else:
                meta = {}
                sample_id = f"sample_{i}"
            
            sample_group = embeddings_group.create_group(sample_id)
            sample_group.create_dataset('mu', data=embedding)
            
            # Save all metadata as attributes
            sample_group.attrs['structure_name'] = label
            
            # Save original CryoLens metadata if available
            if 'coordinates' in meta and meta['coordinates'] is not None:
                sample_group.attrs['coordinates'] = np.array(meta['coordinates'])
            if 'object_name' in meta and meta['object_name'] is not None:
                sample_group.attrs['object_name'] = meta['object_name']
            if 'picks_index' in meta and meta['picks_index'] is not None:
                sample_group.attrs['picks_index'] = meta['picks_index']
            if 'point_index' in meta and meta['point_index'] is not None:
                sample_group.attrs['point_index'] = meta['point_index']
            if 'run_name' in meta and meta['run_name'] is not None:
                sample_group.attrs['run_name'] = meta['run_name']
            if 'voxel_spacing' in meta and meta['voxel_spacing'] is not None:
                sample_group.attrs['voxel_spacing'] = meta['voxel_spacing']
        
        # Save metadata
        metadata_group = f.create_group('metadata')
        metadata_group.attrs['fusion_method'] = 'attention'
        metadata_group.attrs['n_samples'] = len(labels)
        metadata_group.attrs['embedding_dim'] = fused_embeddings.shape[1]
        metadata_group.attrs['n_epochs'] = n_epochs
        metadata_group.attrs['random_seed'] = random_seed
        metadata_group.attrs['n_classes'] = n_classes
        
        # Save class names
        metadata_group.create_dataset(
            'class_names',
            data=np.array(le.classes_, dtype='S')
        )
    
    if verbose:
        print(f"  Saved {len(labels)} fused embeddings (32D)")
        print(f"  Classes: {le.classes_.tolist()}")
    
    # Return embeddings and metadata
    metadata = {
        'n_samples': len(labels),
        'embedding_dim': fused_embeddings.shape[1],
        'n_classes': n_classes,
        'class_names': le.classes_.tolist(),
        'output_path': str(output_h5_path)
    }
    
    return fused_embeddings, metadata
