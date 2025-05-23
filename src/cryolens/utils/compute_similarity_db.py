#!/usr/bin/env python
# Script to compute similarity database from MRCS files for TomotWin
# Author: Kyle Harrington

import argparse
import numpy as np
import os
from pathlib import Path
import threading
from tqdm import tqdm
import numpy.ma as ma
from mrcfile import open as mrc_open
from scipy.ndimage import zoom
import sqlite3
import io
import logging
import sys
import shutil
import torch
from torch.nn import functional as F

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("tomotwin-similarity")

class DatabaseManager:
    """Manages storage and retrieval of features and similarity values"""
    
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    molecule_id TEXT PRIMARY KEY,
                    feature_data BLOB
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS affinities (
                    molecule_id1 TEXT,
                    molecule_id2 TEXT,
                    affinity_value REAL,
                    PRIMARY KEY (molecule_id1, molecule_id2)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS affinity_matrix (
                    id INTEGER PRIMARY KEY,
                    matrix_data BLOB,
                    molecule_order TEXT
                )
            """)
            conn.commit()
            logger.info(f"Database tables initialized in {self.db_path}")

    def _adapt_array(self, arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def _convert_array(self, blob):
        out = io.BytesIO(blob)
        out.seek(0)
        return np.load(out)

    def store_feature(self, molecule_id, feature):
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO features (molecule_id, feature_data) VALUES (?, ?)",
                        (molecule_id, self._adapt_array(feature))
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Error storing feature for {molecule_id}: {e}")

    def get_feature(self, molecule_id):
        """Retrieve feature for a molecule."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT feature_data FROM features WHERE molecule_id = ?",
                    (molecule_id,)
                )
                result = cursor.fetchone()
                if result:
                    return self._convert_array(result[0])
                return None
        except Exception as e:
            logger.error(f"Error retrieving feature for {molecule_id}: {e}")
            return None

    def store_affinity(self, mol_i, mol_j, value):
        """Store affinity value for a molecule pair - store in sorted order."""
        mol1, mol2 = sorted([mol_i, mol_j])
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO affinities (molecule_id1, molecule_id2, affinity_value) VALUES (?, ?, ?)",
                        (mol1, mol2, float(value))
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Error storing affinity for {mol1}-{mol2}: {e}")

    def count_affinities(self):
        """Count the number of affinity pairs in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM affinities")
                count = cursor.fetchone()[0]
                return count
        except Exception as e:
            logger.error(f"Error counting affinities: {e}")
            return 0

    def count_features(self):
        """Count the number of features in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM features")
                count = cursor.fetchone()[0]
                return count
        except Exception as e:
            logger.error(f"Error counting features: {e}")
            return 0

    def store_affinity_matrix(self, matrix, molecule_order):
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    molecule_order_json = ','.join(molecule_order)
                    conn.execute(
                        "INSERT OR REPLACE INTO affinity_matrix (id, matrix_data, molecule_order) VALUES (1, ?, ?)",
                        (self._adapt_array(matrix), molecule_order_json)
                    )
                    conn.commit()
                logger.info(f"Saved affinity matrix of shape {matrix.shape}")
            except Exception as e:
                logger.error(f"Error storing affinity matrix: {e}")

    def load_affinity_matrix(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT matrix_data, molecule_order FROM affinity_matrix WHERE id = 1"
                )
                result = cursor.fetchone()
                if result:
                    matrix = self._convert_array(result[0])
                    molecule_order = result[1].split(',')
                    return matrix, molecule_order
                return None, None
        except Exception as e:
            logger.error(f"Error loading affinity matrix: {e}")
            return None, None

    def clear_database(self):
        """Clear all tables in the database."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM features")
                    conn.execute("DELETE FROM affinities")
                    conn.execute("DELETE FROM affinity_matrix")
                    conn.commit()
                logger.info("Database cleared successfully")
                return True
            except Exception as e:
                logger.error(f"Error clearing database: {e}")
                return False


class MRCProcessor:
    """Loads and processes MRC files"""
    
    def __init__(self, mrc_dir, box_size=48):
        self.mrc_dir = Path(mrc_dir)
        self.box_size = box_size
        self.structures = {}
        self._find_mrcs()
    
    def _find_mrcs(self):
        """Find all MRC files in the directory"""
        mrc_files = {}
        for ext in ['.mrc', '.mrcs']:
            for mrc_path in self.mrc_dir.glob(f"*{ext}"):
                # Handle different naming conventions
                # Convert JCVI-syn3.0_76285.mrc to JCVI-syn3.0-76285
                stem = mrc_path.stem
                parts = stem.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    structure_id = f"{parts[0]}-{parts[1]}"
                else:
                    structure_id = stem
                
                mrc_files[structure_id] = mrc_path
        
        logger.info(f"Found {len(mrc_files)} MRC files in {self.mrc_dir}")
        self.mrc_files = mrc_files
    
    def get_structure_ids(self):
        """Get all available structure IDs"""
        return list(self.mrc_files.keys())
    
    def load_structure(self, structure_id):
        """Load a specific structure"""
        if structure_id in self.structures:
            return True
        
        if structure_id not in self.mrc_files:
            logger.warning(f"Structure {structure_id} not found")
            return False
        
        try:
            mrc_path = self.mrc_files[structure_id]
            with mrc_open(mrc_path, mode='r') as mrc:
                density_map = mrc.data.astype(np.float32)
                
                # Resize if needed
                if density_map.shape != (self.box_size,) * 3:
                    logger.info(f"Resizing {mrc_path.name} from {density_map.shape} to {(self.box_size,) * 3}")
                    scale_factor = self.box_size / density_map.shape[0]
                    density_map = zoom(density_map, (scale_factor,) * 3, 
                                      order=1, mode='constant', cval=0.0,
                                      grid_mode=True)
                
                # Normalize
                if density_map.max() > density_map.min():
                    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min())
                
                self.structures[structure_id] = density_map
                return True
                
        except Exception as e:
            logger.error(f"Failed to load structure {structure_id}: {e}")
            return False
    
    def get_density(self, structure_id):
        """Get density map for a structure"""
        if structure_id not in self.structures:
            if not self.load_structure(structure_id):
                raise KeyError(f"Structure {structure_id} could not be loaded")
        
        return self.structures[structure_id]


class SimilarityCalculator:
    """Calculates feature vectors and similarity matrix"""
    
    def __init__(self, mrc_processor, db_path, voxel_size=5.0):
        self.mrc_processor = mrc_processor
        self.db = DatabaseManager(db_path)
        self.voxel_size = voxel_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def _extract_features(self, density):
        """Extract features from density map using a simple feature extraction method"""
        # Convert to tensor and add batch and channel dimensions
        tensor = torch.from_numpy(density).float().to(self.device)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

        # Simple feature extraction: average pooling to reduce dimensions
        # For a box_size of 48, this will reduce to 6x6x6 = 216 features
        pool_size = 8  # Adjust based on box_size
        pooled = F.avg_pool3d(tensor, kernel_size=pool_size)
        
        # Flatten to 1D feature vector
        features = pooled.reshape(-1).cpu().numpy()
        
        # Scale features by voxel size
        scaling_factor = 1.0 / self.voxel_size if self.voxel_size > 0 else 1.0
        features = features * scaling_factor
        
        return features

    def _calculate_similarity(self, feature1, feature2):
        """Calculate cosine similarity between two feature vectors."""
        # Convert to tensors
        f1 = torch.from_numpy(feature1).float()
        f2 = torch.from_numpy(feature2).float()
        
        # Normalize the features (L2 norm)
        f1_norm = F.normalize(f1, p=2, dim=0)
        f2_norm = F.normalize(f2, p=2, dim=0)
        
        # Compute cosine similarity
        similarity = torch.dot(f1_norm, f2_norm).item()
        
        return similarity

    def check_matrix_completeness(self, structure_ids):
        """Check if the similarity matrix is complete for the given structure IDs."""
        # Count features
        feature_count = self.db.count_features()
        
        # Count affinity pairs (diagonal excluded)
        affinity_count = self.db.count_affinities()
        
        # Calculate expected counts
        expected_features = len(structure_ids)
        expected_affinities = (expected_features * (expected_features - 1)) // 2  # Upper triangle
        
        logger.info(f"Feature count: {feature_count}/{expected_features}")
        logger.info(f"Affinity pair count: {affinity_count}/{expected_affinities}")
        
        # Check if we have all the features and affinities
        features_complete = feature_count >= expected_features
        affinities_complete = affinity_count >= expected_affinities
        
        return features_complete and affinities_complete

    def calculate_matrix(self, structure_ids=None, normalize=True, batch_size=1000):
        """Calculate feature vectors and similarity matrix"""
        if not structure_ids:
            structure_ids = self.mrc_processor.get_structure_ids()
        
        n_structures = len(structure_ids)
        if n_structures == 0:
            logger.error("No structures found for similarity calculation")
            return None, None
        
        # Calculate features
        logger.info("Computing features for all structures...")
        features = []
        valid_structures = []
        
        for struct_id in tqdm(structure_ids, desc="Computing features"):
            try:
                # Check if feature already exists in database
                existing_feature = self.db.get_feature(struct_id)
                if existing_feature is not None:
                    logger.info(f"Using existing feature for {struct_id}")
                    features.append(existing_feature)
                    valid_structures.append(struct_id)
                    continue
                
                # Extract new feature
                density = self.mrc_processor.get_density(struct_id)
                feature = self._extract_features(density)
                features.append(feature)
                valid_structures.append(struct_id)
                self.db.store_feature(struct_id, feature)
            except Exception as e:
                logger.error(f"Error computing features for {struct_id}: {e}")
        
        n_valid = len(valid_structures)
        logger.info(f"Successfully computed features for {n_valid}/{n_structures} structures")
        
        # Initialize affinity matrix
        affinity = np.eye(n_valid)
        
        # Calculate affinities for upper triangle
        i_indices, j_indices = np.triu_indices(n_valid, k=1)  # k=1 to exclude diagonal
        total_pairs = len(i_indices)
        
        logger.info(f"Calculating {total_pairs} pairwise affinities...")
        for idx in tqdm(range(total_pairs), desc="Computing affinities"):
            i, j = i_indices[idx], j_indices[idx]
            mol1, mol2 = valid_structures[i], valid_structures[j]
            
            # Check if affinity already exists in the database
            # Since we store in sorted order, we need to sort the molecule IDs
            sorted_ids = sorted([mol1, mol2])
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT affinity_value FROM affinities WHERE molecule_id1 = ? AND molecule_id2 = ?",
                        (sorted_ids[0], sorted_ids[1])
                    )
                    result = cursor.fetchone()
                    if result:
                        aff_value = result[0]
                        logger.debug(f"Using existing affinity for {mol1}-{mol2}: {aff_value:.4f}")
                    else:
                        # Calculate new affinity
                        aff_value = self._calculate_similarity(features[i], features[j])
                        # Store in database
                        self.db.store_affinity(mol1, mol2, aff_value)
                        
                    # Store in matrix
                    affinity[i, j] = affinity[j, i] = aff_value
                    
                    # Log some example affinities
                    if idx < 5 or idx % (total_pairs // 10) == 0:
                        logger.info(f"Affinity {mol1} <-> {mol2}: {aff_value:.4f}")
            except Exception as e:
                logger.error(f"Error processing affinity for {mol1}-{mol2}: {e}")
        
        # Handle background structures - make their rows negative except diagonal
        for idx, struct_id in enumerate(valid_structures):
            if 'background' in struct_id.lower():
                # Make entire row negative except diagonal
                affinity[idx, :] = -np.abs(affinity[idx, :])
                affinity[idx, idx] = 1.0  # Reset diagonal to 1.0
        
        # Apply global normalization to off-diagonal elements
        if normalize:
            # Create a mask for off-diagonal elements
            off_diag_mask = ~np.eye(n_valid, dtype=bool)
            
            # Get off-diagonal elements
            off_diag_values = affinity[off_diag_mask]
            
            # Find global min and max for off-diagonal elements
            global_min = np.min(off_diag_values)
            global_max = np.max(off_diag_values)
            
            logger.info(f"Global min: {global_min:.4f}, Global max: {global_max:.4f}")
            
            # Only normalize if we have a reasonable range
            if np.abs(global_max - global_min) > 1e-6:
                # Create a normalized copy
                normalized_affinity = np.copy(affinity)
                
                # Normalize off-diagonal elements to [0, 1] range
                normalized_affinity[off_diag_mask] = (off_diag_values - global_min) / (global_max - global_min)
                
                # Optionally scale to [-1, 1] range if using negative similarities
                if global_min < 0 or np.any(affinity < 0):
                    normalized_affinity[off_diag_mask] = 2.0 * normalized_affinity[off_diag_mask] - 1.0
                
                # Ensure the diagonal remains at 1.0
                np.fill_diagonal(normalized_affinity, 1.0)
                
                # Log comparison between old and new normalization
                for i, struct_id in enumerate(valid_structures):
                    min_val = normalized_affinity[i][normalized_affinity[i] < 1.0].min() if np.any(normalized_affinity[i] < 1.0) else 0
                    max_val = normalized_affinity[i].max()
                    logger.info(f"Row {i} ({struct_id}) after global norm - Min: {min_val:.4f}, Max: {max_val:.4f}")
                
                affinity = normalized_affinity
            else:
                logger.warning("No normalization applied due to small range of similarity values")
        
        # Store the affinity matrix
        logger.info("Storing affinity matrix and individual pairs in database...")
        self.db.store_affinity_matrix(affinity, valid_structures)
        
        # Verify we stored all affinities
        pair_count = self.db.count_affinities()
        logger.info(f"Total pairs stored in database: {pair_count}")
        
        return affinity, valid_structures


def main():
    parser = argparse.ArgumentParser(description="Compute similarity database for TomotWin")
    parser.add_argument("--mrcs_dir", type=str, required=True, help="Directory containing MRC files")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite database file")
    parser.add_argument("--box_size", type=int, default=48, help="Box size for density maps")
    parser.add_argument("--voxel_size", type=float, default=5.0, help="Voxel size in Angstroms")
    parser.add_argument("--force_recalculate", action="store_true", help="Force recalculation of matrix")
    parser.add_argument("--clean_db", action="store_true", help="Clean database before calculation")
    parser.add_argument("--backup_db", action="store_true", help="Backup database before modification")
    args = parser.parse_args()

    # Initialize MRC processor
    mrc_processor = MRCProcessor(args.mrcs_dir, box_size=args.box_size)
    
    # Check if we have structures
    structure_ids = mrc_processor.get_structure_ids()
    if not structure_ids:
        logger.error("No MRC files found in the directory. Exiting.")
        return
    
    # Backup database if requested
    if args.backup_db and os.path.exists(args.db_path):
        backup_path = f"{args.db_path}.bak"
        logger.info(f"Backing up database to {backup_path}")
        try:
            shutil.copy2(args.db_path, backup_path)
            logger.info("Backup created successfully")
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
    
    # Initialize similarity calculator
    calculator = SimilarityCalculator(
        mrc_processor=mrc_processor,
        db_path=args.db_path,
        voxel_size=args.voxel_size
    )
    
    # Clean the database if requested
    if args.clean_db:
        logger.info("Cleaning database as requested...")
        calculator.db.clear_database()
    
    # Check if matrix is complete for the given structures
    matrix_complete = calculator.check_matrix_completeness(structure_ids)
    
    # Check if we need to calculate or load
    if not args.force_recalculate and matrix_complete:
        # Try to load existing matrix
        matrix, molecule_order = calculator.db.load_affinity_matrix()
        if matrix is not None:
            logger.info(f"Loaded existing affinity matrix of shape {matrix.shape}")
            logger.info(f"Number of structures: {len(molecule_order)}")
            
            # Print range of values for each row
            for i, struct_id in enumerate(molecule_order):
                min_val = matrix[i][matrix[i] < 1.0].min() if np.any(matrix[i] < 1.0) else 0
                max_val = matrix[i].max()
                logger.info(f"Row {i} ({struct_id}) - Min: {min_val:.4f}, Max: {max_val:.4f}")
            
            return
    else:
        if args.force_recalculate:
            logger.info("Forcing recalculation of similarity matrix.")
        else:
            logger.info("Matrix is incomplete. Calculating similarity matrix.")
    
    # Calculate matrix
    logger.info("Computing features and affinities...")
    affinity_matrix, structures = calculator.calculate_matrix(structure_ids=structure_ids)
    
    if affinity_matrix is not None:
        logger.info(f"Computed affinity matrix of shape {affinity_matrix.shape}")
        logger.info(f"Similarity database has been created at {args.db_path}")
        
        # Print range of values for each row
        for i, struct_id in enumerate(structures):
            min_val = affinity_matrix[i][affinity_matrix[i] < 1.0].min() if np.any(affinity_matrix[i] < 1.0) else 0
            max_val = affinity_matrix[i].max()
            logger.info(f"Row {i} ({struct_id}) - Min: {min_val:.4f}, Max: {max_val:.4f}")

if __name__ == "__main__":
    main()
