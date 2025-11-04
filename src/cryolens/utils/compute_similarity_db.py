#!/usr/bin/env python
# Script to compute similarity database from MRCS files for TomotWin
# Author: Kyle Harrington

import argparse
import numpy as np
import os
from pathlib import Path
import threading
import numpy.ma as ma
from mrcfile import open as mrc_open
from scipy.ndimage import zoom, gaussian_filter
import sqlite3
import io
import logging
import sys
import shutil
import torch
from torch.nn import functional as F
import pandas as pd

from cryolens.utils.optional_deps import get_tqdm, require_gemmi

# Get tqdm (or mock if not available)
tqdm = get_tqdm()

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
    
    def __init__(self, mrc_dir, box_size=48, voxel_size=10.0, resolution=15.0, pdb_dir=None):
        self.mrc_dir = Path(mrc_dir) if mrc_dir else None
        self.pdb_dir = Path(pdb_dir) if pdb_dir else None
        self.box_size = box_size
        self.voxel_size = voxel_size  # Angstroms per voxel
        self.resolution = resolution  # Target resolution in Angstroms
        self.structures = {}
        self.structure_info = {}  # Store metadata about structures
        
        if self.mrc_dir:
            self._find_mrcs()
        if self.pdb_dir:
            self._find_pdbs()
    
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
    
    def _find_pdbs(self):
        """Find all PDB files in the directory"""
        pdb_files = {}
        for pdb_path in self.pdb_dir.glob("*.pdb"):
            structure_id = pdb_path.stem
            pdb_files[structure_id] = pdb_path
        
        logger.info(f"Found {len(pdb_files)} PDB files in {self.pdb_dir}")
        self.pdb_files = pdb_files
    
    def get_structure_ids(self):
        """Get all available structure IDs"""
        ids = set()
        if hasattr(self, 'mrc_files'):
            ids.update(self.mrc_files.keys())
        if hasattr(self, 'pdb_files'):
            ids.update(self.pdb_files.keys())
        return list(ids)
    
    def _pdb_to_density(self, filename, box_size=None, voxel_size=None, resolution=None):
        """Convert PDB structure to density map with proper physical units."""
        box_size = box_size or self.box_size
        voxel_size = voxel_size or self.voxel_size
        resolution = resolution or self.resolution
        
        # Check for gemmi dependency
        gemmi = require_gemmi()
        
        try:
            # Read structure
            st = gemmi.read_structure(str(filename))
            
            # Extract coordinates (already in Angstroms) with mass weighting
            coords = []
            masses = []
            for model in st:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            pos = atom.pos
                            coords.append([pos.x, pos.y, pos.z])
                            # Use actual atomic mass
                            masses.append(atom.element.weight if atom.element else 12.0)
            
            if not coords:
                raise RuntimeError("No atoms found in structure")
                
            coords = np.array(coords)
            masses = np.array(masses)
            
            # Center coordinates by center of mass
            center = np.average(coords, weights=masses, axis=0)
            coords = coords - center
            
            # Convert coordinates to voxel space (no scaling - preserve actual size)
            coords_voxels = coords / voxel_size
            
            # Shift to center of box
            coords_voxels = coords_voxels + box_size/2
            
            # Create weighted density map
            density = np.zeros((box_size, box_size, box_size), dtype=np.float32)
            
            # Check which atoms are within bounds
            in_bounds = np.all((coords_voxels >= 0) & (coords_voxels < box_size - 1), axis=1)
            coords_voxels = coords_voxels[in_bounds]
            masses_in_bounds = masses[in_bounds]
            
            if len(coords_voxels) == 0:
                logger.warning(f"No atoms within box for {filename.name}")
                return density
            
            # Place atoms with trilinear interpolation
            for coord, mass in zip(coords_voxels, masses_in_bounds):
                # Get integer and fractional parts
                idx = coord.astype(int)
                frac = coord - idx
                
                # Trilinear interpolation weights
                for dx in range(2):
                    for dy in range(2):
                        for dz in range(2):
                            weight = (1 - abs(dx - frac[0])) * \
                                    (1 - abs(dy - frac[1])) * \
                                    (1 - abs(dz - frac[2]))
                            density[idx[0]+dx, idx[1]+dy, idx[2]+dz] += mass * weight
            
            # Convert resolution to sigma in voxel units
            sigma_voxels = resolution / (2.355 * voxel_size)  # FWHM to sigma conversion
            
            # Smooth to target resolution
            if sigma_voxels > 0:
                density = gaussian_filter(density, sigma=sigma_voxels)
            
            # Normalize
            if density.max() > density.min():
                density = (density - density.min()) / (density.max() - density.min())
            
            return density
            
        except Exception as e:
            raise RuntimeError(f"Failed to process PDB file: {str(e)}")
    
    def _get_structure_info(self, pdb_path):
        """Extract structure information from PDB file."""
        # Check for gemmi dependency
        gemmi = require_gemmi()
        
        try:
            st = gemmi.read_structure(str(pdb_path))
            
            coords = []
            masses = []
            elements = {}
            chains = []
            
            for model in st:
                for chain in model:
                    chains.append(chain.name)
                    for residue in chain:
                        for atom in residue:
                            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                            elem = atom.element.name if atom.element else 'X'
                            elements[elem] = elements.get(elem, 0) + 1
                            masses.append(atom.element.weight if atom.element else 12.0)
            
            if coords:
                coords = np.array(coords)
                masses = np.array(masses)
                
                # Calculate center of mass
                center = np.average(coords, weights=masses, axis=0)
                coords_centered = coords - center
                
                # Radius of gyration
                rg = np.sqrt(np.average(np.sum(coords_centered**2, axis=1), weights=masses))
                
                # Extent
                extent = coords.max(axis=0) - coords.min(axis=0)
                
                # Molecular weight
                molecular_weight = np.sum(masses)
                
                return {
                    'name': pdb_path.stem,
                    'n_atoms': len(coords),
                    'n_chains': len(set(chains)),
                    'molecular_weight_da': molecular_weight,
                    'molecular_weight_kda': molecular_weight / 1000,
                    'extent_angstroms': extent.tolist(),
                    'max_extent_angstroms': float(np.max(extent)),
                    'radius_of_gyration': float(rg),
                    'estimated_diameter_nm': float(np.max(extent) / 10),
                }
            else:
                return {'name': pdb_path.stem, 'error': 'No atoms found'}
                
        except Exception as e:
            return {'name': pdb_path.stem, 'error': str(e)}
    
    def load_structure(self, structure_id):
        """Load a specific structure from MRC or PDB"""
        if structure_id in self.structures:
            return True
        
        # Try MRC first
        if hasattr(self, 'mrc_files') and structure_id in self.mrc_files:
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
                logger.error(f"Failed to load MRC structure {structure_id}: {e}")
        
        # Try PDB
        if hasattr(self, 'pdb_files') and structure_id in self.pdb_files:
            try:
                pdb_path = self.pdb_files[structure_id]
                
                # Get structure info
                info = self._get_structure_info(pdb_path)
                self.structure_info[structure_id] = info
                
                # Convert to density
                density_map = self._pdb_to_density(pdb_path)
                self.structures[structure_id] = density_map
                
                logger.info(f"Loaded PDB {structure_id}: {info.get('molecular_weight_kda', 0):.1f} kDa, "
                           f"{info.get('estimated_diameter_nm', 0):.1f} nm diameter")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load PDB structure {structure_id}: {e}")
        
        logger.warning(f"Structure {structure_id} not found in MRC or PDB files")
        return False
    
    def get_density(self, structure_id):
        """Get density map for a structure"""
        if structure_id not in self.structures:
            if not self.load_structure(structure_id):
                raise KeyError(f"Structure {structure_id} could not be loaded")
        
        return self.structures[structure_id]


class SimilarityCalculator:
    """Calculates feature vectors and similarity matrix"""
    
    def __init__(self, mrc_processor, db_path):
        self.mrc_processor = mrc_processor
        self.db = DatabaseManager(db_path)
        self.voxel_size = mrc_processor.voxel_size  # Get from processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using voxel size: {self.voxel_size} Å/voxel")

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
    parser.add_argument("--mrcs_dir", type=str, help="Directory containing MRC files")
    parser.add_argument("--pdb_dir", type=str, help="Directory containing PDB files")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite database file")
    parser.add_argument("--box_size", type=int, default=48, help="Box size for density maps")
    parser.add_argument("--voxel_size", type=float, default=10.0, help="Voxel size in Angstroms (default: 10.0)")
    parser.add_argument("--resolution", type=float, default=15.0, help="Target resolution in Angstroms (default: 15.0)")
    parser.add_argument("--force_recalculate", action="store_true", help="Force recalculation of matrix")
    parser.add_argument("--clean_db", action="store_true", help="Clean database before calculation")
    parser.add_argument("--backup_db", action="store_true", help="Backup database before modification")
    args = parser.parse_args()
    
    # Validate that at least one directory is provided
    if not args.mrcs_dir and not args.pdb_dir:
        parser.error("At least one of --mrcs_dir or --pdb_dir must be provided")

    # Initialize MRC processor (now handles both MRC and PDB)
    mrc_processor = MRCProcessor(
        mrc_dir=args.mrcs_dir, 
        pdb_dir=args.pdb_dir,
        box_size=args.box_size,
        voxel_size=args.voxel_size,
        resolution=args.resolution
    )
    
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
        db_path=args.db_path
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
