import sqlite3
import numpy as np

class SimilarityCalculator:
    def __init__(self, db_path):
        self.db_path = db_path
        
    def load_matrix(self, structure_ids=None):
        """
        Load pre-computed affinity matrix for specified structures from database.
        Args:
            structure_ids (List[str]): List of structure IDs to include in matrix.
                                     If None, loads all available structures.
        Returns:
            Tuple[np.ndarray, List[str]]: Normalized affinity matrix and list of molecule IDs
        """
        if not structure_ids:
            raise ValueError("No structure IDs provided")
            
        structure_ids = [sid.lower() for sid in structure_ids]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        print(f"Loading matrix for {structure_ids}")
        
        try:
            # Create a N x N matrix for the specified structures
            n = len(structure_ids)
            matrix = np.zeros((n, n), dtype=np.float32)
            
            # Fetch all pairwise affinities for these structures
            placeholders = ','.join(['?'] * len(structure_ids))
            query = f"""
            SELECT 
                molecule_id1,
                molecule_id2,
                affinity_value
            FROM affinities 
            WHERE (molecule_id1 IN ({placeholders}) AND molecule_id2 IN ({placeholders}))
            """
            cursor.execute(query, structure_ids + structure_ids)
            rows = cursor.fetchall()
            
            # Build matrix from pairwise values
            id_to_idx = {sid: idx for idx, sid in enumerate(structure_ids)}
            for mol1, mol2, value in rows:
                # Only include if both molecules are in our structure list
                if mol1 in id_to_idx and mol2 in id_to_idx:
                    i = id_to_idx[mol1]
                    j = id_to_idx[mol2]
                    matrix[i, j] = value
                    matrix[j, i] = value  # Ensure symmetry
                    
            # Fill diagonal with 1.0
            np.fill_diagonal(matrix, 1.0)
            
            # Handle background structures - make their rows negative except diagonal
            for idx, struct_id in enumerate(structure_ids):
                if 'background' in struct_id.lower():
                    # Make entire row negative except diagonal
                    matrix[idx, :] = -np.abs(matrix[idx, :])
                    matrix[idx, idx] = 1.0  # Reset diagonal to 1.0
            
            # Verify we have all needed similarities
            missing = []
            for i, mol1 in enumerate(structure_ids):
                for j, mol2 in enumerate(structure_ids[i+1:], i+1):  # Only check upper triangle
                    if matrix[i, j] == 0:
                        missing.append((mol1, mol2))
                        
            if missing:
                # Let's also check if these structures exist in the database at all
                for mol1, mol2 in missing[:5]:  # Check first few missing pairs
                    cursor.execute("SELECT COUNT(*) FROM features WHERE molecule_id IN (?, ?)", (mol1, mol2))
                    count = cursor.fetchone()[0]
                    if count < 2:
                        print(f"Warning: One or both molecules {mol1}, {mol2} not found in features table")
                
                missing_pairs = ', '.join([f'({m1}, {m2})' for m1, m2 in missing[:5]])
                extra = f" and {len(missing) - 5} more pairs" if len(missing) > 5 else ""
                breakpoint()
                raise ValueError(f"Missing similarity values for pairs: {missing_pairs}{extra}")
            
            print(f"\nSimilarity Matrix molecules: {structure_ids}")
            
            # Global normalization of off-diagonal elements
            # Create a mask for off-diagonal elements
            off_diag_mask = ~np.eye(n, dtype=bool)
            
            # Get off-diagonal elements
            off_diag_values = matrix[off_diag_mask]
            
            # Find global min and max for off-diagonal elements
            global_min = np.min(off_diag_values)
            global_max = np.max(off_diag_values)
            
            print(f"Global similarity range - Min: {global_min:.4f}, Max: {global_max:.4f}")
            
            # Only normalize if we have a reasonable range
            if np.abs(global_max - global_min) > 1e-6:
                # Create a normalized copy
                normalized_matrix = np.copy(matrix)
                
                # Normalize off-diagonal elements to [0, 1] range
                normalized_matrix[off_diag_mask] = (off_diag_values - global_min) / (global_max - global_min)
                
                # Optionally scale to [-1, 1] range if using negative similarities
                if global_min < 0 or np.any(matrix < 0):
                    normalized_matrix[off_diag_mask] = 2.0 * normalized_matrix[off_diag_mask] - 1.0
                
                # Ensure the diagonal remains at 1.0
                np.fill_diagonal(normalized_matrix, 1.0)
                
                return normalized_matrix, structure_ids
            else:
                print("No normalization applied due to small range of similarity values")
                return matrix, structure_ids
            
        finally:
            conn.close()