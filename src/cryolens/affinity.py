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
            
        # Don't convert to lowercase here - preserve original case
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        print(f"Loading matrix for {structure_ids}")
        
        try:
            # First, get the actual case for each structure ID from the database
            # This handles mixed case databases
            placeholders = ','.join(['?'] * len(structure_ids))
            cursor.execute(f"""
                SELECT DISTINCT molecule_id 
                FROM features 
                WHERE LOWER(molecule_id) IN ({placeholders})
            """, [sid.lower() for sid in structure_ids])
            
            # Create a mapping from lowercase to actual case in DB
            db_molecules = {row[0].lower(): row[0] for row in cursor.fetchall()}
            
            # Map our input IDs to the actual case in the database
            actual_structure_ids = []
            missing_in_db = []
            for sid in structure_ids:
                sid_lower = sid.lower()
                if sid_lower in db_molecules:
                    actual_structure_ids.append(db_molecules[sid_lower])
                else:
                    missing_in_db.append(sid)
            
            if missing_in_db:
                print(f"Warning: The following structure IDs were not found in the database: {missing_in_db}")
                print(f"Continuing with {len(actual_structure_ids)} structures found in database")
            
            if not actual_structure_ids:
                raise ValueError("None of the provided structure IDs were found in the database")
            
            # Create a N x N matrix for the specified structures
            n = len(actual_structure_ids)
            matrix = np.zeros((n, n), dtype=np.float32)
            
            # Fetch all pairwise affinities for these structures using actual case
            placeholders = ','.join(['?'] * len(actual_structure_ids))
            query = f"""
            SELECT 
                molecule_id1,
                molecule_id2,
                affinity_value
            FROM affinities 
            WHERE (molecule_id1 IN ({placeholders}) AND molecule_id2 IN ({placeholders}))
            """
            cursor.execute(query, actual_structure_ids + actual_structure_ids)
            rows = cursor.fetchall()
            
            # Build matrix from pairwise values and track which pairs were found
            id_to_idx = {sid: idx for idx, sid in enumerate(actual_structure_ids)}
            found_pairs = set()
            
            for mol1, mol2, value in rows:
                # Only include if both molecules are in our structure list
                if mol1 in id_to_idx and mol2 in id_to_idx:
                    i = id_to_idx[mol1]
                    j = id_to_idx[mol2]
                    matrix[i, j] = value
                    matrix[j, i] = value  # Ensure symmetry
                    # Track that we found this pair (store sorted pair)
                    pair_key = tuple(sorted([mol1, mol2]))
                    found_pairs.add(pair_key)
                    
            # Fill diagonal with 1.0
            np.fill_diagonal(matrix, 1.0)
            
            # Handle background structures - make their rows negative except diagonal
            for idx, struct_id in enumerate(actual_structure_ids):
                if 'background' in struct_id.lower():
                    # Make entire row negative except diagonal
                    matrix[idx, :] = -np.abs(matrix[idx, :])
                    matrix[idx, idx] = 1.0  # Reset diagonal to 1.0
            
            # Verify we have all needed similarities by checking if pairs exist in database
            missing = []
            for i, mol1 in enumerate(actual_structure_ids):
                for j, mol2 in enumerate(actual_structure_ids[i+1:], i+1):  # Only check upper triangle
                    pair_key = tuple(sorted([mol1, mol2]))
                    if pair_key not in found_pairs:
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
                raise ValueError(f"Missing similarity values for pairs: {missing_pairs}{extra}")
            
            print(f"\nSimilarity Matrix molecules: {actual_structure_ids}")
            
            # Global normalization of off-diagonal elements
            # Create a mask for off-diagonal elements
            off_diag_mask = ~np.eye(n, dtype=bool)
            
            # Get off-diagonal elements
            off_diag_values = matrix[off_diag_mask]
            
            # Find global min and max for off-diagonal elements
            global_min = np.min(off_diag_values)
            global_max = np.max(off_diag_values)
            
            print(f"Global similarity range - Min: {global_min:.4f}, Max: {global_max:.4f}")
            print(f"Mean similarity (off-diagonal): {np.mean(off_diag_values):.4f}")
            
            # Only normalize if we have a reasonable range
            if np.abs(global_max - global_min) > 1e-6:
                # Create a normalized copy
                normalized_matrix = np.copy(matrix)
                
                # Normalize off-diagonal elements to [0, 1] range
                normalized_matrix[off_diag_mask] = (off_diag_values - global_min) / (global_max - global_min)
                
                # Keep similarities in [0, 1] range for contrastive loss
                # Do NOT rescale to [-1, 1] as contrastive loss expects [0, 1]
                
                # Ensure the diagonal remains at 1.0
                np.fill_diagonal(normalized_matrix, 1.0)
                
                # Return with the actual structure IDs as they appear in the database
                return normalized_matrix, actual_structure_ids
            else:
                print("No normalization applied due to small range of similarity values")
                return matrix, actual_structure_ids
            
        finally:
            conn.close()
