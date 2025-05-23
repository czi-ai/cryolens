#!/usr/bin/env python
"""
Script to visualize pairs from the similarity database with their projections and distance measures.
Author: Assistant based on cryolens codebase
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3
import io
from mrcfile import open as mrc_open
from scipy.ndimage import zoom
import sys
import logging

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("visualize-similarity-pairs")

class DatabaseReader:
    """Reads similarity data from the database"""
    
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def _convert_array(self, blob):
        """Convert blob to numpy array"""
        out = io.BytesIO(blob)
        out.seek(0)
        return np.load(out)
    
    def get_affinity_pairs(self, n_pairs=10, min_affinity=None, max_affinity=None):
        """Get pairs from the database, optionally filtered by affinity range"""
        pairs = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT molecule_id1, molecule_id2, affinity_value FROM affinities"
                conditions = []
                params = []
                
                if min_affinity is not None:
                    conditions.append("affinity_value >= ?")
                    params.append(min_affinity)
                if max_affinity is not None:
                    conditions.append("affinity_value <= ?")
                    params.append(max_affinity)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY RANDOM() LIMIT ?"
                params.append(n_pairs)
                
                cursor = conn.execute(query, params)
                for row in cursor:
                    pairs.append({
                        'mol1': row[0],
                        'mol2': row[1],
                        'affinity': row[2]
                    })
                    
        except Exception as e:
            logger.error(f"Error reading pairs from database: {e}")
            
        return pairs
    
    def get_top_and_bottom_pairs(self, n_top=5, n_bottom=5):
        """Get the highest and lowest affinity pairs"""
        pairs = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get top pairs
                cursor = conn.execute(
                    "SELECT molecule_id1, molecule_id2, affinity_value FROM affinities "
                    "ORDER BY affinity_value DESC LIMIT ?", (n_top,)
                )
                for row in cursor:
                    pairs.append({
                        'mol1': row[0],
                        'mol2': row[1],
                        'affinity': row[2],
                        'type': 'top'
                    })
                
                # Get bottom pairs
                cursor = conn.execute(
                    "SELECT molecule_id1, molecule_id2, affinity_value FROM affinities "
                    "ORDER BY affinity_value ASC LIMIT ?", (n_bottom,)
                )
                for row in cursor:
                    pairs.append({
                        'mol1': row[0],
                        'mol2': row[1],
                        'affinity': row[2],
                        'type': 'bottom'
                    })
                    
        except Exception as e:
            logger.error(f"Error reading top/bottom pairs from database: {e}")
            
        return pairs

class MRCLoader:
    """Loads MRC files for visualization"""
    
    def __init__(self, mrc_dir, box_size=48):
        self.mrc_dir = Path(mrc_dir)
        self.box_size = box_size
        self.mrc_files = {}
        self._find_mrcs()
    
    def _find_mrcs(self):
        """Find all MRC files in the directory"""
        for ext in ['.mrc', '.mrcs']:
            for mrc_path in self.mrc_dir.glob(f"*{ext}"):
                # Handle different naming conventions
                stem = mrc_path.stem
                parts = stem.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    structure_id = f"{parts[0]}-{parts[1]}"
                else:
                    structure_id = stem
                
                self.mrc_files[structure_id] = mrc_path
        
        logger.info(f"Found {len(self.mrc_files)} MRC files in {self.mrc_dir}")
    
    def load_structure(self, structure_id):
        """Load and normalize a structure"""
        if structure_id not in self.mrc_files:
            logger.warning(f"Structure {structure_id} not found")
            return None
        
        try:
            mrc_path = self.mrc_files[structure_id]
            with mrc_open(mrc_path, mode='r') as mrc:
                density_map = mrc.data.astype(np.float32)
                
                # Resize if needed
                if density_map.shape != (self.box_size,) * 3:
                    scale_factor = self.box_size / density_map.shape[0]
                    density_map = zoom(density_map, (scale_factor,) * 3, 
                                      order=1, mode='constant', cval=0.0,
                                      grid_mode=True)
                
                # Normalize
                if density_map.max() > density_map.min():
                    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min())
                
                return density_map
                
        except Exception as e:
            logger.error(f"Failed to load structure {structure_id}: {e}")
            return None

class PairVisualizer:
    """Creates visualizations of structure pairs"""
    
    @staticmethod
    def get_projections(volume):
        """Generate sum projections along each axis"""
        projections = {
            'XY (top)': np.sum(volume, axis=2),
            'XZ (front)': np.sum(volume, axis=1),
            'YZ (side)': np.sum(volume, axis=0)
        }
        
        # Normalize each projection
        for key in projections:
            proj = projections[key]
            if proj.max() > proj.min():
                projections[key] = (proj - proj.min()) / (proj.max() - proj.min())
                
        return projections
    
    @staticmethod
    def visualize_pair(mol1_data, mol2_data, mol1_id, mol2_id, affinity, save_path=None):
        """Create visualization for a single pair"""
        # Get projections for both structures
        proj1 = PairVisualizer.get_projections(mol1_data)
        proj2 = PairVisualizer.get_projections(mol2_data)
        
        # Create figure with 2 rows x 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f'Pair: {mol1_id} vs {mol2_id}\nAffinity: {affinity:.4f}', fontsize=14)
        
        # Plot first structure
        for idx, (view_name, proj) in enumerate(proj1.items()):
            ax = axes[0, idx]
            im = ax.imshow(proj, cmap='gray', aspect='equal')
            ax.set_title(f'{mol1_id}\n{view_name}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot second structure
        for idx, (view_name, proj) in enumerate(proj2.items()):
            ax = axes[1, idx]
            im = ax.imshow(proj, cmap='gray', aspect='equal')
            ax.set_title(f'{mol2_id}\n{view_name}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def create_summary_visualization(pairs_data, save_path=None):
        """Create a summary visualization of multiple pairs"""
        n_pairs = len(pairs_data)
        
        # Create figure with appropriate size
        fig_height = 4 * n_pairs
        fig, axes = plt.subplots(n_pairs, 6, figsize=(18, fig_height))
        
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for pair_idx, pair_info in enumerate(pairs_data):
            mol1_data = pair_info['mol1_data']
            mol2_data = pair_info['mol2_data']
            
            # Get projections
            proj1 = PairVisualizer.get_projections(mol1_data)
            proj2 = PairVisualizer.get_projections(mol2_data)
            
            # Plot first structure projections
            for proj_idx, (view_name, proj) in enumerate(proj1.items()):
                ax = axes[pair_idx, proj_idx]
                ax.imshow(proj, cmap='gray', aspect='equal')
                if pair_idx == 0:
                    ax.set_title(f'{view_name}', fontsize=10)
                if proj_idx == 0:
                    ax.set_ylabel(f"{pair_info['mol1']}\nvs\n{pair_info['mol2']}\n\nAffinity: {pair_info['affinity']:.3f}", 
                                 rotation=0, labelpad=40, ha='right', va='center', fontsize=9)
                ax.axis('off')
            
            # Plot second structure projections
            for proj_idx, (view_name, proj) in enumerate(proj2.items()):
                ax = axes[pair_idx, proj_idx + 3]
                ax.imshow(proj, cmap='gray', aspect='equal')
                if pair_idx == 0:
                    ax.set_title(f'{view_name}', fontsize=10)
                ax.axis('off')
        
        # Add column headers
        fig.text(0.25, 0.98, 'Structure 1', ha='center', fontsize=12, weight='bold')
        fig.text(0.75, 0.98, 'Structure 2', ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize pairs from similarity database")
    parser.add_argument("--mrcs_dir", type=str, required=True, help="Directory containing MRC files")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite database file")
    parser.add_argument("--n_pairs", type=int, default=10, help="Number of pairs to visualize")
    parser.add_argument("--output_dir", type=str, default="similarity_visualizations", help="Output directory for visualizations")
    parser.add_argument("--box_size", type=int, default=48, help="Box size for density maps")
    parser.add_argument("--min_affinity", type=float, help="Minimum affinity threshold")
    parser.add_argument("--max_affinity", type=float, help="Maximum affinity threshold")
    parser.add_argument("--top_bottom", action="store_true", help="Show top and bottom affinity pairs")
    parser.add_argument("--individual", action="store_true", help="Create individual visualizations for each pair")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    db_reader = DatabaseReader(args.db_path)
    mrc_loader = MRCLoader(args.mrcs_dir, box_size=args.box_size)
    
    # Get pairs based on arguments
    if args.top_bottom:
        n_each = args.n_pairs // 2
        pairs = db_reader.get_top_and_bottom_pairs(n_top=n_each, n_bottom=n_each)
        logger.info(f"Retrieved {len(pairs)} top and bottom pairs")
    else:
        pairs = db_reader.get_affinity_pairs(
            n_pairs=args.n_pairs,
            min_affinity=args.min_affinity,
            max_affinity=args.max_affinity
        )
        logger.info(f"Retrieved {len(pairs)} pairs from database")
    
    if not pairs:
        logger.error("No pairs found matching the criteria")
        return
    
    # Load structures and prepare visualization data
    pairs_data = []
    for pair in pairs:
        logger.info(f"Loading pair: {pair['mol1']} vs {pair['mol2']} (affinity: {pair['affinity']:.4f})")
        
        mol1_data = mrc_loader.load_structure(pair['mol1'])
        mol2_data = mrc_loader.load_structure(pair['mol2'])
        
        if mol1_data is None or mol2_data is None:
            logger.warning(f"Skipping pair due to loading error")
            continue
        
        pair_info = {
            'mol1': pair['mol1'],
            'mol2': pair['mol2'],
            'affinity': pair['affinity'],
            'mol1_data': mol1_data,
            'mol2_data': mol2_data
        }
        
        if 'type' in pair:
            pair_info['type'] = pair['type']
            
        pairs_data.append(pair_info)
        
        # Create individual visualization if requested
        if args.individual:
            pair_type = pair.get('type', 'pair')
            filename = f"{pair_type}_{pair['mol1']}_{pair['mol2']}_affinity_{pair['affinity']:.4f}.png"
            save_path = output_dir / filename
            PairVisualizer.visualize_pair(
                mol1_data, mol2_data,
                pair['mol1'], pair['mol2'],
                pair['affinity'],
                save_path=save_path
            )
            logger.info(f"Saved individual visualization to {save_path}")
    
    # Create summary visualization
    if pairs_data:
        if args.top_bottom:
            # Separate top and bottom pairs
            top_pairs = [p for p in pairs_data if p.get('type') == 'top']
            bottom_pairs = [p for p in pairs_data if p.get('type') == 'bottom']
            
            if top_pairs:
                save_path = output_dir / "top_affinity_pairs_summary.png"
                PairVisualizer.create_summary_visualization(top_pairs, save_path=save_path)
                logger.info(f"Saved top pairs summary to {save_path}")
            
            if bottom_pairs:
                save_path = output_dir / "bottom_affinity_pairs_summary.png"
                PairVisualizer.create_summary_visualization(bottom_pairs, save_path=save_path)
                logger.info(f"Saved bottom pairs summary to {save_path}")
        else:
            save_path = output_dir / "similarity_pairs_summary.png"
            PairVisualizer.create_summary_visualization(pairs_data, save_path=save_path)
            logger.info(f"Saved summary visualization to {save_path}")
        
        # Print statistics
        affinities = [p['affinity'] for p in pairs_data]
        logger.info(f"\nStatistics for {len(pairs_data)} pairs:")
        logger.info(f"  Min affinity: {min(affinities):.4f}")
        logger.info(f"  Max affinity: {max(affinities):.4f}")
        logger.info(f"  Mean affinity: {np.mean(affinities):.4f}")
        logger.info(f"  Std affinity: {np.std(affinities):.4f}")

if __name__ == "__main__":
    main()
