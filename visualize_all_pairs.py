#!/usr/bin/env python
"""
Script to visualize all pairs from the similarity database with their projections and distance measures.
Usage: uv run visualize_all_pairs.py --mrcs_dir /path/to/mrcs --db_path /path/to/db
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
logger = logging.getLogger("visualize-all-pairs")

def convert_array(blob):
    """Convert blob to numpy array"""
    out = io.BytesIO(blob)
    out.seek(0)
    return np.load(out)

def get_all_pairs(db_path):
    """Get all pairs from the database"""
    pairs = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT molecule_id1, molecule_id2, affinity_value FROM affinities ORDER BY affinity_value DESC"
            )
            for row in cursor:
                pairs.append({
                    'mol1': row[0],
                    'mol2': row[1],
                    'affinity': row[2]
                })
    except Exception as e:
        logger.error(f"Error reading pairs from database: {e}")
    return pairs

def load_mrc(mrc_path, box_size=48):
    """Load and normalize an MRC file"""
    try:
        with mrc_open(mrc_path, mode='r') as mrc:
            density_map = mrc.data.astype(np.float32)
            
            # Resize if needed
            if density_map.shape != (box_size,) * 3:
                scale_factor = box_size / density_map.shape[0]
                density_map = zoom(density_map, (scale_factor,) * 3, 
                                  order=1, mode='constant', cval=0.0,
                                  grid_mode=True)
            
            # Normalize
            if density_map.max() > density_map.min():
                density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min())
            
            return density_map
    except Exception as e:
        logger.error(f"Failed to load MRC: {e}")
        return None

def find_mrc_file(mrcs_dir, structure_id):
    """Find MRC file for a structure ID"""
    mrcs_dir = Path(mrcs_dir)
    
    # Try different naming conventions
    for ext in ['.mrc', '.mrcs']:
        # Direct match
        mrc_path = mrcs_dir / f"{structure_id}{ext}"
        if mrc_path.exists():
            return mrc_path
        
        # Try with underscore instead of dash
        if '-' in structure_id:
            parts = structure_id.rsplit('-', 1)
            if len(parts) == 2 and parts[1].isdigit():
                alt_name = f"{parts[0]}_{parts[1]}"
                mrc_path = mrcs_dir / f"{alt_name}{ext}"
                if mrc_path.exists():
                    return mrc_path
    
    return None

def get_projections(volume):
    """Generate sum projections along each axis"""
    projections = {
        'XY': np.sum(volume, axis=2),
        'XZ': np.sum(volume, axis=1),
        'YZ': np.sum(volume, axis=0)
    }
    
    # Normalize each projection
    for key in projections:
        proj = projections[key]
        if proj.max() > proj.min():
            projections[key] = (proj - proj.min()) / (proj.max() - proj.min())
    
    return projections

def main():
    parser = argparse.ArgumentParser(description="Visualize all pairs from similarity database")
    parser.add_argument("--mrcs_dir", type=str, required=True, help="Directory containing MRC files")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite database file")
    parser.add_argument("--output", type=str, default="all_pairs_visualization.png", help="Output filename")
    parser.add_argument("--box_size", type=int, default=48, help="Box size for density maps")
    parser.add_argument("--max_pairs", type=int, help="Maximum number of pairs to display")
    parser.add_argument("--pairs_per_row", type=int, default=5, help="Number of pairs per row in visualization")
    args = parser.parse_args()
    
    # Get all pairs from database
    logger.info(f"Reading pairs from database: {args.db_path}")
    pairs = get_all_pairs(args.db_path)
    
    if not pairs:
        logger.error("No pairs found in database")
        return
    
    logger.info(f"Found {len(pairs)} pairs in database")
    
    # Limit pairs if requested
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
        logger.info(f"Limiting to {len(pairs)} pairs")
    
    # Calculate grid dimensions
    pairs_per_row = args.pairs_per_row
    n_rows = (len(pairs) + pairs_per_row - 1) // pairs_per_row
    
    # Create figure
    # Each pair needs 2 structures x 3 projections = 6 subplots horizontally
    # Plus spacing between pairs
    fig_width = pairs_per_row * 6 * 2.5
    fig_height = n_rows * 5
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Process each pair
    valid_pairs = 0
    for pair_idx, pair in enumerate(pairs):
        row = pair_idx // pairs_per_row
        col = pair_idx % pairs_per_row
        
        # Find MRC files
        mrc1_path = find_mrc_file(args.mrcs_dir, pair['mol1'])
        mrc2_path = find_mrc_file(args.mrcs_dir, pair['mol2'])
        
        if not mrc1_path or not mrc2_path:
            logger.warning(f"Skipping pair {pair['mol1']} vs {pair['mol2']} - MRC files not found")
            continue
        
        # Load structures
        mol1_data = load_mrc(mrc1_path, args.box_size)
        mol2_data = load_mrc(mrc2_path, args.box_size)
        
        if mol1_data is None or mol2_data is None:
            continue
        
        # Get projections
        proj1 = get_projections(mol1_data)
        proj2 = get_projections(mol2_data)
        
        # Calculate subplot positions
        # Each pair gets 6 columns (3 for each structure)
        base_col = col * 6
        base_row = row * 2
        
        # Plot first structure
        for proj_idx, (view_name, proj) in enumerate(proj1.items()):
            ax_idx = base_row * (pairs_per_row * 6) + base_col + proj_idx + 1
            ax = plt.subplot(n_rows * 2, pairs_per_row * 6, ax_idx)
            ax.imshow(proj, cmap='gray', aspect='equal')
            ax.axis('off')
            if proj_idx == 1:  # Middle projection
                ax.set_title(f"{pair['mol1']}", fontsize=8)
        
        # Plot second structure
        for proj_idx, (view_name, proj) in enumerate(proj2.items()):
            ax_idx = (base_row + 1) * (pairs_per_row * 6) + base_col + proj_idx + 1
            ax = plt.subplot(n_rows * 2, pairs_per_row * 6, ax_idx)
            ax.imshow(proj, cmap='gray', aspect='equal')
            ax.axis('off')
            if proj_idx == 1:  # Middle projection
                ax.set_title(f"{pair['mol2']}\nAffinity: {pair['affinity']:.3f}", fontsize=8)
        
        valid_pairs += 1
        
        if valid_pairs % 10 == 0:
            logger.info(f"Processed {valid_pairs} pairs...")
    
    plt.suptitle(f"All Similarity Pairs ({valid_pairs} pairs)", fontsize=16)
    plt.tight_layout()
    
    # Save figure
    logger.info(f"Saving visualization to {args.output}")
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    affinities = [p['affinity'] for p in pairs[:valid_pairs]]
    logger.info(f"\nVisualized {valid_pairs} pairs")
    logger.info(f"Affinity range: {min(affinities):.4f} to {max(affinities):.4f}")
    logger.info(f"Mean affinity: {np.mean(affinities):.4f} ± {np.std(affinities):.4f}")

if __name__ == "__main__":
    main()
