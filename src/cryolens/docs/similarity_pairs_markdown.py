#!/usr/bin/env python
"""
Script to generate a markdown table with all pairs from the similarity database.
Each row shows orthogonal views of both structures and their similarity score.
Usage: uv run similarity_pairs_markdown.py --mrcs_dir /path/to/mrcs --db_path /path/to/db
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
logger = logging.getLogger("similarity-pairs-markdown")

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

def create_ortho_view(volume, title, output_path):
    """Create orthogonal views of a volume and save as image"""
    projections = get_projections(volume)
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(title, fontsize=10)
    
    for idx, (view_name, proj) in enumerate(projections.items()):
        ax = axes[idx]
        ax.imshow(proj, cmap='gray', aspect='equal')
        ax.set_title(view_name, fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate markdown table with similarity pairs")
    parser.add_argument("--mrcs_dir", type=str, required=True, help="Directory containing MRC files")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite database file")
    parser.add_argument("--output_dir", type=str, default="similarity_pairs_output", help="Output directory")
    parser.add_argument("--box_size", type=int, default=48, help="Box size for density maps")
    parser.add_argument("--max_pairs", type=int, help="Maximum number of pairs to display")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
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
    
    # Open markdown file
    markdown_path = output_dir / "similarity_pairs.md"
    with open(markdown_path, 'w') as md_file:
        # Write header
        md_file.write("# Similarity Pairs Visualization\n\n")
        md_file.write(f"Total pairs: {len(pairs)}\n\n")
        
        # Write table header
        md_file.write("| Structure A | Structure B | Similarity Score |\n")
        md_file.write("|-------------|-------------|------------------|\n")
        
        # Process each pair
        valid_pairs = 0
        for pair_idx, pair in enumerate(pairs):
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
            
            # Create images for this pair
            img1_path = images_dir / f"pair_{pair_idx:04d}_mol1_{pair['mol1']}.png"
            img2_path = images_dir / f"pair_{pair_idx:04d}_mol2_{pair['mol2']}.png"
            
            create_ortho_view(mol1_data, pair['mol1'], img1_path)
            create_ortho_view(mol2_data, pair['mol2'], img2_path)
            
            # Add row to markdown table
            img1_rel = f"images/{img1_path.name}"
            img2_rel = f"images/{img2_path.name}"
            
            md_file.write(f"| ![{pair['mol1']}]({img1_rel}) | ![{pair['mol2']}]({img2_rel}) | {pair['affinity']:.4f} |\n")
            
            valid_pairs += 1
            
            if valid_pairs % 10 == 0:
                logger.info(f"Processed {valid_pairs} pairs...")
        
        # Write statistics
        md_file.write(f"\n## Statistics\n\n")
        affinities = [p['affinity'] for p in pairs[:valid_pairs]]
        md_file.write(f"- **Total pairs visualized**: {valid_pairs}\n")
        md_file.write(f"- **Affinity range**: {min(affinities):.4f} to {max(affinities):.4f}\n")
        md_file.write(f"- **Mean affinity**: {np.mean(affinities):.4f} ± {np.std(affinities):.4f}\n")
    
    logger.info(f"\nMarkdown table saved to: {markdown_path}")
    logger.info(f"Images saved to: {images_dir}")
    logger.info(f"\nVisualized {valid_pairs} pairs")

if __name__ == "__main__":
    main()
