#!/usr/bin/env python
"""
Script to generate a markdown table with all pairs from the similarity database.
Each row shows orthogonal views of both structures and their similarity score.
Updated to match metadata handling from copick-features-and-affinities script.
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
import os

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

def load_mrc(mrc_path, box_size=48, pixel_size=10):
    """Load and normalize an MRC file matching the copick-features approach"""
    try:
        with mrc_open(mrc_path, mode='r') as mrc:
            density_map = mrc.data.astype(np.float32)
            
            # Get metadata from MRC header
            original_shape = density_map.shape
            
            # Get voxel size from header if available
            if hasattr(mrc.header, 'cella'):
                voxel_size = mrc.header.cella.item(0) / original_shape[0] if mrc.header.cella.item(0) > 0 else pixel_size
            else:
                voxel_size = pixel_size
            
            # Physical size in Angstroms
            physical_size = tuple(s * voxel_size for s in original_shape)
            
            logger.debug(f"Loading: shape={original_shape}, voxel_size={voxel_size:.2f}Å, physical_size={physical_size}")
            
            # Efficient downsampling if needed - matching the copick approach
            if density_map.shape != (box_size,) * 3:
                logger.debug(f"Resizing from {density_map.shape} to {(box_size,) * 3}")
                scale_factor = box_size / density_map.shape[0]
                # Use order=1 for linear interpolation (faster than cubic)
                density_map = zoom(density_map, (scale_factor,) * 3, 
                                  order=1, mode='constant', cval=0.0,
                                  grid_mode=True)  # grid_mode=True preserves sum
            
            # Normalize
            if density_map.max() > density_map.min():
                density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min())
            
            return density_map, voxel_size, original_shape, physical_size
            
    except Exception as e:
        logger.error(f"Failed to load MRC: {e}")
        return None, None, None, None

def find_mrc_file(mrcs_dir, structure_id):
    """Find MRC file for a structure ID using the copick naming convention"""
    mrcs_dir = Path(mrcs_dir)
    
    # Handle JCVI-syn3.0-76285 -> JCVI-syn3.0_76285.mrc conversion
    parts = structure_id.rsplit('-', 1)
    if len(parts) == 2 and parts[1].isdigit():
        mrc_name = f"{parts[0]}_{parts[1]}.mrc"
        mrc_path = mrcs_dir / mrc_name
        if mrc_path.exists():
            return mrc_path
    
    # Try direct match with .mrc extension
    mrc_path = mrcs_dir / f"{structure_id}.mrc"
    if mrc_path.exists():
        return mrc_path
    
    # Try with .mrcs extension
    mrc_path = mrcs_dir / f"{structure_id}.mrcs"
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

def create_ortho_view(volume, title, output_path, physical_size=None, pixel_size=10):
    """Create orthogonal views of a volume and save as image"""
    projections = get_projections(volume)
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    # Add size info to title if available
    if physical_size:
        size_nm = physical_size[0] / 10.0  # Convert Å to nm
        title = f"{title} ({size_nm:.1f}nm @ {pixel_size:.1f}Å/px)"
    
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
    parser.add_argument("--output_dir", type=str, default="docs", help="Output directory")
    parser.add_argument("--box_size", type=int, default=48, help="Box size for density maps")
    parser.add_argument("--pixel_size", type=float, default=10, help="Default pixel size in Angstroms")
    parser.add_argument("--max_pairs", type=int, help="Maximum number of pairs to display")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    images_dir = output_dir / "figures"
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
        md_file.write(f"Total pairs: {len(pairs)}\n")
        md_file.write(f"Box size: {args.box_size}³ voxels\n")
        md_file.write(f"Default pixel size: {args.pixel_size}Å\n\n")
        
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
            
            # Load structures using the copick approach
            mol1_data, voxel1, shape1, phys1 = load_mrc(mrc1_path, args.box_size, args.pixel_size)
            mol2_data, voxel2, shape2, phys2 = load_mrc(mrc2_path, args.box_size, args.pixel_size)
            
            if mol1_data is None or mol2_data is None:
                continue
            
            # Create images for this pair
            img1_path = images_dir / f"pair_{pair_idx:04d}_mol1_{pair['mol1']}.png"
            img2_path = images_dir / f"pair_{pair_idx:04d}_mol2_{pair['mol2']}.png"
            
            create_ortho_view(mol1_data, pair['mol1'], img1_path, phys1, voxel1)
            create_ortho_view(mol2_data, pair['mol2'], img2_path, phys2, voxel2)
            
            # Add row to markdown table
            img1_rel = f"figures/{img1_path.name}"
            img2_rel = f"figures/{img2_path.name}"
            
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
