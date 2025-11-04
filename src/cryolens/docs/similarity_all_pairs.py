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
from scipy.ndimage import zoom, gaussian_filter
import sys
import logging

from cryolens.utils.optional_deps import require_gemmi

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

def pdb_to_density(filename, box_size=48, voxel_size=10.0, resolution=15.0):
    """Convert PDB structure to density map with proper physical units."""
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

def find_structure_file(dirs, structure_id):
    """Find structure file (MRC or PDB) for a structure ID"""
    mrcs_dir = dirs.get('mrcs_dir')
    pdb_dir = dirs.get('pdb_dir')
    
    # Try MRC files first
    if mrcs_dir:
        mrcs_dir = Path(mrcs_dir)
        # Try different naming conventions
        for ext in ['.mrc', '.mrcs']:
            # Direct match
            mrc_path = mrcs_dir / f"{structure_id}{ext}"
            if mrc_path.exists():
                return mrc_path, 'mrc'
            
            # Try with underscore instead of dash
            if '-' in structure_id:
                parts = structure_id.rsplit('-', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    alt_name = f"{parts[0]}_{parts[1]}"
                    mrc_path = mrcs_dir / f"{alt_name}{ext}"
                    if mrc_path.exists():
                        return mrc_path, 'mrc'
    
    # Try PDB files
    if pdb_dir:
        pdb_dir = Path(pdb_dir)
        pdb_path = pdb_dir / f"{structure_id}.pdb"
        if pdb_path.exists():
            return pdb_path, 'pdb'
    
    return None, None

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

def get_structure_info(pdb_path):
    """Extract structure information from PDB file."""
    # Check for gemmi dependency
    gemmi = require_gemmi()
    
    try:
        st = gemmi.read_structure(str(pdb_path))
        
        coords = []
        masses = []
        chains = []
        
        for model in st:
            for chain in model:
                chains.append(chain.name)
                for residue in chain:
                    for atom in residue:
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
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
                'molecular_weight_kda': molecular_weight / 1000,
                'estimated_diameter_nm': float(np.max(extent) / 10),
                'radius_of_gyration': float(rg),
            }
        else:
            return {'name': pdb_path.stem, 'error': 'No atoms found'}
            
    except Exception as e:
        return {'name': pdb_path.stem, 'error': str(e)}

def create_ortho_view(volume, title, output_path, structure_info=None, voxel_size=10.0):
    """Create orthogonal views of a volume and save as image with optional structure info"""
    projections = get_projections(volume)
    
    if structure_info:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    # Determine title with additional info if available
    if structure_info and 'molecular_weight_kda' in structure_info:
        full_title = f"{title} - {structure_info['molecular_weight_kda']:.1f} kDa, {structure_info.get('estimated_diameter_nm', 0):.1f} nm"
    else:
        full_title = title
    
    fig.suptitle(full_title, fontsize=10)
    
    # Plot projections
    for idx, (view_name, proj) in enumerate(projections.items()):
        ax = axes[idx]
        im = ax.imshow(proj, cmap='gray', aspect='equal')
        ax.set_title(view_name, fontsize=8)
        ax.axis('off')
        
        # Add scale bar on first projection
        if idx == 0:
            # 50 Å scale bar
            scalebar_pixels = 50 / voxel_size
            ax.plot([5, 5 + scalebar_pixels], [5, 5], 'w-', linewidth=2)
            ax.text(5 + scalebar_pixels/2, 3, '50 Å', color='white', ha='center', fontsize=8)
    
    # Add structure info if available
    if structure_info and len(axes) > 3:
        ax = axes[3]
        ax.axis('off')
        info_text = f"""Structure Info:
        
Mol. weight: {structure_info.get('molecular_weight_kda', 0):.1f} kDa
Diameter: {structure_info.get('estimated_diameter_nm', 0):.1f} nm
Radius of gyr.: {structure_info.get('radius_of_gyration', 0):.1f} Å
Atoms: {structure_info.get('n_atoms', 0):,}
Chains: {structure_info.get('n_chains', 0)}"""
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def preprocess_unique_structures(pairs, dirs, args):
    """Pre-process all unique structures to avoid duplicate computations."""
    logger.info("Pre-processing unique structures to avoid duplicates...")
    
    # Get all unique structure IDs
    unique_structures = set()
    for pair in pairs:
        unique_structures.add(pair['mol1'])
        unique_structures.add(pair['mol2'])
    
    logger.info(f"Found {len(unique_structures)} unique structures")
    
    # Cache for storing structure data and info
    structure_cache = {}
    
    for structure_id in unique_structures:
        logger.info(f"Processing structure {structure_id}")
        
        # Find structure file
        file_path, file_type = find_structure_file(dirs, structure_id)
        
        if not file_path:
            logger.warning(f"Structure file not found for {structure_id}")
            structure_cache[structure_id] = None
            continue
        
        # Load structure based on file type
        if file_type == 'mrc':
            volume_data = load_mrc(file_path, args.box_size)
        else:  # pdb
            volume_data = pdb_to_density(file_path, args.box_size, args.voxel_size, args.resolution)
        
        if volume_data is None:
            logger.warning(f"Failed to load volume data for {structure_id}")
            structure_cache[structure_id] = None
            continue
        
        # Get structure info for PDB files
        structure_info = None
        if file_type == 'pdb':
            structure_info = get_structure_info(file_path)
        
        # Store in cache
        structure_cache[structure_id] = {
            'volume_data': volume_data,
            'structure_info': structure_info,
            'file_type': file_type
        }
    
    logger.info(f"Successfully cached {len([v for v in structure_cache.values() if v is not None])} structures")
    return structure_cache

def main():
    parser = argparse.ArgumentParser(description="Generate markdown table with similarity pairs")
    parser.add_argument("--mrcs_dir", type=str, help="Directory containing MRC files")
    parser.add_argument("--pdb_dir", type=str, help="Directory containing PDB files")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite database file")
    parser.add_argument("--output_dir", type=str, default="docs", help="Output directory")
    parser.add_argument("--box_size", type=int, default=48, help="Box size for density maps")
    parser.add_argument("--voxel_size", type=float, default=10.0, help="Voxel size in Angstroms (default: 10.0)")
    parser.add_argument("--resolution", type=float, default=15.0, help="Target resolution in Angstroms (default: 15.0)")
    parser.add_argument("--max_pairs", type=int, help="Maximum number of pairs to display")
    args = parser.parse_args()
    
    # Validate that at least one directory is provided
    if not args.mrcs_dir and not args.pdb_dir:
        parser.error("At least one of --mrcs_dir or --pdb_dir must be provided")
    
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
    
    # Pre-process all unique structures to avoid duplicates
    dirs = {'mrcs_dir': args.mrcs_dir, 'pdb_dir': args.pdb_dir}
    structure_cache = preprocess_unique_structures(pairs, dirs, args)
    
    # Calculate unique images that will be created
    unique_structures = set()
    for pair in pairs:
        if structure_cache.get(pair['mol1']):
            unique_structures.add(pair['mol1'])
        if structure_cache.get(pair['mol2']):
            unique_structures.add(pair['mol2'])
    
    logger.info(f"Will create {len(unique_structures)} unique structure images (instead of {len(pairs) * 2} duplicate images)")
    
    # Open markdown file
    markdown_path = output_dir / "similarity_pairs.md"
    with open(markdown_path, 'w') as md_file:
        # Write header
        md_file.write("# Similarity Pairs Visualization\n\n")
        md_file.write(f"Total pairs: {len(pairs)}\n")
        md_file.write(f"Box size: {args.box_size}³ voxels\n")
        md_file.write(f"Voxel size: {args.voxel_size} Å/voxel\n")
        md_file.write(f"Physical box size: {args.box_size * args.voxel_size} Å³\n")
        md_file.write(f"Target resolution: {args.resolution} Å\n\n")
        
        # Write table header
        md_file.write("| Structure A | Structure B | Similarity Score |\n")
        md_file.write("|-------------|-------------|------------------|\n")
        
        # Process each pair using cached data
        valid_pairs = 0
        
        for pair_idx, pair in enumerate(pairs):
            # Get cached structure data
            mol1_cache = structure_cache.get(pair['mol1'])
            mol2_cache = structure_cache.get(pair['mol2'])
            
            if not mol1_cache or not mol2_cache:
                logger.warning(f"Skipping pair {pair['mol1']} vs {pair['mol2']} - cached data not available")
                continue
            
            mol1_data = mol1_cache['volume_data']
            mol2_data = mol2_cache['volume_data']
            info1 = mol1_cache['structure_info']
            info2 = mol2_cache['structure_info']
            
            # Create images for this pair (reusing the same image if structure appears multiple times)
            img1_path = images_dir / f"structure_{pair['mol1']}.png"
            img2_path = images_dir / f"structure_{pair['mol2']}.png"
            
            # Only create image if it doesn't exist (avoid duplicates)
            if not img1_path.exists():
                create_ortho_view(mol1_data, pair['mol1'], img1_path, info1, args.voxel_size)
            
            if not img2_path.exists():
                create_ortho_view(mol2_data, pair['mol2'], img2_path, info2, args.voxel_size)
            
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
