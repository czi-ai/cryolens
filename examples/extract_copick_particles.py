#!/usr/bin/env python3
"""
Extract particle subvolumes from Copick projects for use with CryoLens.

This script connects to Copick projects (local or Data Portal), extracts
particle subvolumes at specified positions, and saves them as zarr arrays
with metadata for downstream analysis.

Examples
--------
Extract from Data Portal using embedded config:
    python extract_copick_particles.py \\
        --config mlc_experimental_publictest \\
        --structures ribosome,thyroglobulin \\
        --num-particles 30 \\
        --output ./example_data/

Use custom local Copick project:
    python extract_copick_particles.py \\
        --copick-config /path/to/config.json \\
        --structures ribosome \\
        --num-particles 50 \\
        --output ./my_particles/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import zarr
from tqdm import tqdm

try:
    import copick
except ImportError:
    print("Error: copick is not installed. Install with: pip install cryolens[copick]")
    sys.exit(1)


def extract_particles_from_run(
    run,
    object_name: str,
    voxel_spacing: float,
    box_size: int,
    max_particles: Optional[int] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract particle subvolumes from a Copick run.
    
    Parameters
    ----------
    run : copick.models.CopickRun
        Copick run object
    object_name : str
        Name of the pickable object
    voxel_spacing : float
        Voxel spacing to use (in Angstroms)
    box_size : int
        Size of extracted boxes (cubic)
    max_particles : int, optional
        Maximum number of particles to extract
    user_id : str, optional
        Filter by user ID
    session_id : str, optional
        Filter by session ID
        
    Returns
    -------
    particles : np.ndarray
        Particle subvolumes (N, box_size, box_size, box_size)
    positions : np.ndarray
        Particle positions in Angstroms (N, 3)
    orientations : np.ndarray
        Particle orientations as rotation matrices (N, 3, 3) if available, else None
    """
    # Get picks for this object
    picks = run.get_picks(object_name=object_name, user_id=user_id, session_id=session_id)
    
    if not picks:
        print(f"Warning: No picks found for {object_name} in run {run.name}")
        return np.array([]), np.array([]), None
    
    # Use the most recent picks
    picks_obj = picks[0]
    all_points = picks_obj.points
    
    if not all_points:
        print(f"Warning: No points in picks for {object_name} in run {run.name}")
        return np.array([]), np.array([]), None
    
    # Limit number of particles if requested
    if max_particles and len(all_points) > max_particles:
        # Sample uniformly
        indices = np.linspace(0, len(all_points) - 1, max_particles, dtype=int)
        points = [all_points[i] for i in indices]
    else:
        points = all_points
    
    # Get tomogram for this voxel spacing
    voxel_spacings = run.get_voxel_spacings()
    
    # Find closest voxel spacing
    closest_vs = min(voxel_spacings, key=lambda vs: abs(vs.voxel_size - voxel_spacing))
    
    # Get tomogram (prefer denoised/processed if available)
    tomograms = closest_vs.get_tomograms()
    if not tomograms:
        print(f"Warning: No tomograms found for run {run.name} at voxel spacing {closest_vs.voxel_size}")
        return np.array([]), np.array([]), None
    
    # Try to find denoised tomogram, otherwise use first available
    tomo = None
    for t in tomograms:
        if 'denois' in t.tomo_type.lower() or 'ctf' in t.tomo_type.lower():
            tomo = t
            break
    if tomo is None:
        tomo = tomograms[0]
    
    # Load tomogram data
    print(f"Loading tomogram: {tomo.tomo_type} at {closest_vs.voxel_size}Å")
    tomo_data = np.array(tomo.numpy())
    
    # Extract particles
    particles = []
    positions = []
    orientations = []
    half_box = box_size // 2
    
    for point in tqdm(points, desc=f"Extracting {object_name}"):
        # Convert position from Angstroms to voxels
        pos_voxels = np.array([
            point.location.x / closest_vs.voxel_size,
            point.location.y / closest_vs.voxel_size,
            point.location.z / closest_vs.voxel_size,
        ]).astype(int)
        
        # Check bounds
        if (pos_voxels[0] < half_box or pos_voxels[0] >= tomo_data.shape[2] - half_box or
            pos_voxels[1] < half_box or pos_voxels[1] >= tomo_data.shape[1] - half_box or
            pos_voxels[2] < half_box or pos_voxels[2] >= tomo_data.shape[0] - half_box):
            continue
        
        # Extract subvolume
        particle = tomo_data[
            pos_voxels[2] - half_box:pos_voxels[2] + half_box,
            pos_voxels[1] - half_box:pos_voxels[1] + half_box,
            pos_voxels[0] - half_box:pos_voxels[0] + half_box,
        ]
        
        if particle.shape != (box_size, box_size, box_size):
            continue
        
        particles.append(particle)
        positions.append([point.location.x, point.location.y, point.location.z])
        
        # Extract orientation if available
        if hasattr(point, 'transformation') and point.transformation is not None:
            # Extract rotation matrix from transformation
            transform = np.array(point.transformation).reshape(4, 4)
            rotation = transform[:3, :3]
            orientations.append(rotation)
        else:
            orientations.append(np.eye(3))
    
    if not particles:
        print(f"Warning: No valid particles extracted for {object_name} in run {run.name}")
        return np.array([]), np.array([]), None
    
    particles = np.stack(particles)
    positions = np.array(positions)
    orientations = np.stack(orientations) if orientations else None
    
    return particles, positions, orientations


def save_particles_to_zarr(
    particles: np.ndarray,
    positions: np.ndarray,
    orientations: Optional[np.ndarray],
    metadata: Dict[str, Any],
    output_path: Path,
):
    """
    Save particles and metadata to zarr format.
    
    Parameters
    ----------
    particles : np.ndarray
        Particle subvolumes
    positions : np.ndarray
        Particle positions
    orientations : np.ndarray, optional
        Particle orientations
    metadata : dict
        Metadata to save
    output_path : Path
        Output zarr directory
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create zarr arrays
    root = zarr.open(str(output_path), mode='w')
    
    # Save particles
    root.create_dataset(
        'particles',
        data=particles,
        chunks=(1, particles.shape[1], particles.shape[2], particles.shape[3]),
        dtype=particles.dtype,
    )
    
    # Save positions
    root.create_dataset('positions', data=positions, dtype=np.float32)
    
    # Save orientations if available
    if orientations is not None:
        root.create_dataset('orientations', data=orientations, dtype=np.float32)
    
    # Save metadata as attributes
    root.attrs.update(metadata)
    
    # Also save as JSON for easy reading
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(particles)} particles to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract particle subvolumes from Copick projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Copick config options (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--config',
        type=str,
        help='Name of embedded Copick config (e.g., mlc_experimental_publictest)',
    )
    config_group.add_argument(
        '--copick-config',
        type=str,
        help='Path to custom Copick config JSON file',
    )
    
    # Extraction parameters
    parser.add_argument(
        '--structures',
        type=str,
        required=True,
        help='Comma-separated list of structure names to extract',
    )
    parser.add_argument(
        '--num-particles',
        type=int,
        default=30,
        help='Maximum number of particles to extract per structure (default: 30)',
    )
    parser.add_argument(
        '--box-size',
        type=int,
        default=48,
        help='Size of extracted particle boxes (default: 48)',
    )
    parser.add_argument(
        '--voxel-spacing',
        type=float,
        default=10.0,
        help='Voxel spacing in Angstroms (default: 10.0)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./example_data',
        help='Output directory (default: ./example_data)',
    )
    
    # Optional filters
    parser.add_argument(
        '--runs',
        type=str,
        help='Comma-separated list of run names to process (default: all)',
    )
    parser.add_argument(
        '--user-id',
        type=str,
        help='Filter picks by user ID',
    )
    parser.add_argument(
        '--session-id',
        type=str,
        help='Filter picks by session ID',
    )
    
    args = parser.parse_args()
    
    # Get Copick config path
    if args.config:
        from cryolens.data import get_copick_config
        config_path = get_copick_config(args.config)
        print(f"Using embedded config: {args.config}")
    else:
        config_path = args.copick_config
        print(f"Using custom config: {config_path}")
    
    # Parse structures
    structures = [s.strip() for s in args.structures.split(',')]
    
    # Parse runs if specified
    run_filter = [r.strip() for r in args.runs.split(',')] if args.runs else None
    
    # Load Copick root
    print(f"Loading Copick project from: {config_path}")
    root = copick.from_file(config_path)
    
    # Get runs
    runs = root.runs
    if run_filter:
        runs = [r for r in runs if r.name in run_filter]
    
    print(f"Found {len(runs)} runs to process")
    
    # Extract particles for each structure
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    for structure in structures:
        print(f"\n{'='*60}")
        print(f"Extracting {structure}")
        print(f"{'='*60}")
        
        all_particles = []
        all_positions = []
        all_orientations = []
        run_info = []
        
        for run in runs:
            print(f"\nProcessing run: {run.name}")
            
            particles, positions, orientations = extract_particles_from_run(
                run=run,
                object_name=structure,
                voxel_spacing=args.voxel_spacing,
                box_size=args.box_size,
                max_particles=args.num_particles,
                user_id=args.user_id,
                session_id=args.session_id,
            )
            
            if len(particles) > 0:
                all_particles.append(particles)
                all_positions.append(positions)
                if orientations is not None:
                    all_orientations.append(orientations)
                
                run_info.append({
                    'run_name': run.name,
                    'n_particles': len(particles),
                })
        
        if not all_particles:
            print(f"Warning: No particles extracted for {structure}")
            continue
        
        # Concatenate all particles
        all_particles = np.concatenate(all_particles, axis=0)
        all_positions = np.concatenate(all_positions, axis=0)
        all_orientations = np.concatenate(all_orientations, axis=0) if all_orientations else None
        
        # Prepare metadata
        metadata = {
            'structure': structure,
            'box_size': args.box_size,
            'voxel_spacing': args.voxel_spacing,
            'n_particles': len(all_particles),
            'runs': run_info,
            'copick_config': config_path,
        }
        
        # Save to zarr
        output_path = output_base / f"{structure}_particles.zarr"
        save_particles_to_zarr(
            particles=all_particles,
            positions=all_positions,
            orientations=all_orientations,
            metadata=metadata,
            output_path=output_path,
        )
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Output directory: {output_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
