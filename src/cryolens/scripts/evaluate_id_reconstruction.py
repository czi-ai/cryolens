"""
Evaluate ID (in-distribution) reconstruction performance for validation data.

This script evaluates CryoLens reconstruction quality on in-distribution
validation data stored in parquet files. It uses the same evaluation pipeline
as the OOD reconstruction script but loads data from parquet files instead of
Copick.

Usage:
    # List all available structures in validation data
    python -m cryolens.scripts.evaluate_id_reconstruction \
        --list-structures \
        --validation-dir data/validation/parquet/

    # Evaluate with default weights and default alignment
    python -m cryolens.scripts.evaluate_id_reconstruction \
        --validation-dir data/validation/parquet/ \
        --structures-dir structures/mrcs/ \
        --output-dir results/id_validation/ \
        --structures 1g3i 1n9g 1ss8

    # Evaluate with specific checkpoint and RANSAC-ICP alignment
    python -m cryolens.scripts.evaluate_id_reconstruction \
        --checkpoint models/cryolens_epoch_2600.pt \
        --validation-dir data/validation/parquet/ \
        --structures-dir structures/mrcs/ \
        --output-dir results/id_validation/ \
        --structures 1g3i 1n9g 1ss8 \
        --alignment-method ransac_icp \
        --weight-percentile 48.3 \
        --sphere-radius 15.2

    # Auto-discover and evaluate all structures with MRC files
    python -m cryolens.scripts.evaluate_id_reconstruction \
        --checkpoint models/cryolens_epoch_2600.pt \
        --validation-dir data/validation/parquet/ \
        --structures-dir structures/mrcs/ \
        --output-dir results/id_validation/

The script follows the same two-stage alignment process as OOD evaluation:
1. Align all reconstructions to first particle reconstruction (common reference)
2. Average in reference frame, then align to ground truth for evaluation only
"""

import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from cryolens.utils.checkpoint_loading import load_vae_model, list_available_versions
from cryolens.inference.pipeline import InferencePipeline
from cryolens.data.parquet_loader import extract_volume_from_row
from cryolens.evaluation.ood_reconstruction import (
    evaluate_ood_structure,
    load_mrc_structure,
)


def discover_available_structures(
    validation_dir: Path,
    snr: float = 5.0,
    verbose: bool = True
) -> List[str]:
    """
    Discover all unique structures available in validation parquet files.
    
    Parameters
    ----------
    validation_dir : Path
        Directory containing validation parquet files
    snr : float
        SNR value for file matching
    verbose : bool
        Print discovery progress
        
    Returns
    -------
    List[str]
        List of unique PDB codes found in the data
    """
    validation_dir = Path(validation_dir)
    pattern = f"validation_*_snr{snr}.parquet"
    parquet_files = sorted(validation_dir.glob(pattern))
    
    if not parquet_files:
        raise ValueError(f"No validation parquet files found matching {pattern} in {validation_dir}")
    
    if verbose:
        print(f"Scanning {len(parquet_files)} parquet files...")
    
    all_structures = set()
    
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            
            # Find pdb_code column
            for col_name in ['pdb_code', 'structure', 'pdb_id']:
                if col_name in df.columns:
                    structures = df[col_name].str.lower().unique()
                    all_structures.update(structures)
                    break
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not read {parquet_file.name}: {e}")
            continue
    
    return sorted(list(all_structures))


def load_id_validation_particles(
    validation_dir: Path,
    structure_name: str,
    n_particles: int,
    box_size: int = 48,
    snr: float = 5.0,
    verbose: bool = True
) -> List[np.ndarray]:
    """
    Load validation particles from parquet files for a specific structure.
    
    The validation data is organized as multiple parquet files with pattern:
    validation_NNNN_snr5.0.parquet
    
    Each parquet file contains particles from multiple structures, identified
    by the 'pdb_code' column.
    
    Parameters
    ----------
    validation_dir : Path
        Directory containing validation parquet files
    structure_name : str
        PDB code (e.g., "6qzp", case-insensitive)
    n_particles : int
        Number of particles to load
    box_size : int
        Expected box size (default: 48)
    snr : float
        SNR value to match in filenames (default: 5.0)
    verbose : bool
        Print loading progress
        
    Returns
    -------
    List[np.ndarray]
        List of particle volumes, each of shape (box_size, box_size, box_size)
    """
    validation_dir = Path(validation_dir)
    
    # Find all validation parquet files matching the SNR pattern
    pattern = f"validation_*_snr{snr}.parquet"
    parquet_files = sorted(validation_dir.glob(pattern))
    
    if not parquet_files:
        raise ValueError(f"No validation parquet files found matching {pattern} in {validation_dir}")
    
    if verbose:
        print(f"  Found {len(parquet_files)} validation parquet files")
    
    # Normalize structure name to lowercase for matching
    structure_lower = structure_name.lower()
    
    particles = []
    
    for parquet_file in parquet_files:
        if len(particles) >= n_particles:
            break
        
        if verbose:
            print(f"  Reading {parquet_file.name}...")
        
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not read {parquet_file.name}: {e}")
            continue
        
        # Check for pdb_code column (try alternative names)
        pdb_col = None
        for col_name in ['pdb_code', 'structure', 'pdb_id']:
            if col_name in df.columns:
                pdb_col = col_name
                break
        
        if pdb_col is None:
            if verbose:
                print(f"    Warning: No PDB code column found in {parquet_file.name}")
            continue
        
        # Filter for this structure (case-insensitive)
        structure_df = df[df[pdb_col].str.lower() == structure_lower]
        
        if len(structure_df) == 0:
            continue
        
        if verbose:
            print(f"    Found {len(structure_df)} particles for {structure_name}")
        
        # Extract volumes
        for _, row in structure_df.iterrows():
            if len(particles) >= n_particles:
                break
            
            # Use existing parquet_loader function
            volume = extract_volume_from_row(
                row,
                expected_shape=(box_size, box_size, box_size)
            )
            
            if volume is not None:
                particles.append(volume)
        
        if verbose and len(particles) > 0:
            print(f"    Loaded {len(particles)}/{n_particles} particles so far")
    
    if len(particles) == 0:
        raise ValueError(f"No particles could be loaded for structure {structure_name}")
    
    if verbose:
        print(f"  Successfully loaded {len(particles)} particles for {structure_name}")
    
    return particles[:n_particles]


class ParquetDataLoader:
    """
    Adapter to provide a Copick-like interface for parquet validation data.
    
    This class mimics the interface of CopickDataLoader so that the existing
    evaluate_ood_structure() function can be reused without modification.
    """
    
    def __init__(self, validation_dir: Path, box_size: int = 48, snr: float = 5.0):
        """
        Initialize the parquet data loader.
        
        Parameters
        ----------
        validation_dir : Path
            Directory containing validation parquet files
        box_size : int
            Expected box size
        snr : float
            SNR value for file matching
        """
        self.validation_dir = Path(validation_dir)
        self.box_size = box_size
        self.snr = snr
    
    def load_particles(
        self,
        structure_filter: Optional[List[str]] = None,
        max_particles_per_structure: int = 30,
        target_voxel_spacing: float = 10.0,
        box_size: Optional[int] = None,
        normalize: bool = False,
        verbose: bool = True,
        **kwargs  # Accept and ignore other Copick-specific arguments
    ) -> Dict[str, Dict[str, List[np.ndarray]]]:
        """
        Load particles from parquet files.
        
        This method provides a Copick-compatible interface, returning data
        in the same format as CopickDataLoader.load_particles().
        
        Parameters
        ----------
        structure_filter : List[str], optional
            List of structure names to load
        max_particles_per_structure : int
            Maximum particles per structure
        target_voxel_spacing : float
            Target voxel spacing (not used for parquet data)
        box_size : int, optional
            Box size override
        normalize : bool
            Whether to normalize (not implemented)
        verbose : bool
            Print loading progress
            
        Returns
        -------
        Dict
            Dictionary mapping structure names to particle data:
            {structure_name: {'particles': [np.ndarray, ...], 'metadata': [...]}}
        """
        if box_size is None:
            box_size = self.box_size
        
        if structure_filter is None:
            raise ValueError("structure_filter must be provided")
        
        data = {}
        
        for structure_name in structure_filter:
            try:
                particles = load_id_validation_particles(
                    self.validation_dir,
                    structure_name,
                    max_particles_per_structure,
                    box_size=box_size,
                    snr=self.snr,
                    verbose=verbose
                )
                
                # Format data to match Copick interface
                data[structure_name] = {
                    'particles': particles,
                    'metadata': [{'index': i} for i in range(len(particles))]
                }
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load {structure_name}: {e}")
                continue
        
        return data


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Special modes
    parser.add_argument(
        '--list-structures',
        action='store_true',
        help='List all available structures in validation data and exit'
    )
    parser.add_argument(
        '--list-versions',
        action='store_true',
        help='List available model versions and exit'
    )
    
    # Checkpoint (now optional)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint, version name (e.g., "v001"), or URL. '
             'If not specified, uses default CryoLens weights. '
             'Use --list-versions to see available versions.'
    )
    
    # Required for evaluation mode
    parser.add_argument(
        '--validation-dir',
        type=Path,
        help='Directory containing validation parquet files'
    )
    parser.add_argument(
        '--structures-dir',
        type=Path,
        help='Directory containing ground truth MRC files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--structures',
        nargs='+',
        default=None,
        help='Structures to evaluate (PDB codes, e.g., 1g3i 6qzp). Default: all structures with MRC files'
    )
    parser.add_argument(
        '--n-particles',
        type=int,
        default=25,
        help='Number of particles per structure (default: 25)'
    )
    parser.add_argument(
        '--n-resamples',
        type=int,
        default=10,
        help='Number of resamples for uncertainty (default: 10)'
    )
    parser.add_argument(
        '--particle-counts',
        nargs='+',
        type=int,
        default=[5, 10, 15, 20, 25],
        help='Particle counts to test (default: 5 10 15 20 25)'
    )
    parser.add_argument(
        '--voxel-size',
        type=float,
        default=10.0,
        help='Voxel size in Angstroms (default: 10.0)'
    )
    parser.add_argument(
        '--snr',
        type=float,
        default=5.0,
        help='SNR value for validation data (default: 5.0)'
    )
    
    # Alignment method arguments
    parser.add_argument(
        '--alignment-method',
        type=str,
        default='cross_correlation',
        choices=['cross_correlation', 'fourier', 'gradient_descent', 'ransac_icp'],
        help='Alignment method to use (default: cross_correlation)'
    )
    parser.add_argument(
        '--angular-step',
        type=float,
        default=None,
        help='Angular step for alignment search (method-specific default used if not specified)'
    )
    
    # RANSAC-ICP specific parameters
    parser.add_argument(
        '--weight-percentile',
        type=float,
        default=48.3,
        help='Weight percentile threshold for RANSAC-ICP (default: 48.3)'
    )
    parser.add_argument(
        '--sphere-radius',
        type=float,
        default=15.2,
        help='Sphere radius for RANSAC-ICP filtering (default: 15.2)'
    )
    parser.add_argument(
        '--ransac-iterations',
        type=int,
        default=252,
        help='Number of RANSAC iterations (default: 252)'
    )
    parser.add_argument(
        '--icp-iterations',
        type=int,
        default=17,
        help='Number of ICP refinement iterations (default: 17)'
    )
    
    args = parser.parse_args()
    
    # Handle --list-versions flag
    if args.list_versions:
        print("\nAvailable CryoLens model versions:")
        print("=" * 60)
        for version, description in list_available_versions().items():
            print(f"  {version:10s}: {description}")
        print("=" * 60)
        print("\nUse --checkpoint <version> to specify a version")
        print("Or omit --checkpoint to use the default version")
        return 0
    
    # Handle --list-structures mode
    if args.list_structures:
        if not args.validation_dir:
            parser.error("--validation-dir is required with --list-structures")
        
        print("="*70)
        print("DISCOVERING AVAILABLE STRUCTURES")
        print("="*70)
        print(f"Validation dir: {args.validation_dir}")
        print(f"SNR:            {args.snr}")
        print()
        
        try:
            structures = discover_available_structures(
                args.validation_dir,
                snr=args.snr,
                verbose=True
            )
            
            print(f"\n{'='*70}")
            print(f"Found {len(structures)} unique structures")
            print("="*70)
            
            # Print in columns
            for i in range(0, len(structures), 8):
                row = structures[i:i+8]
                print("  " + "  ".join(f"{s:6s}" for s in row))
            
            print("="*70)
            print(f"\nTo evaluate specific structures, use:")
            print(f"  --structures {' '.join(structures[:5])} ...")
            print(f"\nTo evaluate ALL structures with MRC files:")
            print(f"  (omit --structures to auto-discover)")
            
        except Exception as e:
            print(f"\n✗ Error discovering structures: {e}")
            return 1
        
        return 0
    
    # Validate required arguments for evaluation mode
    if args.validation_dir is None:
        parser.error("--validation-dir is required for evaluation mode")
    if args.structures_dir is None:
        parser.error("--structures-dir is required for evaluation mode")
    if args.output_dir is None:
        parser.error("--output-dir is required for evaluation mode")
    
    # Print configuration
    print("="*70)
    print("ID VALIDATION RECONSTRUCTION EVALUATION")
    print("="*70)
    if args.checkpoint:
        print(f"Checkpoint:        {args.checkpoint}")
    else:
        print(f"Checkpoint:        <default weights>")
    print(f"Validation dir:    {args.validation_dir}")
    print(f"Structures dir:    {args.structures_dir}")
    print(f"Output dir:        {args.output_dir}")
    print(f"Structures:        {', '.join(args.structures) if args.structures else 'auto-discover'}")
    print(f"Particles:         {args.n_particles} per structure")
    print(f"Resamples:         {args.n_resamples} per particle")
    print(f"Voxel size:        {args.voxel_size}Å")
    print(f"SNR:               {args.snr}")
    print(f"Alignment:         {args.alignment_method}")
    print("="*70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading model...")
    try:
        model, config = load_vae_model(
            args.checkpoint,
            device=device,
            load_config=True,
            strict_loading=False
        )
        model.eval()
        print("  Model loaded successfully")
        
        # Check decoder type
        decoder = model.decoder
        if hasattr(decoder, 'affinity_segment_size'):
            print("  Detected: SegmentedGaussianSplatDecoder")
        else:
            print("  Detected: Standard decoder")
            
    except Exception as e:
        print(f"  Error loading model: {e}")
        return 1
    
    # Create pipeline
    pipeline = InferencePipeline(
        model=model,
        device=device,
        normalization_method=config.get('normalization', 'z-score')
    )
    
    # Initialize parquet data loader
    print(f"\nInitializing parquet data loader...")
    try:
        parquet_loader = ParquetDataLoader(
            args.validation_dir,
            box_size=48,
            snr=args.snr
        )
        print("  Parquet loader initialized")
    except Exception as e:
        print(f"  Error initializing parquet loader: {e}")
        return 1
    
    # Determine which structures to process
    if args.structures is None:
        # Auto-discover: find structures that have both parquet data and MRC files
        print("\nAuto-discovering structures with both validation data and MRC files...")
        available_structures = discover_available_structures(
            args.validation_dir,
            snr=args.snr,
            verbose=False
        )
        
        # Check which ones have MRC files
        structures_to_process = []
        for structure in available_structures:
            mrc_path = args.structures_dir / f"{structure}.mrc"
            if mrc_path.exists():
                structures_to_process.append(structure)
        
        if not structures_to_process:
            print(f"\n✗ Error: No structures found with both validation data and MRC files")
            print(f"  Validation data has {len(available_structures)} structures")
            print(f"  MRC directory: {args.structures_dir}")
            print(f"\nFirst 20 structures in validation data:")
            for i in range(0, min(20, len(available_structures)), 5):
                row = available_structures[i:i+5]
                print(f"  {' '.join(row)}")
            if len(available_structures) > 20:
                print(f"  ... and {len(available_structures) - 20} more")
            print(f"\nUse --list-structures to see all available structures")
            return 1
        
        print(f"  Found {len(structures_to_process)} structures with both data and MRC files")
        if len(structures_to_process) <= 10:
            print(f"  Structures: {', '.join(structures_to_process)}")
        else:
            print(f"  First 10: {', '.join(structures_to_process[:10])}")
            print(f"  ... and {len(structures_to_process) - 10} more")
    else:
        structures_to_process = [s.lower() for s in args.structures]
        print(f"\nProcessing {len(structures_to_process)} specified structures")
    
    # Create main output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare alignment kwargs
    alignment_kwargs = {}
    if args.angular_step is not None:
        alignment_kwargs['angular_step'] = args.angular_step
    
    # Add RANSAC-ICP specific parameters
    if args.alignment_method == 'ransac_icp':
        alignment_kwargs.update({
            'weight_percentile': args.weight_percentile,
            'sphere_radius': args.sphere_radius,
            'ransac_iterations': args.ransac_iterations,
            'icp_iterations': args.icp_iterations,
        })
    
    # Evaluate each structure
    results = {}
    errors = {}
    
    for structure in structures_to_process:
        try:
            # Construct ground truth path from structure name
            gt_filename = f"{structure}.mrc"
            gt_path = args.structures_dir / gt_filename
            
            if not gt_path.exists():
                print(f"\n⚠ Warning: Ground truth not found at {gt_path}")
                errors[structure] = f"Ground truth not found: {gt_path}"
                continue
            
            # Evaluate using existing OOD evaluation function
            # This function handles the entire two-stage alignment process
            result = evaluate_ood_structure(
                structure_name=structure,
                model=model,
                pipeline=pipeline,
                copick_loader=parquet_loader,  # Use our parquet adapter!
                ground_truth_path=gt_path,
                output_dir=args.output_dir / structure,
                device=device,
                n_particles=args.n_particles,
                n_resamples=args.n_resamples,
                particle_counts=args.particle_counts,
                voxel_size=args.voxel_size,
                alignment_method=args.alignment_method,
                alignment_kwargs=alignment_kwargs
            )
            
            results[structure] = result
            
        except Exception as e:
            print(f"\n✗ Error processing {structure}: {e}")
            import traceback
            traceback.print_exc()
            errors[structure] = str(e)
            continue
    
    # Save summary JSON
    summary_path = args.output_dir / "evaluation_summary.json"
    
    summary = {
        'config': {
            'checkpoint': args.checkpoint if args.checkpoint else 'default',
            'validation_dir': str(args.validation_dir),
            'structures_dir': str(args.structures_dir),
            'n_particles': args.n_particles,
            'n_resamples': args.n_resamples,
            'voxel_size': args.voxel_size,
            'snr': args.snr,
            'particle_counts': args.particle_counts,
            'alignment_method': args.alignment_method
        },
        'results': {},
        'errors': errors
    }
    
    # Extract key metrics
    for structure, result in results.items():
        if 'metrics' in result:
            max_n = max(result['metrics'].keys())
            summary['results'][structure] = {
                'n_particles': max_n,
                'resolution': result['metrics'][max_n]['resolution'],
                'correlation': result['metrics'][max_n]['correlation'],
                'h5_path': result.get('h5_path', ''),
                'figure_path': result.get('figure_path', '')
            }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary to {summary_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        print("\nSuccessfully evaluated structures:")
        for structure, result in results.items():
            if 'metrics' in result:
                max_n = max(result['metrics'].keys())
                m = result['metrics'][max_n]
                print(f"  {structure:25s}: {m['resolution']:5.1f}Å, r={m['correlation']:.3f}")
    
    if errors:
        print("\nErrors:")
        for structure, error in errors.items():
            print(f"  {structure:25s}: {error}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {args.output_dir}")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())
