"""
Evaluate OOD reconstruction performance for paper figures.

This script evaluates CryoLens reconstruction quality on out-of-distribution
experimental cryo-ET data from the ML Challenge dataset.

Usage:
    # With default weights
    python -m cryolens.scripts.evaluate_ood_reconstruction \
        --copick-config ml_challenge_experimental.json \
        --structures-dir structures/mrcs/ \
        --output-dir results/ood/ \
        --structures ribosome thyroglobulin

    # With specific checkpoint
    python -m cryolens.scripts.evaluate_ood_reconstruction \
        --checkpoint models/cryolens_epoch_2600.pt \
        --copick-config ml_challenge_experimental.json \
        --structures-dir structures/mrcs/ \
        --output-dir results/ood/ \
        --structures ribosome thyroglobulin

For all structures:
    python -m cryolens.scripts.evaluate_ood_reconstruction \
        --checkpoint models/cryolens_epoch_2600.pt \
        --copick-config ml_challenge_experimental.json \
        --structures-dir structures/mrcs/ \
        --output-dir results/ood/
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Optional, List

from cryolens.utils.checkpoint_loading import load_vae_model, list_available_versions
from cryolens.inference.pipeline import InferencePipeline
from cryolens.data import CopickDataLoader
from cryolens.evaluation.ood_reconstruction import evaluate_ood_structure


# ML Challenge structure mapping
# Maps structure names to ground truth MRC filenames
STRUCTURE_MAPPING = {
    "apo-ferritin": "7vd8.mrc",
    "beta-amylase": "1fa2.mrc",
    "beta-galactoside": "6drv.mrc",
    "ribosome": "6qzp.mrc",
    "thyroglobulin": "7b75.mrc",
    "virus-like-particle": "6n4v.mrc"
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    
    # List versions flag
    parser.add_argument(
        '--list-versions',
        action='store_true',
        help='List available model versions and exit'
    )
    
    # Required arguments
    parser.add_argument(
        '--copick-config',
        type=str,
        help='Path to Copick configuration file'
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
        default=list(STRUCTURE_MAPPING.keys()),
        choices=list(STRUCTURE_MAPPING.keys()),
        help='Structures to evaluate (default: all)'
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
    
    # Validate required arguments for evaluation mode
    if args.copick_config is None:
        parser.error("--copick-config is required for evaluation mode")
    if args.structures_dir is None:
        parser.error("--structures-dir is required for evaluation mode")
    if args.output_dir is None:
        parser.error("--output-dir is required for evaluation mode")
    
    # Print configuration
    print("="*70)
    print("OOD RECONSTRUCTION EVALUATION")
    print("="*70)
    if args.checkpoint:
        print(f"Checkpoint:     {args.checkpoint}")
    else:
        print(f"Checkpoint:     <default weights>")
    print(f"Copick config:  {args.copick_config}")
    print(f"Structures dir: {args.structures_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Structures:     {', '.join(args.structures)}")
    print(f"Particles:      {args.n_particles} per structure")
    print(f"Resamples:      {args.n_resamples} per particle")
    print(f"Voxel size:     {args.voxel_size}Å")
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
    
    # Initialize Copick loader
    print(f"\nInitializing Copick data loader...")
    try:
        copick_loader = CopickDataLoader(args.copick_config)
        print("  Copick loader initialized")
    except Exception as e:
        print(f"  Error initializing Copick loader: {e}")
        return 1
    
    # Create main output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each structure
    results = {}
    errors = {}
    
    for structure in args.structures:
        try:
            # Get ground truth path
            gt_filename = STRUCTURE_MAPPING[structure]
            gt_path = args.structures_dir / gt_filename
            
            if not gt_path.exists():
                print(f"\n⚠ Warning: Ground truth not found at {gt_path}")
                errors[structure] = f"Ground truth not found: {gt_path}"
                continue
            
            # Evaluate
            result = evaluate_ood_structure(
                structure_name=structure,
                model=model,
                pipeline=pipeline,
                copick_loader=copick_loader,
                ground_truth_path=gt_path,
                output_dir=args.output_dir / structure,
                device=device,
                n_particles=args.n_particles,
                n_resamples=args.n_resamples,
                particle_counts=args.particle_counts,
                voxel_size=args.voxel_size
            )
            
            results[structure] = result
            
        except Exception as e:
            print(f"\n✗ Error processing {structure}: {e}")
            errors[structure] = str(e)
            continue
    
    # Save summary JSON
    summary_path = args.output_dir / "evaluation_summary.json"
    
    summary = {
        'config': {
            'checkpoint': args.checkpoint if args.checkpoint else 'default',
            'copick_config': args.copick_config,
            'structures_dir': str(args.structures_dir),
            'n_particles': args.n_particles,
            'n_resamples': args.n_resamples,
            'voxel_size': args.voxel_size,
            'particle_counts': args.particle_counts
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
