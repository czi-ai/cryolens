"""
Utility script to explore and analyze extracted embeddings HDF5 files.

This script provides various operations for inspecting HDF5 files created by
extract_validation_embeddings.py.

Usage:
    # Show file info and summary statistics
    python -m cryolens.scripts.explore_embeddings \
        --h5-file /path/to/embeddings.h5 \
        --info

    # List unique structures and their counts
    python -m cryolens.scripts.explore_embeddings \
        --h5-file /path/to/embeddings.h5 \
        --list-structures

    # Extract embeddings for specific structures
    python -m cryolens.scripts.explore_embeddings \
        --h5-file /path/to/embeddings.h5 \
        --filter-structures 1g3i 1n9g \
        --output-filtered filtered_embeddings.h5

    # Export to CSV (first N particles)
    python -m cryolens.scripts.explore_embeddings \
        --h5-file /path/to/embeddings.h5 \
        --export-csv embeddings.csv \
        --max-rows 10000

    # Compute embedding statistics by structure
    python -m cryolens.scripts.explore_embeddings \
        --h5-file /path/to/embeddings.h5 \
        --statistics
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def show_file_info(h5_path: Path):
    """Display comprehensive information about the HDF5 file."""
    print("="*70)
    print("HDF5 FILE INFORMATION")
    print("="*70)
    print(f"File: {h5_path}")
    print()
    
    with h5py.File(h5_path, 'r') as f:
        # Show attributes
        print("Attributes:")
        for key, value in f.attrs.items():
            print(f"  {key:25s}: {value}")
        print()
        
        # Show datasets
        print("Datasets:")
        for name, dataset in f.items():
            print(f"  {name:25s}: shape={dataset.shape}, dtype={dataset.dtype}")
            if dataset.shape[0] > 0:
                if dataset.dtype.kind == 'f':  # float
                    data_sample = dataset[:min(100, dataset.shape[0])]
                    print(f"    {'':25s}  min={np.min(data_sample):.4f}, max={np.max(data_sample):.4f}, mean={np.mean(data_sample):.4f}")
        print()
        
        # Show memory size
        total_size = sum(dataset.size * dataset.dtype.itemsize for dataset in f.values())
        print(f"Total size: {total_size / 1e9:.2f} GB")
    
    print("="*70)


def list_structures(h5_path: Path):
    """List unique structures and their particle counts."""
    print("="*70)
    print("STRUCTURE COUNTS")
    print("="*70)
    
    with h5py.File(h5_path, 'r') as f:
        pdb_codes = f['pdb_codes'][:]
        
        # Count occurrences
        unique, counts = np.unique(pdb_codes, return_counts=True)
        
        # Sort by count (descending)
        sorted_indices = np.argsort(-counts)
        
        print(f"Total structures: {len(unique)}")
        print(f"Total particles:  {len(pdb_codes)}")
        print()
        
        # Print table
        print(f"{'Structure':15s} {'Count':>10s} {'Percentage':>10s}")
        print("-" * 40)
        for idx in sorted_indices:
            structure = unique[idx]
            count = counts[idx]
            percentage = 100.0 * count / len(pdb_codes)
            print(f"{structure:15s} {count:>10d} {percentage:>9.2f}%")
    
    print("="*70)


def filter_by_structures(
    h5_path: Path,
    output_path: Path,
    structures: List[str]
):
    """Create a new HDF5 file with only selected structures."""
    structures_lower = [s.lower() for s in structures]
    logger.info(f"Filtering for structures: {', '.join(structures_lower)}")
    
    with h5py.File(h5_path, 'r') as f_in:
        pdb_codes = f_in['pdb_codes'][:]
        
        # Find matching indices
        mask = np.isin(pdb_codes, structures_lower)
        indices = np.where(mask)[0]
        
        logger.info(f"Found {len(indices)} matching particles")
        
        if len(indices) == 0:
            logger.warning("No particles found for specified structures")
            return
        
        # Create output file
        with h5py.File(output_path, 'w') as f_out:
            # Copy attributes
            for key, value in f_in.attrs.items():
                f_out.attrs[key] = value
            f_out.attrs['filtered_structures'] = ','.join(structures_lower)
            f_out.attrs['original_total'] = len(pdb_codes)
            f_out.attrs['filtered_total'] = len(indices)
            
            # Copy filtered data
            for name, dataset in f_in.items():
                logger.info(f"Copying {name}...")
                if len(dataset.shape) == 1:
                    f_out.create_dataset(
                        name,
                        data=dataset[indices],
                        compression='gzip'
                    )
                else:
                    f_out.create_dataset(
                        name,
                        data=dataset[indices, :],
                        compression='gzip'
                    )
    
    logger.info(f"Filtered file saved to: {output_path}")


def export_to_csv(
    h5_path: Path,
    output_csv: Path,
    max_rows: Optional[int] = None
):
    """Export embeddings to CSV format."""
    logger.info(f"Exporting to CSV: {output_csv}")
    
    with h5py.File(h5_path, 'r') as f:
        n_total = f['pdb_codes'].shape[0]
        n_export = min(max_rows, n_total) if max_rows else n_total
        
        logger.info(f"Exporting {n_export} / {n_total} particles")
        
        # Load data
        data = {
            'pdb_code': f['pdb_codes'][:n_export],
            'source_file': f['source_file'][:n_export],
            'source_row_index': f['source_row_index'][:n_export],
        }
        
        # Add embedding dimensions
        embedding_mean = f['embedding_mean'][:n_export]
        for i in range(embedding_mean.shape[1]):
            data[f'emb_mean_{i}'] = embedding_mean[:, i]
        
        embedding_log_var = f['embedding_log_var'][:n_export]
        for i in range(embedding_log_var.shape[1]):
            data[f'emb_logvar_{i}'] = embedding_log_var[:, i]
        
        # Add pose dimensions
        pose = f['pose'][:n_export]
        for i in range(pose.shape[1]):
            data[f'pose_{i}'] = pose[:, i]
        
        # Add global weight
        data['global_weight'] = f['global_weight'][:n_export].flatten()
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        
        logger.info(f"CSV saved: {output_csv}")
        logger.info(f"Shape: {df.shape}")


def compute_statistics(h5_path: Path):
    """Compute statistics for embeddings grouped by structure."""
    print("="*70)
    print("EMBEDDING STATISTICS BY STRUCTURE")
    print("="*70)
    
    with h5py.File(h5_path, 'r') as f:
        pdb_codes = f['pdb_codes'][:]
        embeddings = f['embedding_mean'][:]
        log_vars = f['embedding_log_var'][:]
        
        unique_structures = np.unique(pdb_codes)
        
        print(f"Computing statistics for {len(unique_structures)} structures...")
        print()
        
        # Compute per-structure statistics
        for structure in unique_structures[:20]:  # Show first 20
            mask = pdb_codes == structure
            structure_embs = embeddings[mask]
            structure_logvars = log_vars[mask]
            
            print(f"Structure: {structure}")
            print(f"  Particles: {np.sum(mask)}")
            print(f"  Embedding mean: {np.mean(structure_embs):.4f} ± {np.std(structure_embs):.4f}")
            print(f"  Embedding norm (avg): {np.mean(np.linalg.norm(structure_embs, axis=1)):.4f}")
            print(f"  Log variance mean: {np.mean(structure_logvars):.4f} ± {np.std(structure_logvars):.4f}")
            print()
        
        if len(unique_structures) > 20:
            print(f"... and {len(unique_structures) - 20} more structures")
    
    print("="*70)


def sample_particles(
    h5_path: Path,
    n_samples: int = 10,
    structure: Optional[str] = None
):
    """Show sample particles from the dataset."""
    print("="*70)
    print("SAMPLE PARTICLES")
    print("="*70)
    
    with h5py.File(h5_path, 'r') as f:
        n_total = f['pdb_codes'].shape[0]
        
        if structure:
            # Sample from specific structure
            pdb_codes = f['pdb_codes'][:]
            structure_lower = structure.lower()
            mask = pdb_codes == structure_lower
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                logger.warning(f"No particles found for structure: {structure}")
                return
            
            sample_indices = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
            print(f"Sampling {len(sample_indices)} particles from structure: {structure}")
        else:
            # Random sample from all
            sample_indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)
            print(f"Sampling {len(sample_indices)} random particles")
        
        print()
        
        for idx in sample_indices:
            pdb_code = f['pdb_codes'][idx]
            emb_mean = f['embedding_mean'][idx]
            pose = f['pose'][idx]
            global_weight = f['global_weight'][idx, 0]
            source_file = f['source_file'][idx]
            source_row = f['source_row_index'][idx]
            
            print(f"Index {idx}:")
            print(f"  Structure:      {pdb_code}")
            print(f"  Source:         {source_file}, row {source_row}")
            print(f"  Embedding norm: {np.linalg.norm(emb_mean):.4f}")
            print(f"  Pose:           [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}, {pose[3]:.3f}]")
            print(f"  Global weight:  {global_weight:.4f}")
            print()
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--h5-file',
        type=Path,
        required=True,
        help='Path to embeddings HDF5 file'
    )
    
    # Actions
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show file information and summary'
    )
    parser.add_argument(
        '--list-structures',
        action='store_true',
        help='List all structures and their counts'
    )
    parser.add_argument(
        '--statistics',
        action='store_true',
        help='Compute statistics by structure'
    )
    parser.add_argument(
        '--sample',
        type=int,
        metavar='N',
        help='Show N sample particles'
    )
    parser.add_argument(
        '--sample-structure',
        type=str,
        help='Structure to sample from (use with --sample)'
    )
    
    # Export options
    parser.add_argument(
        '--filter-structures',
        nargs='+',
        help='Filter and save specific structures'
    )
    parser.add_argument(
        '--output-filtered',
        type=Path,
        help='Output path for filtered HDF5 file'
    )
    parser.add_argument(
        '--export-csv',
        type=Path,
        help='Export to CSV file'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        help='Maximum rows to export to CSV'
    )
    
    args = parser.parse_args()
    
    # Check file exists
    if not args.h5_file.exists():
        logger.error(f"File not found: {args.h5_file}")
        return 1
    
    # Execute actions
    if args.info:
        show_file_info(args.h5_file)
    
    if args.list_structures:
        list_structures(args.h5_file)
    
    if args.statistics:
        compute_statistics(args.h5_file)
    
    if args.sample:
        sample_particles(
            args.h5_file,
            n_samples=args.sample,
            structure=args.sample_structure
        )
    
    if args.filter_structures:
        if not args.output_filtered:
            logger.error("--output-filtered required when using --filter-structures")
            return 1
        filter_by_structures(
            args.h5_file,
            args.output_filtered,
            args.filter_structures
        )
    
    if args.export_csv:
        export_to_csv(
            args.h5_file,
            args.export_csv,
            max_rows=args.max_rows
        )
    
    # If no action specified, show info by default
    if not any([args.info, args.list_structures, args.statistics, args.sample,
                args.filter_structures, args.export_csv]):
        show_file_info(args.h5_file)
    
    return 0


if __name__ == '__main__':
    exit(main())
