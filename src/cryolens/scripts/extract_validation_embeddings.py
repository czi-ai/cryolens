"""
Extract embeddings from validation parquet files.

This script processes validation parquet files to extract latent embeddings
(mean and log variance), pose, and global weight for each particle. Results
are saved to an HDF5 file with periodic flushing to manage memory.

Usage:
    python -m cryolens.scripts.extract_validation_embeddings \
        --checkpoint /path/to/model_epoch_2600.pt \
        --validation-dir /path/to/validation/parquet/ \
        --output-h5 /path/to/output/embeddings.h5 \
        --batch-size 32 \
        --flush-every 1000 \
        --snr 5.0

The output HDF5 file contains:
    - pdb_codes: string dataset with particle types
    - embedding_mean: (N, latent_dims) float32 dataset
    - embedding_log_var: (N, latent_dims) float32 dataset
    - pose: (N, 4) float32 dataset
    - global_weight: (N, 1) float32 dataset
    - source_file: string dataset with source parquet filename
    - source_row_index: int64 dataset with row index in source file

Attributes stored in HDF5:
    - checkpoint_path: path to model checkpoint
    - validation_dir: path to validation directory
    - snr: SNR value used
    - total_particles: total number of particles processed
    - latent_dims: dimension of latent space
    - creation_date: ISO timestamp
"""

import argparse
import h5py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
import logging

from cryolens.utils.checkpoint_loading import load_vae_model
from cryolens.inference.pipeline import InferencePipeline
from cryolens.data.parquet_loader import extract_volume_from_row

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extract embeddings from validation data with batching and periodic flushing.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        pipeline: InferencePipeline,
        output_h5_path: Path,
        latent_dims: int,
        batch_size: int = 32,
        flush_every: int = 1000
    ):
        """
        Initialize the embedding extractor.
        
        Parameters
        ----------
        model : torch.nn.Module
            Trained CryoLens model
        pipeline : InferencePipeline
            Inference pipeline for processing volumes
        output_h5_path : Path
            Path to output HDF5 file
        latent_dims : int
            Dimension of latent space
        batch_size : int
            Number of volumes to process in each batch
        flush_every : int
            Number of particles to accumulate before flushing to disk
        """
        self.model = model
        self.pipeline = pipeline
        self.output_h5_path = Path(output_h5_path)
        self.latent_dims = latent_dims
        self.batch_size = batch_size
        self.flush_every = flush_every
        
        # Buffers for accumulating data before flushing
        self.buffer = {
            'pdb_codes': [],
            'embedding_mean': [],
            'embedding_log_var': [],
            'pose': [],
            'global_weight': [],
            'source_file': [],
            'source_row_index': []
        }
        
        # Counter for total particles processed
        self.total_processed = 0
        
        # Initialize HDF5 file
        self._initialize_h5_file()
    
    def _initialize_h5_file(self):
        """Initialize the HDF5 file with expandable datasets."""
        logger.info(f"Initializing HDF5 file: {self.output_h5_path}")
        
        with h5py.File(self.output_h5_path, 'w') as f:
            # Create expandable datasets
            # Using maxshape=(None, ...) allows unlimited expansion along first axis
            
            # String datasets for pdb_codes and source_file
            dt_string = h5py.string_dtype(encoding='utf-8')
            f.create_dataset(
                'pdb_codes',
                shape=(0,),
                maxshape=(None,),
                dtype=dt_string,
                chunks=True,
                compression='gzip'
            )
            f.create_dataset(
                'source_file',
                shape=(0,),
                maxshape=(None,),
                dtype=dt_string,
                chunks=True,
                compression='gzip'
            )
            
            # Numeric datasets
            f.create_dataset(
                'embedding_mean',
                shape=(0, self.latent_dims),
                maxshape=(None, self.latent_dims),
                dtype='float32',
                chunks=True,
                compression='gzip'
            )
            f.create_dataset(
                'embedding_log_var',
                shape=(0, self.latent_dims),
                maxshape=(None, self.latent_dims),
                dtype='float32',
                chunks=True,
                compression='gzip'
            )
            f.create_dataset(
                'pose',
                shape=(0, 4),
                maxshape=(None, 4),
                dtype='float32',
                chunks=True,
                compression='gzip'
            )
            f.create_dataset(
                'global_weight',
                shape=(0, 1),
                maxshape=(None, 1),
                dtype='float32',
                chunks=True,
                compression='gzip'
            )
            f.create_dataset(
                'source_row_index',
                shape=(0,),
                maxshape=(None,),
                dtype='int64',
                chunks=True,
                compression='gzip'
            )
            
            # Store metadata as attributes
            f.attrs['creation_date'] = datetime.now().isoformat()
            f.attrs['latent_dims'] = self.latent_dims
            f.attrs['total_particles'] = 0  # Will be updated as we process
        
        logger.info("HDF5 file initialized successfully")
    
    def _flush_buffer(self):
        """Flush accumulated data in buffer to HDF5 file."""
        if len(self.buffer['pdb_codes']) == 0:
            return
        
        n_new = len(self.buffer['pdb_codes'])
        logger.info(f"Flushing {n_new} particles to HDF5 (total so far: {self.total_processed + n_new})")
        
        with h5py.File(self.output_h5_path, 'a') as f:
            # Get current size and resize datasets
            current_size = f['pdb_codes'].shape[0]
            new_size = current_size + n_new
            
            # Resize all datasets
            f['pdb_codes'].resize((new_size,))
            f['embedding_mean'].resize((new_size, self.latent_dims))
            f['embedding_log_var'].resize((new_size, self.latent_dims))
            f['pose'].resize((new_size, 4))
            f['global_weight'].resize((new_size, 1))
            f['source_file'].resize((new_size,))
            f['source_row_index'].resize((new_size,))
            
            # Write buffered data
            f['pdb_codes'][current_size:new_size] = self.buffer['pdb_codes']
            f['embedding_mean'][current_size:new_size] = np.vstack(self.buffer['embedding_mean'])
            f['embedding_log_var'][current_size:new_size] = np.vstack(self.buffer['embedding_log_var'])
            f['pose'][current_size:new_size] = np.vstack(self.buffer['pose'])
            f['global_weight'][current_size:new_size] = np.vstack(self.buffer['global_weight'])
            f['source_file'][current_size:new_size] = self.buffer['source_file']
            f['source_row_index'][current_size:new_size] = self.buffer['source_row_index']
            
            # Update total count
            self.total_processed += n_new
            f.attrs['total_particles'] = self.total_processed
        
        # Clear buffer
        for key in self.buffer:
            self.buffer[key] = []
        
        logger.info(f"Flush complete. Total particles: {self.total_processed}")
    
    def process_batch(
        self,
        volumes: List[np.ndarray],
        pdb_codes: List[str],
        source_files: List[str],
        row_indices: List[int]
    ):
        """
        Process a batch of volumes and add results to buffer.
        
        Parameters
        ----------
        volumes : List[np.ndarray]
            List of volume arrays
        pdb_codes : List[str]
            Corresponding PDB codes
        source_files : List[str]
            Source parquet filenames
        row_indices : List[int]
            Row indices in source files
        """
        # Stack volumes into batch
        volumes_batch = np.stack(volumes)
        
        # Process through pipeline to get embeddings
        # We only need embeddings, not reconstructions
        with torch.no_grad():
            results = self.pipeline.process_batch(
                volumes_batch,
                batch_size=self.batch_size,
                return_embeddings=True,
                return_reconstruction=False
            )
        
        # Add to buffer
        for i in range(len(volumes)):
            self.buffer['pdb_codes'].append(pdb_codes[i])
            self.buffer['embedding_mean'].append(results['embeddings'][i])
            self.buffer['embedding_log_var'].append(results['log_var'][i])
            self.buffer['pose'].append(results['pose'][i])
            self.buffer['global_weight'].append(results['global_weight'][i])
            self.buffer['source_file'].append(source_files[i])
            self.buffer['source_row_index'].append(row_indices[i])
        
        # Check if we should flush
        if len(self.buffer['pdb_codes']) >= self.flush_every:
            self._flush_buffer()
    
    def finalize(self):
        """Flush any remaining data in buffer and close."""
        if len(self.buffer['pdb_codes']) > 0:
            logger.info("Flushing final buffer")
            self._flush_buffer()
        
        logger.info(f"Extraction complete. Total particles processed: {self.total_processed}")


def process_validation_directory(
    validation_dir: Path,
    extractor: EmbeddingExtractor,
    snr: float = 5.0,
    box_size: int = 48
):
    """
    Process all validation parquet files in a directory.
    
    Parameters
    ----------
    validation_dir : Path
        Directory containing validation parquet files
    extractor : EmbeddingExtractor
        Embedding extractor instance
    snr : float
        SNR value for file pattern matching
    box_size : int
        Expected box size for volumes
    """
    validation_dir = Path(validation_dir)
    
    # Find all validation parquet files
    pattern = f"validation_*_snr{snr}.parquet"
    parquet_files = sorted(validation_dir.glob(pattern))
    
    if not parquet_files:
        raise ValueError(f"No validation parquet files found matching {pattern} in {validation_dir}")
    
    logger.info(f"Found {len(parquet_files)} validation parquet files")
    
    # Process each file
    for file_idx, parquet_file in enumerate(parquet_files, 1):
        logger.info(f"Processing file {file_idx}/{len(parquet_files)}: {parquet_file.name}")
        
        try:
            df = pd.read_parquet(parquet_file)
            logger.info(f"  Loaded {len(df)} rows from {parquet_file.name}")
        except Exception as e:
            logger.error(f"  Could not read {parquet_file.name}: {e}")
            continue
        
        # Find pdb_code column
        pdb_col = None
        for col_name in ['pdb_code', 'structure', 'pdb_id']:
            if col_name in df.columns:
                pdb_col = col_name
                break
        
        if pdb_col is None:
            logger.warning(f"  No PDB code column found in {parquet_file.name}, skipping")
            continue
        
        # Process in batches
        batch_volumes = []
        batch_pdb_codes = []
        batch_source_files = []
        batch_row_indices = []
        
        for row_idx, row in df.iterrows():
            # Extract volume
            volume = extract_volume_from_row(row, expected_shape=(box_size, box_size, box_size))
            
            if volume is None:
                logger.debug(f"  Could not extract volume from row {row_idx}")
                continue
            
            # Extract PDB code
            pdb_code = str(row[pdb_col]).lower()
            
            # Add to batch
            batch_volumes.append(volume)
            batch_pdb_codes.append(pdb_code)
            batch_source_files.append(parquet_file.name)
            batch_row_indices.append(int(row_idx))
            
            # Process batch when it reaches the desired size
            if len(batch_volumes) >= extractor.batch_size:
                extractor.process_batch(
                    batch_volumes,
                    batch_pdb_codes,
                    batch_source_files,
                    batch_row_indices
                )
                
                # Clear batch
                batch_volumes = []
                batch_pdb_codes = []
                batch_source_files = []
                batch_row_indices = []
        
        # Process remaining volumes in last batch
        if len(batch_volumes) > 0:
            extractor.process_batch(
                batch_volumes,
                batch_pdb_codes,
                batch_source_files,
                batch_row_indices
            )
        
        logger.info(f"  Completed processing {parquet_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--validation-dir',
        type=Path,
        required=True,
        help='Directory containing validation parquet files'
    )
    parser.add_argument(
        '--output-h5',
        type=Path,
        required=True,
        help='Output HDF5 file path'
    )
    
    # Optional arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--flush-every',
        type=int,
        default=1000,
        help='Number of particles to accumulate before flushing to disk (default: 1000)'
    )
    parser.add_argument(
        '--snr',
        type=float,
        default=5.0,
        help='SNR value for validation data (default: 5.0)'
    )
    parser.add_argument(
        '--box-size',
        type=int,
        default=48,
        help='Expected box size for volumes (default: 48)'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("EXTRACT VALIDATION EMBEDDINGS")
    print("="*70)
    print(f"Checkpoint:        {args.checkpoint}")
    print(f"Validation dir:    {args.validation_dir}")
    print(f"Output HDF5:       {args.output_h5}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Flush every:       {args.flush_every} particles")
    print(f"SNR:               {args.snr}")
    print(f"Box size:          {args.box_size}")
    print("="*70)
    
    # Check if validation directory exists
    if not args.validation_dir.exists():
        logger.error(f"Validation directory does not exist: {args.validation_dir}")
        return 1
    
    # Check if output file already exists
    if args.output_h5.exists():
        logger.warning(f"Output file already exists: {args.output_h5}")
        response = input("Overwrite? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Aborting")
            return 0
        args.output_h5.unlink()
    
    # Create output directory if needed
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    try:
        model, config = load_vae_model(
            args.checkpoint,
            device=device,
            load_config=True,
            strict_loading=False
        )
        model.eval()
        logger.info("Model loaded successfully")
        
        # Get latent dimensions
        latent_dims = model.latent_dims
        logger.info(f"Latent dimensions: {latent_dims}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create pipeline
    logger.info("Creating inference pipeline...")
    pipeline = InferencePipeline(
        model=model,
        device=device,
        normalization_method=config.get('normalization', 'z-score')
    )
    logger.info("Inference pipeline created")
    
    # Create extractor
    logger.info("Initializing embedding extractor...")
    extractor = EmbeddingExtractor(
        model=model,
        pipeline=pipeline,
        output_h5_path=args.output_h5,
        latent_dims=latent_dims,
        batch_size=args.batch_size,
        flush_every=args.flush_every
    )
    
    # Store metadata in HDF5
    with h5py.File(args.output_h5, 'a') as f:
        f.attrs['checkpoint_path'] = str(args.checkpoint)
        f.attrs['validation_dir'] = str(args.validation_dir)
        f.attrs['snr'] = args.snr
        f.attrs['box_size'] = args.box_size
        f.attrs['batch_size'] = args.batch_size
        f.attrs['flush_every'] = args.flush_every
    
    # Process validation directory
    try:
        logger.info("Starting processing...")
        process_validation_directory(
            validation_dir=args.validation_dir,
            extractor=extractor,
            snr=args.snr,
            box_size=args.box_size
        )
        
        # Finalize
        extractor.finalize()
        
        logger.info("="*70)
        logger.info("EXTRACTION COMPLETE")
        logger.info(f"Total particles processed: {extractor.total_processed}")
        logger.info(f"Output saved to: {args.output_h5}")
        logger.info("="*70)
        
        # Print HDF5 file info
        logger.info("\nHDF5 file contents:")
        with h5py.File(args.output_h5, 'r') as f:
            logger.info(f"  Datasets:")
            for name, dataset in f.items():
                logger.info(f"    {name}: shape={dataset.shape}, dtype={dataset.dtype}")
            logger.info(f"  Attributes:")
            for key, value in f.attrs.items():
                logger.info(f"    {key}: {value}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
