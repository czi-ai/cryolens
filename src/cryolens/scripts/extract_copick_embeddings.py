"""
Extract embeddings from Copick projects.

This script processes Copick projects run-by-run to extract VAE embeddings
(mean, sigma), pose, global weight, structure labels, and coordinates for
all particle picks. It is designed to be memory-efficient by loading one
tomogram at a time and flushing results after each run.

Usage:
    python -m cryolens.scripts.extract_copick_embeddings \
        --checkpoint models/cryolens_epoch_2600.pt \
        --copick-config data/copick_config.json \
        --output-h5 embeddings/copick_embeddings.h5 \
        --batch-size 64

For specific structures:
    python -m cryolens.scripts.extract_copick_embeddings \
        --checkpoint models/cryolens_epoch_2600.pt \
        --copick-config data/copick_config.json \
        --output-h5 embeddings/copick_embeddings.h5 \
        --structure-filter ribosome thyroglobulin
"""

import argparse
import json
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import h5py
from tqdm import tqdm

from cryolens.utils.checkpoint_loading import load_vae_model
from cryolens.data import CopickDataLoader


# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*weights_only.*')


class CopickEmbeddingExtractor:
    """Extract embeddings from Copick projects run-by-run."""
    
    def __init__(
        self,
        checkpoint_path: str,
        copick_config_path: str,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the Copick embedding extractor.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to VAE checkpoint file
        copick_config_path : str
            Path to Copick configuration JSON
        device : Optional[torch.device]
            Device to run on (default: cuda if available)
        """
        self.checkpoint_path = checkpoint_path
        self.copick_config_path = copick_config_path
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Model and config
        self.model = None
        self.config = None
        self.normalization_method = 'z-score'
        
        # Copick loader
        self.copick_loader = None
        
        print(f"Initialized extractor with device: {self.device}")
    
    def load_model(self):
        """Load the VAE model from checkpoint."""
        print(f"Loading VAE from checkpoint: {self.checkpoint_path}")
        
        try:
            self.model, self.config = load_vae_model(
                self.checkpoint_path,
                device=self.device,
                load_config=True,
                strict_loading=False
            )
            self.model.eval()
            
            # Get normalization method from config
            self.normalization_method = self.config.get('normalization', 'z-score')
            
            print("Model loaded successfully")
            print(f"  Box size: {self.config.get('box_size', 48)}")
            print(f"  Latent dims: {self.config.get('latent_dims', 40)}")
            print(f"  Normalization: {self.normalization_method}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def initialize_copick(self):
        """Initialize Copick data loader."""
        print(f"Initializing Copick loader: {self.copick_config_path}")
        
        try:
            self.copick_loader = CopickDataLoader(self.copick_config_path)
            
            runs = self.copick_loader.list_runs()
            structures = self.copick_loader.list_available_structures()
            
            print(f"  Found {len(runs)} runs")
            print(f"  Found {len(structures)} structures: {', '.join(structures)}")
            
        except Exception as e:
            print(f"Error initializing Copick: {e}")
            raise
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize a volume using the same method as training.
        
        Parameters
        ----------
        volume : np.ndarray
            Input volume
            
        Returns
        -------
        np.ndarray
            Normalized volume
        """
        if self.normalization_method == "z-score":
            mean_val = np.mean(volume)
            std_val = np.std(volume)
            if std_val > 1e-6:
                return (volume - mean_val) / std_val
            return volume - mean_val
            
        elif self.normalization_method == "min-max":
            min_val = np.min(volume)
            max_val = np.max(volume)
            return (volume - min_val) / (max_val - min_val + 1e-6)
            
        elif self.normalization_method == "percentile":
            p1 = np.percentile(volume, 1)
            p99 = np.percentile(volume, 99)
            return np.clip((volume - p1) / (p99 - p1 + 1e-6), 0, 1)
            
        else:
            return volume
    
    def extract_run_embeddings(
        self,
        run,
        target_voxel_spacing: float = 10.0,
        voxel_spacing_tolerance: float = 1.0,
        box_size: int = 48,
        batch_size: int = 64,
        structure_filter: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Optional[Dict]:
        """
        Extract embeddings for all picks in a single run.
        
        Parameters
        ----------
        run : copick.Run
            Copick run object
        target_voxel_spacing : float
            Target voxel spacing in Angstroms
        voxel_spacing_tolerance : float
            Tolerance for voxel spacing matching
        box_size : int
            Size of particle box in voxels
        batch_size : int
            Batch size for processing
        structure_filter : Optional[List[str]]
            List of structure names to process (None for all)
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        Optional[Dict]
            Dictionary containing embeddings and metadata, or None if no data
        """
        if verbose:
            print(f"\nProcessing run: {run.name}")
        
        # Find best voxel spacing
        best_voxel_spacing = self._find_best_voxel_spacing(
            run, target_voxel_spacing, voxel_spacing_tolerance
        )
        
        if best_voxel_spacing is None:
            if verbose:
                available = [vs.voxel_size for vs in run.voxel_spacings]
                print(f"  No tomogram near {target_voxel_spacing}Å (available: {available})")
            return None
        
        if verbose:
            print(f"  Using voxel spacing: {best_voxel_spacing.voxel_size}Å")
        
        # Get tomogram
        tomogram = self._get_tomogram(best_voxel_spacing)
        if tomogram is None:
            if verbose:
                print("  No tomograms found")
            return None
        
        # Load tomogram data
        try:
            tomogram_data = self._load_tomogram_data(tomogram)
            if tomogram_data is None:
                return None
            
            if verbose:
                print(f"  Loaded tomogram shape: {tomogram_data.shape}")
                
        except Exception as e:
            if verbose:
                print(f"  Error loading tomogram: {e}")
            return None
        
        # Collect all particles from picks
        particles_data = []
        
        for picks in run.picks:
            structure_name = picks.pickable_object_name
            
            # Skip if no points
            if len(picks.points) == 0:
                continue
            
            # Apply structure filter
            if structure_filter and structure_name not in structure_filter:
                continue
            
            if verbose:
                print(f"  Processing {structure_name}: {len(picks.points)} picks")
            
            for point in picks.points:
                # Extract particle subvolume
                particle_info = self._extract_particle(
                    tomogram_data,
                    point,
                    best_voxel_spacing.voxel_size,
                    box_size,
                    structure_name,
                    run.name,
                    picks
                )
                
                if particle_info is not None:
                    particles_data.append(particle_info)
        
        # Clean up tomogram
        del tomogram_data
        gc.collect()
        
        if not particles_data:
            if verbose:
                print("  No valid particles extracted")
            return None
        
        if verbose:
            print(f"  Extracted {len(particles_data)} valid particles")
        
        # Process in batches to extract embeddings
        all_embeddings = []
        all_log_vars = []
        all_poses = []
        all_global_weights = []
        
        for i in tqdm(
            range(0, len(particles_data), batch_size),
            desc=f"  Extracting embeddings",
            disable=not verbose
        ):
            batch_data = particles_data[i:i + batch_size]
            batch_volumes = [d['volume'] for d in batch_data]
            
            # Normalize volumes
            batch_normalized = [self.normalize_volume(v) for v in batch_volumes]
            
            # Convert to tensor
            batch_tensor = torch.stack([
                torch.tensor(vol, dtype=torch.float32).unsqueeze(0)
                for vol in batch_normalized
            ]).to(self.device)
            
            # Extract features
            with torch.no_grad():
                mu, log_var, pose, global_weight = self.model.encode(batch_tensor)
            
            # Convert to numpy
            all_embeddings.extend(mu.cpu().numpy())
            all_log_vars.extend(log_var.cpu().numpy())
            all_poses.extend(pose.cpu().numpy())
            all_global_weights.extend(global_weight.cpu().numpy())
        
        # Compile results
        result = {
            'run_name': run.name,
            'n_particles': len(particles_data),
            'structure_labels': np.array([d['structure_name'] for d in particles_data], dtype='S'),
            'embeddings': np.array(all_embeddings, dtype=np.float32),
            'log_var': np.array(all_log_vars, dtype=np.float32),
            'pose': np.array(all_poses, dtype=np.float32),
            'global_weight': np.array(all_global_weights, dtype=np.float32),
            'coordinates_angstrom': np.array(
                [d['position_angstrom'] for d in particles_data],
                dtype=np.float32
            ),
            'coordinates_voxel': np.array(
                [d['position_voxel'] for d in particles_data],
                dtype=np.float32
            ),
            'orientations': np.array(
                [d['orientation'] for d in particles_data],
                dtype=np.float32
            ),
            'voxel_spacings': np.array(
                [d['voxel_spacing'] for d in particles_data],
                dtype=np.float32
            ),
            'session_ids': np.array(
                [d['session_id'] for d in particles_data],
                dtype='S'
            ),
            'user_ids': np.array(
                [d['user_id'] for d in particles_data],
                dtype='S'
            )
        }
        
        if verbose:
            # Print structure distribution
            unique_structures, counts = np.unique(
                result['structure_labels'],
                return_counts=True
            )
            print(f"  Structure distribution:")
            for struct, count in zip(unique_structures, counts):
                print(f"    {struct.decode('utf-8')}: {count}")
        
        return result
    
    def _find_best_voxel_spacing(self, run, target_spacing: float, tolerance: float):
        """Find the voxel spacing closest to the target."""
        best_spacing = None
        best_diff = float('inf')
        
        for vs in run.voxel_spacings:
            diff = abs(vs.voxel_size - target_spacing)
            if diff < best_diff and diff <= tolerance:
                best_diff = diff
                best_spacing = vs
        
        return best_spacing
    
    def _get_tomogram(self, voxel_spacing):
        """Get the first available tomogram, preferring denoised."""
        tomograms = list(voxel_spacing.tomograms)
        if not tomograms:
            return None
        
        # Prefer denoised tomograms
        for tomo in tomograms:
            if hasattr(tomo, 'tomo_type') and tomo.tomo_type == 'denoised':
                return tomo
        
        # Fall back to first available
        return tomograms[0]
    
    def _load_tomogram_data(self, tomogram) -> Optional[np.ndarray]:
        """Load tomogram data from zarr store."""
        try:
            import zarr
            tomo_zarr = zarr.open(tomogram.zarr(), mode='r')
            
            # Handle different zarr structures
            for key in ['0', 's0', 'data']:
                if key in tomo_zarr:
                    return np.array(tomo_zarr[key])
            
            # Try first available key
            keys = list(tomo_zarr.keys())
            if keys:
                return np.array(tomo_zarr[keys[0]])
            
            return None
            
        except Exception as e:
            warnings.warn(f"Error loading tomogram data: {e}")
            return None
    
    def _extract_particle(
        self,
        tomogram_data: np.ndarray,
        point,
        voxel_spacing: float,
        box_size: int,
        structure_name: str,
        run_name: str,
        picks
    ) -> Optional[Dict]:
        """
        Extract a particle subvolume and metadata from the tomogram.
        
        Parameters
        ----------
        tomogram_data : np.ndarray
            Tomogram data array (ZYX order)
        point : copick.PickPoint
            Pick point object
        voxel_spacing : float
            Voxel spacing in Angstroms
        box_size : int
            Size of particle box
        structure_name : str
            Name of the structure
        run_name : str
            Name of the run
        picks : copick.Picks
            Picks object for metadata
            
        Returns
        -------
        Optional[Dict]
            Dictionary with particle data and metadata, or None if invalid
        """
        # Get position in Angstroms
        location = point.location
        if hasattr(location, 'x'):
            position_angstrom = np.array([location.x, location.y, location.z])
        elif isinstance(location, (list, tuple, np.ndarray)):
            position_angstrom = np.array(location)
        else:
            return None
        
        # Convert to voxel coordinates
        position_voxel = (position_angstrom / voxel_spacing).astype(int)
        
        # Calculate bounds
        half_box = box_size // 2
        z_min = position_voxel[2] - half_box
        z_max = position_voxel[2] + half_box
        y_min = position_voxel[1] - half_box
        y_max = position_voxel[1] + half_box
        x_min = position_voxel[0] - half_box
        x_max = position_voxel[0] + half_box
        
        # Check bounds
        if (z_min < 0 or z_max > tomogram_data.shape[0] or
            y_min < 0 or y_max > tomogram_data.shape[1] or
            x_min < 0 or x_max > tomogram_data.shape[2]):
            return None
        
        # Extract subvolume (tomogram is in ZYX order)
        subvolume = tomogram_data[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Check if we got the right size
        if subvolume.shape != (box_size, box_size, box_size):
            return None
        
        # Extract orientation/rotation matrix
        orientation = self._extract_orientation(point)
        
        # Get metadata
        session_id = picks.session_id if hasattr(picks, 'session_id') else 'unknown'
        user_id = picks.user_id if hasattr(picks, 'user_id') else 'unknown'
        
        return {
            'volume': subvolume.copy(),
            'position_angstrom': position_angstrom,
            'position_voxel': position_voxel,
            'orientation': orientation,
            'structure_name': structure_name,
            'run_name': run_name,
            'voxel_spacing': voxel_spacing,
            'session_id': session_id,
            'user_id': user_id
        }
    
    def _extract_orientation(self, point) -> np.ndarray:
        """Extract orientation/rotation matrix from a pick point."""
        try:
            if hasattr(point, 'transformation') and point.transformation is not None:
                transformation = point.transformation
                
                # Handle different transformation formats
                if hasattr(transformation, 'shape'):
                    if transformation.shape == (4, 4):
                        return transformation[:3, :3]
                    elif transformation.shape == (3, 3):
                        return transformation
                
                # Try to convert to array
                transformation = np.array(transformation)
                if transformation.shape == (4, 4):
                    return transformation[:3, :3]
                elif transformation.shape == (3, 3):
                    return transformation
        except Exception:
            pass
        
        # Return identity if no valid transformation
        return np.eye(3, dtype=np.float32)
    
    def extract_all_embeddings(
        self,
        output_h5_path: str,
        target_voxel_spacing: float = 10.0,
        voxel_spacing_tolerance: float = 1.0,
        box_size: int = 48,
        batch_size: int = 64,
        structure_filter: Optional[List[str]] = None,
        max_runs: Optional[int] = None,
        resume: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Extract embeddings for all runs in the Copick project.
        
        Parameters
        ----------
        output_h5_path : str
            Path to output HDF5 file
        target_voxel_spacing : float
            Target voxel spacing in Angstroms
        voxel_spacing_tolerance : float
            Tolerance for voxel spacing matching
        box_size : int
            Size of particle box in voxels
        batch_size : int
            Batch size for processing
        structure_filter : Optional[List[str]]
            List of structure names to process
        max_runs : Optional[int]
            Maximum number of runs to process (for testing)
        resume : bool
            Whether to skip existing runs in output file
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        Dict
            Summary statistics
        """
        output_path = Path(output_h5_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get list of runs
        runs = list(self.copick_loader.root.runs)
        if max_runs:
            runs = runs[:max_runs]
        
        print(f"\nProcessing {len(runs)} runs")
        print("=" * 60)
        
        # Check for existing runs if resuming
        existing_runs = set()
        if resume and output_path.exists():
            with h5py.File(output_path, 'r') as f:
                existing_runs = set(f.keys()) - {'metadata'}
            if existing_runs:
                print(f"Resuming: found {len(existing_runs)} existing runs")
        
        # Statistics
        total_particles = 0
        runs_processed = 0
        structure_counts = {}
        
        # Process each run
        for run in tqdm(runs, desc="Processing runs"):
            # Skip if resuming and run exists
            if resume and run.name in existing_runs:
                if verbose:
                    print(f"Skipping existing run: {run.name}")
                continue
            
            try:
                # Extract embeddings for this run
                run_data = self.extract_run_embeddings(
                    run,
                    target_voxel_spacing=target_voxel_spacing,
                    voxel_spacing_tolerance=voxel_spacing_tolerance,
                    box_size=box_size,
                    batch_size=batch_size,
                    structure_filter=structure_filter,
                    verbose=verbose
                )
                
                if run_data is None:
                    continue
                
                # Write to HDF5
                self._write_run_to_h5(output_path, run_data)
                
                # Update statistics
                runs_processed += 1
                total_particles += run_data['n_particles']
                
                for struct in run_data['structure_labels']:
                    struct_str = struct.decode('utf-8')
                    structure_counts[struct_str] = structure_counts.get(struct_str, 0) + 1
                
                # Memory cleanup
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing run {run.name}: {e}")
                continue
        
        # Write metadata
        self._write_metadata(
            output_path,
            total_particles,
            runs_processed,
            structure_counts,
            box_size,
            target_voxel_spacing
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Runs processed: {runs_processed}")
        print(f"Total particles: {total_particles}")
        print(f"\nStructure distribution:")
        for struct, count in sorted(structure_counts.items(), key=lambda x: -x[1]):
            print(f"  {struct}: {count}")
        print(f"\nOutput saved to: {output_path}")
        print("=" * 60)
        
        return {
            'runs_processed': runs_processed,
            'total_particles': total_particles,
            'structure_counts': structure_counts
        }
    
    def _write_run_to_h5(self, h5_path: Path, run_data: Dict):
        """Write run data to HDF5 file."""
        with h5py.File(h5_path, 'a') as f:
            # Create group for this run
            if run_data['run_name'] in f:
                del f[run_data['run_name']]
            
            grp = f.create_group(run_data['run_name'])
            
            # Write datasets
            grp.create_dataset('structure_labels', data=run_data['structure_labels'])
            grp.create_dataset('embeddings', data=run_data['embeddings'], compression='gzip')
            grp.create_dataset('log_var', data=run_data['log_var'], compression='gzip')
            grp.create_dataset('pose', data=run_data['pose'], compression='gzip')
            grp.create_dataset('global_weight', data=run_data['global_weight'], compression='gzip')
            grp.create_dataset('coordinates_angstrom', data=run_data['coordinates_angstrom'])
            grp.create_dataset('coordinates_voxel', data=run_data['coordinates_voxel'])
            grp.create_dataset('orientations', data=run_data['orientations'], compression='gzip')
            grp.create_dataset('voxel_spacings', data=run_data['voxel_spacings'])
            grp.create_dataset('session_ids', data=run_data['session_ids'])
            grp.create_dataset('user_ids', data=run_data['user_ids'])
            
            # Attributes
            grp.attrs['n_particles'] = run_data['n_particles']
    
    def _write_metadata(
        self,
        h5_path: Path,
        total_particles: int,
        runs_processed: int,
        structure_counts: Dict,
        box_size: int,
        target_voxel_spacing: float
    ):
        """Write metadata to HDF5 file."""
        with h5py.File(h5_path, 'a') as f:
            if 'metadata' in f:
                del f['metadata']
            
            meta_grp = f.create_group('metadata')
            
            meta_grp.attrs['checkpoint_path'] = str(self.checkpoint_path)
            meta_grp.attrs['copick_config_path'] = str(self.copick_config_path)
            meta_grp.attrs['box_size'] = box_size
            meta_grp.attrs['normalization_method'] = self.normalization_method
            meta_grp.attrs['total_particles'] = total_particles
            meta_grp.attrs['runs_processed'] = runs_processed
            meta_grp.attrs['target_voxel_spacing'] = target_voxel_spacing
            meta_grp.attrs['extraction_date'] = datetime.now().isoformat()
            
            # Store structure counts as dataset
            if structure_counts:
                structures = list(structure_counts.keys())
                counts = [structure_counts[s] for s in structures]
                meta_grp.create_dataset(
                    'structures',
                    data=np.array(structures, dtype='S')
                )
                meta_grp.create_dataset(
                    'structure_counts',
                    data=np.array(counts, dtype=np.int32)
                )


def main():
    """Main function for extracting Copick embeddings."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to VAE checkpoint file'
    )
    parser.add_argument(
        '--copick-config',
        type=str,
        required=True,
        help='Path to Copick configuration JSON file'
    )
    parser.add_argument(
        '--output-h5',
        type=str,
        required=True,
        help='Path to output HDF5 file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for processing (default: 64)'
    )
    parser.add_argument(
        '--box-size',
        type=int,
        default=48,
        help='Size of particle box in voxels (default: 48)'
    )
    parser.add_argument(
        '--target-voxel-spacing',
        type=float,
        default=10.0,
        help='Target voxel spacing in Angstroms (default: 10.0)'
    )
    parser.add_argument(
        '--voxel-spacing-tolerance',
        type=float,
        default=1.0,
        help='Tolerance for voxel spacing matching (default: 1.0)'
    )
    parser.add_argument(
        '--structure-filter',
        nargs='+',
        help='List of structure names to process (default: all)'
    )
    parser.add_argument(
        '--max-runs',
        type=int,
        help='Maximum number of runs to process (for testing)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip runs that already exist in output file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("COPICK EMBEDDINGS EXTRACTION")
    print("=" * 60)
    print(f"Checkpoint:             {args.checkpoint}")
    print(f"Copick config:          {args.copick_config}")
    print(f"Output HDF5:            {args.output_h5}")
    print(f"Batch size:             {args.batch_size}")
    print(f"Box size:               {args.box_size}")
    print(f"Target voxel spacing:   {args.target_voxel_spacing}Å")
    print(f"Voxel spacing tolerance: {args.voxel_spacing_tolerance}Å")
    if args.structure_filter:
        print(f"Structure filter:       {', '.join(args.structure_filter)}")
    if args.max_runs:
        print(f"Max runs:               {args.max_runs}")
    print(f"Resume:                 {args.resume}")
    print("=" * 60)
    
    # Initialize extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    extractor = CopickEmbeddingExtractor(
        args.checkpoint,
        args.copick_config,
        device=device
    )
    
    # Load model
    extractor.load_model()
    
    # Initialize Copick
    extractor.initialize_copick()
    
    # Extract embeddings
    summary = extractor.extract_all_embeddings(
        output_h5_path=args.output_h5,
        target_voxel_spacing=args.target_voxel_spacing,
        voxel_spacing_tolerance=args.voxel_spacing_tolerance,
        box_size=args.box_size,
        batch_size=args.batch_size,
        structure_filter=args.structure_filter,
        max_runs=args.max_runs,
        resume=args.resume,
        verbose=not args.quiet
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
