# CryoLens Examples

Examples demonstrating CryoLens functionality for particle reconstruction and analysis.

## Prerequisites

```bash
# Install CryoLens with Copick support
pip install cryolens[copick]
```

## Quick Start

### 1. Extract Particles from Data Portal

Extract particles from the ML Challenge public test set:

```bash
python examples/extract_copick_particles.py \
    --config mlc_experimental_publictest \
    --structures ribosome \
    --num-particles 30 \
    --output ./example_data/
```

This will:
- Connect to the CZ cryoET Data Portal
- Extract 30 ribosome particles
- Save them as `./example_data/ribosome_particles.zarr/`

### 2. Reconstruct Particles

Reconstruct extracted particles using CryoLens:

```bash
python examples/reconstruct_particles.py \
    --particles ./example_data/ribosome_particles.zarr \
    --checkpoint-epoch 2600 \
    --output ./reconstructions/
```

This will:
- Load the pre-trained CryoLens model (epoch 2600)
- Reconstruct all particles
- Save mean reconstruction as MRC file
- Generate FSC curve if ground truth provided

### 3. Extract Multiple Structures

```bash
python examples/extract_copick_particles.py \
    --config mlc_experimental_publictest \
    --structures ribosome,thyroglobulin,apo-ferritin \
    --num-particles 30 \
    --output ./example_data/
```

### 3. Use Custom Copick Project

```bash
python examples/extract_copick_particles.py \
    --copick-config /path/to/your/copick_config.json \
    --structures your_structure \
    --num-particles 50 \
    --output ./my_particles/
```

## Available Examples

### `reconstruct_particles.py`
Reconstruct particles using pre-trained CryoLens model.

**Features:**
- Automatic checkpoint fetching via pooch
- Batch reconstruction with GPU support
- Uncertainty estimation with multiple samples
- FSC calculation with ground truth
- Saves MRC files and analysis plots

**Usage:**
```bash
python examples/reconstruct_particles.py --help
```

**Example with uncertainty estimation:**
```bash
python examples/reconstruct_particles.py \
    --particles ./example_data/ribosome_particles.zarr \
    --checkpoint-epoch 2600 \
    --num-samples 10 \
    --output ./reconstructions/
```

**Example with FSC analysis:**
```bash
python examples/reconstruct_particles.py \
    --particles ./example_data/ribosome_particles.zarr \
    --checkpoint-epoch 2600 \
    --ground-truth ./references/ribosome.mrc \
    --output ./reconstructions/
```

### `extract_copick_particles.py`
Extract particle subvolumes from Copick projects.

**Features:**
- Direct Data Portal access via embedded configs
- Custom local Copick project support
- Batch extraction of multiple structures
- Saves particles as zarr with metadata

**Usage:**
```bash
python examples/extract_copick_particles.py --help
```

## Data Sources

Examples use embedded Copick configs that connect to CZ cryoET Data Portal:

- **`mlc_experimental_publictest`**: 6 structures for out-of-distribution testing
  - apo-ferritin, beta-amylase, beta-galactoside, ribosome, thyroglobulin, virus-like-particle
  
- **`mlc_experimental_privatetest`**: Private test set

- **`mlc_experimental_training`**: ML Challenge training set

- **`training_synthetic`**: 104 structures from synthetic training dataset

List available configs:
```python
from cryolens.data import list_available_configs
configs = list_available_configs()
for name, desc in configs.items():
    print(f"{name}: {desc}")
```

## Reconstruction Output Format

Reconstructions are saved with the following structure:

```
reconstructio ns/ribosome/
├── ribosome_mean_reconstruction.mrc    # Mean reconstruction
├── ribosome_uncertainty.mrc            # Uncertainty map (if num-samples > 1)
├── ribosome_fsc.png                     # FSC curve plot (if ground truth provided)
├── ribosome_fsc.npz                     # FSC data (if ground truth provided)
└── ribosome_summary.json                # Reconstruction metadata
```

Load reconstructions in Python:
```python
import mrcfile
import json

# Load reconstruction
with mrcfile.open('reconstructions/ribosome/ribosome_mean_reconstruction.mrc') as mrc:
    reconstruction = mrc.data
    voxel_size = mrc.voxel_size.x

# Load summary
with open('reconstructions/ribosome/ribosome_summary.json') as f:
    summary = json.load(f)
    
print(f"Resolution (FSC=0.5): {summary.get('resolution_at_fsc05', 'N/A')} Å")
```

## Output Format

Extracted particles are saved as zarr arrays with the following structure:

```
ribosome_particles.zarr/
├── particles              # (N, 48, 48, 48) particle volumes
├── positions              # (N, 3) positions in Angstroms
├── orientations           # (N, 3, 3) rotation matrices (if available)
└── metadata.json          # Extraction metadata
```

Load particles in Python:
```python
import zarr
import json

# Load particles
root = zarr.open('ribosome_particles.zarr', mode='r')
particles = root['particles'][:]
positions = root['positions'][:]

# Load metadata
with open('ribosome_particles.zarr/metadata.json') as f:
    metadata = json.load(f)
    
print(f"Loaded {len(particles)} particles")
print(f"Box size: {metadata['box_size']}")
print(f"Voxel spacing: {metadata['voxel_spacing']} Å")
```

## Troubleshooting

### "copick is not installed"
```bash
pip install cryolens[copick]
```

### "No picks found"
Make sure the structure name matches exactly what's in the Copick config. Check available structures:
```python
import copick
from cryolens.data import get_copick_config

config_path = get_copick_config("mlc_experimental_publictest")
root = copick.from_file(config_path)
print("Available objects:", [obj.name for obj in root.config.pickable_objects])
```

### "No tomograms found"
The specified voxel spacing may not be available. Try `--voxel-spacing 10.0` (default) or check what's available in the dataset.

## Next Steps

Complete workflow:
1. Extract particles with `extract_copick_particles.py`
2. Reconstruct with `reconstruct_particles.py`
3. Analyze results (FSC curves, uncertainty maps)
4. Use interactive picking with napari-cryolens (see PR #4)
