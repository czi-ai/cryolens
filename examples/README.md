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

### 2. Extract Multiple Structures

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

After extracting particles, you can:
1. Run reconstructions with CryoLens (see future examples)
2. Perform alignment and averaging
3. Calculate FSC curves
4. Analyze embeddings
