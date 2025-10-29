# API Reference

Detailed API documentation for CryoLens components.

## Inference Pipeline

### `create_inference_pipeline`

Create an inference pipeline for processing particles.

```python
from cryolens.inference import create_inference_pipeline

pipeline = create_inference_pipeline(
    checkpoint="v001",           # Model checkpoint (name or path)
    device="auto",               # Device: "auto", "cuda", or "cpu"
    use_segmented_decoder=True,  # Use segmented decoder architecture
    use_final_convolution=True,  # Apply final convolution layer
)
```

**Parameters:**
- `checkpoint` (str): Model checkpoint name (e.g., "v001") or path to checkpoint file
- `device` (str, optional): Device for inference. Default: "auto"
- `use_segmented_decoder` (bool, optional): Use segmented decoder. Default: True
- `use_final_convolution` (bool, optional): Apply final convolution. Default: True

**Returns:**
- `InferencePipeline`: Configured inference pipeline

### `InferencePipeline.process_volume`

Process a single particle volume.

```python
result = pipeline.process_volume(
    volume,                      # (48, 48, 48) numpy array
    return_splat_data=False,     # Return Gaussian splat parameters
    return_embeddings=True,      # Return embeddings
)
```

**Parameters:**
- `volume` (np.ndarray): Input volume of shape (48, 48, 48)
- `return_splat_data` (bool, optional): Return Gaussian splat parameters. Default: False
- `return_embeddings` (bool, optional): Return embeddings. Default: True

**Returns:**
- `dict`: Results dictionary with keys:
  - `reconstruction` (np.ndarray): 3D density map, shape (48, 48, 48)
  - `embedding` (np.ndarray): Structural features, shape (32,) [if return_embeddings=True]
  - `pose` (np.ndarray): Orientation quaternion, shape (4,)
  - `global_weight` (float): Amplitude multiplier
  - `splat_data` (dict): Gaussian splat parameters [if return_splat_data=True]

### `InferencePipeline.batch_process`

Process multiple particles efficiently.

```python
results = pipeline.batch_process(
    volumes,                     # (N, 48, 48, 48) numpy array
    batch_size=32,              # Batch size for processing
)
```

**Parameters:**
- `volumes` (np.ndarray): Input volumes of shape (N, 48, 48, 48)
- `batch_size` (int, optional): Batch size for GPU processing. Default: 32

**Returns:**
- `dict`: Results dictionary with keys:
  - `reconstructions` (np.ndarray): Density maps, shape (N, 48, 48, 48)
  - `embeddings` (np.ndarray): Features, shape (N, 32)
  - `poses` (np.ndarray): Quaternions, shape (N, 4)
  - `global_weights` (np.ndarray): Multipliers, shape (N,)

## Data Loading

### `CopickDataLoader`

Load particles from Copick projects.

```python
from cryolens.data import CopickDataLoader

loader = CopickDataLoader(
    config_path="copick_config.json",  # Path to Copick config
)
```

**Parameters:**
- `config_path` (str): Path to Copick configuration file

### `CopickDataLoader.load_particles`

Load particles from the Copick project.

```python
data = loader.load_particles(
    structure_filter=["ribosome"],     # Filter by structure names
    max_particles_per_structure=100,   # Max particles per structure
    target_voxel_spacing=10.0,         # Target voxel spacing in Å
    box_size=48,                       # Box size in voxels
)
```

**Parameters:**
- `structure_filter` (list, optional): List of structure names to load
- `max_particles_per_structure` (int, optional): Maximum particles per structure
- `target_voxel_spacing` (float, optional): Voxel spacing in Angstroms. Default: 10.0
- `box_size` (int, optional): Box size in voxels. Default: 48

**Returns:**
- `dict`: Dictionary mapping structure names to data dictionaries with keys:
  - `particles` (np.ndarray): Particle volumes, shape (N, 48, 48, 48)
  - `orientations` (np.ndarray): Rotation matrices, shape (N, 3, 3)
  - `positions` (np.ndarray): Positions in Å, shape (N, 3)

### `CopickDataLoader.list_available_structures`

List all structures available in the project.

```python
structures = loader.list_available_structures()
```

**Returns:**
- `list`: List of structure names (str)

## Gaussian Splats

### Extract Splat Parameters

Extract Gaussian splat parameters from a volume.

```python
from cryolens.splats import extract_gaussian_splats

splat_data = extract_gaussian_splats(
    pipeline=pipeline,
    volume=particle,
    include_free_segment=True,   # Include free segment splats
)
```

**Parameters:**
- `pipeline` (InferencePipeline): Configured inference pipeline
- `volume` (np.ndarray): Input volume, shape (48, 48, 48)
- `include_free_segment` (bool, optional): Include free segment. Default: True

**Returns:**
- `dict`: Dictionary with keys:
  - `centers` (np.ndarray): Gaussian centers, shape (N, 3)
  - `weights` (np.ndarray): Gaussian weights, shape (N,)
  - `scales` (np.ndarray): Gaussian scales, shape (N,)
  - `n_affinity` (int): Number of affinity segment Gaussians
  - `n_free` (int): Number of free segment Gaussians

## Utilities

### Rotation Utilities

Convert between rotation representations.

```python
from cryolens.utils import (
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
)

# Quaternion to rotation matrix
R = quaternion_to_rotation_matrix(quaternion, scalar_first=True)

# Rotation matrix to quaternion (w, x, y, z)
quat = rotation_matrix_to_quaternion(R, scalar_first=True)

# Rotation matrix to Euler angles
euler = rotation_matrix_to_euler(R, convention="XYZ", degrees=True)
```

### Alignment Utilities

Align point clouds and rotations.

```python
from cryolens.utils import kabsch_alignment, compute_geodesic_distance

# Kabsch alignment
R_optimal, rmsd = kabsch_alignment(points_source, points_target)

# Geodesic distance between rotations (in radians)
distance = compute_geodesic_distance(R1, R2)
```

## Model Classes

### `SegmentedVAE`

The core VAE model with segmented decoder.

```python
from cryolens.models import SegmentedVAE

model = SegmentedVAE(
    latent_dim=40,               # Total latent dimensions
    latent_ratio=0.8,            # Ratio for affinity segment
    n_gaussians=768,             # Number of Gaussian splats
)
```

### Loading Models

Load a trained model checkpoint.

```python
from cryolens.models import load_vae_model

model, config = load_vae_model(
    checkpoint_path="path/to/checkpoint.pt",
    device="cuda",
    load_config=True,
)
```

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint file
- `device` (str): Device for model ("cuda" or "cpu")
- `load_config` (bool, optional): Load configuration from checkpoint. Default: True

**Returns:**
- `model` (SegmentedVAE): Loaded model
- `config` (dict): Model configuration [if load_config=True]

## See Also

- **[Getting Started](getting_started.md)** - Installation and basic usage
- **[Evaluation Guide](evaluation.md)** - Model evaluation
- **[Server Documentation](server.md)** - REST API reference
