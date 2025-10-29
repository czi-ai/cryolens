# Getting Started with CryoLens

This guide will help you get started with CryoLens for cryo-ET particle analysis.

## Installation

### Requirements

- Python 3.11 or 3.12
- CUDA-capable GPU (recommended for inference, though CPU is supported)
- At least 8GB RAM

### Install from PyPI

```bash
pip install cryolens
```

### Install from Source

For development or to access the latest features:

```bash
git clone https://github.com/czi-ai/cryolens.git
cd cryolens
pip install -e .
```

### Optional Dependencies

For Copick integration (loading particles from cryoET datasets):

```bash
pip install cryolens[copick]
```

For development tools:

```bash
pip install cryolens[dev]
```

## First Steps

### Loading a Pre-trained Model

CryoLens comes with pre-trained model checkpoints that can be loaded by name:

```python
from cryolens.inference import create_inference_pipeline

# Load the v001 checkpoint (automatically downloads if needed)
pipeline = create_inference_pipeline("v001")
```

You can also load a local checkpoint:

```python
pipeline = create_inference_pipeline("/path/to/checkpoint.pt")
```

### Processing Your First Particle

CryoLens expects input particles as 48×48×48 numpy arrays with voxel spacing of 10 Å.

```python
import numpy as np
from cryolens.inference import create_inference_pipeline

# Load model
pipeline = create_inference_pipeline("v001")

# Load your particle (48×48×48 array)
particle = np.load("my_particle.npy")

# Process the particle
result = pipeline.process_volume(particle)

# Access the outputs
reconstruction = result['reconstruction']  # 3D density map
embedding = result['embedding']           # 32D structural features
pose = result['pose']                     # Orientation (quaternion)
global_weight = result['global_weight']   # Amplitude multiplier
```

### Batch Processing

For processing multiple particles efficiently:

```python
# particles_array shape: (N, 48, 48, 48)
particles_array = np.load("particles.npy")

# Process all particles
results = pipeline.batch_process(particles_array, batch_size=32)

# Access results for each particle
embeddings = results['embeddings']        # (N, 32)
reconstructions = results['reconstructions']  # (N, 48, 48, 48)
poses = results['poses']                  # (N, 4)
```

## Loading Data from Copick

If you have data in Copick format (including datasets from the CZ cryoET Data Portal):

```python
from cryolens.data import CopickDataLoader

# Initialize with a Copick configuration
loader = CopickDataLoader("copick_config.json")

# List available structures
structures = loader.list_available_structures()
print(f"Available: {structures}")

# Load particles for specific structures
data = loader.load_particles(
    structure_filter=["ribosome", "proteasome"],
    max_particles_per_structure=100,
    target_voxel_spacing=10.0,  # Angstroms
    box_size=48
)

# Process loaded particles
for structure_name, structure_data in data.items():
    particles = structure_data['particles']  # (N, 48, 48, 48)
    results = pipeline.batch_process(particles)
    print(f"{structure_name}: processed {len(particles)} particles")
```

## Using the Napari Plugin

For interactive particle picking with real-time feedback:

1. Install napari-cryolens:
   ```bash
   pip install napari-cryolens
   ```

2. Launch napari:
   ```bash
   napari
   ```

3. Open your tomogram and start picking particles - CryoLens will provide immediate structural feedback

See the [napari-cryolens repository](https://github.com/kephale/napari-cryolens) for detailed usage instructions.

## Next Steps

- **[API Reference](api_reference.md)** - Detailed API documentation
- **[Evaluation Guide](evaluation.md)** - Evaluate model performance
- **[Server Documentation](server.md)** - Deploy CryoLens as a REST API
- **[Examples](../examples/)** - Example scripts and notebooks

## Common Issues

### CUDA Out of Memory

If you encounter CUDA out of memory errors, reduce the batch size:

```python
results = pipeline.batch_process(particles, batch_size=8)
```

Or use CPU inference:

```python
pipeline = create_inference_pipeline("v001", device="cpu")
```

### Missing Dependencies

If you get import errors, ensure all dependencies are installed:

```bash
pip install -e ".[copick,dev]"
```

## Getting Help

- **Issues**: Report bugs at [GitHub Issues](https://github.com/czi-ai/cryolens/issues)
- **Discussions**: Ask questions at [GitHub Discussions](https://github.com/czi-ai/cryolens/discussions)
