# CryoLens

[![Video from napari that shows a synthetic cryoET tomogram with high signal being populated by CryoLens reconstructions. There are 10 differeny protein structures in the tomogram. Over a hundred reconstructions are added then the word "CryoLens" is shown.](https://github.com/czi-ai/cryolens/assets/docs/img/cryolens_video.gif)](https://github.com/czi-ai/cryolens/assets/docs/img/cryolens_video.mp4)

A pre-trained generative model for particle analysis in cryo-electron tomography.

## Overview

CryoLens uses a variational autoencoder with a segmented Gaussian splat decoder to provide:
- **Zero-shot structural reconstructions** from single cryo-ET particles
- **Learned embeddings** that capture structural features
- **Real-time feedback** during particle picking workflows

The model is trained on 5.8 million synthetic particles spanning 103 protein structures and generalizes to experimental data.

## Quick Start

### Installation

```bash
pip install cryolens
```

Or from source:
```bash
git clone https://github.com/czi-ai/cryolens.git
cd cryolens
pip install -e .
```

### Basic Usage

```python
from cryolens.inference import create_inference_pipeline
import numpy as np

# Load pre-trained model (v001 checkpoint)
pipeline = create_inference_pipeline("v001")

# Process a particle (48x48x48 voxels)
particle = np.load("particle.npy")
result = pipeline.process_volume(particle)

# Access outputs
reconstruction = result['reconstruction']  # 3D density map
embedding = result['embedding']           # Structural features
pose = result['pose']                     # Orientation estimate
```

## Features

### Particle Reconstruction
Generate interpretable 3D density estimates from individual particles for immediate structural feedback.

### Feature Extraction
Extract structural embeddings suitable for:
- Particle classification
- Quality assessment
- Clustering and analysis

### Interactive Tools
- **napari plugin**: Real-time feedback during particle picking ([napari-cryolens](https://github.com/kephale/napari-cryolens))
- **FastAPI server**: REST API for integration into analysis pipelines

## Model Weights

Pre-trained model weights are automatically downloaded when using named versions of cryolens models. Currently this includes:

- `v001`, the first publicly available set of cryolens weights

## Documentation

- **[Getting Started Guide](docs/getting_started.md)** - Installation and basic usage
- **[API Reference](docs/api_reference.md)** - Detailed API documentation
- **[Evaluation Guide](docs/evaluation.md)** - Model evaluation and metrics
- **[Server Documentation](docs/server.md)** - REST API deployment
- **[Examples](examples/)** - Jupyter notebooks and scripts

## Citation

CryoLens is developed by Kyle Harrington, Utz Ermel, Ritvik Vasan, Ashley Anderson, Dan Lu, Ryan Lim, Mikala Caton, and Alan R. Lowe at the Chan Zuckerberg Initiative.

Publication details to be announced.

## Advanced Usage

### Copick Integration

Load particles directly from Copick projects:

```python
from cryolens.data import CopickDataLoader

loader = CopickDataLoader("copick_config.json")
particles = loader.load_particles(
    structure_filter=["ribosome"],
    max_particles_per_structure=100
)
```

### Batch Processing

```python
# Process multiple particles
results = pipeline.batch_process(particles_array)
```

### Server Deployment

```bash
python -m cryolens.server.minimal_cryolens_server --checkpoint v001 --port 8023
```

See [Server Documentation](docs/server.md) for API endpoints and deployment options.

## Development

### Running Tests

```bash
pytest tests/
```

## License

See [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/czi-ai/cryolens/issues)
- **Discussions**: [GitHub Discussions](https://github.com/czi-ai/cryolens/discussions)
- **Security**: security@chanzuckerberg.com

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.
