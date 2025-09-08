## CryoLens

CryoLens is a generative model and toolkit for 3D reconstruction of molecular structures from cryoET tomograms.

## Installation

CryoLens requires Python 3.11 or 3.12. We recommend using `uv` for package management:

```bash
# Install uv if you haven't already
pip install uv

# Clone the repository
git clone https://github.com/czi-ai/cryolens.git
cd cryolens

# Switch to the refactor-core-utilities branch
git checkout refactor-core-utilities

# Install dependencies with uv
uv sync
```

Alternatively, you can use pip:

```bash
pip install -e .
```

### Optional: Copick Integration

To use the Copick integration for loading particles from cryoET datasets:

```bash
# Install with Copick support
pip install -e ".[copick]"
```

## Copick Integration

CryoLens includes support for loading particles directly from Copick projects, including datasets from the CZ cryoET Data Portal.

### Basic Usage

```python
from cryolens.data.copick import CopickDataLoader

# Load particles from a Copick project
loader = CopickDataLoader("path/to/copick_config.json")

# List available structures
structures = loader.list_available_structures()
print(f"Available structures: {structures}")

# Load particles for specific structures
results = loader.load_particles(
    structure_filter=["ribosome", "proteasome"],
    max_particles_per_structure=100,
    target_voxel_spacing=10.0,  # in Angstroms
    box_size=48
)

# Access loaded data
for structure_name, data in results.items():
    particles = data['particles']  # (N, 48, 48, 48) array
    orientations = data['orientations']  # (N, 3, 3) rotation matrices
    positions = data['positions']  # (N, 3) positions in Angstroms
    print(f"{structure_name}: {len(particles)} particles loaded")
```

### ML Challenge Datasets

CryoLens provides easy access to ML Challenge datasets:

```python
from cryolens.data.copick import load_ml_challenge_configs

# Get available ML Challenge configurations
configs = load_ml_challenge_configs()

# Load experimental test data
if "experimental_public_test" in configs:
    loader = CopickDataLoader(configs["experimental_public_test"])
    results = loader.load_particles(
        structure_filter=["ribosome"],
        max_particles_per_structure=50
    )
```

## Running the CryoLens Server

CryoLens includes a FastAPI server that provides REST API endpoints for VAE functionality including volume reconstruction, feature extraction, and Gaussian splat extraction.

### Basic Usage

To start the server with a trained model checkpoint:

```bash
uv run src/cryolens/server/minimal_cryolens_server.py --checkpoint ./my_training_run/last.pt --port 8023
```

### Command-Line Options

The server supports the following options:

- `--checkpoint, -c` (required): Path to VAE checkpoint file
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port, -p`: Port to bind the server to (default: 8023)
- `--cors`: CORS origins, comma-separated or * for all (default: *)
- `--device`: Device to use for inference: cuda, cpu, or auto (default: auto)
- `--reload`: Enable auto-reload for development
- `--log-level`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

### Examples

Start server on default port with GPU acceleration:
```bash
uv run src/cryolens/server/minimal_cryolens_server.py --checkpoint ./checkpoints/model.pt
```

Start server on custom port with specific host:
```bash
uv run src/cryolens/server/minimal_cryolens_server.py \
    --checkpoint ./checkpoints/model.pt \
    --host localhost \
    --port 8080
```

Start server with CPU-only inference:
```bash
uv run src/cryolens/server/minimal_cryolens_server.py \
    --checkpoint ./checkpoints/model.pt \
    --device cpu \
    --port 8023
```

Development mode with auto-reload:
```bash
uv run src/cryolens/server/minimal_cryolens_server.py \
    --checkpoint ./checkpoints/model.pt \
    --reload \
    --log-level DEBUG
```

### API Endpoints

Once the server is running, you can access the following endpoints:

- `GET /` - Server metadata and available endpoints
- `GET /health` - Health check endpoint
- `GET /model_info` - Detailed model configuration
- `POST /reconstruct` - Reconstruct a volume using the VAE
- `POST /extract_features` - Extract embeddings, pose, and global weight
- `POST /extract_gaussian_splats` - Extract Gaussian splat parameters
- `POST /batch_process` - Process multiple volumes in batch
- `GET /statistics` - Get statistics for a volume

Visit `http://localhost:8023/` (or your configured host:port) to see the full API documentation and available endpoints.

### Testing the Server

You can test the server using curl:

```bash
# Check server health
curl http://localhost:8023/health

# Get model information
curl http://localhost:8023/model_info
```

Or use Python with the requests library:

```python
import requests
import numpy as np

# Create a test volume
test_volume = np.random.randn(48, 48, 48).tolist()

# Send reconstruction request
response = requests.post(
    "http://localhost:8023/reconstruct",
    json={
        "input_volume": test_volume,
        "use_segmented_decoder": True,
        "use_final_convolution": True,
        "return_splat_data": False,
        "return_embeddings": False
    }
)

result = response.json()
print(f"Input shape: {result['input_shape']}")
print(f"Reconstruction shape: {len(result['reconstruction'])}")
```

## License

See the LICENSE file for details.

## Contributing

This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.

## Reporting Security Issues

Please note: If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.
