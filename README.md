## CryoLens

CryoLens is a generative model and toolkit for 3D reconstruction of molecular structures from cryoET tomograms.

## Features

- **Generative modeling**: VAE-based architecture for learning molecular structure representations
- **3D reconstruction**: Reconstruct molecular structures from cryoET particle data
- **Pose analysis**: Comprehensive utilities for 3D rotation analysis and alignment
- **Inference server**: FastAPI-based server for model inference
- **Gaussian splats**: Extract Gaussian splat representations for visualization

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

## Pose Analysis Utilities

CryoLens includes comprehensive utilities for analyzing and aligning 3D rotations and poses.

### Kabsch Alignment

Align point clouds and find optimal rotations:

```python
from cryolens.utils import kabsch_alignment

# Align two point clouds
R_optimal, rmsd = kabsch_alignment(points_source, points_target)
print(f"RMSD after alignment: {rmsd:.3f}")
```

### Rotation Metrics

Compute distances and metrics between rotations:

```python
from cryolens.utils import compute_geodesic_distance, compute_rotation_metrics

# Geodesic distance between rotations
dist = compute_geodesic_distance(R1, R2)
print(f"Rotation angle: {np.degrees(dist):.1f} degrees")

# Comprehensive metrics for pose predictions
metrics = compute_rotation_metrics(predicted_poses, ground_truth_poses)
print(f"Mean angular error: {metrics['mean_geodesic_error']:.2f} degrees")
```

### Rotation Set Alignment

Align sets of rotations to find global transformations:

```python
from cryolens.utils import align_rotation_sets

# Find global rotation aligning recovered poses to ground truth
aligned_poses, R_global, metrics = align_rotation_sets(
    recovered_poses, ground_truth_poses
)
print(f"Alignment error: {metrics['mean_angular_error']:.2f} degrees")
```

### Format Conversions

Convert between rotation representations:

```python
from cryolens.utils import (
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
)

# Convert rotation matrix to Euler angles
euler_angles = rotation_matrix_to_euler(R, convention="XYZ", degrees=True)

# Convert to quaternion (w, x, y, z format)
quaternion = rotation_matrix_to_quaternion(R, scalar_first=True)

# Convert to axis-angle
axis, angle = rotation_matrix_to_axis_angle(R)
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
