# Server Documentation

CryoLens includes a FastAPI server for deploying the model as a REST API.

## Quick Start

Start the server with a pre-trained model:

```bash
python -m cryolens.server.minimal_cryolens_server --checkpoint v001 --port 8023
```

The server will be available at `http://localhost:8023`.

## Command-Line Options

### Required Arguments

- `--checkpoint, -c`: Path to VAE checkpoint file or checkpoint name (e.g., "v001")

### Optional Arguments

- `--host`: Host to bind the server to (default: `0.0.0.0`)
- `--port, -p`: Port to bind the server to (default: `8023`)
- `--cors`: CORS origins, comma-separated or `*` for all (default: `*`)
- `--device`: Device for inference: `cuda`, `cpu`, or `auto` (default: `auto`)
- `--reload`: Enable auto-reload for development
- `--log-level`: Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

## Usage Examples

### Basic Usage

Start server with default settings:

```bash
python -m cryolens.server.minimal_cryolens_server --checkpoint v001
```

### Custom Configuration

Start on specific host and port:

```bash
python -m cryolens.server.minimal_cryolens_server \
    --checkpoint v001 \
    --host localhost \
    --port 8080
```

### CPU-Only Inference

Force CPU usage (useful for systems without GPU):

```bash
python -m cryolens.server.minimal_cryolens_server \
    --checkpoint v001 \
    --device cpu
```

### Development Mode

Enable auto-reload and debug logging:

```bash
python -m cryolens.server.minimal_cryolens_server \
    --checkpoint v001 \
    --reload \
    --log-level DEBUG
```

### CORS Configuration

Restrict CORS to specific origins:

```bash
python -m cryolens.server.minimal_cryolens_server \
    --checkpoint v001 \
    --cors "https://example.com,https://app.example.com"
```

## API Endpoints

### Server Information

#### `GET /`

Returns server metadata and available endpoints.

**Response:**
```json
{
  "service": "CryoLens VAE Server",
  "version": "0.1.0",
  "status": "running",
  "endpoints": [...]
}
```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

#### `GET /model_info`

Returns detailed model configuration.

**Response:**
```json
{
  "latent_dim": 40,
  "latent_ratio": 0.8,
  "n_gaussians": 768,
  "input_shape": [48, 48, 48],
  "device": "cuda"
}
```

### Volume Processing

#### `POST /reconstruct`

Reconstruct a volume using the VAE decoder.

**Request Body:**
```json
{
  "input_volume": [...],  // 48×48×48 array as nested list
  "use_segmented_decoder": true,
  "use_final_convolution": true,
  "return_splat_data": false,
  "return_embeddings": false
}
```

**Response:**
```json
{
  "reconstruction": [...],  // 48×48×48 array
  "input_shape": [48, 48, 48],
  "reconstruction_shape": [48, 48, 48]
}
```

#### `POST /extract_features`

Extract embeddings, pose, and global weight from a volume.

**Request Body:**
```json
{
  "input_volume": [...]  // 48×48×48 array
}
```

**Response:**
```json
{
  "embedding": [...],      // 40D latent vector
  "pose": [...],           // 4D quaternion (w, x, y, z)
  "global_weight": 1.23,   // Amplitude multiplier
  "input_shape": [48, 48, 48]
}
```

#### `POST /extract_gaussian_splats`

Extract Gaussian splat parameters from a volume.

**Request Body:**
```json
{
  "input_volume": [...],  // 48×48×48 array
  "include_free_segment": true
}
```

**Response:**
```json
{
  "centers": [...],        // (N, 3) array of positions
  "weights": [...],        // (N,) array of weights
  "scales": [...],         // (N,) array of scales
  "n_affinity": 614,       // Number of affinity gaussians
  "n_free": 154,           // Number of free gaussians
  "n_total": 768,
  "input_shape": [48, 48, 48]
}
```

#### `POST /batch_process`

Process multiple volumes in batch.

**Request Body:**
```json
{
  "input_volumes": [...],  // (N, 48, 48, 48) array
  "batch_size": 32,
  "return_reconstructions": true,
  "return_embeddings": true,
  "return_poses": true,
  "return_global_weights": true
}
```

**Response:**
```json
{
  "reconstructions": [...],   // (N, 48, 48, 48)
  "embeddings": [...],        // (N, 40)
  "poses": [...],             // (N, 4)
  "global_weights": [...],    // (N,)
  "n_volumes": 100,
  "input_shape": [48, 48, 48]
}
```

### Statistics

#### `GET /statistics?volume=[...]`

Get statistics for a volume (query parameter).

**Query Parameters:**
- `volume`: URL-encoded JSON array of the volume

**Response:**
```json
{
  "mean": 0.123,
  "std": 0.456,
  "min": -1.234,
  "max": 2.345,
  "shape": [48, 48, 48]
}
```

## Python Client Example

```python
import requests
import numpy as np

# Server URL
base_url = "http://localhost:8023"

# Health check
response = requests.get(f"{base_url}/health")
print(response.json())

# Load and process a volume
volume = np.random.randn(48, 48, 48)

# Reconstruct
response = requests.post(
    f"{base_url}/reconstruct",
    json={
        "input_volume": volume.tolist(),
        "use_segmented_decoder": True,
        "use_final_convolution": True,
        "return_embeddings": True
    }
)
result = response.json()
reconstruction = np.array(result['reconstruction'])

# Extract features
response = requests.post(
    f"{base_url}/extract_features",
    json={"input_volume": volume.tolist()}
)
features = response.json()
embedding = np.array(features['embedding'])
pose = np.array(features['pose'])

# Batch processing
volumes = np.random.randn(10, 48, 48, 48)
response = requests.post(
    f"{base_url}/batch_process",
    json={
        "input_volumes": volumes.tolist(),
        "batch_size": 5
    }
)
batch_results = response.json()
```

## Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8023/docs`
- **ReDoc**: `http://localhost:8023/redoc`

These provide interactive API documentation where you can test endpoints directly from your browser.

## Deployment Considerations

### Production Deployment

For production use, consider:

1. **Use a production ASGI server** (e.g., uvicorn with multiple workers):
   ```bash
   uvicorn cryolens.server.minimal_cryolens_server:app \
       --host 0.0.0.0 \
       --port 8023 \
       --workers 4
   ```

2. **Set up a reverse proxy** (e.g., nginx) for load balancing and SSL

3. **Limit CORS origins** to trusted domains

4. **Monitor GPU memory usage** if using CUDA

### Docker Deployment

Example Dockerfile:

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY . /app
RUN pip install -e .

EXPOSE 8023

CMD ["python", "-m", "cryolens.server.minimal_cryolens_server", \
     "--checkpoint", "v001", "--port", "8023"]
```

Build and run:
```bash
docker build -t cryolens-server .
docker run -p 8023:8023 --gpus all cryolens-server
```

## Performance Tuning

### Batch Size

Adjust batch size based on available GPU memory:

```python
# Client-side batch processing
volumes = np.random.randn(100, 48, 48, 48)

# Process in smaller batches for limited GPU memory
response = requests.post(
    f"{base_url}/batch_process",
    json={
        "input_volumes": volumes.tolist(),
        "batch_size": 8  # Reduce if OOM errors occur
    }
)
```

### CPU vs GPU

The server automatically selects CUDA if available. For CPU-only:

```bash
python -m cryolens.server.minimal_cryolens_server \
    --checkpoint v001 \
    --device cpu
```

## Troubleshooting

### Port Already in Use

If port 8023 is busy, use a different port:

```bash
python -m cryolens.server.minimal_cryolens_server \
    --checkpoint v001 \
    --port 8024
```

### CUDA Out of Memory

Reduce batch size or switch to CPU:

```bash
python -m cryolens.server.minimal_cryolens_server \
    --checkpoint v001 \
    --device cpu
```

### Checkpoint Not Found

Ensure checkpoint exists or use a named checkpoint:

```bash
# Use named checkpoint (downloads automatically)
python -m cryolens.server.minimal_cryolens_server --checkpoint v001

# Use local checkpoint
python -m cryolens.server.minimal_cryolens_server --checkpoint /path/to/model.pt
```

## See Also

- **[API Reference](api_reference.md)** - Python API documentation
- **[Getting Started](getting_started.md)** - Installation and setup
- **[Examples](../examples/)** - Example client scripts
