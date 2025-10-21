## CryoLens

CryoLens is a generative model and toolkit for 3D reconstruction of molecular structures from cryoET tomograms.

## Features

- **Generative modeling**: VAE-based architecture for learning molecular structure representations
- **3D reconstruction**: Reconstruct molecular structures from cryoET particle data
- **Evaluation metrics**: Comprehensive metrics for embeddings, reconstructions, and poses
- **Classification evaluation**: Statistical validation for classification tasks with cross-validation
- **ID and OOD reconstruction evaluation**: Zero-shot evaluation on validation and experimental data
- **OOD reconstruction evaluation**: Zero-shot evaluation on out-of-distribution experimental data
- **Particle picking quality assessment**: Detect contaminated particle sets through reconstruction distance analysis
- **Pose analysis**: Utilities for 3D rotation analysis and alignment
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

## Evaluation Metrics

CryoLens provides comprehensive metrics for evaluating model performance across embeddings, reconstructions, and poses.

### Embedding Space Metrics

Evaluate the quality of learned embeddings and clustering:

```python
from cryolens.evaluation import (
    compute_class_separation_metrics,
    compute_embedding_diversity,
    compute_mahalanobis_overlap
)

# Analyze class separation in embedding space
metrics = compute_class_separation_metrics(embeddings, labels)
print(f"Separation ratio: {metrics['separation_ratio']:.2f}")
print(f"Silhouette score: {metrics['silhouette_score']:.3f}")

# Compute embedding diversity
diversity = compute_embedding_diversity(embeddings, method='variance')
print(f"Embedding diversity: {diversity:.2f}")

# Analyze class overlap with Mahalanobis distance
overlap_matrix, classes = compute_mahalanobis_overlap(embeddings, labels)
```

### Classification Evaluation

Evaluate classification performance with proper statistical validation:

```bash
# Create a configuration file
cp examples/classification_config.yaml my_classification_config.yaml
# Edit my_classification_config.yaml with your embedding paths

# Run evaluation with 10-fold cross-validation
python -m cryolens.scripts.evaluate_classification \
    --config my_classification_config.yaml \
    --output-dir ./results/classification/ \
    --n-folds 10
```

This will generate:
- `classification_results.json` - All metrics with error bars and statistical significance
- `classification_performance.png` - Comprehensive figure with overall and per-class results

See `examples/classification_config.yaml` for configuration options.

### In-Distribution (ID) Reconstruction Evaluation

Evaluate reconstruction performance on validation data stored in parquet files:

```bash
# Run ID evaluation on validation data
python -m cryolens.scripts.evaluate_id_reconstruction \
    --checkpoint models/cryolens_epoch_2600.pt \
    --validation-dir data/validation/parquet/ \
    --structures-dir structures/mrcs/ \
    --output-dir results/id_validation/ \
    --structures 6qzp 7vd8 \
    --n-particles 100 \
    --particle-counts 5 10 15 20 25 50 75 100
```

This evaluates the model on validation data with the same methodology as OOD evaluation:
- **Two-stage alignment**: Aligns reconstructions to first particle, then averages and aligns to ground truth
- **FSC resolution** computation at different particle counts
- **Correlation metrics** with ground truth
- **Uncertainty estimation** through resampling

The validation data should be organized as:
```
validation/
├── validation_0001_snr5.0.parquet
├── validation_0002_snr5.0.parquet
└── ...
```

Each parquet file should contain columns: `subvolume` (or `volume`), `pdb_code` (or `structure`), and optionally `orientation_quaternion`.

### Out-of-Distribution Reconstruction Evaluation

Evaluate zero-shot reconstruction performance on experimental data:

```bash
# Run OOD evaluation on ML Challenge data
python -m cryolens.scripts.evaluate_ood_reconstruction \
    --checkpoint models/cryolens_epoch_2600.pt \
    --copick-config ml_challenge_experimental.json \
    --structures-dir structures/mrcs/ \
    --output-dir results/ood/ \
    --structures ribosome thyroglobulin \
    --n-particles 25
```

This will generate for each structure:
- `{structure}_results.h5` - HDF5 file with all data for reproducibility
- `{structure}_results.png` - Figure showing ground truth, reconstruction, and metrics
- `evaluation_summary.json` - Summary of all metrics

The evaluation computes:
- **FSC resolution** at 0.5 threshold vs particle count
- **Correlation** with ground truth vs particle count
- **Uncertainty estimates** through resampling

See `examples/ood_reconstruction_config.yaml` for configuration options.

**Programmatic usage:**

```python
from cryolens.evaluation import evaluate_ood_structure
from cryolens.inference.pipeline import create_inference_pipeline
from cryolens.data import CopickDataLoader

# Setup
device = torch.device("cuda")
model, config = load_vae_model("checkpoint.pt", device=device, load_config=True)
pipeline = create_inference_pipeline("checkpoint.pt", device=device)
copick_loader = CopickDataLoader("copick_config.json")

# Evaluate single structure
result = evaluate_ood_structure(
    structure_name="ribosome",
    model=model,
    pipeline=pipeline,
    copick_loader=copick_loader,
    ground_truth_path=Path("structures/6qzp.mrc"),
    output_dir=Path("results/ribosome"),
    device=device,
    n_particles=25,
    n_resamples=10
)

# Access metrics
for n, metrics in result['metrics'].items():
    print(f"n={n}: {metrics['resolution']:.1f}Å, r={metrics['correlation']:.3f}")
```

### Particle Picking Quality Assessment

Evaluate CryoLens's ability to detect contaminated particle sets by measuring reconstruction distances:

```bash
# Assess quality for structure pairs
python -m cryolens.scripts.evaluate_picking_quality \
    --checkpoint models/cryolens_epoch_2600.pt \
    --copick-config ml_challenge_experimental.json \
    --output-dir results/picking_quality/ \
    --structure-pairs ribosome,thyroglobulin beta-galactoside,virus-like-particle \
    --n-particles 100
```

This will generate:
- **Per-scenario results**: Distance distributions (MSE and Missing Wedge Loss) for each contamination level
- **Separability analysis**: Cohen's d effect sizes showing class discrimination ability
- **Contamination heatmap**: Visual summary of detection performance across all structure pairs
- **Distance arrays**: Raw distance measurements for further analysis

The evaluation tests contamination ratios:
- Pure cases: 100/0, 0/100
- Light contamination: 99/1, 1/99
- Moderate contamination: 90/10, 10/90
- Heavy mixing: 50/50

Key metrics computed:
- **MSE distance**: Direct L2 difference between reconstructions
- **Missing Wedge Loss distance**: Fourier-space weighted distance accounting for missing wedge
- **Cohen's d**: Effect size measuring separability between contaminating classes
- **Distribution statistics**: Mean, std, and overlap for each class

See `examples/picking_quality_config.yaml` for configuration options.

### Reconstruction Quality Metrics

Evaluate 3D reconstruction quality:

```python
from cryolens.evaluation import (
    compute_reconstruction_metrics,
    compute_fourier_shell_correlation,
    compute_fsc_with_threshold
)

# Compute reconstruction metrics
metrics = compute_reconstruction_metrics(
    reconstruction,
    ground_truth=reference_volume  # optional
)
print(f"Contrast: {metrics['contrast']:.2f}")
print(f"Edge strength: {metrics['edge_strength']:.3f}")
print(f"SNR estimate: {metrics['snr_estimate']:.1f} dB")

# Compute FSC with threshold-based resolution estimation
resolutions, fsc_values, resolution_at_half = compute_fsc_with_threshold(
    volume1, volume2,
    voxel_size=10.0,
    threshold=0.5,
    mask_radius=20.0
)
print(f"Resolution at FSC=0.5: {resolution_at_half:.1f}Å")
```

### Integrated Evaluation

Comprehensive model evaluation across all metrics:

```python
from cryolens.evaluation import evaluate_model_performance

# Evaluate complete model performance
results = evaluate_model_performance(
    embeddings=latent_vectors,
    labels=class_labels,
    reconstructions=reconstructed_volumes,
    ground_truth=reference_volumes,
    predicted_poses=predicted_rotations,
    true_poses=ground_truth_rotations
)

# Access different metric categories
print(f"Embedding separation: {results['embedding_metrics']['separation_ratio']:.2f}")
print(f"Mean reconstruction correlation: {results['reconstruction_summary']['mean_correlation']:.3f}")
print(f"Pose error: {results['pose_metrics']['mean_geodesic_error']:.1f}°")
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
    print(f"{structure_name}: {len(particles)} loaded")
```

### ML Challenge Datasets

CryoLens provides easy access to ML Challenge datasets:

```python
from cryolens.data.copick import load_ml_challenge_configs
import os

# Set the path to ML Challenge configs (or use environment variable)
os.environ["CRYOLENS_ML_CHALLENGE_PATH"] = "/path/to/ml_challenge/configs"

# Get available ML Challenge configurations
configs = load_ml_challenge_configs()
# Or provide path directly:
# configs = load_ml_challenge_configs("/path/to/ml_challenge/configs")

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
