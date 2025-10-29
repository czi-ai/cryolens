# Evaluation Guide

Comprehensive guide for evaluating CryoLens model performance.

## Overview

CryoLens provides metrics for evaluating:
- **Embedding quality** - Clustering and separation of structural features
- **Reconstruction quality** - Density map fidelity and resolution
- **Classification performance** - Downstream task performance
- **Pose estimation** - Orientation prediction accuracy

## Classification Evaluation

Evaluate classification performance with proper statistical validation using cross-validation.

### Configuration

Create a configuration file (e.g., `classification_config.yaml`):

```yaml
embeddings:
  cryolens: "embeddings/cryolens_embeddings.npy"
  tomotwin: "embeddings/tomotwin_embeddings.npy"

labels: "labels/particle_labels.npy"

output_dir: "results/classification"

cross_validation:
  n_folds: 10
  random_state: 42

fusion:
  enabled: true
  attention_based: true
```

### Running Evaluation

```bash
python -m cryolens.scripts.evaluate_classification \
    --config classification_config.yaml \
    --output-dir ./results/classification/ \
    --n-folds 10
```

### Output

This generates:
- `classification_results.json` - All metrics with error bars and statistical significance
- `classification_performance.png` - Overall and per-class performance visualization

### Programmatic Usage

```python
from cryolens.evaluation import evaluate_classification

results = evaluate_classification(
    embeddings=cryolens_embeddings,
    labels=particle_labels,
    n_folds=10,
    random_state=42
)

print(f"Mean Average Precision: {results['map']:.3f} ± {results['map_std']:.3f}")
```

## In-Distribution (ID) Reconstruction Evaluation

Evaluate reconstruction performance on held-out validation data.

### Data Format

Validation data should be in parquet format with columns:
- `subvolume` or `volume`: Particle volumes
- `pdb_code` or `structure`: Structure identifiers
- `orientation_quaternion` (optional): Ground truth orientations

```
validation/
├── validation_0001_snr5.0.parquet
├── validation_0002_snr5.0.parquet
└── ...
```

### Running Evaluation

```bash
python -m cryolens.scripts.evaluate_id_reconstruction \
    --checkpoint v001 \
    --validation-dir data/validation/parquet/ \
    --structures-dir structures/mrcs/ \
    --output-dir results/id_validation/ \
    --structures 6qzp 7vd8 \
    --n-particles 100 \
    --particle-counts 5 10 15 20 25 50 75 100
```

### Parameters

- `--checkpoint`: Model checkpoint (name or path)
- `--validation-dir`: Directory containing validation parquet files
- `--structures-dir`: Directory containing ground truth MRC files
- `--structures`: Space-separated list of structure codes to evaluate
- `--n-particles`: Total particles to use per structure
- `--particle-counts`: Particle counts for averaging analysis

### Output

For each structure:
- `{structure}_results.h5` - Complete results in HDF5 format
- `{structure}_results.png` - Visualization with metrics
- `evaluation_summary.json` - Summary metrics

### Programmatic Usage

```python
from cryolens.evaluation import evaluate_id_reconstruction
from cryolens.inference import create_inference_pipeline

pipeline = create_inference_pipeline("v001")

results = evaluate_id_reconstruction(
    pipeline=pipeline,
    validation_dir="data/validation/parquet/",
    structures=["6qzp", "7vd8"],
    structures_dir="structures/mrcs/",
    n_particles=100,
    particle_counts=[5, 10, 25, 50, 100],
)

for structure, metrics in results.items():
    print(f"{structure}: {metrics['resolution_50']:.1f}Å @ n=50")
```

## Out-of-Distribution (OOD) Reconstruction Evaluation

Evaluate zero-shot reconstruction on experimental data not seen during training.

### Setup

Requires:
- Copick configuration file for the dataset
- Ground truth structures in MRC format

### Running Evaluation

```bash
python -m cryolens.scripts.evaluate_ood_reconstruction \
    --checkpoint v001 \
    --copick-config ml_challenge_experimental.json \
    --structures-dir structures/mrcs/ \
    --output-dir results/ood/ \
    --structures ribosome thyroglobulin \
    --n-particles 25
```

### Parameters

- `--checkpoint`: Model checkpoint (name or path)
- `--copick-config`: Copick configuration file path
- `--structures-dir`: Directory with ground truth MRC files
- `--structures`: Structure names to evaluate
- `--n-particles`: Number of particles to use

### Metrics Computed

- **FSC Resolution**: Resolution at 0.5 FSC threshold
- **Correlation**: Cross-correlation with ground truth
- **Uncertainty**: Variance across resampled reconstructions

### Output

For each structure:
- `{structure}_results.h5` - All data for reproducibility
- `{structure}_results.png` - Visualization of results
- `evaluation_summary.json` - Summary statistics

### Programmatic Usage

```python
from cryolens.evaluation import evaluate_ood_structure
from cryolens.inference import create_inference_pipeline
from cryolens.data import CopickDataLoader

# Setup
pipeline = create_inference_pipeline("v001")
copick_loader = CopickDataLoader("copick_config.json")

# Evaluate
result = evaluate_ood_structure(
    structure_name="ribosome",
    pipeline=pipeline,
    copick_loader=copick_loader,
    ground_truth_path="structures/ribosome.mrc",
    output_dir="results/ribosome",
    n_particles=25,
    n_resamples=10
)

# Access metrics
print(f"Resolution: {result['resolution']:.1f}Å")
print(f"Correlation: {result['correlation']:.3f}")
```

## Embedding Space Metrics

Evaluate the quality of learned embeddings.

### Class Separation

Measure how well embeddings separate different structural classes:

```python
from cryolens.evaluation import compute_class_separation_metrics

metrics = compute_class_separation_metrics(embeddings, labels)

print(f"Separation ratio: {metrics['separation_ratio']:.2f}")
print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
print(f"Davies-Bouldin index: {metrics['davies_bouldin']:.2f}")
```

### Embedding Diversity

Measure the diversity of the embedding space:

```python
from cryolens.evaluation import compute_embedding_diversity

diversity = compute_embedding_diversity(embeddings, method='variance')
print(f"Embedding diversity: {diversity:.2f}")
```

### Mahalanobis Overlap

Analyze overlap between classes using Mahalanobis distance:

```python
from cryolens.evaluation import compute_mahalanobis_overlap

overlap_matrix, class_names = compute_mahalanobis_overlap(embeddings, labels)
```

## Reconstruction Quality Metrics

Evaluate 3D reconstruction quality without ground truth.

```python
from cryolens.evaluation import compute_reconstruction_metrics

metrics = compute_reconstruction_metrics(reconstruction)

print(f"Contrast: {metrics['contrast']:.2f}")
print(f"Edge strength: {metrics['edge_strength']:.3f}")
print(f"SNR estimate: {metrics['snr_estimate']:.1f} dB")
```

### Fourier Shell Correlation (FSC)

Compute FSC between reconstructions:

```python
from cryolens.evaluation import compute_fsc_with_threshold

resolutions, fsc_values, resolution_at_half = compute_fsc_with_threshold(
    volume1, volume2,
    voxel_size=10.0,
    threshold=0.5,
    mask_radius=20.0
)

print(f"Resolution at FSC=0.5: {resolution_at_half:.1f}Å")
```

## Pose Analysis

Evaluate pose estimation accuracy.

### Geodesic Distance

Measure angular distance between predicted and ground truth orientations:

```python
from cryolens.utils import compute_geodesic_distance, compute_rotation_metrics

# Single pair
distance = compute_geodesic_distance(R_pred, R_true)
print(f"Angular error: {np.degrees(distance):.1f}°")

# Multiple pairs
metrics = compute_rotation_metrics(predicted_poses, ground_truth_poses)
print(f"Mean error: {metrics['mean_geodesic_error']:.2f}°")
print(f"Median error: {metrics['median_geodesic_error']:.2f}°")
```

### Alignment

Find global transformation between predicted and ground truth pose sets:

```python
from cryolens.utils import align_rotation_sets

aligned_poses, R_global, metrics = align_rotation_sets(
    recovered_poses, ground_truth_poses
)

print(f"Alignment error: {metrics['mean_angular_error']:.2f}°")
```

## Integrated Evaluation

Comprehensive evaluation across all metrics:

```python
from cryolens.evaluation import evaluate_model_performance

results = evaluate_model_performance(
    embeddings=latent_vectors,
    labels=class_labels,
    reconstructions=reconstructed_volumes,
    ground_truth=reference_volumes,
    predicted_poses=predicted_rotations,
    true_poses=ground_truth_rotations
)

# Access results by category
print("Embedding Metrics:")
print(f"  Separation: {results['embedding_metrics']['separation_ratio']:.2f}")

print("Reconstruction Metrics:")
print(f"  Mean correlation: {results['reconstruction_summary']['mean_correlation']:.3f}")

print("Pose Metrics:")
print(f"  Mean error: {results['pose_metrics']['mean_geodesic_error']:.1f}°")
```

## See Also

- **[API Reference](api_reference.md)** - Detailed API documentation
- **[Getting Started](getting_started.md)** - Installation and basic usage
- **[Examples](../examples/)** - Example evaluation scripts
