# CryoLens API Reference

## Core Modules Overview

CryoLens provides a comprehensive set of tools for cryo-ET particle reconstruction, analysis, and visualization. This document provides the API reference for the core functionality.

## Table of Contents

- [Alignment Module](#alignment-module)
- [Analysis Module](#analysis-module)
- [Reconstruction Module](#reconstruction-module)
- [Visualization Module](#visualization-module)

---

## Alignment Module

### `cryolens.alignment`

Tools for aligning and averaging multiple reconstructions.

#### PCA-based Alignment

```python
from cryolens.alignment import PCAAlignment, align_reconstructions_pca

# Basic usage
aligner = PCAAlignment(
    use_weighted_pca=True,
    spherical_mask_radius=18.0,
    center_volumes=True
)

# Align multiple volumes
aligned_volumes, rotations, info = aligner.align_multiple(
    volumes=[vol1, vol2, vol3],
    weights=None,
    reference_idx=0
)

# Convenience function
result = align_reconstructions_pca(
    reconstructions=[vol1, vol2, vol3],
    spherical_mask_radius=18.0,
    return_average=True
)
aligned = result['aligned']
average = result['average']
```

**Key Functions:**
- `PCAAlignment.align_multiple()`: Align multiple volumes to common reference frame
- `PCAAlignment.compute_average()`: Compute weighted average of aligned volumes
- `align_reconstructions_pca()`: Convenience function for PCA alignment

#### Kabsch Alignment

```python
from cryolens.alignment import KabschAlignment, kabsch_rotation

# Compute optimal rotation between point sets
rotation = kabsch_rotation(points1, points2)

# Align volumes using Kabsch algorithm
aligner = KabschAlignment()
aligned_volume, rotation = aligner.align_volumes(volume1, volume2)
```

---

## Analysis Module

### `cryolens.analysis`

Tools for analyzing reconstruction quality, diversity, and classification performance.

#### Quality Metrics

```python
from cryolens.analysis import calculate_quality_metrics, calculate_progressive_metrics

# Calculate metrics for single reconstruction
metrics = calculate_quality_metrics(
    reconstruction,
    reference=None  # Optional reference for comparison
)
# Returns: mean_intensity, std_intensity, contrast, edge_strength, SNR, etc.

# Analyze progressive reconstruction quality
progressive_data = calculate_progressive_metrics(
    reconstructions=[recon1, recon2, recon3],
    particle_counts=[10, 20, 50],
    reference=ground_truth  # Optional
)
```

**Available Metrics:**
- Basic statistics: mean, std, min, max, dynamic range
- Contrast metrics: Weber, Michelson, RMS contrast
- Sharpness metrics: edge strength, Laplacian variance, Tenengrad
- Noise metrics: SNR, PSNR, entropy
- Sparsity metrics: sparsity ratio, L0 norm, Gini coefficient
- Spectrum metrics: power spectrum, frequency ratios

#### Diversity Analysis

```python
from cryolens.analysis import DiversityAnalyzer

analyzer = DiversityAnalyzer(
    compute_soap=True,
    compute_embeddings=True
)

# Analyze structural diversity
diversity_metrics = analyzer.analyze_diversity(
    structures=protein_structures,
    embeddings=latent_embeddings
)
# Returns: pairwise distances, clustering metrics, UMAP projections
```

#### Classification Performance

```python
from cryolens.analysis import ClassificationEvaluator

evaluator = ClassificationEvaluator()

# Evaluate classification performance
results = evaluator.evaluate(
    embeddings=latent_embeddings,
    labels=structure_labels,
    test_size=0.2
)
# Returns: accuracy, F1 score, confusion matrix, per-class metrics
```

---

## Reconstruction Module

### `cryolens.reconstruction`

Tools for cumulative and progressive reconstruction from particles.

#### Cumulative Reconstruction

```python
from cryolens.reconstruction import accumulate_reconstructions

# Basic cumulative reconstruction
reconstruction = accumulate_reconstructions(
    model=cryolens_model,
    particles=particle_batch,
    use_identity_pose=True,
    method='mean'  # or 'weighted', 'median'
)
```

#### Progressive Reconstruction

```python
from cryolens.reconstruction import progressive_reconstruction

# Generate reconstructions with increasing particle counts
reconstructions = progressive_reconstruction(
    model=cryolens_model,
    particles=all_particles,
    particle_counts=[1, 5, 10, 20, 50],
    return_metrics=False
)

# With full metrics
results = progressive_reconstruction(
    model=cryolens_model,
    particles=all_particles,
    particle_counts=[1, 5, 10, 20, 50],
    return_metrics=True
)
# Access: results['reconstructions'], results['embeddings'], results['info']
```

#### Advanced Reconstruction

```python
from cryolens.reconstruction import CumulativeReconstructor

reconstructor = CumulativeReconstructor(
    use_identity_pose=True,
    normalize_before_accumulation=False,
    accumulation_method='mean'
)

# Custom accumulation
reconstruction, info = reconstructor.accumulate_splats(
    model=cryolens_model,
    particles=particles,
    device=torch.device('cuda')
)
```

---

## Visualization Module

### `cryolens.visualization`

Tools for visualizing reconstructions and analysis results.

#### Orthogonal Views

```python
from cryolens.visualization import create_orthoviews, plot_orthoviews

# Single volume orthogonal views
fig = create_orthoviews(
    volume=reconstruction,
    title="Ribosome Reconstruction",
    cmap='gray',
    show_colorbar=True
)

# Multiple volumes comparison
fig = plot_orthoviews(
    volumes=[recon1, recon2],
    titles=['10 particles', '50 particles'],
    comparison=True  # Side-by-side with difference
)
```

#### Maximum Intensity Projections

```python
from cryolens.visualization import create_mip, create_progressive_mip

# Basic MIP visualization
fig = create_mip(
    volume=reconstruction,
    title="Maximum Intensity Projection",
    cmap='hot'
)

# Progressive MIP showing improvement
fig = create_progressive_mip(
    volumes=progressive_recons,
    particle_counts=[1, 5, 10, 20, 50],
    axis=0  # 0=XY, 1=XZ, 2=YZ
)
```

#### Projection Comparisons

```python
from cryolens.visualization import create_projection_comparison

# Compare different projection types
fig = create_projection_comparison(
    volume=reconstruction,
    projection_types=['max', 'mean', 'std'],
    axis=0
)
```

#### Advanced Visualization

```python
from cryolens.visualization import ProjectionVisualizer

visualizer = ProjectionVisualizer(
    cmap='hot',
    projection_type='max',
    show_colorbar=True
)

# Colored MIP with depth encoding
fig = visualizer.create_colored_mip(
    volume=reconstruction,
    axis=0,
    colormap='viridis',
    alpha_threshold=0.1
)

# Slab projections
fig = visualizer.create_projections(
    volume=reconstruction,
    projection_type='mean',
    slab_thickness=10  # 10-voxel thick slab
)
```

---

## Complete Workflow Example

```python
import numpy as np
from cryolens.reconstruction import progressive_reconstruction
from cryolens.alignment import align_reconstructions_pca
from cryolens.analysis import calculate_progressive_metrics
from cryolens.visualization import create_progressive_mip

# 1. Load particles
particles = load_particles_from_copick(copick_config)

# 2. Generate progressive reconstructions
reconstructions = progressive_reconstruction(
    model=cryolens_model,
    particles=particles,
    particle_counts=[1, 5, 10, 20, 50]
)

# 3. Align reconstructions
aligned_result = align_reconstructions_pca(
    reconstructions=reconstructions,
    spherical_mask_radius=18.0,
    return_average=True
)

# 4. Calculate quality metrics
metrics = calculate_progressive_metrics(
    reconstructions=aligned_result['aligned'],
    particle_counts=[1, 5, 10, 20, 50]
)

# 5. Visualize results
fig = create_progressive_mip(
    volumes=aligned_result['aligned'],
    particle_counts=[1, 5, 10, 20, 50]
)

# 6. Access results
print(f"SNR improvement: {metrics['trends']['snr']['slope']:.2f} per particle")
print(f"Final average reconstruction shape: {aligned_result['average'].shape}")
```

---

## Module Dependencies

### Required Dependencies
- numpy >= 1.20
- torch >= 1.10
- scipy >= 1.7
- matplotlib >= 3.3
- scikit-learn >= 0.24 (for analysis.classification)
- umap-learn >= 0.5 (for analysis.diversity)

### Optional Dependencies
- dscribe (for SOAP descriptors in diversity analysis)
- copick (for data loading)

---

## Notes on Usage

### Memory Management
- For large datasets, use batch processing with appropriate batch sizes
- Progressive reconstruction automatically handles memory by processing subsets
- Alignment operations can be memory-intensive for large volumes

### GPU Acceleration
- Reconstruction module supports GPU acceleration via PyTorch
- Set `device=torch.device('cuda')` for GPU processing
- Batch size affects GPU memory usage

### Normalization
- Input particles are automatically normalized in reconstruction
- Alignment assumes volumes are already normalized
- Metrics calculation does not modify input volumes

### Error Handling
- All modules include logging via Python's logging module
- Set logging level to DEBUG for detailed information
- Functions validate inputs and provide informative error messages
