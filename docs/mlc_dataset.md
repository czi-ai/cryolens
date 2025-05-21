# CryoLens MLC Dataset Visualization

This document provides an overview of the CryoLens MLC dataset through various visualizations. These visualizations help in understanding the dataset's composition, structure distributions, and quality variations across different SNR levels.

## Dataset Structure Distribution

The distribution of structures in the dataset shows the relative frequency of each molecule type, including background samples:

![Structure Distribution](figures/dataset_viz_structure_distribution.png)

## Sample Structures

Sample views of various structures from the dataset with projections along the XY, XZ, and YZ axes:

![Sample Structures](figures/dataset_viz_sample_structures.png)

## Density Distribution

The distribution of density values across structures in the dataset:

![Density Distribution](figures/dataset_viz_density_distribution.png)

## SNR Comparison

Comparison of the same structure across different Signal-to-Noise Ratio (SNR) levels:

![SNR Comparison](figures/dataset_viz_snr_comparison.png)

## Structure Comparison at Different SNR Levels

### High SNR

Comparison of different structures at high SNR levels:

![High SNR Comparison](figures/dataset_viz_structure_high_snr.png)

### Low SNR

Comparison of the same structures at low SNR levels:

![Low SNR Comparison](figures/dataset_viz_structure_low_snr.png)

## Source Type Distribution

Distribution of source types in the dataset:

![Source Type Distribution](figures/dataset_viz_source_type_distribution.png)

## Curriculum Weights

Distribution of weights in the curriculum specification (if available):

![Curriculum Weights](figures/dataset_viz_curriculum_weights.png)

## Individual Molecule Projections

The following sections display three examples of each molecule type with sum projections along each axis.

### Apo-Ferritin

![Apo-Ferritin Sample 1](molecule_projections/apo-ferritin_sample_1.png)
![Apo-Ferritin Sample 2](molecule_projections/apo-ferritin_sample_2.png)
![Apo-Ferritin Sample 3](molecule_projections/apo-ferritin_sample_3.png)

### Beta-Amylase

![Beta-Amylase Sample 1](molecule_projections/beta-amylase_sample_1.png)
![Beta-Amylase Sample 2](molecule_projections/beta-amylase_sample_2.png)
![Beta-Amylase Sample 3](molecule_projections/beta-amylase_sample_3.png)

### Beta-Galactoside

![Beta-Galactoside Sample 1](molecule_projections/beta-galactoside_sample_1.png)
![Beta-Galactoside Sample 2](molecule_projections/beta-galactoside_sample_2.png)
![Beta-Galactoside Sample 3](molecule_projections/beta-galactoside_sample_3.png)

### Ribosome

![Ribosome Sample 1](molecule_projections/ribosome_sample_1.png)
![Ribosome Sample 2](molecule_projections/ribosome_sample_2.png)
![Ribosome Sample 3](molecule_projections/ribosome_sample_3.png)

### Thyroglobulin

![Thyroglobulin Sample 1](molecule_projections/thyroglobulin_sample_1.png)
![Thyroglobulin Sample 2](molecule_projections/thyroglobulin_sample_2.png)
![Thyroglobulin Sample 3](molecule_projections/thyroglobulin_sample_3.png)

### Virus-Like-Particle

![Virus-Like-Particle Sample 1](molecule_projections/virus-like-particle_sample_1.png)
![Virus-Like-Particle Sample 2](molecule_projections/virus-like-particle_sample_2.png)
![Virus-Like-Particle Sample 3](molecule_projections/virus-like-particle_sample_3.png)

### Background

![Background Sample 1](molecule_projections/background_sample_1.png)
![Background Sample 2](molecule_projections/background_sample_2.png)
![Background Sample 3](molecule_projections/background_sample_3.png)

## Regenerate images

```
uv run src/cryolens/docs/viz_dataset.py --parquet_path ./dataset_20250322_051824.parquet --copick_config ./ml_challenge.json --curriculum_path ./curriculum.json --output_dir ./docs --normalization z-score
```