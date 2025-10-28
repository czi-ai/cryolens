# PR Summary: Add option to save final fused embeddings from attention fusion

## Summary

This PR adds the ability to save fused embeddings from attention fusion classification experiments. After cross-validation completes, the system can now train a final fusion model on the full dataset and save the resulting fused embeddings for reuse.

## Motivation

When running classification evaluation with attention fusion, the current implementation trains a separate fusion model for each CV fold (to prevent leakage), but doesn't provide a way to save fused embeddings for downstream use. This PR addresses that by:

1. Training a final fusion model on the **full aligned dataset** after CV completes
2. Saving the fused embeddings in HDF5 format compatible with CryoLens embeddings
3. Making these embeddings reusable without needing to retrain the fusion model

## Changes

### New Function: `train_final_fusion_and_save()`
Located in `src/cryolens/evaluation/attention_fusion_cv.py`

- Trains attention fusion model on entire dataset
- Generates 32D fused embeddings for all samples
- Saves to HDF5 with structure:
  ```
  /embeddings/{sample_id}/mu  # Fused embedding
  /metadata/                   # Training metadata
  ```
- Returns both embeddings array and metadata dict

### CLI Enhancement
In `src/cryolens/scripts/evaluate_classification.py`

- New argument: `--save-fused-embeddings PATH`
- Only works with `--fusion-method attention`
- Runs after CV evaluation completes
- Uses same hyperparameters as CV (epochs, seed, device)

## Usage

```bash
python -m cryolens.scripts.evaluate_classification \
    --config /mnt/czi-sci-ai/imaging-models/kyle/cryolens_paper/classification/cryolens/classification.yaml \
    --output-dir /mnt/czi-sci-ai/imaging-models/cryolens/mlflow/outputs/alternating_curriculum/cryolens-sim-015/classification_with_stats/ \
    --n-folds 10 \
    --fusion-method attention \
    --attention-epochs 20 \
    --save-fused-embeddings /mnt/czi-sci-ai/imaging-models/cryolens/mlflow/outputs/alternating_curriculum/cryolens-sim-015/classification_with_stats/fused_embeddings.h5
```

## Output Format

The saved HDF5 file is compatible with existing CryoLens embedding loaders:

```python
import h5py
with h5py.File('fused_embeddings.h5', 'r') as f:
    # Load embeddings
    sample_emb = f['embeddings']['sample_0']['mu'][:]
    
    # Load metadata
    n_samples = f['metadata'].attrs['n_samples']
    class_names = f['metadata']['class_names'][:]
```

## Benefits

1. **Reusability**: Fused embeddings can be used for multiple downstream tasks without retraining
2. **Consistency**: Single model trained on full data (standard ML practice)
3. **Compatibility**: HDF5 format matches existing CryoLens embeddings structure
4. **Metadata**: Preserves training parameters and class information

## Branch Information

- Branch: `feature/save-fused-embeddings-final`
- Base: `main`
- Commit: f578c477caf31af58232d09dfb417474eaf97d91

## Files Changed

- `src/cryolens/evaluation/attention_fusion_cv.py`: Added `train_final_fusion_and_save()` function
- `src/cryolens/scripts/evaluate_classification.py`: Added CLI argument and post-CV training logic

## Notes

- The final model is trained on **all data** after CV, not from any specific fold
- This follows standard practice: CV for evaluation, final model on full data for deployment
- Only available for attention fusion (other methods don't require training)
- Embeddings are saved with sample IDs as `sample_0`, `sample_1`, etc. (can be customized if needed)
