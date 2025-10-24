# Alignment Method Implementation - Testing Guide

## Summary of Changes

This implementation adds flexible alignment method support to CryoLens reconstruction evaluation scripts with minimal code changes while maintaining backward compatibility.

### Files Modified

1. **src/cryolens/splats/alignment_methods.py** (moved from evaluation/)
   - Unified alignment interface with 4 methods: cross_correlation, fourier, gradient_descent, ransac_icp
   - Helper function `align_volumes()` for consistent API

2. **src/cryolens/evaluation/ood_reconstruction.py**
   - Added `average_splat_params()` helper function
   - Updated `evaluate_ood_structure()` to accept `alignment_method` and `alignment_kwargs`
   - Added splat extraction when using ransac_icp method
   - STAGE 1 alignment now uses `align_volumes()` unified interface

3. **src/cryolens/scripts/evaluate_ood_reconstruction.py**
   - Added CLI arguments for alignment method selection
   - Added method-specific parameters (angular-step, weight-percentile, etc.)
   - Passes alignment configuration to evaluate_ood_structure()

4. **src/cryolens/scripts/evaluate_id_reconstruction.py**
   - Same CLI argument additions as OOD script
   - Works seamlessly via ParquetDataLoader adapter

## Testing Commands

### Test 1: Backward Compatibility (Default cross_correlation)

Test that the default behavior works exactly as before:

```bash
python -m cryolens.scripts.evaluate_ood_reconstruction \
    --checkpoint v001 \
    --copick-config /mnt/czi-sci-ai/imaging-models/data/cryolens/mlc/copick_czcdp/ml_challenge_experimental_publictest.json \
    --structures-dir /mnt/czi-sci-ai/imaging-models/data/cryolens/mlc/structures/mrcs \
    --output-dir /tmp/test_default_alignment/ \
    --structures ribosome \
    --n-particles 5 \
    --n-resamples 3
```

**Expected:** Should run without errors, using cross_correlation method (default)

### Test 2: Fourier Alignment Method

Test the Fourier-based phase correlation method:

```bash
python -m cryolens.scripts.evaluate_ood_reconstruction \
    --checkpoint v001 \
    --copick-config /mnt/czi-sci-ai/imaging-models/data/cryolens/mlc/copick_czcdp/ml_challenge_experimental_publictest.json \
    --structures-dir /mnt/czi-sci-ai/imaging-models/data/cryolens/mlc/structures/mrcs \
    --output-dir /tmp/test_fourier_alignment/ \
    --structures thyroglobulin \
    --n-particles 5 \
    --n-resamples 3 \
    --alignment-method fourier \
    --angular-step 15.0
```

**Expected:** Should use Fourier method, faster but potentially less accurate

### Test 3: Gradient Descent Alignment

Test gradient descent refinement:

```bash
python -m cryolens.scripts.evaluate_ood_reconstruction \
    --checkpoint v001 \
    --copick-config /mnt/czi-sci-ai/imaging-models/data/cryolens/mlc/copick_czcdp/ml_challenge_experimental_publictest.json \
    --structures-dir /mnt/czi-sci-ai/imaging-models/data/cryolens/mlc/structures/mrcs \
    --output-dir /tmp/test_gradient_alignment/ \
    --structures ribosome \
    --n-particles 5 \
    --n-resamples 3 \
    --alignment-method gradient_descent
```

**Expected:** Should use gradient descent, slower but potentially more accurate

### Test 4: RANSAC-ICP with Splats

Test the splat-based RANSAC-ICP alignment (MOST IMPORTANT TEST):

```bash
python -m cryolens.scripts.evaluate_ood_reconstruction \
    --checkpoint v001 \
    --copick-config /mnt/czi-sci-ai/imaging-models/data/cryolens/mlc/copick_czcdp/ml_challenge_experimental_publictest.json \
    --structures-dir /mnt/czi-sci-ai/imaging-models/data/cryolens/mlc/structures/mrcs \
    --output-dir /tmp/test_ransac_icp_alignment/ \
    --structures ribosome thyroglobulin \
    --n-particles 5 \
    --n-resamples 3 \
    --alignment-method ransac_icp \
    --weight-percentile 48.3 \
    --sphere-radius 15.2 \
    --ransac-iterations 252 \
    --icp-iterations 17
```

**Expected:** 
- Should extract splat parameters during reconstruction
- Should use RANSAC-ICP for STAGE 1 alignment
- Should print "Alignment method: ransac_icp" in output

### Test 5: ID Validation with RANSAC-ICP

Test the ID reconstruction script with splat-based alignment:

```bash
python -m cryolens.scripts.evaluate_id_reconstruction \
    --checkpoint /mnt/czi-sci-ai/imaging-models/cryolens/mlflow/outputs/alternating_curriculum/cryolens-sim-015/checkpoints/model_epoch_2600_train_loss_6.436.pt \
    --validation-dir /mnt/czi-sci-ai/imaging-models/data/cryolens/tomotwin/subvolume_zarr_v03/parquet_by_structSNR_05/validation \
    --structures-dir /mnt/czi-sci-ai/imaging-models/data/cryolens/tomotwin/structures/mrcs \
    --output-dir /tmp/test_id_ransac_icp/ \
    --structures 1g3i 1n9g \
    --n-particles 5 \
    --n-resamples 3 \
    --alignment-method ransac_icp \
    --weight-percentile 48.3 \
    --sphere-radius 15.2
```

**Expected:** Works seamlessly via ParquetDataLoader adapter

### Test 6: Custom Angular Step

Test custom angular step parameter:

```bash
python -m cryolens.scripts.evaluate_ood_reconstruction \
    --checkpoint v001 \
    --copick-config /mnt/czi-sci-ai/imaging-models/data/cryolens/mlc/copick_czcdp/ml_challenge_experimental_publictest.json \
    --structures-dir /mnt/czi-sci-ai/imaging-models/data/cryolens/mlc/structures/mrcs \
    --output-dir /tmp/test_custom_angular/ \
    --structures ribosome \
    --n-particles 5 \
    --n-resamples 3 \
    --alignment-method cross_correlation \
    --angular-step 15.0
```

**Expected:** Should use finer angular step (15° instead of default 30°)

## Verification Checklist

After running tests, verify:

- [ ] Default behavior (no --alignment-method) works exactly as before
- [ ] Each alignment method runs without errors
- [ ] RANSAC-ICP extracts and uses splat parameters
- [ ] Non-splat methods do NOT extract splats (check via debug prints if needed)
- [ ] Output H5 files include alignment_method attribute
- [ ] Results are scientifically reasonable (FSC curves, correlations)
- [ ] Performance: RANSAC-ICP should be comparable to cross_correlation
- [ ] Memory usage: No excessive memory consumption from splat extraction

## Expected Output

Each test should produce:
1. Console output showing alignment method being used
2. H5 file with results in output directory
3. PNG figure with 3 panels (GT, reconstruction, metrics plot)
4. evaluation_summary.json with configuration and results

## Debug Tips

If tests fail:

1. **Import errors**: Check that alignment_methods.py is in src/cryolens/splats/
2. **Splat extraction fails**: Verify decoder has 'affinity_segment_size' attribute
3. **RANSAC-ICP errors**: Check splat_params dict has 'coordinates' key (not 'centroids')
4. **Performance issues**: Reduce --n-particles and --n-resamples for faster testing

## Key Implementation Details

1. **Backward Compatibility**: Default `--alignment-method cross_correlation` maintains exact current behavior
2. **Splat Extraction**: Only happens when `method == 'ransac_icp'`
3. **Two-Stage Alignment**: 
   - STAGE 1: Align reconstructions to first particle (uses selected method)
   - STAGE 2: Align average to GT (always uses cross_correlation)
4. **Splat Averaging**: Splat parameters averaged across resamples before alignment
5. **Key Name Translation**: 'centroids' → 'coordinates' in average_splat_params()

## Performance Expectations

- **cross_correlation**: Baseline performance (current method)
- **fourier**: ~2x faster, similar or slightly worse accuracy
- **gradient_descent**: ~2x slower, similar or slightly better accuracy  
- **ransac_icp**: Similar speed to cross_correlation, potentially better for certain structures

## Next Steps

After successful testing:

1. Compare alignment methods on full dataset
2. Analyze which structures benefit from different methods
3. Consider adding method selection to evaluate_picking_quality if needed
4. Document best practices for method selection in user guide
