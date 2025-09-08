"""
Tests for evaluation metrics module.
"""

import numpy as np
import pytest
import warnings

from cryolens.evaluation.metrics import (
    compute_davies_bouldin_index,
    compute_silhouette_score,
    compute_class_separation_metrics,
    compute_mahalanobis_overlap,
    compute_embedding_diversity,
    compute_reconstruction_metrics,
    compute_ssim,
    compute_fourier_shell_correlation,
    evaluate_model_performance,
    SKLEARN_AVAILABLE,
)


class TestEmbeddingMetrics:
    """Test embedding space metrics."""
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_davies_bouldin_index(self):
        """Test Davies-Bouldin index computation."""
        # Create well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(30, 10) + np.array([5] * 10)
        cluster2 = np.random.randn(30, 10) + np.array([-5] * 10)
        embeddings = np.vstack([cluster1, cluster2])
        labels = np.array([0] * 30 + [1] * 30)
        
        dbi = compute_davies_bouldin_index(embeddings, labels)
        
        # Well-separated clusters should have low DBI
        assert dbi < 1.0
        assert dbi > 0.0
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_davies_bouldin_single_cluster(self):
        """Test DBI with single cluster."""
        embeddings = np.random.randn(50, 10)
        labels = np.zeros(50)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dbi = compute_davies_bouldin_index(embeddings, labels)
        
        assert dbi == 0.0
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_silhouette_score(self):
        """Test silhouette score computation."""
        # Create well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(25, 10) * 0.5 + np.array([10] * 10)
        cluster2 = np.random.randn(25, 10) * 0.5 + np.array([-10] * 10)
        embeddings = np.vstack([cluster1, cluster2])
        labels = np.array([0] * 25 + [1] * 25)
        
        score = compute_silhouette_score(embeddings, labels)
        
        # Well-separated clusters should have high silhouette score
        assert score > 0.5
        assert score <= 1.0
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_silhouette_score_overlapping(self):
        """Test silhouette score with overlapping clusters."""
        # Create overlapping clusters
        np.random.seed(42)
        embeddings = np.random.randn(100, 10)
        labels = np.random.randint(0, 3, 100)
        
        score = compute_silhouette_score(embeddings, labels)
        
        # Random overlapping clusters should have low score
        assert -1.0 <= score <= 1.0
        assert abs(score) < 0.2  # Should be close to 0
    
    def test_class_separation_metrics(self):
        """Test comprehensive class separation metrics."""
        # Create 3 well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(20, 5) * 0.5 + np.array([10, 0, 0, 0, 0])
        cluster2 = np.random.randn(20, 5) * 0.5 + np.array([0, 10, 0, 0, 0])
        cluster3 = np.random.randn(20, 5) * 0.5 + np.array([0, 0, 10, 0, 0])
        
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        
        metrics = compute_class_separation_metrics(embeddings, labels)
        
        # Check all expected keys
        assert 'mean_within_class_distance' in metrics
        assert 'mean_between_class_distance' in metrics
        assert 'separation_ratio' in metrics
        
        # Well-separated clusters should have high separation ratio
        assert metrics['separation_ratio'] > 5.0
        assert metrics['mean_within_class_distance'] < metrics['mean_between_class_distance']
    
    def test_mahalanobis_overlap(self):
        """Test Mahalanobis overlap computation."""
        # Create non-overlapping clusters
        np.random.seed(42)
        cluster1 = np.random.randn(50, 10) * 0.5 + np.array([20] * 10)
        cluster2 = np.random.randn(50, 10) * 0.5 + np.array([-20] * 10)
        
        embeddings = np.vstack([cluster1, cluster2])
        labels = np.array([0] * 50 + [1] * 50)
        
        overlap_matrix, unique_labels = compute_mahalanobis_overlap(embeddings, labels)
        
        assert overlap_matrix.shape == (2, 2)
        assert len(unique_labels) == 2
        
        # Diagonal should be 1 (perfect self-overlap)
        assert np.allclose(np.diag(overlap_matrix), 1.0)
        
        # Off-diagonal should be small (low overlap)
        assert overlap_matrix[0, 1] < 0.01
        assert overlap_matrix[1, 0] < 0.01
    
    def test_embedding_diversity_variance(self):
        """Test embedding diversity with variance method."""
        # High diversity embeddings
        np.random.seed(42)
        diverse_embeddings = np.random.randn(100, 20) * 5
        
        # Low diversity embeddings
        uniform_embeddings = np.ones((100, 20)) + np.random.randn(100, 20) * 0.1
        
        div_high = compute_embedding_diversity(diverse_embeddings, 'variance')
        div_low = compute_embedding_diversity(uniform_embeddings, 'variance')
        
        assert div_high > div_low
        assert div_high > 10.0  # High variance
        assert div_low < 1.0    # Low variance
    
    def test_embedding_diversity_pairwise(self):
        """Test embedding diversity with pairwise distance method."""
        # Spread out embeddings
        np.random.seed(42)
        spread_embeddings = np.random.randn(50, 10) * 10
        
        # Clustered embeddings
        clustered_embeddings = np.random.randn(50, 10) * 0.1
        
        div_spread = compute_embedding_diversity(spread_embeddings, 'pairwise')
        div_clustered = compute_embedding_diversity(clustered_embeddings, 'pairwise')
        
        assert div_spread > div_clustered
    
    def test_embedding_diversity_determinant(self):
        """Test embedding diversity with determinant method."""
        # Full rank embeddings
        np.random.seed(42)
        full_rank = np.random.randn(50, 10)
        
        # Low rank embeddings (correlated dimensions)
        base = np.random.randn(50, 2)
        low_rank = np.hstack([base, base * 2, base * 3, base * 4, base * 5])
        
        div_full = compute_embedding_diversity(full_rank, 'determinant')
        div_low = compute_embedding_diversity(low_rank, 'determinant')
        
        # Full rank should have higher determinant
        assert div_full > div_low


class TestReconstructionMetrics:
    """Test reconstruction quality metrics."""
    
    def test_reconstruction_metrics_basic(self):
        """Test basic reconstruction metrics."""
        # Create a synthetic reconstruction
        np.random.seed(42)
        reconstruction = np.random.randn(48, 48, 48)
        
        metrics = compute_reconstruction_metrics(reconstruction)
        
        # Check all expected keys
        expected_keys = [
            'mean_intensity', 'std_intensity', 'contrast',
            'edge_strength', 'dynamic_range', 'sparsity', 'snr_estimate'
        ]
        for key in expected_keys:
            assert key in metrics
        
        # Check reasonable values
        assert metrics['std_intensity'] > 0
        assert metrics['dynamic_range'] > 0
        assert 0 <= metrics['sparsity'] <= 1
    
    def test_reconstruction_metrics_with_ground_truth(self):
        """Test reconstruction metrics with ground truth."""
        np.random.seed(42)
        ground_truth = np.random.randn(32, 32, 32)
        noise = np.random.randn(32, 32, 32) * 0.1
        reconstruction = ground_truth + noise
        
        metrics = compute_reconstruction_metrics(reconstruction, ground_truth)
        
        # Check comparison metrics
        assert 'mse' in metrics
        assert 'psnr' in metrics
        assert 'correlation' in metrics
        assert 'ssim' in metrics
        
        # Should have high correlation and low MSE
        assert metrics['correlation'] > 0.9
        assert metrics['mse'] < 0.02
        assert metrics['psnr'] > 20
    
    def test_reconstruction_metrics_with_mask(self):
        """Test reconstruction metrics with mask."""
        np.random.seed(42)
        reconstruction = np.random.randn(32, 32, 32)
        
        # Create a spherical mask
        center = 16
        radius = 10
        x, y, z = np.ogrid[:32, :32, :32]
        mask = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2
        
        metrics = compute_reconstruction_metrics(reconstruction, mask=mask)
        
        # Metrics should be computed only within mask
        assert 'mean_intensity' in metrics
        assert 'sparsity' in metrics
    
    def test_reconstruction_metrics_2d(self):
        """Test reconstruction metrics for 2D images."""
        np.random.seed(42)
        reconstruction = np.random.randn(64, 64)
        
        metrics = compute_reconstruction_metrics(reconstruction)
        
        # Should work for 2D as well
        assert 'edge_strength' in metrics
        assert metrics['edge_strength'] > 0
    
    def test_ssim_identical(self):
        """Test SSIM for identical images."""
        img = np.random.randn(32, 32)
        ssim = compute_ssim(img, img)
        
        # Identical images should have SSIM = 1
        assert np.isclose(ssim, 1.0, atol=0.01)
    
    def test_ssim_different(self):
        """Test SSIM for different images."""
        np.random.seed(42)
        img1 = np.random.randn(32, 32)
        img2 = np.random.randn(32, 32)
        
        ssim = compute_ssim(img1, img2)
        
        # Random images should have low SSIM
        assert 0 <= ssim <= 1
        assert ssim < 0.3
    
    def test_fourier_shell_correlation(self):
        """Test FSC computation."""
        np.random.seed(42)
        # Create two similar volumes
        volume1 = np.random.randn(32, 32, 32)
        volume2 = volume1 + np.random.randn(32, 32, 32) * 0.5
        
        fsc_result = compute_fourier_shell_correlation(volume1, volume2)
        
        assert 'resolution' in fsc_result
        assert 'fsc_curve' in fsc_result
        assert 'frequencies' in fsc_result
        
        # FSC curve should start high and decrease
        fsc_curve = fsc_result['fsc_curve']
        assert fsc_curve[0] > 0.5
        assert fsc_curve[0] > fsc_curve[-1]
    
    def test_fsc_identical_volumes(self):
        """Test FSC for identical volumes."""
        volume = np.random.randn(32, 32, 32)
        
        fsc_result = compute_fourier_shell_correlation(volume, volume)
        
        # Identical volumes should have FSC = 1 everywhere
        fsc_curve = fsc_result['fsc_curve']
        assert np.all(fsc_curve > 0.99)


class TestIntegratedEvaluation:
    """Test integrated evaluation pipeline."""
    
    def test_evaluate_model_performance_embeddings(self):
        """Test model evaluation with embeddings."""
        np.random.seed(42)
        # Create clustered embeddings
        embeddings = np.vstack([
            np.random.randn(20, 10) + [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            np.random.randn(20, 10) + [-5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        labels = np.array([0] * 20 + [1] * 20)
        
        results = evaluate_model_performance(
            embeddings=embeddings,
            labels=labels,
            verbose=False
        )
        
        assert 'embedding_metrics' in results
        assert 'embedding_diversity' in results
        assert results['embedding_metrics']['separation_ratio'] > 1.0
    
    def test_evaluate_model_performance_reconstructions(self):
        """Test model evaluation with reconstructions."""
        np.random.seed(42)
        reconstructions = np.random.randn(5, 32, 32, 32)
        ground_truth = reconstructions + np.random.randn(5, 32, 32, 32) * 0.1
        
        results = evaluate_model_performance(
            reconstructions=reconstructions,
            ground_truth=ground_truth,
            verbose=False
        )
        
        assert 'reconstruction_metrics' in results
        assert 'reconstruction_summary' in results
        assert len(results['reconstruction_metrics']) == 5
        assert 'mean_correlation' in results['reconstruction_summary']
    
    def test_evaluate_model_performance_complete(self):
        """Test complete model evaluation."""
        np.random.seed(42)
        
        # Create test data
        n_samples = 10
        embeddings = np.random.randn(n_samples, 20)
        labels = np.random.randint(0, 3, n_samples)
        reconstructions = np.random.randn(n_samples, 16, 16, 16)
        ground_truth = reconstructions + np.random.randn(n_samples, 16, 16, 16) * 0.1
        
        # Note: Pose metrics would require pose utilities module
        
        results = evaluate_model_performance(
            embeddings=embeddings,
            labels=labels,
            reconstructions=reconstructions,
            ground_truth=ground_truth,
            verbose=False
        )
        
        # Check that all components are evaluated
        assert 'embedding_metrics' in results
        assert 'reconstruction_metrics' in results
        assert 'reconstruction_summary' in results
    
    def test_shape_mismatches(self):
        """Test error handling for shape mismatches."""
        embeddings = np.random.randn(10, 20)
        labels = np.random.randint(0, 2, 15)  # Wrong size
        
        with pytest.raises(ValueError):
            compute_class_separation_metrics(embeddings, labels)
        
        recon = np.random.randn(32, 32, 32)
        gt = np.random.randn(32, 32, 16)  # Wrong shape
        
        with pytest.raises(ValueError):
            compute_reconstruction_metrics(recon, gt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
