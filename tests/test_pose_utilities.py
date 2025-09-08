"""
Tests for pose analysis utilities.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from cryolens.utils.pose import (
    kabsch_alignment,
    compute_geodesic_distance,
    align_rotation_sets,
    quaternion_distance,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
    random_rotation_matrix,
    compute_rotation_metrics,
)


class TestKabschAlignment:
    """Test the Kabsch alignment function."""
    
    def test_perfect_alignment(self):
        """Test alignment of identical point clouds."""
        P = np.random.randn(100, 3)
        Q = P.copy()
        
        R_opt, rmsd = kabsch_alignment(P, Q)
        
        assert np.allclose(R_opt, np.eye(3))
        assert rmsd < 1e-10
    
    def test_known_rotation(self):
        """Test alignment with known rotation."""
        P = np.random.randn(100, 3)
        R_true = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        Q = P @ R_true
        
        R_opt, rmsd = kabsch_alignment(P, Q, center=False)
        
        assert np.allclose(R_opt, R_true, atol=1e-10)
        assert rmsd < 1e-10
    
    def test_with_translation(self):
        """Test alignment with translation."""
        P = np.random.randn(50, 3)
        R_true = R.random().as_matrix()
        t = np.array([1.0, 2.0, 3.0])
        Q = P @ R_true + t
        
        R_opt, rmsd, centroid_P, centroid_Q = kabsch_alignment(
            P, Q, center=True, return_transform=True
        )
        
        assert np.allclose(R_opt, R_true, atol=1e-10)
        assert np.allclose(centroid_Q - centroid_P @ R_true, t, atol=1e-10)
    
    def test_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        P = np.random.randn(10, 3)
        Q = np.random.randn(15, 3)
        
        with pytest.raises(ValueError):
            kabsch_alignment(P, Q)
    
    def test_2d_points(self):
        """Test that 2D points raise error."""
        P = np.random.randn(10, 2)
        Q = np.random.randn(10, 2)
        
        with pytest.raises(ValueError):
            kabsch_alignment(P, Q)


class TestGeodesicDistance:
    """Test geodesic distance computation."""
    
    def test_identity(self):
        """Test distance between identical rotations."""
        R1 = np.eye(3)
        R2 = np.eye(3)
        
        dist = compute_geodesic_distance(R1, R2)
        assert abs(dist) < 1e-10
    
    def test_90_degree_rotation(self):
        """Test distance for 90-degree rotation."""
        R1 = np.eye(3)
        R2 = R.from_euler('z', 90, degrees=True).as_matrix()
        
        dist = compute_geodesic_distance(R1, R2)
        assert np.isclose(dist, np.pi/2, atol=1e-10)
    
    def test_180_degree_rotation(self):
        """Test distance for 180-degree rotation."""
        R1 = np.eye(3)
        R2 = R.from_euler('z', 180, degrees=True).as_matrix()
        
        dist = compute_geodesic_distance(R1, R2)
        assert np.isclose(dist, np.pi, atol=1e-10)
    
    def test_symmetry(self):
        """Test that distance is symmetric."""
        R1 = R.random().as_matrix()
        R2 = R.random().as_matrix()
        
        dist12 = compute_geodesic_distance(R1, R2)
        dist21 = compute_geodesic_distance(R2, R1)
        
        assert np.isclose(dist12, dist21)
    
    def test_triangle_inequality(self):
        """Test triangle inequality."""
        R1 = R.random().as_matrix()
        R2 = R.random().as_matrix()
        R3 = R.random().as_matrix()
        
        dist12 = compute_geodesic_distance(R1, R2)
        dist23 = compute_geodesic_distance(R2, R3)
        dist13 = compute_geodesic_distance(R1, R3)
        
        # Triangle inequality: d(1,3) <= d(1,2) + d(2,3)
        assert dist13 <= dist12 + dist23 + 1e-10


class TestAlignRotationSets:
    """Test rotation set alignment."""
    
    def test_perfect_recovery(self):
        """Test perfect recovery of global rotation."""
        n_poses = 20
        ground_truth = np.array([R.random().as_matrix() for _ in range(n_poses)])
        R_global = R.random().as_matrix()
        recovered = np.array([R_global @ gt for gt in ground_truth])
        
        aligned, R_est, metrics = align_rotation_sets(recovered, ground_truth)
        
        # Check that estimated rotation is inverse of applied rotation
        assert np.allclose(R_est @ R_global, np.eye(3), atol=1e-6)
        
        # Check that aligned poses match ground truth
        for i in range(n_poses):
            assert np.allclose(aligned[i], ground_truth[i], atol=1e-6)
        
        # Check metrics
        assert metrics['mean_angular_error'] < 0.01  # Less than 0.01 degrees
    
    def test_with_noise(self):
        """Test alignment with noisy rotations."""
        n_poses = 30
        ground_truth = np.array([R.random().as_matrix() for _ in range(n_poses)])
        R_global = R.random().as_matrix()
        
        # Add noise to rotations
        recovered = []
        for gt in ground_truth:
            noisy = R_global @ gt
            # Add small perturbation
            perturbation = R.from_rotvec(np.random.randn(3) * 0.05)
            noisy = perturbation.as_matrix() @ noisy
            recovered.append(noisy)
        recovered = np.array(recovered)
        
        aligned, R_est, metrics = align_rotation_sets(recovered, ground_truth)
        
        # Should still achieve reasonable alignment
        assert metrics['mean_angular_error'] < 5.0  # Less than 5 degrees
    
    def test_procrustes_method(self):
        """Test Procrustes alignment method."""
        n_poses = 15
        ground_truth = np.array([R.random().as_matrix() for _ in range(n_poses)])
        R_global = R.random().as_matrix()
        recovered = np.array([R_global @ gt for gt in ground_truth])
        
        aligned, R_est, metrics = align_rotation_sets(
            recovered, ground_truth, method="procrustes"
        )
        
        # Check alignment quality
        assert metrics['mean_angular_error'] < 0.01


class TestQuaternionDistance:
    """Test quaternion distance computation."""
    
    def test_identical_quaternions(self):
        """Test distance between identical quaternions."""
        q = np.array([1, 0, 0, 0])
        dist = quaternion_distance(q, q)
        assert abs(dist) < 1e-10
    
    def test_opposite_quaternions(self):
        """Test that q and -q have zero distance."""
        q1 = np.array([0.5, 0.5, 0.5, 0.5])
        q2 = -q1
        dist = quaternion_distance(q1, q2)
        assert abs(dist) < 1e-10
    
    def test_orthogonal_quaternions(self):
        """Test distance between orthogonal quaternions."""
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        dist = quaternion_distance(q1, q2)
        assert np.isclose(dist, 1.0)


class TestRotationConversions:
    """Test rotation format conversions."""
    
    def test_euler_round_trip(self):
        """Test Euler angle conversion round trip."""
        R_orig = R.random().as_matrix()
        
        euler = rotation_matrix_to_euler(R_orig, convention="XYZ", degrees=True)
        R_recovered = euler_to_rotation_matrix(euler, convention="XYZ", degrees=True)
        
        assert np.allclose(R_orig, R_recovered)
    
    def test_quaternion_round_trip(self):
        """Test quaternion conversion round trip."""
        R_orig = R.random().as_matrix()
        
        # Test scalar-first format
        q = rotation_matrix_to_quaternion(R_orig, scalar_first=True)
        R_recovered = quaternion_to_rotation_matrix(q, scalar_first=True)
        assert np.allclose(R_orig, R_recovered)
        
        # Test scalar-last format
        q = rotation_matrix_to_quaternion(R_orig, scalar_first=False)
        R_recovered = quaternion_to_rotation_matrix(q, scalar_first=False)
        assert np.allclose(R_orig, R_recovered)
    
    def test_axis_angle_round_trip(self):
        """Test axis-angle conversion round trip."""
        R_orig = R.random().as_matrix()
        
        axis, angle = rotation_matrix_to_axis_angle(R_orig)
        R_recovered = axis_angle_to_rotation_matrix(axis, angle)
        
        assert np.allclose(R_orig, R_recovered)
    
    def test_batch_conversions(self):
        """Test batch conversions."""
        n = 10
        R_batch = np.array([R.random().as_matrix() for _ in range(n)])
        
        # Test Euler batch conversion
        euler_batch = rotation_matrix_to_euler(R_batch)
        assert euler_batch.shape == (n, 3)
        R_recovered = euler_to_rotation_matrix(euler_batch)
        assert np.allclose(R_batch, R_recovered)
        
        # Test quaternion batch conversion
        q_batch = rotation_matrix_to_quaternion(R_batch)
        assert q_batch.shape == (n, 4)
        R_recovered = quaternion_to_rotation_matrix(q_batch)
        assert np.allclose(R_batch, R_recovered)
    
    def test_zero_rotation(self):
        """Test conversions for zero rotation."""
        R_identity = np.eye(3)
        
        # Axis-angle for identity should have zero angle
        axis, angle = rotation_matrix_to_axis_angle(R_identity)
        assert abs(angle) < 1e-10
        
        # Euler angles should be zero
        euler = rotation_matrix_to_euler(R_identity)
        assert np.allclose(euler, 0)
        
        # Quaternion should be [1, 0, 0, 0] (scalar first)
        q = rotation_matrix_to_quaternion(R_identity, scalar_first=True)
        assert np.allclose(q, [1, 0, 0, 0])


class TestRandomRotation:
    """Test random rotation generation."""
    
    def test_single_rotation(self):
        """Test single random rotation."""
        R_random = random_rotation_matrix(n=1, seed=42)
        
        assert R_random.shape == (3, 3)
        assert np.allclose(R_random @ R_random.T, np.eye(3))
        assert np.isclose(np.linalg.det(R_random), 1.0)
    
    def test_multiple_rotations(self):
        """Test multiple random rotations."""
        n = 100
        R_batch = random_rotation_matrix(n=n, seed=123)
        
        assert R_batch.shape == (n, 3, 3)
        
        for i in range(n):
            assert np.allclose(R_batch[i] @ R_batch[i].T, np.eye(3))
            assert np.isclose(np.linalg.det(R_batch[i]), 1.0)
    
    def test_reproducibility(self):
        """Test that seed gives reproducible results."""
        R1 = random_rotation_matrix(n=5, seed=999)
        R2 = random_rotation_matrix(n=5, seed=999)
        
        assert np.allclose(R1, R2)


class TestRotationMetrics:
    """Test rotation metrics computation."""
    
    def test_perfect_predictions(self):
        """Test metrics for perfect predictions."""
        n = 20
        ground_truth = np.array([R.random().as_matrix() for _ in range(n)])
        predictions = ground_truth.copy()
        
        metrics = compute_rotation_metrics(predictions, ground_truth, degrees=True)
        
        assert metrics['mean_geodesic_error'] < 1e-10
        assert metrics['std_geodesic_error'] < 1e-10
        assert metrics['max_geodesic_error'] < 1e-10
        assert metrics['mean_relative_error'] < 1e-10
    
    def test_noisy_predictions(self):
        """Test metrics for noisy predictions."""
        n = 30
        ground_truth = np.array([R.random().as_matrix() for _ in range(n)])
        
        # Add noise to predictions
        predictions = []
        for gt in ground_truth:
            noise = R.from_rotvec(np.random.randn(3) * 0.1)
            predictions.append(noise.as_matrix() @ gt)
        predictions = np.array(predictions)
        
        metrics = compute_rotation_metrics(predictions, ground_truth, degrees=True)
        
        # Should have non-zero but reasonable errors
        assert 0 < metrics['mean_geodesic_error'] < 10.0
        assert metrics['std_geodesic_error'] > 0
        assert metrics['mean_relative_error'] > 0
    
    def test_radians_output(self):
        """Test metrics in radians."""
        n = 10
        ground_truth = np.array([R.random().as_matrix() for _ in range(n)])
        predictions = ground_truth.copy()
        
        # Add 90-degree error to first prediction
        predictions[0] = R.from_euler('z', 90, degrees=True).as_matrix() @ predictions[0]
        
        metrics = compute_rotation_metrics(predictions, ground_truth, degrees=False)
        
        # First prediction should contribute π/2 error
        assert metrics['max_geodesic_error'] >= np.pi/2 - 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
