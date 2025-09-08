"""
Tests for Copick integration module.
"""

import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

from cryolens.data.copick import (
    CopickDataLoader,
    extract_particles_from_tomogram,
    load_ml_challenge_configs,
    COPICK_AVAILABLE
)


@pytest.fixture
def mock_copick_config():
    """Create a temporary mock Copick configuration file."""
    config = {
        "name": "test_project",
        "description": "Test Copick project",
        "version": "0.1.0",
        "pickable_objects": [
            {"name": "ribosome", "label": 1},
            {"name": "proteasome", "label": 2}
        ],
        "runs_dir": "/tmp/test_runs",
        "static_dir": "/tmp/test_static"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        return Path(f.name)


@pytest.fixture
def mock_tomogram_data():
    """Create mock tomogram data."""
    # Create a 200x200x200 tomogram with some structure
    tomogram = np.random.randn(200, 200, 200).astype(np.float32)
    
    # Add some particles at known locations
    particle_locations = [
        (100, 100, 100),
        (50, 50, 50),
        (150, 150, 150)
    ]
    
    for x, y, z in particle_locations:
        # Add a sphere-like structure
        for dx in range(-10, 11):
            for dy in range(-10, 11):
                for dz in range(-10, 11):
                    if dx**2 + dy**2 + dz**2 <= 100:
                        if 0 <= x+dx < 200 and 0 <= y+dy < 200 and 0 <= z+dz < 200:
                            tomogram[z+dz, y+dy, x+dx] += 2.0
    
    return tomogram, particle_locations


class TestCopickDataLoader:
    """Test the CopickDataLoader class."""
    
    @pytest.mark.skipif(not COPICK_AVAILABLE, reason="Copick not installed")
    def test_initialization_with_valid_config(self, mock_copick_config):
        """Test initialization with a valid configuration file."""
        with patch('cryolens.data.copick.copick.from_file') as mock_from_file:
            mock_root = Mock()
            mock_from_file.return_value = mock_root
            
            loader = CopickDataLoader(mock_copick_config)
            
            assert loader.config_path == mock_copick_config
            assert loader.root == mock_root
            mock_from_file.assert_called_once_with(str(mock_copick_config))
    
    @pytest.mark.skipif(not COPICK_AVAILABLE, reason="Copick not installed")
    def test_initialization_with_missing_config(self):
        """Test initialization with a missing configuration file."""
        with pytest.raises(FileNotFoundError):
            CopickDataLoader("/path/to/nonexistent/config.json")
    
    @pytest.mark.skipif(COPICK_AVAILABLE, reason="Testing without Copick")
    def test_initialization_without_copick(self, mock_copick_config):
        """Test initialization when Copick is not installed."""
        with pytest.raises(ImportError, match="Copick is required"):
            CopickDataLoader(mock_copick_config)
    
    @pytest.mark.skipif(not COPICK_AVAILABLE, reason="Copick not installed")
    def test_list_available_structures(self, mock_copick_config):
        """Test listing available structures."""
        with patch('cryolens.data.copick.copick.from_file') as mock_from_file:
            mock_root = Mock()
            mock_obj1 = Mock(name="ribosome")
            mock_obj2 = Mock(name="proteasome")
            mock_root.pickable_objects = [mock_obj1, mock_obj2]
            mock_from_file.return_value = mock_root
            
            loader = CopickDataLoader(mock_copick_config)
            structures = loader.list_available_structures()
            
            assert structures == ["ribosome", "proteasome"]
    
    @pytest.mark.skipif(not COPICK_AVAILABLE, reason="Copick not installed")
    def test_list_runs(self, mock_copick_config):
        """Test listing available runs."""
        with patch('cryolens.data.copick.copick.from_file') as mock_from_file:
            mock_root = Mock()
            mock_run1 = Mock(name="run001")
            mock_run2 = Mock(name="run002")
            mock_root.runs = [mock_run1, mock_run2]
            mock_from_file.return_value = mock_root
            
            loader = CopickDataLoader(mock_copick_config)
            runs = loader.list_runs()
            
            assert runs == ["run001", "run002"]
    
    @pytest.mark.skipif(not COPICK_AVAILABLE, reason="Copick not installed")
    def test_extract_particle(self, mock_copick_config, mock_tomogram_data):
        """Test particle extraction from tomogram."""
        with patch('cryolens.data.copick.copick.from_file') as mock_from_file:
            mock_root = Mock()
            mock_from_file.return_value = mock_root
            
            loader = CopickDataLoader(mock_copick_config)
            tomogram, locations = mock_tomogram_data
            
            # Create mock point
            mock_point = Mock()
            mock_point.location = Mock(x=100.0, y=100.0, z=100.0)
            
            # Extract particle
            particle, position = loader._extract_particle(
                tomogram,
                mock_point,
                voxel_spacing=1.0,
                box_size=48,
                normalize=True
            )
            
            assert particle is not None
            assert particle.shape == (48, 48, 48)
            assert position is not None
            assert np.allclose(position, [100.0, 100.0, 100.0])
            
            # Check normalization
            assert abs(np.mean(particle)) < 0.1  # Should be close to 0
            assert abs(np.std(particle) - 1.0) < 0.2  # Should be close to 1
    
    @pytest.mark.skipif(not COPICK_AVAILABLE, reason="Copick not installed")
    def test_extract_orientation(self, mock_copick_config):
        """Test orientation extraction from pick point."""
        with patch('cryolens.data.copick.copick.from_file') as mock_from_file:
            mock_root = Mock()
            mock_from_file.return_value = mock_root
            
            loader = CopickDataLoader(mock_copick_config)
            
            # Test with 4x4 transformation matrix
            mock_point = Mock()
            transformation = np.eye(4)
            transformation[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            mock_point.transformation = transformation
            
            orientation = loader._extract_orientation(mock_point)
            assert orientation.shape == (3, 3)
            assert np.allclose(orientation, transformation[:3, :3])
            
            # Test with 3x3 rotation matrix
            mock_point.transformation = transformation[:3, :3]
            orientation = loader._extract_orientation(mock_point)
            assert orientation.shape == (3, 3)
            assert np.allclose(orientation, transformation[:3, :3])
            
            # Test with no transformation
            mock_point.transformation = None
            orientation = loader._extract_orientation(mock_point)
            assert orientation.shape == (3, 3)
            assert np.allclose(orientation, np.eye(3))


class TestStandaloneFunctions:
    """Test standalone functions."""
    
    def test_extract_particles_from_tomogram(self, mock_tomogram_data):
        """Test the standalone particle extraction function."""
        tomogram, locations = mock_tomogram_data
        
        # Convert locations to Angstroms (assuming 1Å voxel spacing)
        positions = [(x, y, z) for x, y, z in locations]
        
        particles = extract_particles_from_tomogram(
            tomogram,
            positions,
            voxel_spacing=1.0,
            box_size=20,
            normalize=True
        )
        
        assert len(particles) == len(locations)
        for particle in particles:
            assert particle.shape == (20, 20, 20)
            # Check normalization
            assert abs(np.mean(particle)) < 0.1
            assert abs(np.std(particle) - 1.0) < 0.2
    
    def test_extract_particles_out_of_bounds(self, mock_tomogram_data):
        """Test particle extraction with out-of-bounds positions."""
        tomogram, _ = mock_tomogram_data
        
        # Positions that would extend beyond tomogram boundaries
        positions = [
            (10, 10, 10),  # Too close to edge for box_size=48
            (190, 190, 190),  # Too close to other edge
            (100, 100, 100)  # This one should work
        ]
        
        particles = extract_particles_from_tomogram(
            tomogram,
            positions,
            voxel_spacing=1.0,
            box_size=48,
            normalize=True
        )
        
        # Only the third position should yield a particle
        assert len(particles) == 1
        assert particles[0].shape == (48, 48, 48)
    
    def test_load_ml_challenge_configs(self):
        """Test loading ML Challenge configurations."""
        with patch('cryolens.data.copick.Path.exists') as mock_exists:
            # Mock that only synthetic config exists
            def exists_side_effect(self):
                return 'synthetic' in str(self)
            
            mock_exists.side_effect = exists_side_effect
            
            configs = load_ml_challenge_configs()
            
            assert 'synthetic' in configs
            assert configs['synthetic'].endswith('ml_challenge_synthetic.json')


class TestIntegration:
    """Integration tests with mocked Copick and zarr."""
    
    @pytest.mark.skipif(not COPICK_AVAILABLE, reason="Copick not installed")
    def test_full_particle_loading_workflow(self, mock_copick_config, mock_tomogram_data):
        """Test the full particle loading workflow."""
        with patch('cryolens.data.copick.copick.from_file') as mock_from_file, \
             patch('cryolens.data.copick.zarr.open') as mock_zarr_open:
            
            # Setup mock Copick structure
            mock_root = Mock()
            mock_from_file.return_value = mock_root
            
            # Create mock run
            mock_run = Mock(name="run001")
            mock_root.runs = [mock_run]
            
            # Create mock voxel spacing
            mock_vs = Mock(voxel_size=10.0)
            mock_run.voxel_spacings = [mock_vs]
            
            # Create mock tomogram
            mock_tomogram = Mock()
            mock_vs.tomograms = [mock_tomogram]
            
            # Setup zarr mock
            tomogram_data, locations = mock_tomogram_data
            mock_zarr_store = {'0': tomogram_data}
            mock_zarr_open.return_value = mock_zarr_store
            
            # Create mock picks
            mock_picks = Mock(pickable_object_name="ribosome")
            mock_points = []
            for x, y, z in locations:
                point = Mock()
                point.location = Mock(x=float(x*10), y=float(y*10), z=float(z*10))  # Convert to Angstroms
                point.transformation = np.eye(4)
                mock_points.append(point)
            mock_picks.points = mock_points
            mock_run.picks = [mock_picks]
            
            # Create loader and load particles
            loader = CopickDataLoader(mock_copick_config)
            results = loader.load_particles(
                structure_filter=["ribosome"],
                max_particles_per_structure=10,
                target_voxel_spacing=10.0,
                box_size=20,
                verbose=False
            )
            
            assert "ribosome" in results
            assert "particles" in results["ribosome"]
            assert "orientations" in results["ribosome"]
            assert "positions" in results["ribosome"]
            
            particles = results["ribosome"]["particles"]
            assert len(particles) == len(locations)
            assert particles.shape == (len(locations), 20, 20, 20)
            
            orientations = results["ribosome"]["orientations"]
            assert orientations.shape == (len(locations), 3, 3)
            
            positions = results["ribosome"]["positions"]
            assert positions.shape == (len(locations), 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
