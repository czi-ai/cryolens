import os
import sys
import torch
import pytest
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from cryolens.models.encoders import Encoder3D
from cryolens.models.decoders import SegmentedGaussianSplatDecoder
from cryolens.models.vae import AffinityVAE
from cryolens.training.losses import MissingWedgeLoss, AffinityCosineLoss

# Test configuration
TEST_CONFIG = {
    "structure_ids": ['PDB1', 'PDB2', 'PDB3', 'PDB4'],
    "box_size": 48,
    "latent_dims": 16,
    "num_splats": 256,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "beta": 0.05,
    "gamma": 0.2,
    "latent_ratio": 0.4,
    "wedge_weight_factor": 0.001
}

class MockDataset(torch.utils.data.Dataset):
    """Simple mock dataset for testing."""
    def __init__(self, num_samples=10, box_size=48, structure_ids=None):
        self.num_samples = num_samples
        self.box_size = box_size
        self.structure_ids = structure_ids or TEST_CONFIG["structure_ids"]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create a random volume with single channel
        volume = torch.randn(1, self.box_size, self.box_size, self.box_size)
        
        # Assign a random structure ID index
        structure_idx = torch.tensor(idx % len(self.structure_ids))
        
        return volume, structure_idx


@pytest.mark.integration
def test_vae_training_integration():
    """
    Integration test for VAE training with similarity loss.
    
    This test verifies the core functionality needed by mlc.py:
    1. Creating a mock similarity matrix (bypassing database loading)
    2. Creating encoder/decoder
    3. Setting up VAE model
    4. Training loop with reconstruction and similarity losses
    """
    # Use CPU for testing unless explicitly requested to use GPU
    use_cuda = torch.cuda.is_available() and os.environ.get("TEST_CUDA", "0") == "1"
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    print(f"Running integration test on device: {device}")
    
    # Step 1: Create a mock similarity matrix directly (no database needed)
    n_structures = len(TEST_CONFIG["structure_ids"])
    similarity_matrix = torch.ones((n_structures, n_structures), device=device)
    
    # Set the diagonal to 1.0 (self-similarity)
    torch.diagonal(similarity_matrix).fill_(1.0)
    
    # Set random similarity values for off-diagonal elements (between 0.1 and 0.9)
    mask = ~torch.eye(n_structures, dtype=bool, device=device)
    similarity_matrix[mask] = torch.rand(mask.sum(), device=device) * 0.8 + 0.1
    
    # Make matrix symmetric
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    
    # Step 2: Create encoder
    encoder = Encoder3D(
        input_shape=(TEST_CONFIG["box_size"], TEST_CONFIG["box_size"], TEST_CONFIG["box_size"]),
        layer_channels=(8, 16, 32, 64)
    ).to(device)
    
    # Step 3: Create decoder
    decoder = SegmentedGaussianSplatDecoder(
        (TEST_CONFIG["box_size"], TEST_CONFIG["box_size"], TEST_CONFIG["box_size"]),
        latent_dims=TEST_CONFIG["latent_dims"],
        n_splats=TEST_CONFIG["num_splats"],
        output_channels=1,
        device=device,
        splat_sigma_range=(0.005, 0.1),
        padding=9,
        latent_ratio=TEST_CONFIG["latent_ratio"]
    ).to(device)
    
    # Step 4: Create VAE model
    model = AffinityVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dims=TEST_CONFIG["latent_dims"],
        pose_channels=4
    ).to(device)
    
    # Step 5: Create loss functions
    reconstruction_loss = MissingWedgeLoss(
        volume_size=TEST_CONFIG["box_size"],
        wedge_angle=90.0,
        weight_factor=TEST_CONFIG["wedge_weight_factor"]
    )
    
    similarity_loss = AffinityCosineLoss(
        lookup=similarity_matrix,
        device=device
    )
    
    # Step 6: Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG["learning_rate"])
    
    # Step 7: Create a small mock dataset and dataloader
    dataset = MockDataset(num_samples=10, box_size=TEST_CONFIG["box_size"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=TEST_CONFIG["batch_size"], shuffle=True)
    
    # Store initial weights to verify training changes them
    initial_weight = next(model.parameters()).clone().detach()
    
    # Step 8: Mini training loop
    model.train()
    for epoch in range(2):  # Just 2 epochs for testing
        epoch_losses = []
        
        for batch_idx, (data, target_idx) in enumerate(dataloader):
            # Move data to device
            data = data.to(device)
            target_idx = target_idx.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, z, pose, mu, log_var = model(data)
            
            # Compute reconstruction loss
            r_loss = reconstruction_loss(output, data)
            
            # Compute KL divergence loss
            kld = -0.5 * torch.mean(torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
            kld = TEST_CONFIG["beta"] * kld
            
            # Compute similarity loss
            s_loss = TEST_CONFIG["gamma"] * similarity_loss(target_idx, mu)
            
            # Total loss
            loss = r_loss + kld + s_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Store loss
            epoch_losses.append(loss.item())
            
            # For testing, just run a few batches
            if batch_idx >= 2:
                break
                
        # Print epoch stats
        print(f"Epoch {epoch+1}, Loss: {np.mean(epoch_losses):.4f}")
    
    # Verify model learned something
    assert not torch.isnan(torch.tensor(epoch_losses)).any(), "NaN loss detected"
    
    # Verify that weights have changed
    current_weight = next(model.parameters()).clone().detach()
    weight_diff = (initial_weight - current_weight).abs().sum().item()
    assert weight_diff > 0, "Model weights did not change during training"
    
    # Step 9: Test inference
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 1, TEST_CONFIG["box_size"], TEST_CONFIG["box_size"], TEST_CONFIG["box_size"], device=device)
        output, z, pose, mu, log_var = model(test_input)
        
        # Check output shape
        assert output.shape == test_input.shape, f"Expected output shape {test_input.shape}, got {output.shape}"
        assert z.shape[1] == TEST_CONFIG["latent_dims"], f"Expected latent dim {TEST_CONFIG['latent_dims']}, got {z.shape[1]}"
    
    print("Integration test completed successfully!")


@pytest.mark.integration
def test_vae_with_mocked_similarity_calculator():
    """
    Integration test that mocks the SimilarityCalculator but tests the rest of the pipeline.
    This approach keeps the test more aligned with how mlc.py would use the components.
    """
    # Import the SimilarityCalculator here to be mocked
    from cryolens.affinity import SimilarityCalculator
    
    # Use CPU for testing unless explicitly requested to use GPU
    use_cuda = torch.cuda.is_available() and os.environ.get("TEST_CUDA", "0") == "1"
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    print(f"Running integration test with mocked similarity calculator on device: {device}")
    
    # Create a mock similarity matrix
    n_structures = len(TEST_CONFIG["structure_ids"])
    mock_matrix = np.ones((n_structures, n_structures))
    np.fill_diagonal(mock_matrix, 1.0)
    
    # Set off-diagonal elements
    for i in range(n_structures):
        for j in range(i+1, n_structures):
            val = 0.1 + 0.8 * np.random.random()
            mock_matrix[i, j] = val
            mock_matrix[j, i] = val  # Ensure symmetry
            
    # Create the mock calculator
    with patch.object(SimilarityCalculator, 'load_matrix', return_value=(mock_matrix, TEST_CONFIG["structure_ids"])):
        # Create a real calculator instance but with mocked method
        calculator = SimilarityCalculator("mocked_path_doesnt_matter.db")
        
        # Call the mocked method
        similarity_matrix, molecule_order = calculator.load_matrix(structure_ids=TEST_CONFIG["structure_ids"])
        
        # Convert to torch tensor
        similarity_matrix = torch.from_numpy(similarity_matrix).float().to(device)
        
        # Continue with the same test flow as before
        # Create encoder
        encoder = Encoder3D(
            input_shape=(TEST_CONFIG["box_size"], TEST_CONFIG["box_size"], TEST_CONFIG["box_size"]),
            layer_channels=(8, 16, 32, 64)
        ).to(device)
        
        # Create decoder
        decoder = SegmentedGaussianSplatDecoder(
            (TEST_CONFIG["box_size"], TEST_CONFIG["box_size"], TEST_CONFIG["box_size"]),
            latent_dims=TEST_CONFIG["latent_dims"],
            n_splats=TEST_CONFIG["num_splats"],
            output_channels=1,
            device=device,
            splat_sigma_range=(0.005, 0.1),
            padding=9,
            latent_ratio=TEST_CONFIG["latent_ratio"]
        ).to(device)
        
        # Create VAE model
        model = AffinityVAE(
            encoder=encoder,
            decoder=decoder,
            latent_dims=TEST_CONFIG["latent_dims"],
            pose_channels=4
        ).to(device)
        
        # Create loss functions
        reconstruction_loss = MissingWedgeLoss(
            volume_size=TEST_CONFIG["box_size"],
            wedge_angle=90.0,
            weight_factor=TEST_CONFIG["wedge_weight_factor"]
        )
        
        similarity_loss = AffinityCosineLoss(
            lookup=similarity_matrix,
            device=device
        )
        
        # Mini training loop (1 batch only to keep it fast)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=TEST_CONFIG["learning_rate"])
        
        # Make a single batch of data
        data = torch.randn(TEST_CONFIG["batch_size"], 1, 
                         TEST_CONFIG["box_size"], 
                         TEST_CONFIG["box_size"], 
                         TEST_CONFIG["box_size"]).to(device)
        target_idx = torch.tensor([0, 1]).to(device)  # Just two different indices
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output, z, pose, mu, log_var = model(data)
        
        # Compute losses
        r_loss = reconstruction_loss(output, data)
        kld = -0.5 * torch.mean(torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
        kld = TEST_CONFIG["beta"] * kld
        s_loss = TEST_CONFIG["gamma"] * similarity_loss(target_idx, mu)
        
        # Total loss
        loss = r_loss + kld + s_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Verify no NaN loss
        assert not torch.isnan(loss).any(), "NaN loss detected"
        
        # Test inference
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 1, TEST_CONFIG["box_size"], 
                                   TEST_CONFIG["box_size"], 
                                   TEST_CONFIG["box_size"]).to(device)
            output, z, pose, mu, log_var = model(test_input)
            
            # Check output shape
            assert output.shape == test_input.shape, "Output shape mismatch"
        
        print("Integration test with mocked similarity calculator completed successfully!")


if __name__ == "__main__":
    # Allow running as a standalone script
    print("Running first integration test...")
    test_vae_training_integration()
    print("\nRunning integration test with mocked similarity calculator...")
    test_vae_with_mocked_similarity_calculator()