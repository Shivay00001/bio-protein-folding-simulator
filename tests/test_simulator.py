"""
Unit tests for the Protein Folding Simulator.
"""
import pytest
import numpy as np
import torch
from src.simulation.simulator import FoldingSimulator
from src.models.folding_model import ProteinFoldingModel, load_folding_model


class TestProteinFoldingModel:
    """Tests for the ProteinFoldingModel neural network."""

    def test_model_initialization(self):
        """Test that the model initializes correctly with default parameters."""
        model = ProteinFoldingModel()
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'fc')

    def test_model_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = ProteinFoldingModel()
        model.eval()
        
        # Create dummy input: sequence length 10, batch size 1
        dummy_input = torch.randint(0, 20, (10, 1))
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Output should be [seq_len, batch_size, 3] for 3D coordinates
        assert output.shape == (10, 1, 3)

    def test_model_custom_params(self):
        """Test model with custom hyperparameters."""
        model = ProteinFoldingModel(vocab_size=30, d_model=128, nhead=4, num_layers=3)
        model.eval()
        
        dummy_input = torch.randint(0, 20, (5, 2))
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (5, 2, 3)

    def test_load_folding_model(self):
        """Test the model loading utility function."""
        model = load_folding_model(weights_path=None)
        assert isinstance(model, ProteinFoldingModel)
        assert not model.training  # Should be in eval mode


class TestFoldingSimulator:
    """Tests for the FoldingSimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create a FoldingSimulator instance for testing."""
        return FoldingSimulator(device="cpu")

    def test_simulator_initialization(self, simulator):
        """Test simulator initializes correctly."""
        assert simulator.device == torch.device("cpu")
        assert hasattr(simulator, 'model')
        assert hasattr(simulator, 'aa_map')
        assert len(simulator.aa_map) == 20  # 20 standard amino acids

    def test_sequence_to_tensor(self, simulator):
        """Test amino acid sequence conversion to tensor."""
        sequence = "MKVIFL"
        tensor = simulator.sequence_to_tensor(sequence)
        
        assert tensor.shape == (6, 1)
        assert tensor.dtype == torch.long

    def test_sequence_handles_unknown_residues(self, simulator):
        """Test that unknown amino acids are handled gracefully."""
        sequence = "MKV-XFL"  # Contains invalid characters
        tensor = simulator.sequence_to_tensor(sequence)
        
        # Should still produce valid tensor (unknown mapped to 0)
        assert tensor.shape == (7, 1)

    def test_predict_structure(self, simulator):
        """Test structure prediction produces valid 3D coordinates."""
        sequence = "MKVIFLALLV"
        coords = simulator.predict_structure(sequence)
        
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (10, 3)  # 10 residues, 3D coordinates

    def test_calculate_energy(self, simulator):
        """Test energy calculation."""
        # Simple test with known coordinates
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ])
        
        energy = simulator.calculate_energy(coords)
        
        assert isinstance(energy, float)
        assert energy > 0  # Energy should be positive

    def test_energy_increases_with_distance(self, simulator):
        """Test that energy correlates with pairwise distances."""
        # Compact structure
        compact_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        
        # Extended structure
        extended_coords = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0]
        ])
        
        compact_energy = simulator.calculate_energy(compact_coords)
        extended_energy = simulator.calculate_energy(extended_coords)
        
        assert extended_energy > compact_energy


class TestIntegration:
    """Integration tests for the complete simulation pipeline."""

    def test_end_to_end_simulation(self):
        """Test complete simulation from sequence to energy."""
        simulator = FoldingSimulator()
        sequence = "ACDEFGHIKLMNPQRSTVWY"  # All 20 standard amino acids
        
        # Predict structure
        coords = simulator.predict_structure(sequence)
        
        # Calculate energy
        energy = simulator.calculate_energy(coords)
        
        # Verify outputs
        assert coords.shape == (20, 3)
        assert isinstance(energy, float)
        assert not np.isnan(energy)
        assert not np.isinf(energy)

    def test_reproducibility(self):
        """Test that predictions are deterministic."""
        simulator = FoldingSimulator()
        sequence = "MKVIFL"
        
        torch.manual_seed(42)
        coords1 = simulator.predict_structure(sequence)
        
        torch.manual_seed(42)
        coords2 = simulator.predict_structure(sequence)
        
        # In eval mode, results should be deterministic
        np.testing.assert_array_almost_equal(coords1, coords2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
