import torch
import numpy as np
from src.models.folding_model import ProteinFoldingModel

class FoldingSimulator:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = ProteinFoldingModel().to(self.device)
        self.model.eval()
        
        # Amino acid mapping
        self.aa_map = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        
    def sequence_to_tensor(self, seq):
        """Converts an amino acid sequence string to a PyTorch tensor."""
        indices = [self.aa_map.get(aa, 0) for aa in seq.upper()]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(1).to(self.device)

    def predict_structure(self, sequence):
        """Predicts the 3D coordinates of a protein sequence."""
        tensor = self.sequence_to_tensor(sequence)
        with torch.no_grad():
            coords = self.model(tensor)
        return coords.squeeze(1).cpu().numpy()

    def calculate_energy(self, coords):
        """Calculates pseudo-energy of the conformation."""
        # Simplified distance-based energy
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        energy = np.sum(dist_matrix) / 2.0
        return energy
