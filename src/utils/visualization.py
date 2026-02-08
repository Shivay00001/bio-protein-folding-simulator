"""
Visualization utilities for protein structure analysis.

Provides 3D structure visualization and analysis plotting capabilities.
"""
import numpy as np
from typing import Optional, Dict, List, Tuple
import json


class StructureVisualizer:
    """
    Visualizes protein 3D structures and analysis results.
    
    Provides methods for generating structure data suitable for
    various visualization backends (matplotlib, plotly, pymol, etc.)
    """
    
    # Color schemes based on amino acid properties
    PROPERTY_COLORS = {
        'hydrophobic': '#FFD700',  # Gold
        'hydrophilic': '#00CED1',  # Dark Turquoise
        'charged_positive': '#0000FF',  # Blue
        'charged_negative': '#FF0000',  # Red
        'aromatic': '#800080',  # Purple
        'polar': '#00FF00',  # Green
    }
    
    # Secondary structure colors
    SECONDARY_COLORS = {
        'helix': '#FF6B6B',
        'sheet': '#4ECDC4',
        'coil': '#95A5A6',
    }
    
    def __init__(self, coordinates: np.ndarray, sequence: str):
        """
        Initialize the visualizer with structure data.
        
        Args:
            coordinates: Nx3 array of atom coordinates
            sequence: Amino acid sequence
        """
        if len(coordinates) != len(sequence):
            raise ValueError("Coordinates and sequence length mismatch")
        
        self.coordinates = coordinates
        self.sequence = sequence.upper()
        self._cached_distances: Optional[np.ndarray] = None
    
    @property
    def center_of_mass(self) -> np.ndarray:
        """Calculate the center of mass of the structure."""
        return np.mean(self.coordinates, axis=0)
    
    @property
    def radius_of_gyration(self) -> float:
        """Calculate the radius of gyration."""
        com = self.center_of_mass
        distances = np.linalg.norm(self.coordinates - com, axis=1)
        return np.sqrt(np.mean(distances ** 2))
    
    @property
    def distance_matrix(self) -> np.ndarray:
        """Compute pairwise distance matrix."""
        if self._cached_distances is None:
            n = len(self.coordinates)
            self._cached_distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
                    self._cached_distances[i, j] = dist
                    self._cached_distances[j, i] = dist
        return self._cached_distances.copy()
    
    def get_contact_map(self, threshold: float = 8.0) -> np.ndarray:
        """
        Generate a contact map based on distance threshold.
        
        Args:
            threshold: Distance threshold in Angstroms (default 8.0)
            
        Returns:
            Binary NxN contact map
        """
        dist_matrix = self.distance_matrix
        return (dist_matrix < threshold).astype(int)
    
    def get_backbone_curve(self, smoothing: int = 3) -> np.ndarray:
        """
        Get smoothed backbone curve coordinates.
        
        Args:
            smoothing: Window size for moving average smoothing
            
        Returns:
            Smoothed Nx3 coordinate array
        """
        if smoothing < 2:
            return self.coordinates.copy()
        
        smoothed = np.zeros_like(self.coordinates)
        half_window = smoothing // 2
        
        for i in range(len(self.coordinates)):
            start = max(0, i - half_window)
            end = min(len(self.coordinates), i + half_window + 1)
            smoothed[i] = np.mean(self.coordinates[start:end], axis=0)
        
        return smoothed
    
    def get_ramachandran_data(self) -> Dict[str, List[float]]:
        """
        Calculate phi and psi angles for Ramachandran plot.
        
        Returns:
            Dictionary with 'phi' and 'psi' angle lists
        """
        phi_angles = []
        psi_angles = []
        
        for i in range(1, len(self.coordinates) - 1):
            # Simplified calculation using consecutive C-alpha positions
            v1 = self.coordinates[i] - self.coordinates[i-1]
            v2 = self.coordinates[i+1] - self.coordinates[i]
            
            # Calculate dihedral-like angle
            cross = np.cross(v1, v2)
            angle = np.arctan2(np.linalg.norm(cross), np.dot(v1, v2))
            
            phi_angles.append(np.degrees(angle))
            psi_angles.append(np.degrees(-angle))  # Simplified
        
        return {'phi': phi_angles, 'psi': psi_angles}
    
    def to_pdb_format(self, chain_id: str = 'A') -> str:
        """
        Convert structure to PDB format string.
        
        Args:
            chain_id: Chain identifier (default 'A')
            
        Returns:
            PDB formatted string
        """
        lines = []
        lines.append("HEADER    PREDICTED PROTEIN STRUCTURE")
        lines.append("TITLE     PROTEIN FOLDING SIMULATION RESULT")
        
        for i, (coord, aa) in enumerate(zip(self.coordinates, self.sequence)):
            aa_3letter = self._one_to_three(aa)
            atom_line = (
                f"ATOM  {i+1:5d}  CA  {aa_3letter} {chain_id}{i+1:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           C"
            )
            lines.append(atom_line)
        
        lines.append("END")
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export structure data as JSON."""
        return json.dumps({
            'sequence': self.sequence,
            'coordinates': self.coordinates.tolist(),
            'center_of_mass': self.center_of_mass.tolist(),
            'radius_of_gyration': float(self.radius_of_gyration),
            'num_residues': len(self.sequence),
        }, indent=2)
    
    def get_secondary_structure_prediction(self) -> List[str]:
        """
        Predict secondary structure based on local geometry.
        
        Returns:
            List of secondary structure assignments ('H', 'E', 'C')
        """
        assignments = []
        
        for i in range(len(self.coordinates)):
            if i < 2 or i >= len(self.coordinates) - 2:
                assignments.append('C')  # Coil at termini
                continue
            
            # Use local curvature as proxy for secondary structure
            v1 = self.coordinates[i] - self.coordinates[i-2]
            v2 = self.coordinates[i+2] - self.coordinates[i]
            
            angle = np.arccos(np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8),
                -1, 1
            ))
            
            if angle < 0.5:  # Low curvature - sheet
                assignments.append('E')
            elif angle < 1.2:  # Medium curvature - helix
                assignments.append('H')
            else:
                assignments.append('C')
        
        return assignments
    
    @staticmethod
    def _one_to_three(aa: str) -> str:
        """Convert one-letter amino acid code to three-letter."""
        mapping = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
        }
        return mapping.get(aa.upper(), 'UNK')


class AnalysisReport:
    """Generates comprehensive analysis reports for protein structures."""
    
    def __init__(self, visualizer: StructureVisualizer):
        """Initialize with a StructureVisualizer instance."""
        self.viz = visualizer
    
    def generate_summary(self) -> Dict:
        """Generate a summary of structural properties."""
        ss = self.viz.get_secondary_structure_prediction()
        
        return {
            'sequence_length': len(self.viz.sequence),
            'radius_of_gyration': float(self.viz.radius_of_gyration),
            'center_of_mass': self.viz.center_of_mass.tolist(),
            'secondary_structure': {
                'helix_percent': ss.count('H') / len(ss) * 100,
                'sheet_percent': ss.count('E') / len(ss) * 100,
                'coil_percent': ss.count('C') / len(ss) * 100,
            },
            'compactness': self._calculate_compactness(),
        }
    
    def _calculate_compactness(self) -> float:
        """Calculate structure compactness score (0-1)."""
        # Higher Rg = less compact
        # Normalize by expected Rg for a random coil
        expected_rg = 2.0 * (len(self.viz.sequence) ** 0.6)
        actual_rg = self.viz.radius_of_gyration
        
        compactness = 1.0 - min(actual_rg / expected_rg, 1.0)
        return max(0.0, compactness)
