"""
Sequence utilities for protein folding simulation.

Provides validation, processing, and analysis utilities for amino acid sequences.
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AminoAcidProperty(Enum):
    """Classification of amino acid properties."""
    HYDROPHOBIC = "hydrophobic"
    HYDROPHILIC = "hydrophilic"
    CHARGED_POSITIVE = "charged_positive"
    CHARGED_NEGATIVE = "charged_negative"
    AROMATIC = "aromatic"
    POLAR = "polar"


@dataclass
class AminoAcidInfo:
    """Information about a single amino acid."""
    code: str
    name: str
    three_letter: str
    molecular_weight: float
    property: AminoAcidProperty


class SequenceValidator:
    """Validates amino acid sequences for proper formatting and content."""
    
    STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
    EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYUBXZ")
    
    # Amino acid properties database
    AA_DATABASE: Dict[str, AminoAcidInfo] = {
        'A': AminoAcidInfo('A', 'Alanine', 'Ala', 89.1, AminoAcidProperty.HYDROPHOBIC),
        'C': AminoAcidInfo('C', 'Cysteine', 'Cys', 121.2, AminoAcidProperty.POLAR),
        'D': AminoAcidInfo('D', 'Aspartic Acid', 'Asp', 133.1, AminoAcidProperty.CHARGED_NEGATIVE),
        'E': AminoAcidInfo('E', 'Glutamic Acid', 'Glu', 147.1, AminoAcidProperty.CHARGED_NEGATIVE),
        'F': AminoAcidInfo('F', 'Phenylalanine', 'Phe', 165.2, AminoAcidProperty.AROMATIC),
        'G': AminoAcidInfo('G', 'Glycine', 'Gly', 75.1, AminoAcidProperty.HYDROPHOBIC),
        'H': AminoAcidInfo('H', 'Histidine', 'His', 155.2, AminoAcidProperty.CHARGED_POSITIVE),
        'I': AminoAcidInfo('I', 'Isoleucine', 'Ile', 131.2, AminoAcidProperty.HYDROPHOBIC),
        'K': AminoAcidInfo('K', 'Lysine', 'Lys', 146.2, AminoAcidProperty.CHARGED_POSITIVE),
        'L': AminoAcidInfo('L', 'Leucine', 'Leu', 131.2, AminoAcidProperty.HYDROPHOBIC),
        'M': AminoAcidInfo('M', 'Methionine', 'Met', 149.2, AminoAcidProperty.HYDROPHOBIC),
        'N': AminoAcidInfo('N', 'Asparagine', 'Asn', 132.1, AminoAcidProperty.POLAR),
        'P': AminoAcidInfo('P', 'Proline', 'Pro', 115.1, AminoAcidProperty.HYDROPHOBIC),
        'Q': AminoAcidInfo('Q', 'Glutamine', 'Gln', 146.2, AminoAcidProperty.POLAR),
        'R': AminoAcidInfo('R', 'Arginine', 'Arg', 174.2, AminoAcidProperty.CHARGED_POSITIVE),
        'S': AminoAcidInfo('S', 'Serine', 'Ser', 105.1, AminoAcidProperty.POLAR),
        'T': AminoAcidInfo('T', 'Threonine', 'Thr', 119.1, AminoAcidProperty.POLAR),
        'V': AminoAcidInfo('V', 'Valine', 'Val', 117.1, AminoAcidProperty.HYDROPHOBIC),
        'W': AminoAcidInfo('W', 'Tryptophan', 'Trp', 204.2, AminoAcidProperty.AROMATIC),
        'Y': AminoAcidInfo('Y', 'Tyrosine', 'Tyr', 181.2, AminoAcidProperty.AROMATIC),
    }
    
    @classmethod
    def validate(cls, sequence: str, allow_extended: bool = False) -> Tuple[bool, List[str]]:
        """
        Validates an amino acid sequence.
        
        Args:
            sequence: The amino acid sequence to validate
            allow_extended: Whether to allow extended amino acid codes (B, X, Z, U)
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        sequence = sequence.upper().strip()
        
        if not sequence:
            errors.append("Sequence is empty")
            return False, errors
        
        if len(sequence) < 2:
            errors.append("Sequence must be at least 2 residues long")
            
        if len(sequence) > 10000:
            errors.append("Sequence exceeds maximum length of 10,000 residues")
        
        valid_aa = cls.EXTENDED_AA if allow_extended else cls.STANDARD_AA
        invalid_chars = set(sequence) - valid_aa
        
        if invalid_chars:
            errors.append(f"Invalid amino acid codes: {', '.join(sorted(invalid_chars))}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def get_amino_acid_info(cls, code: str) -> Optional[AminoAcidInfo]:
        """Get detailed information about an amino acid."""
        return cls.AA_DATABASE.get(code.upper())


class SequenceProcessor:
    """Processes amino acid sequences for analysis and simulation."""
    
    def __init__(self, sequence: str):
        """
        Initialize the processor with an amino acid sequence.
        
        Args:
            sequence: The amino acid sequence to process
        """
        is_valid, errors = SequenceValidator.validate(sequence)
        if not is_valid:
            raise ValueError(f"Invalid sequence: {', '.join(errors)}")
        
        self.sequence = sequence.upper().strip()
        self._composition_cache: Optional[Dict[str, int]] = None
    
    @property
    def length(self) -> int:
        """Return the length of the sequence."""
        return len(self.sequence)
    
    @property
    def composition(self) -> Dict[str, int]:
        """Calculate amino acid composition."""
        if self._composition_cache is None:
            self._composition_cache = {}
            for aa in self.sequence:
                self._composition_cache[aa] = self._composition_cache.get(aa, 0) + 1
        return self._composition_cache.copy()
    
    @property
    def molecular_weight(self) -> float:
        """Calculate the molecular weight of the protein."""
        total = 0.0
        water_loss = (self.length - 1) * 18.015  # Peptide bond formation
        
        for aa in self.sequence:
            info = SequenceValidator.get_amino_acid_info(aa)
            if info:
                total += info.molecular_weight
        
        return total - water_loss
    
    def get_hydrophobicity_profile(self, window_size: int = 9) -> List[float]:
        """
        Calculate hydrophobicity profile using Kyte-Doolittle scale.
        
        Args:
            window_size: Size of the sliding window (default 9)
            
        Returns:
            List of hydrophobicity values for each position
        """
        kyte_doolittle = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        profile = []
        half_window = window_size // 2
        
        for i in range(len(self.sequence)):
            start = max(0, i - half_window)
            end = min(len(self.sequence), i + half_window + 1)
            
            window = self.sequence[start:end]
            avg_hydro = sum(kyte_doolittle.get(aa, 0) for aa in window) / len(window)
            profile.append(avg_hydro)
        
        return profile
    
    def find_motifs(self, pattern: str) -> List[Tuple[int, str]]:
        """
        Find occurrences of a sequence motif.
        
        Args:
            pattern: Regex pattern to search for
            
        Returns:
            List of (position, matched_sequence) tuples
        """
        matches = []
        for match in re.finditer(pattern, self.sequence):
            matches.append((match.start(), match.group()))
        return matches
    
    def get_property_distribution(self) -> Dict[str, float]:
        """Calculate the distribution of amino acid properties."""
        distribution = {prop.value: 0 for prop in AminoAcidProperty}
        
        for aa in self.sequence:
            info = SequenceValidator.get_amino_acid_info(aa)
            if info:
                distribution[info.property.value] += 1
        
        # Convert to percentages
        total = len(self.sequence)
        return {k: v / total * 100 for k, v in distribution.items()}
    
    def to_fasta(self, header: str = "protein") -> str:
        """Convert sequence to FASTA format."""
        lines = [f">{header}"]
        for i in range(0, len(self.sequence), 70):
            lines.append(self.sequence[i:i+70])
        return "\n".join(lines)
