# Bio-Protein Folding Simulator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade protein folding simulator** leveraging deep learning for structural biology. Built with PyTorch, this repository provides a framework for predicting the 3D conformation of proteins from their amino acid sequences, facilitating drug discovery and biological research.

## 🔬 Scientific Context

Protein folding is one of the most complex problems in biology. Predicting the 3D structure of a protein from its amino acid sequence is crucial for **drug discovery**, **enzyme engineering**, and understanding **genetic diseases**. This tool uses a simplified Transformer-based model to approximate this process, demonstrating the application of Deep Learning to biological constraints.

## 🏗️ Architecture

The system uses a custom Transformer Encoder to capture long-range interactions between amino acids, regressing directly to 3D Cartesian coordinates.

```mermaid
graph TD
    A[Input Sequence] -->|Embedding| B(Transformer Encoder)
    B -->|Self-Attention| B
    B -->|Feature Extraction| C{3D Coordinate Head}
    C -->|Regression| D[Predicted Structure (x,y,z)]
    D -->|Physics Loss| E[Energy Minimization]
    E -->|Backprop| B
```

## 🚀 Features

- **Deep Learning Core**: Transformer-based architecture for learning long-range dependencies in protein sequences.
- **Structural Prediction**: Predicts secondary structures (alpha-helices, beta-sheets) and 3D coordinates.
- **Physical Invariants**: Incorporates differentiable physical constraints to ensure structural validity.
- **High Performance**: Optimized tensor operations for fast inference on CPUs and GPUs.
- **Visualization Export**: Generates PDB (Protein Data Bank) files for visualization in standard software (PyMOL, Chimera).
- **Containerized**: Modular Docker setup for easy scaling and deployment in HPC environments.

## 📁 Project Structure

```text
bio-protein-folding-simulator/
├── src/
│   ├── models/       # Neural network architectures
│   ├── simulation/   # Physics-based simulation logic
│   └── main.py       # Job entrypoint
├── data/             # Sample amino acid datasets
├── tests/            # Validation and regression tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🛠️ Quick Start

```bash
# Clone
git clone https://github.com/Shivay00001/bio-protein-folding-simulator.git

# Install
pip install -r requirements.txt

# Run Simulator
python src/main.py --sequence MKVIFLALLVSTISSV
```

### Sample Output

```json
{
    "sequence": "MKVIFLALLVSTISSV",
    "energy": -42.85,
    "residues": 16,
    "coordinates": [
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [5.2, 2.4, 1.1]
    ]
}
```

*Generated structure saved to `output/structure.json` and converted to PDB format.*

## 📄 License

MIT License
