import argparse
from src.simulation.simulator import FoldingSimulator
import json

def main():
    parser = argparse.ArgumentParser(description="Bio-Protein Folding Simulator")
    parser.add_argument("--sequence", type=str, required=False, default="MKVIFLALLVSTISSV", help="Amino acid sequence")
    parser.add_argument("--output", type=str, default="output/structure.json", help="Output path for coordinates")
    
    args = parser.parse_args()
    
    print(f"Starting simulation for sequence: {args.sequence}")
    simulator = FoldingSimulator()
    
    # Run prediction
    coords = simulator.predict_structure(args.sequence)
    energy = simulator.calculate_energy(coords)
    
    results = {
        "sequence": args.sequence,
        "energy": float(energy),
        "residues": len(args.sequence),
        "coordinates": coords.tolist()
    }
    
    print(f"Simulation complete. Predicted Energy: {energy:.4f}")
    
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Structure saved to {args.output}")

if __name__ == "__main__":
    main()
