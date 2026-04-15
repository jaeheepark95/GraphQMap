"""Build GraphQMap training dataset from labels and circuits.

Constructs PyG graph objects and saves everything as a pickle file
for fast loading.

Usage:
    python scripts/build_dataset.py --labels data/labels/labels.json \
        --circuit-dir data/circuits/mqt_bench --output data/dataset.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import QuantumCircuit

from data.circuit_graph import build_circuit_graph
from data.dataset import MappingDataset, MappingSample
from data.hardware_graph import build_hardware_graph, get_backend, precompute_error_distance
from data.label_generation import layout_to_permutation_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_dataset_from_labels(
    labels: list[dict],
    circuit_dir: Path,
    include_labels: bool = True,
) -> MappingDataset:
    """Build MappingDataset from label entries.

    Args:
        labels: List of label dicts from generate_labels.py.
        circuit_dir: Directory with .qasm files.
        include_labels: Whether to include label matrices.

    Returns:
        MappingDataset.
    """
    dataset = MappingDataset()

    # Cache hardware graphs per backend
    hw_graph_cache: dict[str, object] = {}
    backend_cache: dict[str, object] = {}

    for i, entry in enumerate(labels):
        circuit_stem = entry["circuit"]
        backend_name = entry["backend"]

        qasm_path = circuit_dir / f"{circuit_stem}.qasm"
        if not qasm_path.exists():
            continue

        try:
            circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
        except Exception:
            continue

        # Get backend
        if backend_name not in backend_cache:
            backend_cache[backend_name] = get_backend(backend_name)
        backend = backend_cache[backend_name]

        # Hardware graph (cached)
        if backend_name not in hw_graph_cache:
            hw_graph_cache[backend_name] = build_hardware_graph(backend)
        hw_graph = hw_graph_cache[backend_name]

        num_physical = backend.target.num_qubits
        num_logical = circuit.num_qubits

        # Circuit graph (4-dim node features, no global summary)
        circuit_graph = build_circuit_graph(circuit)

        # Label
        label_matrix = None
        layout = None
        if include_labels and "layout" in entry:
            layout = entry["layout"]
            label_matrix = layout_to_permutation_matrix(layout, num_physical)

        sample = MappingSample(
            circuit_graph=circuit_graph,
            hardware_graph=hw_graph,
            backend_name=backend_name,
            num_logical=num_logical,
            num_physical=num_physical,
            label_matrix=label_matrix,
            layout=layout,
        )
        dataset.add_sample(sample)

        if (i + 1) % 500 == 0:
            logger.info(f"Built {i+1}/{len(labels)} samples")

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GraphQMap dataset")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to labels.json")
    parser.add_argument("--circuit-dir", type=str, required=True,
                        help="Directory with .qasm files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output pickle file path")
    parser.add_argument("--no-labels", action="store_true",
                        help="Build without labels (Stage 2)")
    args = parser.parse_args()

    circuit_dir = Path(args.circuit_dir)

    # Load labels
    logger.info(f"Loading labels from {args.labels}")
    with open(args.labels) as f:
        labels = json.load(f)
    logger.info(f"Loaded {len(labels)} label entries")

    # Build dataset
    logger.info("Building dataset...")
    t0 = time.time()
    dataset = build_dataset_from_labels(
        labels, circuit_dir,
        include_labels=not args.no_labels,
    )
    elapsed = time.time() - t0
    logger.info(f"Built {len(dataset)} samples in {elapsed:.1f}s")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    logger.info(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
