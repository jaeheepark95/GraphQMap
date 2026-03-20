"""Build GraphQMap training dataset from labels and circuits.

Constructs PyG graph objects, computes dataset statistics,
and saves everything as a pickle file for fast loading.

Usage:
    python scripts/build_dataset.py --labels data/labels/labels.json \
        --circuit-dir data/circuits/mqt_bench --output data/dataset_stage1.pkl
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

import numpy as np
import torch
from qiskit import QuantumCircuit

from data.circuit_graph import build_circuit_graph, extract_circuit_features
from data.dataset import MappingDataset, MappingSample
from data.hardware_graph import build_hardware_graph, get_backend, precompute_error_distance
from data.label_generation import layout_to_permutation_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_global_summary_stats(
    circuit_dir: Path,
    qasm_stems: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute dataset-level mean/std for global summary features.

    Args:
        circuit_dir: Directory with .qasm files.
        qasm_stems: List of circuit stems to include.

    Returns:
        Dict mapping feature name to (mean, std).
    """
    summaries = []

    for stem in qasm_stems:
        qasm_path = circuit_dir / f"{stem}.qasm"
        if not qasm_path.exists():
            continue
        try:
            circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
            feats = extract_circuit_features(circuit)
            summaries.append(feats["global_summary"].numpy())
        except Exception:
            continue

    if not summaries:
        return {
            "total_qubits": (0.0, 1.0),
            "total_2q_gates": (0.0, 1.0),
            "total_depth": (0.0, 1.0),
            "gate_density": (0.0, 1.0),
        }

    arr = np.stack(summaries)  # (N, 4)
    names = ["total_qubits", "total_2q_gates", "total_depth", "gate_density"]
    stats = {}
    for i, name in enumerate(names):
        mean = float(arr[:, i].mean())
        std = float(arr[:, i].std())
        stats[name] = (mean, max(std, 1e-8))

    return stats


def build_dataset_from_labels(
    labels: list[dict],
    circuit_dir: Path,
    summary_stats: dict[str, tuple[float, float]],
    include_labels: bool = True,
) -> MappingDataset:
    """Build MappingDataset from label entries.

    Args:
        labels: List of label dicts from generate_labels.py.
        circuit_dir: Directory with .qasm files.
        summary_stats: Global summary normalization stats.
        include_labels: Whether to include label matrices (Stage 1).

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

        # Circuit graph with normalized global summary
        feats = extract_circuit_features(circuit)
        global_summary = feats["global_summary"]

        # Normalize global summary
        names = ["total_qubits", "total_2q_gates", "total_depth", "gate_density"]
        normalized_summary = []
        for j, name in enumerate(names):
            mean, std = summary_stats[name]
            normalized_summary.append((global_summary[j].item() - mean) / std)
        norm_summary = torch.tensor(normalized_summary, dtype=torch.float32)

        circuit_graph = build_circuit_graph(circuit, global_summary=norm_summary)

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

    # Compute global summary stats
    logger.info("Computing global summary statistics...")
    qasm_stems = list(set(entry["circuit"] for entry in labels))
    summary_stats = compute_global_summary_stats(circuit_dir, qasm_stems)
    for name, (mean, std) in summary_stats.items():
        logger.info(f"  {name}: mean={mean:.4f}, std={std:.4f}")

    # Build dataset
    logger.info("Building dataset...")
    t0 = time.time()
    dataset = build_dataset_from_labels(
        labels, circuit_dir, summary_stats,
        include_labels=not args.no_labels,
    )
    elapsed = time.time() - t0
    logger.info(f"Dataset built: {len(dataset)} samples in {elapsed:.1f}s")

    # Per-backend summary
    for backend in dataset.backend_names:
        count = len(dataset.indices_for_backend(backend))
        logger.info(f"  {backend}: {count} samples")

    # Precompute error distances for all backends
    logger.info("Precomputing error distance matrices...")
    error_distances = {}
    for backend_name in dataset.backend_names:
        backend = get_backend(backend_name)
        error_distances[backend_name] = precompute_error_distance(backend)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "dataset": dataset,
        "summary_stats": summary_stats,
        "error_distances": error_distances,
    }

    with open(output_path, "wb") as f:
        pickle.dump(save_data, f)

    logger.info(f"Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
