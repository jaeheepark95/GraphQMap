"""Preprocess all QASM circuits into cached PyG .pt files.

Parses each .qasm file, extracts circuit features, builds the PyG graph,
and saves everything needed for training into a single .pt file per circuit.

This avoids repeated QASM parsing during training (the main OOM bottleneck
when loading thousands of circuits at once).

Usage:
    python scripts/preprocess_circuits.py --data-root data/circuits

Cache layout:
    data/circuits/cache/{source}/{filename}.pt
    Each .pt contains:
        - circuit_graph: PyG Data (z-score normalized node features)
        - circuit_edge_pairs: list of (i, j) tuples
        - qubit_importance: numpy array (l,)
        - num_logical: int
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.circuit_graph import build_circuit_graph, extract_circuit_features, load_circuit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def preprocess_one(qasm_path: Path, cache_path: Path) -> bool:
    """Preprocess a single QASM file and save as .pt cache.

    Returns True on success, False on failure.
    """
    try:
        circuit = load_circuit(qasm_path)
    except Exception as e:
        logger.debug("Failed to load %s: %s", qasm_path.name, e)
        return False

    if circuit.num_qubits < 2:
        return False

    try:
        circuit_graph = build_circuit_graph(circuit)
        feats = extract_circuit_features(circuit)
    except Exception as e:
        logger.debug("Failed to build graph for %s: %s", qasm_path.name, e)
        return False

    # Qubit importance: 2Q gate count per logical qubit, normalized
    qi = feats["node_features"][:, 1].numpy()  # two_qubit_gate_count
    qi_sum = qi.sum()
    qubit_importance = qi / qi_sum if qi_sum > 0 else np.ones(circuit.num_qubits) / circuit.num_qubits

    # Raw interaction counts per edge (for loss computation, before normalization)
    edge_weights = feats["edge_features"][:, 0].tolist() if len(feats["edge_list"]) > 0 else []

    cache_data = {
        "circuit_graph": circuit_graph,
        "circuit_edge_pairs": feats["edge_list"],
        "circuit_edge_weights": edge_weights,
        "qubit_importance": qubit_importance,
        "num_logical": circuit.num_qubits,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_data, cache_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess QASM circuits to cached .pt files")
    parser.add_argument("--data-root", type=str, default="data/circuits", help="Root of data/circuits/")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    qasm_root = data_root / "qasm"
    cache_root = data_root / "cache"

    if not qasm_root.exists():
        logger.error("QASM directory not found: %s", qasm_root)
        return

    t0 = time.time()
    total, cached, skipped, failed = 0, 0, 0, 0

    for source_dir in sorted(qasm_root.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        qasm_files = sorted(source_dir.glob("*.qasm"))
        src_cached, src_failed = 0, 0

        for qasm_path in qasm_files:
            total += 1
            cache_path = cache_root / source / (qasm_path.stem + ".pt")

            if cache_path.exists() and not args.force:
                skipped += 1
                continue

            if preprocess_one(qasm_path, cache_path):
                cached += 1
                src_cached += 1
            else:
                failed += 1
                src_failed += 1

        logger.info(
            "%s: %d files, %d cached, %d failed",
            source, len(qasm_files), src_cached, src_failed,
        )

    # Build metadata index: {source/filename.qasm: num_logical}
    # This avoids reading .pt files just to get num_logical during dataset loading
    metadata = {}
    for source_dir in sorted(cache_root.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for pt_file in sorted(source_dir.glob("*.pt")):
            try:
                data = torch.load(pt_file, weights_only=False, map_location="cpu")
                qasm_name = pt_file.stem + ".qasm"
                metadata[f"{source}/{qasm_name}"] = data["num_logical"]
            except Exception:
                pass

    meta_path = cache_root / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    logger.info("Metadata index written: %d entries -> %s", len(metadata), meta_path)

    elapsed = time.time() - t0
    logger.info(
        "Done in %.1fs. Total: %d, New cached: %d, Already cached: %d, Failed: %d",
        elapsed, total, cached, skipped, failed,
    )


if __name__ == "__main__":
    main()
