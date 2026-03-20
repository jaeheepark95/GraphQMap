"""Batch label generation for GraphQMap.

For each (circuit, backend) pair, generates the best initial layout
via candidate search + SABRE transpile.

Usage:
    python scripts/generate_labels.py --circuit-dir data/circuits/mqt_bench \
        --output-dir data/labels --backends manila jakarta guadalupe montreal kolkata mumbai
    python scripts/generate_labels.py --circuit-dir data/circuits/mqt_bench \
        --output-dir data/labels --backends manila --max-circuits 100 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import QuantumCircuit

from data.hardware_graph import get_backend
from data.label_generation import generate_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def process_single(
    qasm_path: str,
    backend_name: str,
    num_sabre_seeds: int,
    num_random: int,
) -> dict | None:
    """Process a single (circuit, backend) pair. Runs in worker process."""
    try:
        circuit = QuantumCircuit.from_qasm_file(qasm_path)
        backend = get_backend(backend_name)

        # Skip if circuit has more qubits than backend
        if circuit.num_qubits > backend.target.num_qubits:
            return None

        # Skip circuits with no 2-qubit gates (trivial mapping)
        has_2q = any(
            len(inst.qubits) == 2
            for inst in circuit.data
        )
        if not has_2q:
            return None

        result = generate_label(
            circuit, backend,
            num_sabre_seeds=num_sabre_seeds,
            num_random=num_random,
            rng_seed=42,
        )

        return {
            "circuit": Path(qasm_path).stem,
            "backend": backend_name,
            "layout": result.layout,
            "swap_count": result.swap_count,
            "depth": result.depth,
            "num_candidates": result.num_candidates,
            "num_qubits": circuit.num_qubits,
        }
    except Exception as e:
        logger.warning(f"Failed {qasm_path} on {backend_name}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labels for GraphQMap")
    parser.add_argument("--circuit-dir", type=str, required=True,
                        help="Directory with .qasm files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for labels")
    parser.add_argument("--backends", nargs="+", required=True,
                        help="Backend names (e.g. manila jakarta)")
    parser.add_argument("--max-circuits", type=int, default=None,
                        help="Max circuits to process (for testing)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--sabre-seeds", type=int, default=10,
                        help="Number of SABRE seeds")
    parser.add_argument("--random-layouts", type=int, default=8,
                        help="Number of random layouts")
    args = parser.parse_args()

    circuit_dir = Path(args.circuit_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qasm_files = sorted(circuit_dir.glob("*.qasm"))
    if args.max_circuits:
        qasm_files = qasm_files[: args.max_circuits]

    logger.info(f"Circuits: {len(qasm_files)}, Backends: {args.backends}")
    logger.info(f"Workers: {args.workers}, Candidates per pair: {args.sabre_seeds + args.random_layouts + 2}")

    total_pairs = len(qasm_files) * len(args.backends)
    results: list[dict] = []
    t0 = time.time()

    if args.workers <= 1:
        # Sequential processing
        for i, qasm_path in enumerate(qasm_files):
            for backend_name in args.backends:
                result = process_single(
                    str(qasm_path), backend_name,
                    args.sabre_seeds, args.random_layouts,
                )
                if result:
                    results.append(result)

                done = len(results)
                if done % 50 == 0 and done > 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    logger.info(f"Progress: {done} labels generated ({rate:.1f}/s)")
    else:
        # Parallel processing
        tasks = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for qasm_path in qasm_files:
                for backend_name in args.backends:
                    future = executor.submit(
                        process_single,
                        str(qasm_path), backend_name,
                        args.sabre_seeds, args.random_layouts,
                    )
                    tasks.append(future)

            for i, future in enumerate(as_completed(tasks)):
                result = future.result()
                if result:
                    results.append(result)
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    logger.info(f"Progress: {i+1}/{total_pairs} ({len(results)} labels, {(i+1)/elapsed:.1f}/s)")

    elapsed = time.time() - t0
    logger.info(f"Done: {len(results)} labels in {elapsed:.1f}s")

    # Save results
    output_file = output_dir / "labels.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Labels saved to {output_file}")

    # Per-backend summary
    from collections import Counter
    backend_counts = Counter(r["backend"] for r in results)
    for b, c in sorted(backend_counts.items()):
        avg_swap = sum(r["swap_count"] for r in results if r["backend"] == b) / c
        logger.info(f"  {b}: {c} labels, avg SWAP={avg_swap:.1f}")


if __name__ == "__main__":
    main()
