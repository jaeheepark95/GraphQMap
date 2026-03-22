"""Normalize all QASM circuits to a standard basis gate set.

Transpiles every .qasm file under data/circuits/qasm/ to the Qiskit standard
basis gates {cx, id, rz, sx, x} using optimization_level=0 (pure decomposition,
no gate optimization/merging). This ensures all datasets use a consistent gate
representation so that circuit_graph.py feature extraction captures all
multi-qubit interactions correctly.

Each file is processed in a subprocess to prevent OOM from large circuits
from killing the entire run.

Usage:
    python scripts/normalize_gates.py [--data-root data/circuits] [--dry-run]
    python scripts/normalize_gates.py --source mqt_bench   # single dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASIS_GATES = ["cx", "id", "rz", "sx", "x"]

# Subprocess worker script — processes a single file
WORKER_SCRIPT = '''
import sys, json, logging
logging.getLogger("qiskit").setLevel(logging.ERROR)
from qiskit import QuantumCircuit, qasm2, transpile

path = sys.argv[1]
dry_run = sys.argv[2] == "true"
BASIS = ["cx", "id", "rz", "sx", "x"]
SKIP = {"measure", "barrier", "reset"}

try:
    circuit = QuantumCircuit.from_qasm_file(path)
except Exception as e:
    print(json.dumps({"status": "load_failed", "error": str(e)}))
    sys.exit(0)

orig_q = circuit.num_qubits
orig_g = circuit.size()
gate_names = {inst.operation.name for inst in circuit.data}
non_basis = gate_names - set(BASIS) - SKIP

if not non_basis:
    print(json.dumps({"status": "already_basis", "qubits": orig_q}))
    sys.exit(0)

try:
    normalized = transpile(circuit, basis_gates=BASIS, optimization_level=0)
except Exception as e:
    print(json.dumps({"status": "transpile_failed", "error": str(e)}))
    sys.exit(0)

new_q = normalized.num_qubits
new_g = normalized.size()

if new_q != orig_q:
    print(json.dumps({"status": "qubit_mismatch", "orig": orig_q, "new": new_q}))
    sys.exit(0)

if not dry_run:
    qasm2.dump(normalized, path)

print(json.dumps({"status": "converted", "qubits": orig_q, "orig_gates": orig_g, "new_gates": new_g, "non_basis": sorted(non_basis)}))
'''


def process_one(qasm_path: Path, dry_run: bool, timeout: int = 300) -> dict:
    """Process a single file in a subprocess with timeout."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKER_SCRIPT, str(qasm_path), str(dry_run).lower()],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return {"status": "subprocess_error", "error": result.stderr[:200]}
        output = result.stdout.strip()
        if not output:
            return {"status": "no_output"}
        return json.loads(output)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize QASM circuits to basis gates")
    parser.add_argument("--data-root", type=str, default="data/circuits")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    parser.add_argument("--source", type=str, default=None, help="Process only this dataset")
    parser.add_argument("--timeout", type=int, default=300, help="Per-file timeout in seconds")
    args = parser.parse_args()

    qasm_root = Path(args.data_root) / "qasm"
    if not qasm_root.exists():
        logger.error("QASM directory not found: %s", qasm_root)
        return

    t0 = time.time()

    for source_dir in sorted(qasm_root.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        if args.source and source != args.source:
            continue

        qasm_files = sorted(source_dir.glob("*.qasm"))
        converted, already_basis, failed = 0, 0, 0
        failed_files = []

        for i, qasm_path in enumerate(qasm_files):
            result = process_one(qasm_path, args.dry_run, timeout=args.timeout)
            status = result["status"]
            if status == "converted":
                converted += 1
            elif status == "already_basis":
                already_basis += 1
            else:
                failed += 1
                failed_files.append((qasm_path.name, status))

            if (i + 1) % 100 == 0:
                logger.info("  %s: %d/%d processed...", source, i + 1, len(qasm_files))

        logger.info(
            "%s: %d files — %d converted, %d already basis, %d failed",
            source, len(qasm_files), converted, already_basis, failed,
        )
        for name, status in failed_files:
            logger.warning("  FAILED: %s (%s)", name, status)

    elapsed = time.time() - t0
    prefix = "[DRY RUN] " if args.dry_run else ""
    logger.info("%sDone in %.1fs", prefix, elapsed)


if __name__ == "__main__":
    main()
