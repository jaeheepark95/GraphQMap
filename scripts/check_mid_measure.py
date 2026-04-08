"""Scan all QASM circuits for mid-circuit measurements.

A mid-circuit measurement is defined as: any non-measure, non-barrier
operation on a qubit that occurs AFTER that qubit has been measured.
End-of-circuit measurements (the standard pattern) are not flagged.

Outputs:
- Per-dataset counts
- List of offending files
- JSON log at data/circuits/splits/mid_measure_log.json
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from qiskit import QuantumCircuit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QASM_ROOT = Path("data/circuits/qasm")
LOG_PATH = Path("data/circuits/splits/mid_measure_log.json")


def has_mid_measurement(circuit: QuantumCircuit) -> tuple[bool, int]:
    """Return (has_mid_measure, num_offending_qubits).

    A qubit is offending if it has any non-measure/barrier op after a measure.
    """
    measured: set[int] = set()
    offending: set[int] = set()
    qubit_index = {q: i for i, q in enumerate(circuit.qubits)}

    for instr in circuit.data:
        name = instr.operation.name
        qubits = [qubit_index[q] for q in instr.qubits]
        if name == "measure":
            measured.update(qubits)
        elif name == "barrier":
            continue
        else:
            for q in qubits:
                if q in measured:
                    offending.add(q)
    return (len(offending) > 0, len(offending))


def main() -> None:
    if not QASM_ROOT.exists():
        logger.error("QASM root not found: %s", QASM_ROOT)
        return

    results: dict[str, dict] = {}
    total_files = 0
    total_offending = 0

    for source_dir in sorted(QASM_ROOT.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        files = sorted(source_dir.glob("*.qasm"))
        offenders: list[dict] = []
        load_failed: list[str] = []

        for path in files:
            try:
                qc = QuantumCircuit.from_qasm_file(str(path))
            except Exception as e:
                load_failed.append(f"{path.name}: {e}")
                continue
            mid, n_off = has_mid_measurement(qc)
            if mid:
                offenders.append({
                    "file": path.name,
                    "num_qubits": qc.num_qubits,
                    "offending_qubits": n_off,
                })

        results[source] = {
            "total": len(files),
            "mid_measure_count": len(offenders),
            "load_failed": len(load_failed),
            "offenders": offenders,
            "load_failed_files": load_failed,
        }
        total_files += len(files)
        total_offending += len(offenders)
        logger.info(
            "%s: %d files — %d mid-measure, %d load_failed",
            source, len(files), len(offenders), len(load_failed),
        )
        for o in offenders[:5]:
            logger.info("  %s (q=%d, offending=%d)",
                        o["file"], o["num_qubits"], o["offending_qubits"])
        if len(offenders) > 5:
            logger.info("  ... and %d more", len(offenders) - 5)

    summary = {
        "total_files": total_files,
        "total_mid_measure": total_offending,
        "by_source": results,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("TOTAL: %d files scanned, %d mid-measure detected",
                total_files, total_offending)
    logger.info("Log written to %s", LOG_PATH)


if __name__ == "__main__":
    main()
