"""QUEKO dataset label loader for GraphQMap.

QUEKO provides circuits with known optimal (zero-SWAP) initial layouts (τ⁻¹).
These layouts are used directly as ground truth labels — no self-generation needed.

Expected QUEKO directory structure:
  queko_dir/
    ├── <circuit_name>.qasm    # circuit file
    └── <circuit_name>.layout  # optimal layout (one physical qubit per line)
"""

from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit


def load_queko_layout(layout_path: str | Path) -> list[int]:
    """Load an optimal layout from a QUEKO .layout file.

    The file contains one physical qubit index per line,
    where line i specifies the physical qubit for logical qubit i.

    Args:
        layout_path: Path to the .layout file.

    Returns:
        Layout as a list mapping logical qubit index -> physical qubit index.
    """
    path = Path(layout_path)
    if not path.exists():
        raise FileNotFoundError(f"Layout file not found: {path}")

    layout = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                layout.append(int(line))
    return layout


def load_queko_pair(
    qasm_path: str | Path,
    layout_path: str | Path,
) -> tuple[QuantumCircuit, list[int]]:
    """Load a QUEKO circuit and its optimal layout.

    Args:
        qasm_path: Path to the .qasm circuit file.
        layout_path: Path to the .layout file.

    Returns:
        Tuple of (circuit, optimal_layout).
    """
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    layout = load_queko_layout(layout_path)

    if len(layout) != circuit.num_qubits:
        raise ValueError(
            f"Layout length ({len(layout)}) != circuit qubits ({circuit.num_qubits}) "
            f"for {qasm_path}"
        )

    return circuit, layout


def discover_queko_pairs(
    queko_dir: str | Path,
) -> list[tuple[Path, Path]]:
    """Discover all (qasm, layout) file pairs in a QUEKO directory.

    Args:
        queko_dir: Root directory containing QUEKO files.

    Returns:
        List of (qasm_path, layout_path) tuples.
    """
    queko_dir = Path(queko_dir)
    pairs = []

    for qasm_file in sorted(queko_dir.rglob("*.qasm")):
        layout_file = qasm_file.with_suffix(".layout")
        if layout_file.exists():
            pairs.append((qasm_file, layout_file))

    return pairs
