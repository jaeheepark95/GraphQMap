"""Self-generated label pipeline for GraphQMap.

For each (circuit, backend) pair, generates candidate initial layouts,
transpiles with SABRE routing, and selects the best layout based on
minimum SWAP count (tiebreaker: minimum depth).

Candidate sources (20 total):
  - SabreLayout with 10 different random seeds
  - DenseLayout (1)
  - TrivialLayout (1)
  - Fully random layouts (8)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile


@dataclass
class LabelResult:
    """Result of label generation for a single (circuit, backend) pair."""

    layout: list[int]           # best initial layout (logical -> physical)
    swap_count: int             # estimated SWAP count of best layout
    depth: int                  # compiled circuit depth of best layout
    num_candidates: int         # total candidates evaluated
    all_results: list[dict]     # all candidate results for analysis


def count_additional_2q_gates(
    circuit: QuantumCircuit,
    compiled: QuantumCircuit,
    backend: Any = None,
) -> int:
    """Count additional 2-qubit gates introduced by routing.

    In Qiskit >= 1.0, SWAP gates are decomposed into native 2-qubit gates
    during transpilation (3 per SWAP for cx/ecr/cz). We count the difference
    between compiled and original 2Q gate counts as a routing overhead proxy.

    Detects the native 2-qubit gate from the backend, or falls back to
    checking cx/ecr/cz in the compiled circuit.

    Args:
        circuit: Original circuit before transpilation.
        compiled: Transpiled circuit.
        backend: Optional FakeBackendV2 to detect native 2Q gate.

    Returns:
        Number of additional 2-qubit gates (>= 0).
    """
    gate_name = None
    if backend is not None:
        from data.hardware_graph import _get_two_qubit_gate_name
        try:
            gate_name = _get_two_qubit_gate_name(backend)
        except ValueError:
            pass

    if gate_name is None:
        # Fallback: check which 2Q gate appears in the compiled circuit
        compiled_ops = compiled.count_ops()
        for name in ("cx", "ecr", "cz"):
            if compiled_ops.get(name, 0) > 0:
                gate_name = name
                break
        if gate_name is None:
            return 0

    original_count = circuit.count_ops().get(gate_name, 0)
    compiled_count = compiled.count_ops().get(gate_name, 0)
    return max(0, compiled_count - original_count)


def generate_candidate_layouts(
    circuit: QuantumCircuit,
    backend: Any,
    num_sabre_seeds: int = 10,
    num_random: int = 8,
    rng_seed: int | None = None,
) -> list[tuple[str, list[int]]]:
    """Generate candidate initial layouts for a (circuit, backend) pair.

    Args:
        circuit: Quantum circuit.
        backend: FakeBackendV2 instance.
        num_sabre_seeds: Number of SabreLayout candidates with different seeds.
        num_random: Number of fully random layout candidates.
        rng_seed: Random seed for reproducibility.

    Returns:
        List of (source_name, layout) tuples.
    """
    num_logical = circuit.num_qubits
    num_physical = backend.target.num_qubits
    rng = random.Random(rng_seed)

    candidates: list[tuple[str, list[int]]] = []

    # SabreLayout with different seeds
    for seed in range(num_sabre_seeds):
        compiled = transpile(
            circuit, backend=backend,
            layout_method="sabre", routing_method="sabre",
            optimization_level=1, seed_transpiler=seed,
        )
        layout = _extract_initial_layout(compiled, num_logical)
        if layout is not None:
            candidates.append((f"sabre_seed_{seed}", layout))

    # DenseLayout
    try:
        compiled = transpile(
            circuit, backend=backend,
            layout_method="dense", routing_method="sabre",
            optimization_level=1, seed_transpiler=0,
        )
        layout = _extract_initial_layout(compiled, num_logical)
        if layout is not None:
            candidates.append(("dense", layout))
    except Exception:
        pass

    # TrivialLayout
    try:
        compiled = transpile(
            circuit, backend=backend,
            layout_method="trivial", routing_method="sabre",
            optimization_level=1, seed_transpiler=0,
        )
        layout = _extract_initial_layout(compiled, num_logical)
        if layout is not None:
            candidates.append(("trivial", layout))
    except Exception:
        pass

    # Fully random layouts
    physical_qubits = list(range(num_physical))
    for i in range(num_random):
        perm = physical_qubits.copy()
        rng.shuffle(perm)
        layout = perm[:num_logical]
        candidates.append((f"random_{i}", layout))

    return candidates


def _extract_initial_layout(
    compiled: QuantumCircuit,
    num_logical: int,
) -> list[int] | None:
    """Extract initial layout from a transpiled circuit.

    Args:
        compiled: Transpiled QuantumCircuit with layout info.
        num_logical: Number of logical qubits.

    Returns:
        List mapping logical qubit index -> physical qubit index, or None.
    """
    if compiled.layout is None:
        return None

    initial_layout = compiled.layout.initial_layout
    # initial_layout maps Qubit objects -> physical indices
    # We need logical qubit index -> physical qubit index
    layout_dict: dict[int, int] = {}
    for virtual_qubit, physical_qubit in initial_layout.get_virtual_bits().items():
        # Filter out ancilla qubits
        if hasattr(virtual_qubit, "_register") and virtual_qubit._register is not None:
            if virtual_qubit._register.name == "ancilla":
                continue
        if hasattr(virtual_qubit, "_index") and virtual_qubit._index is not None:
            idx = virtual_qubit._index
            if idx < num_logical:
                layout_dict[idx] = physical_qubit

    if len(layout_dict) != num_logical:
        return None

    return [layout_dict[i] for i in range(num_logical)]


def evaluate_layout(
    circuit: QuantumCircuit,
    backend: Any,
    layout: list[int],
    seed: int = 0,
) -> dict:
    """Evaluate a single layout by transpiling and measuring quality.

    Args:
        circuit: Original quantum circuit.
        backend: FakeBackendV2 instance.
        layout: Initial layout (logical -> physical mapping).
        seed: Transpiler seed.

    Returns:
        Dict with swap_count, depth, compiled circuit.
    """
    compiled = transpile(
        circuit, backend=backend,
        initial_layout=layout, routing_method="sabre",
        optimization_level=1, seed_transpiler=seed,
    )
    swap_count = count_additional_2q_gates(circuit, compiled, backend=backend) // 3
    depth = compiled.depth()

    return {
        "layout": layout,
        "swap_count": swap_count,
        "depth": depth,
    }


def generate_label(
    circuit: QuantumCircuit,
    backend: Any,
    num_sabre_seeds: int = 10,
    num_random: int = 8,
    rng_seed: int | None = None,
) -> LabelResult:
    """Generate the best layout label for a (circuit, backend) pair.

    Selection: primary = minimum SWAP count, tiebreaker = minimum depth.

    Args:
        circuit: Quantum circuit.
        backend: FakeBackendV2 instance.
        num_sabre_seeds: Number of SabreLayout seeds.
        num_random: Number of random layout candidates.
        rng_seed: Random seed for reproducibility.

    Returns:
        LabelResult with the best layout and evaluation details.
    """
    candidates = generate_candidate_layouts(
        circuit, backend,
        num_sabre_seeds=num_sabre_seeds,
        num_random=num_random,
        rng_seed=rng_seed,
    )

    all_results = []
    for source_name, layout in candidates:
        result = evaluate_layout(circuit, backend, layout)
        result["source"] = source_name
        all_results.append(result)

    # Sort: primary = swap_count (ascending), secondary = depth (ascending)
    all_results.sort(key=lambda r: (r["swap_count"], r["depth"]))

    best = all_results[0]
    return LabelResult(
        layout=best["layout"],
        swap_count=best["swap_count"],
        depth=best["depth"],
        num_candidates=len(all_results),
        all_results=all_results,
    )


def layout_to_permutation_matrix(
    layout: list[int],
    num_physical: int,
) -> np.ndarray:
    """Convert a layout to a binary permutation matrix Y (h×h).

    Y[i, layout[i]] = 1 for logical qubits.
    Dummy rows are assigned to remaining physical qubits to complete
    the permutation.

    Args:
        layout: List of physical qubit assignments for each logical qubit.
        num_physical: Total number of physical qubits (h).

    Returns:
        Y: (h, h) binary permutation matrix.
    """
    num_logical = len(layout)
    Y = np.zeros((num_physical, num_physical), dtype=np.float32)

    # Assign logical qubits
    assigned_physical = set()
    for logical_idx, physical_idx in enumerate(layout):
        Y[logical_idx, physical_idx] = 1.0
        assigned_physical.add(physical_idx)

    # Assign dummy rows to remaining physical qubits
    remaining = [p for p in range(num_physical) if p not in assigned_physical]
    for i, physical_idx in enumerate(remaining):
        dummy_row = num_logical + i
        Y[dummy_row, physical_idx] = 1.0

    return Y
