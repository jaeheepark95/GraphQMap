"""Baseline layout methods for comparison with GraphQMap.

Baselines:
  - SabreLayout: Qiskit default SABRE-based layout
  - DenseLayout: Map to densely connected subgraph
  - TrivialLayout: Identity mapping (qubit i → physical i)
  - RandomLayout: Uniformly random physical qubit assignment
  - NaiveMultiProgramming: Independent layout per circuit (no conflict avoidance)
"""

from __future__ import annotations

import random
from typing import Any

from qiskit import QuantumCircuit, transpile

from evaluation.benchmark import execute_on_simulators
from evaluation.pst import create_ideal_simulator, create_noisy_simulator, measure_pst
from evaluation.transpiler import transpile_with_timing


def sabre_layout(
    circuit: QuantumCircuit,
    backend: Any,
    seed: int = 0,
    optimization_level: int = 3,
) -> list[int]:
    """Get SabreLayout initial layout."""
    compiled = transpile(
        circuit, backend=backend,
        layout_method="sabre", routing_method="sabre",
        optimization_level=optimization_level, seed_transpiler=seed,
    )
    return _extract_layout(compiled, circuit.num_qubits)


def dense_layout(
    circuit: QuantumCircuit,
    backend: Any,
    seed: int = 0,
    optimization_level: int = 3,
) -> list[int]:
    """Get DenseLayout initial layout."""
    compiled = transpile(
        circuit, backend=backend,
        layout_method="dense", routing_method="sabre",
        optimization_level=optimization_level, seed_transpiler=seed,
    )
    return _extract_layout(compiled, circuit.num_qubits)


def trivial_layout(
    circuit: QuantumCircuit,
    backend: Any,
) -> list[int]:
    """Get TrivialLayout (identity mapping)."""
    return list(range(circuit.num_qubits))


def random_layout(
    circuit: QuantumCircuit,
    backend: Any,
    seed: int = 0,
) -> list[int]:
    """Get a uniformly random layout."""
    rng = random.Random(seed)
    physical = list(range(backend.target.num_qubits))
    rng.shuffle(physical)
    return physical[: circuit.num_qubits]


def naive_multi_programming_layout(
    circuits: list[QuantumCircuit],
    backend: Any,
    seed: int = 0,
) -> list[int]:
    """Naive baseline for multi-programming: independent layout per circuit.

    Each circuit gets its own SabreLayout independently.
    Conflicts are resolved by greedy first-come assignment.
    """
    num_physical = backend.target.num_qubits
    used_physical: set[int] = set()
    combined_layout: list[int] = []

    for i, circuit in enumerate(circuits):
        compiled = transpile(
            circuit, backend=backend,
            layout_method="sabre", routing_method="sabre",
            optimization_level=3, seed_transpiler=seed + i,
        )
        layout = _extract_layout(compiled, circuit.num_qubits)

        resolved: list[int] = []
        available = [q for q in range(num_physical) if q not in used_physical]

        for phys in layout:
            if phys not in used_physical:
                resolved.append(phys)
                used_physical.add(phys)
            else:
                if available:
                    alt = available.pop(0)
                    resolved.append(alt)
                    used_physical.add(alt)
                else:
                    resolved.append(phys)

        combined_layout.extend(resolved)

    return combined_layout


def _extract_layout(
    compiled: QuantumCircuit,
    num_logical: int,
) -> list[int]:
    """Extract initial layout from transpiled circuit."""
    if compiled.layout is None:
        return list(range(num_logical))

    layout_dict: dict[int, int] = {}
    for virtual_qubit, physical_qubit in compiled.layout.initial_layout.get_virtual_bits().items():
        if hasattr(virtual_qubit, "_register") and virtual_qubit._register is not None:
            if virtual_qubit._register.name == "ancilla":
                continue
        if hasattr(virtual_qubit, "_index") and virtual_qubit._index is not None:
            idx = virtual_qubit._index
            if idx < num_logical:
                layout_dict[idx] = physical_qubit

    if len(layout_dict) != num_logical:
        return list(range(num_logical))

    return [layout_dict[i] for i in range(num_logical)]


def evaluate_baseline(
    circuit: QuantumCircuit,
    backend: Any,
    method: str = "sabre",
    routing_method: str = "sabre",
    shots: int = 8192,
    seed: int = 0,
    optimization_level: int = 3,
) -> dict[str, Any]:
    """Evaluate a baseline method and return PST + metrics.

    Supports both layout-only baselines (sabre, dense, trivial, random)
    and layout×routing combinations via the custom transpiler.

    Args:
        circuit: Quantum circuit.
        backend: FakeBackendV2.
        method: Layout method ('sabre', 'dense', 'trivial', 'random', 'noise_adaptive').
        routing_method: Routing method ('sabre', 'nassc').
        shots: Simulation shots.
        seed: Random seed.
        optimization_level: Qiskit transpiler optimization level (0-3).

    Returns:
        Dict with pst, swap_count, depth, method, layout.
    """
    if method in ("noise_adaptive",) or routing_method == "nassc":
        # Use custom transpiler for advanced combinations
        tc, metadata = transpile_with_timing(
            circuit, backend,
            layout_method=method,
            routing_method=routing_method,
            seed=seed,
        )
        ideal_sim = create_ideal_simulator(backend)
        noisy_sim = create_noisy_simulator(backend)
        avg_pst, depth, _ = execute_on_simulators(
            tc, ideal_sim, noisy_sim, shots=shots,
        )
        return {
            "pst": avg_pst,
            "swap_count": metadata["map_cx"],
            "depth": depth,
            "method": f"{method}+{routing_method}",
            "layout_time": metadata["layout_time"],
            "total_time": metadata["total_time"],
        }
    else:
        # Use simple layout + measure_pst for basic baselines
        layout_fns = {
            "sabre": lambda: sabre_layout(circuit, backend, seed, optimization_level),
            "dense": lambda: dense_layout(circuit, backend, seed, optimization_level),
            "trivial": lambda: trivial_layout(circuit, backend),
            "random": lambda: random_layout(circuit, backend, seed),
        }

        if method not in layout_fns:
            raise ValueError(f"Unknown method '{method}'. Available: {list(layout_fns.keys())}")

        layout = layout_fns[method]()
        result = measure_pst(
            circuit, backend, layout, shots=shots,
            optimization_level=optimization_level,
        )
        result["method"] = method
        result["layout"] = layout
        return result
