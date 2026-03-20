"""PST (Probability of Successful Trials) measurement for GraphQMap.

Pipeline:
  Model → P matrix → Hungarian → discrete layout
  → qiskit.transpile(initial_layout, routing_method='sabre')
  → noise simulation on FakeBackendV2
  → PST computation (Hellinger fidelity between ideal and noisy distributions)
"""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, hellinger_fidelity
from qiskit_aer import AerSimulator


def compute_ideal_distribution(circuit: QuantumCircuit) -> dict[str, float]:
    """Compute the ideal output probability distribution via statevector simulation.

    Args:
        circuit: Quantum circuit (without measurements).

    Returns:
        Dict mapping bitstrings to probabilities.
    """
    # Remove measurements if present
    bare = circuit.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(bare)
    probs = sv.probabilities_dict()
    # Filter near-zero entries and convert numpy types to float
    return {k: float(v) for k, v in probs.items() if v > 1e-10}


def run_noisy_simulation(
    compiled_circuit: QuantumCircuit,
    backend: Any,
    shots: int = 4096,
) -> dict[str, float]:
    """Run noisy simulation on a FakeBackendV2 using AerSimulator.

    Args:
        compiled_circuit: Transpiled circuit (must include measurements).
        backend: FakeBackendV2 instance.
        shots: Number of shots.

    Returns:
        Dict mapping bitstrings to probabilities.
    """
    sim = AerSimulator.from_backend(backend)
    result = sim.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def _count_2q_gates(circuit: QuantumCircuit, backend: Any = None) -> int:
    """Count native 2-qubit gates in a circuit, auto-detecting gate type."""
    ops = circuit.count_ops()
    # Try backend-specific gate first
    if backend is not None:
        from data.hardware_graph import _get_two_qubit_gate_name
        try:
            gate_name = _get_two_qubit_gate_name(backend)
            return ops.get(gate_name, 0)
        except ValueError:
            pass
    # Fallback: check common 2Q gates
    for name in ("cx", "ecr", "cz"):
        if ops.get(name, 0) > 0:
            return ops[name]
    return 0


def measure_pst(
    circuit: QuantumCircuit,
    backend: Any,
    layout: list[int] | dict[int, int],
    shots: int = 4096,
    seed_transpiler: int = 0,
    optimization_level: int = 3,
) -> dict[str, Any]:
    """Measure PST for a given circuit + layout + backend.

    Full pipeline: transpile → noise sim → Hellinger fidelity.

    Args:
        circuit: Original quantum circuit (with or without measurements).
        backend: FakeBackendV2 instance.
        layout: Initial layout (list or dict mapping logical → physical).
        shots: Number of simulation shots.
        seed_transpiler: Transpiler random seed.
        optimization_level: Qiskit transpiler optimization level (0-3).

    Returns:
        Dict with:
        - pst: Hellinger fidelity (float in [0, 1])
        - swap_count: estimated SWAP count
        - depth: compiled circuit depth
        - compiled_2q: total native 2Q gates in compiled circuit
    """
    # Ensure circuit has measurements
    if circuit.num_clbits == 0:
        meas_circuit = circuit.copy()
        meas_circuit.measure_all()
    else:
        meas_circuit = circuit

    # Transpile with given layout
    compiled = transpile(
        meas_circuit,
        backend=backend,
        initial_layout=layout,
        routing_method="sabre",
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )

    # Metrics — auto-detect native 2Q gate (cx/ecr/cz)
    original_2q = _count_2q_gates(circuit, backend)
    compiled_2q = _count_2q_gates(compiled, backend)
    swap_count = max(0, compiled_2q - original_2q) // 3
    depth = compiled.depth()

    # Ideal distribution
    ideal_dist = compute_ideal_distribution(circuit)

    # Noisy simulation
    noisy_dist = run_noisy_simulation(compiled, backend, shots=shots)

    # PST via Hellinger fidelity
    pst = hellinger_fidelity(ideal_dist, noisy_dist)

    return {
        "pst": pst,
        "swap_count": swap_count,
        "depth": depth,
        "compiled_2q": compiled_2q,
    }


def measure_pst_batch(
    circuits: list[QuantumCircuit],
    backend: Any,
    layouts: list[list[int] | dict[int, int]],
    shots: int = 4096,
    seed_transpiler: int = 0,
    optimization_level: int = 3,
) -> list[dict[str, Any]]:
    """Measure PST for a batch of circuits.

    Args:
        circuits: List of quantum circuits.
        backend: FakeBackendV2 instance.
        layouts: List of layouts, one per circuit.
        shots: Simulation shots per circuit.
        seed_transpiler: Transpiler seed.
        optimization_level: Qiskit transpiler optimization level (0-3).

    Returns:
        List of result dicts from measure_pst.
    """
    results = []
    for circuit, layout in zip(circuits, layouts):
        result = measure_pst(
            circuit, backend, layout,
            shots=shots, seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
        )
        results.append(result)
    return results
