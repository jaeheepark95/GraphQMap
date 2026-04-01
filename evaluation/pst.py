"""PST (Probability of Successful Trials) measurement for GraphQMap.

Pipeline:
  Model → P matrix → Hungarian → discrete layout
  → transpile(initial_layout, routing_method='sabre')
  → noise simulation on FakeBackendV2
  → PST computation

PST metric:
  - pst: P(correct output) — probability of the ideal most-probable bitstring
    appearing in noisy execution (standard definition, used in QUEKO/multi-prog papers)

Simulation strategy (following MQM pattern):
  - tensor_network + GPU (cuQuantum) as default — handles any circuit size
  - Simulators created once per backend, reused across all circuits
"""

from __future__ import annotations

import logging
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)

# Detect best simulation config once at import time
_SIM_CONFIG: dict[str, str] | None = None


def _detect_sim_config() -> dict[str, str]:
    """Detect the best available simulation method. Cached after first call."""
    global _SIM_CONFIG
    if _SIM_CONFIG is not None:
        return _SIM_CONFIG

    # Try tensor_network + GPU first (handles any circuit size)
    try:
        sim = AerSimulator(method="tensor_network", device="GPU")
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        sim.run(qc, shots=1).result()
        _SIM_CONFIG = {"method": "tensor_network", "device": "GPU"}
        logger.info("Simulation method: tensor_network + GPU (cuQuantum)")
        return _SIM_CONFIG
    except Exception:
        pass

    # Try tensor_network on CPU
    try:
        sim = AerSimulator(method="tensor_network", device="CPU")
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        sim.run(qc, shots=1).result()
        _SIM_CONFIG = {"method": "tensor_network", "device": "CPU"}
        logger.info("Simulation method: tensor_network + CPU")
        return _SIM_CONFIG
    except Exception:
        pass

    raise RuntimeError(
        "No supported simulation method available. "
        "tensor_network (GPU or CPU) is required. "
        "Install qiskit-aer with tensor_network support."
    )


def get_sim_config() -> dict[str, str]:
    """Get the detected simulation configuration."""
    return _detect_sim_config()


def create_ideal_simulator(backend: Any) -> AerSimulator:
    """Create a noiseless simulator for a backend. Reuse across circuits.

    Args:
        backend: FakeBackendV2 instance.

    Returns:
        AerSimulator configured without noise model.
    """
    sim_config = _detect_sim_config()
    return AerSimulator.from_backend(backend, noise_model=None, **sim_config)


def create_noisy_simulator(backend: Any) -> AerSimulator:
    """Create a noisy simulator for a backend. Reuse across circuits.

    Args:
        backend: FakeBackendV2 instance.

    Returns:
        AerSimulator configured with backend noise model.
    """
    sim_config = _detect_sim_config()
    return AerSimulator.from_backend(backend, **sim_config)



def _count_2q_gates(circuit: QuantumCircuit, backend: Any = None) -> int:
    """Count native 2-qubit gates in a circuit, auto-detecting gate type."""
    ops = circuit.count_ops()
    if backend is not None:
        from data.hardware_graph import _get_two_qubit_gate_name
        try:
            gate_name = _get_two_qubit_gate_name(backend)
            return ops.get(gate_name, 0)
        except ValueError:
            pass
    for name in ("cx", "ecr", "cz"):
        if ops.get(name, 0) > 0:
            return ops[name]
    return 0


def compute_pst(
    result_counts: dict[str, int],
    ideal_counts: dict[str, int],
) -> float | list[float]:
    """Compute PST from result and ideal count dicts.

    Supports multi-register circuits (space-separated bitstrings).
    Follows MQM PSTv2 pattern.

    Args:
        result_counts: Noisy simulation counts.
        ideal_counts: Ideal simulation counts.

    Returns:
        PST value (float) or list of per-register PSTs for multi-register circuits.
    """
    ideal_result = max(ideal_counts, key=lambda k: ideal_counts[k])
    total_shots = sum(result_counts.values())

    if " " in ideal_result:
        # Multi-register: compute per-register PST
        ideal_parts = ideal_result.split(" ")
        psts = []
        for idx, ideal_part in enumerate(ideal_parts):
            matching_counts = []
            for key, count in result_counts.items():
                parts = key.split(" ")
                if ideal_part == parts[idx]:
                    matching_counts.append(count)
            psts.append(sum(matching_counts) / total_shots)
        return psts
    else:
        return result_counts.get(ideal_result, 0) / total_shots


def measure_pst(
    circuit: QuantumCircuit,
    backend: Any,
    layout: list[int] | dict[int, int],
    shots: int = 8192,
    seed_transpiler: int = 0,
    optimization_level: int = 3,
    ideal_sim: AerSimulator | None = None,
    noisy_sim: AerSimulator | None = None,
) -> dict[str, Any]:
    """Measure PST for a given circuit + layout + backend.

    Both ideal and noisy simulations run on the SAME transpiled circuit.
    Ideal = transpiled circuit without noise (fair comparison).
    Noisy = transpiled circuit with backend noise model.

    Args:
        circuit: Original quantum circuit (with or without measurements).
        backend: FakeBackendV2 instance.
        layout: Initial layout (list or dict mapping logical → physical).
        shots: Number of simulation shots.
        seed_transpiler: Transpiler random seed.
        optimization_level: Qiskit transpiler optimization level (0-3).
        ideal_sim: Pre-created ideal simulator (avoids re-creation per circuit).
        noisy_sim: Pre-created noisy simulator (avoids re-creation per circuit).

    Returns:
        Dict with:
        - pst: P(correct output) (float in [0, 1])
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

    # Create simulators if not provided (backward compatibility)
    if ideal_sim is None:
        ideal_sim = create_ideal_simulator(backend)
    if noisy_sim is None:
        noisy_sim = create_noisy_simulator(backend)

    # Ideal simulation
    ideal_counts = ideal_sim.run(compiled, shots=shots).result().get_counts()

    # Noisy simulation
    noisy_counts = noisy_sim.run(compiled, shots=shots).result().get_counts()

    # PST = P(correct output)
    pst = compute_pst(noisy_counts, ideal_counts)
    if isinstance(pst, list):
        pst = sum(pst) / len(pst)

    return {
        "pst": pst,
        "swap_count": swap_count,
        "depth": depth,
        "compiled_2q": compiled_2q,
    }


