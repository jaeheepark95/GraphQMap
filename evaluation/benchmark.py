"""Structured benchmark runner for GraphQMap evaluation.

Runs baseline layout×routing combinations across backends:
  - Simulators created once per backend, reused for all circuits
  - DataFrame-based results with PST, Depth, CX count, Time
  - Simulator recovery on GPU crash (tensor_network failures)

Usage:
    python -m evaluation.benchmark --backend toronto --reps 2
    python -m evaluation.benchmark --backend toronto brooklyn torino --reps 1
    python -m evaluation.benchmark --backend toronto --circuit-dir references/colleague/tests2/benchmarks
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider

from evaluation.pst import (
    compute_pst,
    create_ideal_simulator,
    create_noisy_simulator,
)
from evaluation.transpiler import transpile_with_timing

logger = logging.getLogger(__name__)

# Pozzi benchmark split — ALL 25 used as test circuits (never training).
# Test set 1 (12 circuits): smaller/simpler, matches colleague's paper's "train" split
POZZI_TEST1_CIRCUITS = [
    "bv_n3",
    "bv_n4",
    "peres_3",
    "toffoli_3",
    "fredkin_3",
    "xor5_254",
    "3_17_13",
    "4mod5-v1_22",
    "mod5mils_65",
    "alu-v0_27",
    "decod24-v2_43",
    "4gt13_92",
]

# Test set 2 (13 circuits): matches colleague's paper's "test" split
POZZI_TEST2_CIRCUITS = [
    "ham3_102",
    "miller_11",
    "decod24-v0_38",
    "rd32-v0_66",
    "4gt5_76",
    "4mod7-v0_94",
    "alu-v2_32",
    "hwb4_49",
    "ex1_226",
    "decod24-bdd_294",
    "ham7_104",
    "rd53_138",
    "qft_10",
]

# All 25 Pozzi benchmark circuits
BENCHMARK_CIRCUITS = POZZI_TEST1_CIRCUITS + POZZI_TEST2_CIRCUITS

# Named circuit sets for --circuit-set CLI option
CIRCUIT_SETS = {
    "test1": POZZI_TEST1_CIRCUITS,
    "test2": POZZI_TEST2_CIRCUITS,
    "all": BENCHMARK_CIRCUITS,
}

# Default method combinations: (routing_method, layout_method)
DEFAULT_COMBINATIONS = [
    ("sabre", "sabre"),
    ("sabre", "noise_adaptive"),
    ("sabre", "qap"),
    ("nassc", "sabre"),
    ("nassc", "noise_adaptive"),
    ("nassc", "qap"),
]

# Default test backends
DEFAULT_BACKENDS = {
    "toronto": fake_provider.FakeTorontoV2,
    "brooklyn": fake_provider.FakeBrooklynV2,
    "torino": fake_provider.FakeTorino,
    "prague": fake_provider.FakePrague,
}

BENCHMARK_CIRCUIT_DIR = "data/circuits/qasm/benchmarks"


def add_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Add measurements to the qubits actually used by any gate.

    Mirrors the reference Attention_Qubit_Mapping implementation so PST
    results are comparable. Reuses any pre-existing classical register from
    the QASM file (e.g. ``creg c[16]``) instead of appending a new ``meas``
    register, which would cause multi-register result bitstrings and the
    0.5-floor PST bug.
    """
    if qc.count_ops().get("measure", 0) > 0:
        return qc
    used = sorted({qc.find_bit(q).index for inst in qc.data for q in inst.qubits})
    new_qc = qc.copy()
    if len(new_qc.clbits) == 0:
        new_qc.add_register(ClassicalRegister(len(used), "meas"))
    for idx, q in enumerate(used):
        if idx < len(new_qc.clbits):
            new_qc.measure(q, idx)
    return new_qc


def load_benchmark_circuit(
    name: str,
    circuit_dir: str = BENCHMARK_CIRCUIT_DIR,
    measure: bool = True,
) -> QuantumCircuit:
    """Load a benchmark circuit from QASM file.

    Args:
        name: Circuit name (without .qasm extension).
        circuit_dir: Directory containing .qasm files.
        measure: Whether to add measurements if missing.

    Returns:
        QuantumCircuit instance.
    """
    path = os.path.join(circuit_dir, f"{name}.qasm")
    circuit = QuantumCircuit.from_qasm_file(path)
    if measure:
        circuit = add_measurements(circuit)
    return circuit


def execute_on_simulators(
    transpiled_circuit: QuantumCircuit,
    ideal_simulator: AerSimulator,
    noisy_simulator: AerSimulator,
    shots: int = 8192,
) -> tuple[float, float, float | list[float]]:
    """Execute a transpiled circuit on ideal and noisy simulators.

    Args:
        transpiled_circuit: Already transpiled circuit.
        ideal_simulator: Noiseless simulator.
        noisy_simulator: Noisy simulator.
        shots: Number of shots.

    Returns:
        Tuple of (avg_pst, depth, pst_value_or_list).

    Raises:
        QiskitError: If simulation fails (e.g. tensor_network GPU crash).
    """
    ideal_result = ideal_simulator.run(
        transpiled_circuit, shots=shots
    ).result().get_counts()
    noisy_result = noisy_simulator.run(
        transpiled_circuit, shots=shots
    ).result().get_counts()

    psts = compute_pst(noisy_result, ideal_result)
    avg_pst = sum(psts) / len(psts) if isinstance(psts, list) else psts
    depth = transpiled_circuit.depth()

    return avg_pst, depth, psts


def run_benchmark_single(
    circuit_names: list[str] | None = None,
    backend_names: list[str] | None = None,
    combinations: list[tuple[str, str]] | None = None,
    reps: int = 1,
    shots: int = 8192,
    circuit_dir: str = BENCHMARK_CIRCUIT_DIR,
    seed: int = 43,
    warm_up: int = 1,
) -> dict[str, pd.DataFrame]:
    """Run single-circuit benchmarks across backends and method combinations.

    On simulation failure (e.g. tensor_network GPU crash), the simulators
    are recreated and the failed run is recorded as NaN.

    Args:
        circuit_names: List of circuit names. Defaults to BENCHMARK_CIRCUITS.
        backend_names: List of backend names. Defaults to all DEFAULT_BACKENDS.
        combinations: List of (routing, layout) tuples. Defaults to DEFAULT_COMBINATIONS.
        reps: Number of repetitions per circuit (skip first `warm_up` for averaging).
        shots: Simulation shots.
        circuit_dir: Directory containing benchmark .qasm files.
        seed: Random seed.
        warm_up: Number of warm-up runs to skip in averaging.

    Returns:
        Dict mapping backend_name → DataFrame with results.
    """
    if circuit_names is None:
        circuit_names = BENCHMARK_CIRCUITS
    if backend_names is None:
        backend_names = list(DEFAULT_BACKENDS.keys())
    if combinations is None:
        combinations = DEFAULT_COMBINATIONS

    results = {}

    for backend_name in backend_names:
        logger.info("=== Backend: %s ===", backend_name)

        if backend_name not in DEFAULT_BACKENDS:
            logger.warning("Unknown backend: %s, skipping", backend_name)
            continue

        backend = DEFAULT_BACKENDS[backend_name]()
        ideal_sim = create_ideal_simulator(backend)
        noisy_sim = create_noisy_simulator(backend)

        def recreate_simulators():
            nonlocal ideal_sim, noisy_sim
            logger.info("  Recreating simulators after failure...")
            ideal_sim = create_ideal_simulator(backend)
            noisy_sim = create_noisy_simulator(backend)

        method_labels = [f"{r}+{l}" for r, l in combinations]

        all_pst = {label: [] for label in method_labels}
        all_time = {label: [] for label in method_labels}
        all_cx = {label: [] for label in method_labels}
        all_total_time = {label: [] for label in method_labels}

        for cname in circuit_names:
            logger.info("  Circuit: %s", cname)
            circuit = load_benchmark_circuit(cname, circuit_dir)

            for (routing, layout), label in zip(combinations, method_labels):
                psts_runs = []
                times_runs = []
                cx_runs = []
                total_time_runs = []

                total_runs = warm_up + reps
                for j in range(total_runs):
                    tc, metadata = transpile_with_timing(
                        circuit,
                        backend,
                        layout_method=layout,
                        routing_method=routing,
                        seed=seed + j,
                    )
                    try:
                        avg_pst, depth, _ = execute_on_simulators(
                            tc, ideal_sim, noisy_sim, shots=shots
                        )
                    except Exception as e:
                        logger.warning("  Simulation failed: %s", e)
                        recreate_simulators()
                        avg_pst = float("nan")

                    if j >= warm_up:
                        psts_runs.append(avg_pst)
                        times_runs.append(metadata["layout_time"])
                        cx_runs.append(metadata["map_cx"])
                        total_time_runs.append(metadata["total_time"])

                all_pst[label].append(np.mean(psts_runs) if psts_runs else 0.0)
                all_time[label].append(np.mean(times_runs) if times_runs else 0.0)
                all_cx[label].append(np.mean(cx_runs) if cx_runs else 0.0)
                all_total_time[label].append(np.mean(total_time_runs) if total_time_runs else 0.0)

        # Build MultiIndex DataFrame
        data = {}
        for label in method_labels:
            data[("PST", label)] = all_pst[label]
            data[("TIME", label)] = all_time[label]
            data[("CX", label)] = all_cx[label]
            data[("TOTAL_TIME", label)] = all_total_time[label]

        df = pd.DataFrame(data, index=circuit_names)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.loc["Avg"] = df.mean(numeric_only=True, axis=0)

        results[backend_name] = df

        pd.options.display.float_format = "{:.2f}".format
        print(f"\nResults for backend {backend_name}:")
        print(df)
        print()

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="GraphQMap Benchmark")
    parser.add_argument(
        "--backend",
        nargs="+",
        default=["toronto", "brooklyn", "torino"],
        help="Backend names",
    )
    parser.add_argument(
        "--circuits",
        nargs="+",
        default=None,
        help="Circuit names (default: standard 8 circuits)",
    )
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument(
        "--circuit-dir",
        default=BENCHMARK_CIRCUIT_DIR,
        help="Directory containing benchmark .qasm files",
    )
    args = parser.parse_args()

    run_benchmark_single(
        circuit_names=args.circuits,
        backend_names=args.backend,
        reps=args.reps,
        shots=args.shots,
        circuit_dir=args.circuit_dir,
    )


if __name__ == "__main__":
    main()
