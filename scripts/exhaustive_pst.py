"""Exhaustive PST measurement over all possible initial layouts.

Measures PST for every P(n_physical, n_logical) permutation of physical qubits
for a given circuit on a given backend.

Usage:
    python scripts/exhaustive_pst.py --circuit 3_17_13 --backend toronto --routing nassc
"""

import sys
import time
import csv
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import QuantumCircuit
from data.hardware_graph import get_backend, _get_two_qubit_gate_name
from evaluation.pst import (
    create_ideal_simulator,
    create_noisy_simulator,
    compute_pst,
)
from evaluation.transpiler import build_transpiler


def _count_2q_gates(circuit: QuantumCircuit, backend) -> int:
    ops = circuit.count_ops()
    try:
        gate_name = _get_two_qubit_gate_name(backend)
        return ops.get(gate_name, 0)
    except ValueError:
        pass
    for name in ("cx", "ecr", "cz"):
        if ops.get(name, 0) > 0:
            return ops[name]
    return 0


def measure_pst_custom(
    circuit: QuantumCircuit,
    backend,
    layout: list[int],
    routing_method: str = "nassc",
    shots: int = 8192,
    seed: int = 42,
    ideal_sim=None,
    noisy_sim=None,
) -> dict:
    """Measure PST using custom transpiler with configurable routing."""
    # Ensure measurements
    if circuit.num_clbits == 0:
        meas_circuit = circuit.copy()
        meas_circuit.measure_all()
    else:
        meas_circuit = circuit

    # Build and run custom transpiler
    pm = build_transpiler(
        backend=backend,
        initial_layout=layout,
        routing_method=routing_method,
        seed=seed,
        optimization_level=3,
    )
    compiled = pm.run(meas_circuit)

    # Metrics
    original_2q = _count_2q_gates(circuit, backend)
    compiled_2q = _count_2q_gates(compiled, backend)
    swap_count = max(0, compiled_2q - original_2q) // 3
    depth = compiled.depth()

    # Simulate
    ideal_counts = ideal_sim.run(compiled, shots=shots).result().get_counts()
    noisy_counts = noisy_sim.run(compiled, shots=shots).result().get_counts()

    pst = compute_pst(noisy_counts, ideal_counts)
    if isinstance(pst, list):
        pst = sum(pst) / len(pst)

    return {
        "pst": pst,
        "swap_count": swap_count,
        "depth": depth,
        "compiled_2q": compiled_2q,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", required=True, help="Circuit name (e.g. 3_17_13)")
    parser.add_argument("--backend", default="toronto")
    parser.add_argument("--routing", default="nassc", choices=["sabre", "nassc"])
    args = parser.parse_args()

    backend_name = args.backend
    routing = args.routing
    circuit_path = f"data/circuits/qasm/benchmarks/{args.circuit}.qasm"
    output_path = f"runs/exhaustive_pst_{args.circuit}_{backend_name}_{routing}.csv"

    # Load circuit and backend
    circuit = QuantumCircuit.from_qasm_file(circuit_path)
    backend = get_backend(backend_name)

    num_logical = circuit.num_qubits
    num_physical = backend.num_qubits
    total = 1
    for i in range(num_logical):
        total *= (num_physical - i)

    print(f"Circuit: {circuit_path} ({num_logical} qubits)")
    print(f"Backend: {backend_name} ({num_physical} qubits)")
    print(f"Routing: {routing}")
    print(f"Total layouts: P({num_physical},{num_logical}) = {total:,}")

    # Create simulators once
    print("Creating simulators...")
    ideal_sim = create_ideal_simulator(backend)
    noisy_sim = create_noisy_simulator(backend)

    # Run exhaustive search
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layout", "pst", "swap_count", "depth", "compiled_2q"])

        start = time.time()
        for i, perm in enumerate(permutations(range(num_physical), num_logical)):
            layout = list(perm)
            try:
                result = measure_pst_custom(
                    circuit=circuit,
                    backend=backend,
                    layout=layout,
                    routing_method=routing,
                    shots=8192,
                    seed=42,
                    ideal_sim=ideal_sim,
                    noisy_sim=noisy_sim,
                )
                writer.writerow([
                    layout,
                    f"{result['pst']:.6f}",
                    result["swap_count"],
                    result["depth"],
                    result["compiled_2q"],
                ])
            except Exception as e:
                writer.writerow([layout, "ERROR", str(e), "", ""])

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate
                print(
                    f"  [{i + 1:,}/{total:,}] "
                    f"{rate:.1f} layouts/s, "
                    f"~{remaining / 60:.1f} min remaining",
                    flush=True,
                )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f} min. Results saved to {output_path}")

    # Print summary
    import pandas as pd
    df = pd.read_csv(output_path)
    df_valid = df[df["pst"] != "ERROR"].copy()
    df_valid["pst"] = df_valid["pst"].astype(float)

    print(f"\n--- Summary ---")
    print(f"Valid layouts: {len(df_valid):,} / {total:,}")
    print(f"PST: min={df_valid['pst'].min():.4f}, max={df_valid['pst'].max():.4f}, "
          f"mean={df_valid['pst'].mean():.4f}, std={df_valid['pst'].std():.4f}")

    best = df_valid.loc[df_valid["pst"].idxmax()]
    print(f"Best layout: {best['layout']} → PST={best['pst']:.4f}")

    worst = df_valid.loc[df_valid["pst"].idxmin()]
    print(f"Worst layout: {worst['layout']} → PST={worst['pst']:.4f}")


if __name__ == "__main__":
    main()
