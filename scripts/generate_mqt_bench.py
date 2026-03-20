"""Generate MQT Bench quantum circuits and export as .qasm files.

Generates circuits across all available MQT Bench algorithm types
with qubit counts from 2 to 127. For parameterized circuits
(VQE, QAOA), multiple variants are generated with different seeds/reps.

Includes a per-circuit timeout to skip circuits that take too long to generate.

Output: data/circuits/qasm/mqt_bench/{algorithm}_{num_qubits}[_variant].qasm
"""

import importlib
import os
import signal
import sys
from pathlib import Path

import numpy as np
from qiskit import qasm2

# Ensure mqt.bench is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tmp_downloads" / "bench" / "src"))

# Import all benchmark modules to populate registry
bench_dir = Path(__file__).resolve().parent.parent / "tmp_downloads" / "bench" / "src" / "mqt" / "bench" / "benchmarks"
for f in sorted(os.listdir(bench_dir)):
    if f.endswith(".py") and not f.startswith("_"):
        importlib.import_module(f"mqt.bench.benchmarks.{f[:-3]}")

from mqt.bench.benchmarks._registry import _REGISTRY

TIMEOUT_SECONDS = 30  # Per-circuit generation timeout


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Circuit generation timed out")


# Benchmarks to skip entirely
SKIP_BENCHMARKS = {
    "seven_qubit_steane_code",
    "shors_nine_qubit_code",
    "hrs_cumulative_multiplier",
    "shor",
    "ghz_dynamic",
}

MIN_QUBITS = {
    "cdkm_ripple_carry_adder": 4,
    "full_adder": 4,
    "graphstate": 3,
    "hhl": 3,
    "qwalk": 3,
    "vbe_ripple_carry_adder": 4,
}

EVEN_ONLY = {
    "bmw_quark_copula",
    "draper_qft_adder",
    "modular_adder",
    "multiplier",
    "rg_qft_multiplier",
    "cdkm_ripple_carry_adder",
    "full_adder",
    "vbe_ripple_carry_adder",
}

FIXED_SIZE = {
    "half_adder": [3],
}

BASE_QUBIT_COUNTS = (
    list(range(2, 21))
    + list(range(22, 51, 2))
    + list(range(55, 101, 5))
    + [110, 120, 127]
)

VQE_BENCHMARKS = {"vqe_real_amp", "vqe_su2", "vqe_two_local"}
QAOA_BENCHMARK = "qaoa"
REPS_VALUES = [1, 2, 3]
SEED_VALUES = [0, 1, 2]


def generate_and_save(name: str, num_qubits: int, output_dir: Path, suffix: str = "", **kwargs) -> bool:
    """Generate a circuit with timeout and save as QASM. Returns True on success."""
    factory = _REGISTRY[name].factory

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    try:
        qc = factory(num_qubits, **kwargs)
        if len(qc.parameters) > 0:
            rng = np.random.default_rng(42)
            param_dict = {p: rng.uniform(0, 2 * np.pi) for p in qc.parameters}
            qc.assign_parameters(param_dict, inplace=True)
        fname = f"{name}_n{qc.num_qubits}{suffix}.qasm"
        qasm2.dump(qc, str(output_dir / fname))
        signal.alarm(0)
        return True
    except TimeoutError:
        signal.alarm(0)
        return False
    except Exception:
        signal.alarm(0)
        return False


def main() -> None:
    output_dir = Path(__file__).resolve().parent.parent / "data" / "circuits" / "qasm" / "mqt_bench"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    failed = 0
    timed_out = 0

    for name in sorted(_REGISTRY.keys()):
        if name in SKIP_BENCHMARKS:
            continue

        if name in FIXED_SIZE:
            for n in FIXED_SIZE[name]:
                if generate_and_save(name, n, output_dir):
                    total += 1
                else:
                    failed += 1
            print(f"  {name}: fixed size done")
            continue

        min_q = MIN_QUBITS.get(name, 2)
        qubit_counts = [q for q in BASE_QUBIT_COUNTS if q >= min_q]
        if name in EVEN_ONLY:
            qubit_counts = [q for q in qubit_counts if q % 2 == 0]

        bench_total = 0
        max_succeeded_n = 0

        if name in VQE_BENCHMARKS:
            for n in qubit_counts:
                any_success = False
                for reps in REPS_VALUES:
                    if generate_and_save(name, n, output_dir, suffix=f"_reps{reps}", reps=reps):
                        total += 1
                        bench_total += 1
                        any_success = True
                        max_succeeded_n = max(max_succeeded_n, n)
                    else:
                        failed += 1
                # If all reps failed at this qubit count, skip larger sizes
                if not any_success and n > 20:
                    break
        elif name == QAOA_BENCHMARK:
            for n in qubit_counts:
                any_success = False
                for seed in SEED_VALUES:
                    if generate_and_save(name, n, output_dir, suffix=f"_seed{seed}", seed=seed):
                        total += 1
                        bench_total += 1
                        any_success = True
                        max_succeeded_n = max(max_succeeded_n, n)
                    else:
                        failed += 1
                if not any_success and n > 20:
                    break
        else:
            consecutive_failures = 0
            for n in qubit_counts:
                if generate_and_save(name, n, output_dir):
                    total += 1
                    bench_total += 1
                    max_succeeded_n = max(max_succeeded_n, n)
                    consecutive_failures = 0
                else:
                    failed += 1
                    consecutive_failures += 1
                    # Stop if 3 consecutive failures at larger sizes
                    if consecutive_failures >= 3 and n > 20:
                        break

        print(f"  {name}: {bench_total} circuits (max {max_succeeded_n}Q)")

    print(f"\nTotal: {total} circuits generated ({failed} failed)")
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()
