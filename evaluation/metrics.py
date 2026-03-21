"""Metrics collection and reporting for GraphQMap evaluation.

Reports: PST, SWAP count, circuit depth, inference latency.
All metrics with mean ± std across multiple repetitions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from qiskit import QuantumCircuit

from evaluation.pst import create_ideal_simulator, create_noisy_simulator, measure_pst


@dataclass
class EvalResult:
    """Evaluation result for a single circuit across repetitions."""

    circuit_name: str
    backend_name: str
    method: str
    pst_values: list[float] = field(default_factory=list)
    swap_counts: list[int] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)
    inference_times: list[float] = field(default_factory=list)

    @property
    def pst_mean(self) -> float:
        return float(np.mean(self.pst_values)) if self.pst_values else 0.0

    @property
    def pst_std(self) -> float:
        return float(np.std(self.pst_values)) if self.pst_values else 0.0

    @property
    def swap_mean(self) -> float:
        return float(np.mean(self.swap_counts)) if self.swap_counts else 0.0

    @property
    def depth_mean(self) -> float:
        return float(np.mean(self.depths)) if self.depths else 0.0

    @property
    def inference_time_mean(self) -> float:
        return float(np.mean(self.inference_times)) if self.inference_times else 0.0

    def summary(self) -> dict[str, Any]:
        """Return a summary dict."""
        return {
            "circuit": self.circuit_name,
            "backend": self.backend_name,
            "method": self.method,
            "pst": f"{self.pst_mean:.4f} ± {self.pst_std:.4f}",
            "swap_count": f"{self.swap_mean:.1f}",
            "depth": f"{self.depth_mean:.1f}",
            "inference_ms": f"{self.inference_time_mean * 1000:.1f}",
            "n_reps": len(self.pst_values),
        }


def evaluate_model_on_circuit(
    model: torch.nn.Module,
    circuit: QuantumCircuit,
    circuit_graph: Any,
    hardware_graph: Any,
    backend: Any,
    circuit_name: str = "",
    backend_name: str = "",
    num_logical: int = 0,
    num_physical: int = 0,
    tau: float = 0.05,
    shots: int = 4096,
    num_repetitions: int = 1,
) -> EvalResult:
    """Evaluate the model on a single circuit with multiple repetitions.

    Each repetition uses a different transpiler seed for statistical reliability.

    Args:
        model: GraphQMap model.
        circuit: Quantum circuit.
        circuit_graph: Batched PyG Data for the circuit (batch_size=1).
        hardware_graph: Batched PyG Data for the hardware (batch_size=1).
        backend: FakeBackendV2 instance.
        circuit_name: Name for reporting.
        backend_name: Backend name for reporting.
        num_logical: Number of logical qubits.
        num_physical: Number of physical qubits.
        tau: Sinkhorn temperature.
        shots: Simulation shots.
        num_repetitions: Number of repetitions.

    Returns:
        EvalResult with all metrics.
    """
    result = EvalResult(
        circuit_name=circuit_name,
        backend_name=backend_name,
        method="graphqmap",
    )

    # Create simulators once, reuse across repetitions
    ideal_sim = create_ideal_simulator(backend)
    noisy_sim = create_noisy_simulator(backend)

    for rep in range(num_repetitions):
        # Inference timing
        t0 = time.perf_counter()
        layouts = model.predict(
            circuit_graph, hardware_graph,
            batch_size=1,
            num_logical=num_logical,
            num_physical=num_physical,
            tau=tau,
        )
        t1 = time.perf_counter()

        layout = layouts[0]
        result.inference_times.append(t1 - t0)

        # PST measurement with different transpiler seed per rep
        pst_result = measure_pst(
            circuit, backend, list(layout.values()),
            shots=shots, seed_transpiler=rep,
            ideal_sim=ideal_sim, noisy_sim=noisy_sim,
        )

        result.pst_values.append(pst_result["pst"])
        result.swap_counts.append(pst_result["swap_count"])
        result.depths.append(pst_result["depth"])

    return result


def format_results_table(results: list[EvalResult]) -> str:
    """Format evaluation results as a text table.

    Args:
        results: List of EvalResult objects.

    Returns:
        Formatted table string.
    """
    header = f"{'Circuit':<25} {'Backend':<12} {'Method':<12} {'PST':>14} {'SWAP':>8} {'Depth':>8} {'Time(ms)':>10}"
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        s = r.summary()
        line = (
            f"{s['circuit']:<25} {s['backend']:<12} {s['method']:<12} "
            f"{s['pst']:>14} {s['swap_count']:>8} {s['depth']:>8} {s['inference_ms']:>10}"
        )
        lines.append(line)

    return "\n".join(lines)


def aggregate_results(results: list[EvalResult]) -> dict[str, Any]:
    """Aggregate results across circuits for a given method+backend.

    Args:
        results: List of EvalResult objects.

    Returns:
        Dict with aggregated mean ± std for each metric.
    """
    all_pst = [r.pst_mean for r in results]
    all_swap = [r.swap_mean for r in results]
    all_depth = [r.depth_mean for r in results]
    all_time = [r.inference_time_mean for r in results]

    return {
        "num_circuits": len(results),
        "pst_mean": float(np.mean(all_pst)),
        "pst_std": float(np.std(all_pst)),
        "swap_mean": float(np.mean(all_swap)),
        "swap_std": float(np.std(all_swap)),
        "depth_mean": float(np.mean(all_depth)),
        "depth_std": float(np.std(all_depth)),
        "inference_ms_mean": float(np.mean(all_time)) * 1000,
        "inference_ms_std": float(np.std(all_time)) * 1000,
    }
