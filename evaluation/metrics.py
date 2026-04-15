"""Metrics collection and reporting for GraphQMap evaluation.

Reports: PST, SWAP count, circuit depth, inference latency.
All metrics with mean ± std across multiple repetitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
        if not self.pst_values:
            return 0.0
        if any(np.isnan(v) for v in self.pst_values):
            return float("nan")
        return float(np.mean(self.pst_values))

    @property
    def pst_std(self) -> float:
        if not self.pst_values:
            return 0.0
        if any(np.isnan(v) for v in self.pst_values):
            return float("nan")
        return float(np.std(self.pst_values))

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
            "inference_ms": f"{self.inference_time_mean * 1000:.1f}" if self.inference_times else "N/A",
            "n_reps": len(self.pst_values),
        }



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
    all_time = [r.inference_time_mean for r in results if r.inference_times]

    any_pst_nan = any(np.isnan(v) for v in all_pst)
    pst_mean_val = float("nan") if any_pst_nan else float(np.mean(all_pst))
    pst_std_val = float("nan") if any_pst_nan else float(np.std(all_pst))

    agg = {
        "num_circuits": len(results),
        "pst_mean": pst_mean_val,
        "pst_std": pst_std_val,
        "swap_mean": float(np.mean(all_swap)),
        "swap_std": float(np.std(all_swap)),
        "depth_mean": float(np.mean(all_depth)),
        "depth_std": float(np.std(all_depth)),
    }
    if all_time:
        agg["inference_ms_mean"] = float(np.mean(all_time)) * 1000
        agg["inference_ms_std"] = float(np.std(all_time)) * 1000
    else:
        agg["inference_ms_mean"] = "N/A"
        agg["inference_ms_std"] = "N/A"

    return agg
