# =============================================================================
# UPDATE LOG (2026-03-30, v1.4.1)
# - Updated evaluation-time proxy metrics to the v1.4.1 execution-surrogate
#   family:
#     * L_1q, L_ro, L_2q, L_native, L_route, L_task
#     * score_proxy_exec and component surrogate scores
# - Evaluation batches now consume the raw loss-path tensors and logical-count
#   tensors: D_raw, e1q, ero, e2q, n1q, nmeas.
# - The final score is reported as a surrogate score, not literal PST.
# =============================================================================
# =============================================================================
# UPDATE LOG (2026-03-30)
# - Added evaluation-time support for freeze_hardware_reindex / identity-R_H
#   ablations so inference uses the same hardware-side semantics as training.
# =============================================================================

# src/kmw/evaluation/evaluate.py

"""
Evaluation / inference pipeline for the KMW project.

What this file is responsible for
---------------------------------
This file owns the evaluation-time path of the project:

1. run inference on a dataset
2. harden the reindexer outputs for inference
3. run the mapper in inference mode
4. decode latent logits back to native frame
5. compute final hard assignment using Hungarian
6. compute per-circuit research metrics
7. optionally run real routed downstream evaluation
8. write:
      - per-circuit CSV
      - summary JSON
      - console summary

Important implementation notes
------------------------------
- Training uses soft assignment (Sinkhorn).
- Inference uses hard reindexing + hard final assignment.
- This file uses HARD permutations for:
      R_L_hat
      R_H_hat
      M_map
- The model is still an initial mapper.
  Real routing is attached only in evaluation when route_final_eval=True.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from kmw.losses.loss import LossConfig, compute_task_loss_from_assignment
from kmw.models.model import KMWModel, decode_to_native


# =============================================================================
# 1) Optional SciPy import for Hungarian
# =============================================================================

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


# =============================================================================
# 2) Evaluation configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation-time configuration."""

    per_circuit_csv_path: str = "artifacts/eval/per_circuit_metrics.csv"
    summary_json_path: str = "artifacts/eval/summary.json"
    print_console_summary: bool = True

    # If routed evaluation is disabled, optionally keep routing columns as
    # not-run placeholders for schema stability.
    include_routing_placeholders_in_csv: bool = True

    # If True, hard-fail on exceptions. Otherwise, record a failure row and continue.
    fail_fast: bool = False

    # Tolerance used when checking whether matrices are valid permutations.
    perm_tol: float = 1e-6

    # Optional tag describing the current evaluation split.
    eval_split_name: str = "eval"

    # Routed downstream evaluation settings.
    project_root: str = "."
    backend_name: str = "fake_toronto_v2"
    route_final_eval: bool = False
    routing_method: str = "sabre"
    transpile_optimization_level: int = 0
    seed_transpiler: int | None = None
    include_readout_in_pst: bool = True
    save_routed_qasm_dir: str | None = None
    save_routed_qpy_dir: str | None = None

    # If True, force inference-time R_H = I and only harden/use logical reindexing.
    freeze_hardware_reindex: bool = False


# =============================================================================
# 3) Core tensor / shape helpers
# =============================================================================

def _assert_finite_tensor(x: torch.Tensor, name: str) -> None:
    """Fail loudly if a tensor contains NaN or Inf."""
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def _ensure_batched_square_matrix(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Accept:
        (N, N)    -> convert to (1, N, N)
        (B, N, N) -> keep as is
    """
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"{name} must have shape (N,N) or (B,N,N), got {tuple(x.shape)}")


def _ensure_batched_vector(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Accept:
        (N,)   -> convert to (1, N)
        (B, N) -> keep as is
    """
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    raise ValueError(f"{name} must have shape (N,) or (B,N), got {tuple(x.shape)}")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move tensor values in a batch dict to the chosen device.
    Non-tensor values pass through unchanged.
    """
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out

#============================================================================


# def extract_native_batch(sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
#     """
#     Extract the native tensors needed by the model / losses.

#     Required keys:
#         A, m, B, c1, c2, D
#     """
#     required = ("A", "m", "B", "c1", "c2", "D", "D_raw", "e1q", "ero", "e2q", "n1q", "nmeas")
#     missing = [k for k in required if k not in sample]
#     if missing:
#         raise KeyError(f"Sample is missing required native keys: {missing}")

#     native = {
#         "A": sample["A"],
#         "m": sample["m"],
#         "B": sample["B"],
#         "c1": sample["c1"],
#         "c2": sample["c2"],
#         "D": sample["D"],
#     }

#     for key, value in native.items():
#         if not torch.is_tensor(value):
#             raise TypeError(f"Sample key '{key}' must be a tensor, got {type(value)}")
#         _assert_finite_tensor(value, key)

#     return native

# fix: 
def extract_native_batch(sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Extract the native tensors needed by the model / losses.

    Required keys:
        A, m, B, c1, c2, D,
        D_raw, e1q, ero, e2q,
        n1q, nmeas
    """
    required = ("A", "m", "B", "c1", "c2", "D", "D_raw", "e1q", "ero", "e2q", "n1q", "nmeas")
    missing = [k for k in required if k not in sample]
    if missing:
        raise KeyError(f"Sample is missing required native keys: {missing}")

    native = {
        "A": sample["A"],
        "m": sample["m"],
        "B": sample["B"],
        "c1": sample["c1"],
        "c2": sample["c2"],
        "D": sample["D"],
        "D_raw": sample["D_raw"],
        "e1q": sample["e1q"],
        "ero": sample["ero"],
        "e2q": sample["e2q"],
        "n1q": sample["n1q"],
        "nmeas": sample["nmeas"],
    }

    for key, value in native.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Sample key '{key}' must be a tensor, got {type(value)}")
        _assert_finite_tensor(value, key)

    return native

#=============================================================================

# =============================================================================
# 4) Batch splitting helpers
# =============================================================================

def _infer_batch_size_from_batch(batch: Dict[str, Any]) -> int:
    """
    Infer batch size from the first tensor-like value with a leading batch axis.

    Because the project currently defaults to batch_size=1, this helper mainly
    exists so evaluation code stays somewhat robust if later run with B > 1.
    """
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return len(value)
    return 1


def _slice_nested_value(value: Any, index: int, batch_size: int) -> Any:
    """
    Slice one sample out of a nested collated structure.

    Rules:
    - tensor with leading dim == batch_size -> take value[index:index+1]
      (we keep the batch dimension to stay compatible with model code)
    - list/tuple of length batch_size -> take value[index]
    - dict -> recurse
    - otherwise -> return as-is
    """
    if torch.is_tensor(value):
        if value.ndim >= 1 and value.shape[0] == batch_size:
            return value[index:index + 1]
        return value

    if isinstance(value, list):
        if len(value) == batch_size:
            return value[index]
        return value

    if isinstance(value, tuple):
        if len(value) == batch_size:
            return value[index]
        return value

    if isinstance(value, dict):
        return {k: _slice_nested_value(v, index, batch_size) for k, v in value.items()}

    return value


def iter_samples_from_batch(batch: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Split a possibly-batched dictionary into per-sample dictionaries.

    Each tensor sample keeps a leading batch dimension of size 1 so the rest of
    the model/loss code can be reused directly.
    """
    batch_size = _infer_batch_size_from_batch(batch)

    for i in range(batch_size):
        yield {k: _slice_nested_value(v, i, batch_size) for k, v in batch.items()}


# =============================================================================
# 5) Metadata helpers
# =============================================================================

def _get_meta_dict(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return a metadata dictionary if present, else {}."""
    meta = sample.get("metadata", {})
    if isinstance(meta, dict):
        return meta
    return {}


def get_meta_value(sample: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Search for a metadata value in:
    1. sample[key]
    2. sample["metadata"][key]
    """
    if key in sample:
        return sample[key]

    meta = _get_meta_dict(sample)
    return meta.get(key, default)


def infer_circuit_id(sample: Dict[str, Any]) -> str:
    """
    Best-effort circuit_id extraction.

    Preferred order:
    1. explicit "circuit_id"
    2. manifest "id"
    3. qasm_relpath stem
    4. fallback "unknown"
    """
    circuit_id = get_meta_value(sample, "circuit_id", None)
    if circuit_id is not None:
        return str(circuit_id)

    manifest_id = get_meta_value(sample, "id", None)
    if manifest_id is not None:
        return str(manifest_id)

    qasm_relpath = get_meta_value(sample, "qasm_relpath", None)
    if qasm_relpath is not None:
        return Path(str(qasm_relpath)).stem

    return "unknown"


def infer_source(sample: Dict[str, Any]) -> str:
    """Best-effort source extraction from sample metadata."""
    return str(get_meta_value(sample, "source", "unknown"))


def infer_k_logical(sample: Dict[str, Any], native: Dict[str, torch.Tensor]) -> int:
    """
    Preferred source: manifest metadata field "k_logical".
    Fallback: sum of m.
    """
    k_logical = get_meta_value(sample, "k_logical", None)
    if k_logical is not None:
        return int(k_logical)

    m = _ensure_batched_vector(native["m"], "m")
    return int(round(float(m[0].sum().item())))


def infer_num_1q(sample: Dict[str, Any]) -> Optional[int]:
    """Try to recover num_1q from metadata."""
    value = get_meta_value(sample, "num_1q", None)
    return None if value is None else int(value)


def infer_num_2q(sample: Dict[str, Any]) -> Optional[int]:
    """Try to recover num_2q from metadata."""
    value = get_meta_value(sample, "num_2q", None)
    return None if value is None else int(value)


# =============================================================================
# 6) Hungarian helpers
# =============================================================================

def require_hungarian_available() -> None:
    """Fail with a clear error if SciPy's Hungarian solver is unavailable."""
    if linear_sum_assignment is None:
        raise ImportError(
            "SciPy is required for inference-time Hungarian assignment, "
            "but scipy.optimize.linear_sum_assignment could not be imported."
        )


def hungarian_permutation_matrix(score_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert a square score matrix into a hard permutation matrix using Hungarian.

    Input:
        score_matrix : (N, N) or (1, N, N)

    Output:
        P : (1, N, N)

    Convention:
        We maximize total score, so we pass -scores to the minimization solver.
    """
    require_hungarian_available()

    score_matrix = _ensure_batched_square_matrix(score_matrix, "score_matrix")
    if score_matrix.shape[0] != 1:
        raise ValueError("hungarian_permutation_matrix expects exactly one sample")

    S = score_matrix[0].detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-S)

    N = score_matrix.shape[-1]
    P = torch.zeros((1, N, N), device=score_matrix.device, dtype=score_matrix.dtype)
    P[0, row_ind, col_ind] = 1.0
    return P


def harden_soft_permutation(R_soft: torch.Tensor) -> torch.Tensor:
    """
    Turn a soft permutation matrix into a hard permutation matrix with Hungarian.
    """
    return hungarian_permutation_matrix(R_soft)


# =============================================================================
# 7) Permutation / entropy / validity helpers
# =============================================================================

def mean_row_entropy(P: torch.Tensor, normalized: bool = True, eps: float = 1e-12) -> float:
    """
    Compute the mean row entropy of a soft permutation matrix.
    """
    P = _ensure_batched_square_matrix(P, "P")
    if P.shape[0] != 1:
        raise ValueError("mean_row_entropy expects exactly one sample")

    probs = P[0].clamp(min=eps)
    H = -(probs * probs.log()).sum(dim=-1)

    if normalized:
        N = probs.shape[-1]
        denom = math.log(max(N, 2))
        if denom > 0:
            H = H / denom

    return float(H.mean().item())


def is_valid_permutation_matrix(P: torch.Tensor, tol: float = 1e-6) -> bool:
    """
    Check whether a matrix is (approximately) a valid permutation matrix.
    """
    P = _ensure_batched_square_matrix(P, "P")
    if P.shape[0] != 1:
        return False

    M = P[0]
    if M.shape[0] != M.shape[1]:
        return False

    if not torch.isfinite(M).all():
        return False

    row_sums = M.sum(dim=-1)
    col_sums = M.sum(dim=-2)

    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=tol, rtol=0.0):
        return False
    if not torch.allclose(col_sums, torch.ones_like(col_sums), atol=tol, rtol=0.0):
        return False

    rounded = torch.round(M)
    if not torch.allclose(M, rounded, atol=tol, rtol=0.0):
        return False

    return True


def permutation_matrix_to_assignment_vector(P: torch.Tensor) -> torch.Tensor:
    """
    Convert a hard permutation matrix P[u, j] into an assignment vector:
        assign[u] = j
    """
    P = _ensure_batched_square_matrix(P, "P")
    if P.shape[0] != 1:
        raise ValueError("permutation_matrix_to_assignment_vector expects exactly one sample")

    return P[0].argmax(dim=-1)


# =============================================================================
# 8) Research metrics on a hard mapping
# =============================================================================

def _offdiag(A: torch.Tensor) -> torch.Tensor:
    """Zero the diagonal of a batched square matrix."""
    A = _ensure_batched_square_matrix(A, "A")
    _, N, _ = A.shape
    eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
    return A * (1.0 - eye)


def _safe_scalar_div(num: float, den: float) -> float:
    """Safe scalar division used for summary-style metrics."""
    if abs(den) < 1e-12:
        return 0.0
    return float(num / den)


def compute_weighted_adj_hit_rate(A: torch.Tensor, Bmat: torch.Tensor, M_map: torch.Tensor) -> float:
    """
    Weighted adjacency hit rate under the hard final mapping.
    """
    A = _ensure_batched_square_matrix(A, "A")
    Bmat = _ensure_batched_square_matrix(Bmat, "Bmat")
    M_map = _ensure_batched_square_matrix(M_map, "M_map")

    A_off = _offdiag(A)[0]
    B = Bmat[0]
    assign = permutation_matrix_to_assignment_vector(M_map)

    N = A_off.shape[0]
    weighted_hit = 0.0
    total_mass = 0.0

    for u in range(N):
        ju = int(assign[u].item())
        for v in range(N):
            jv = int(assign[v].item())
            w = float(A_off[u, v].item())
            total_mass += w
            weighted_hit += w * float(B[ju, jv].item())

    return _safe_scalar_div(weighted_hit, total_mass)


def compute_avg_qubit_cost_used(m: torch.Tensor, c1: torch.Tensor, M_map: torch.Tensor) -> float:
    """
    Average per-qubit cost used by the hard final mapping, weighted by logical mask m.
    """
    m = _ensure_batched_vector(m, "m")
    c1 = _ensure_batched_vector(c1, "c1")
    M_map = _ensure_batched_square_matrix(M_map, "M_map")

    assign = permutation_matrix_to_assignment_vector(M_map)
    m0 = m[0]
    c10 = c1[0]

    weighted = 0.0
    total = 0.0

    for u in range(m0.shape[0]):
        w = float(m0[u].item())
        j = int(assign[u].item())
        weighted += w * float(c10[j].item())
        total += w

    return _safe_scalar_div(weighted, total)


def compute_avg_edge_cost_used(A: torch.Tensor, Bmat: torch.Tensor, c2: torch.Tensor, M_map: torch.Tensor) -> float:
    """
    Average per-edge cost used by the hard final mapping, weighted by A_off.
    """
    A = _ensure_batched_square_matrix(A, "A")
    Bmat = _ensure_batched_square_matrix(Bmat, "Bmat")
    c2 = _ensure_batched_square_matrix(c2, "c2")
    M_map = _ensure_batched_square_matrix(M_map, "M_map")

    A_off = _offdiag(A)[0]
    C2_use = (Bmat * c2)[0]
    assign = permutation_matrix_to_assignment_vector(M_map)

    N = A_off.shape[0]
    weighted = 0.0
    total_mass = 0.0

    for u in range(N):
        ju = int(assign[u].item())
        for v in range(N):
            jv = int(assign[v].item())
            w = float(A_off[u, v].item())
            weighted += w * float(C2_use[ju, jv].item())
            total_mass += w

    return _safe_scalar_div(weighted, total_mass)


# =============================================================================
# 9) Routed downstream evaluation helpers
# =============================================================================

ROUTED_STATUS_NOT_RUN = "not_run"
ROUTED_STATUS_SUCCESS = "success"
ROUTED_STATUS_CIRCUIT_LOAD_ERROR = "circuit_load_error"
ROUTED_STATUS_LAYOUT_BUILD_ERROR = "layout_build_error"
ROUTED_STATUS_BACKEND_RESOLVE_ERROR = "backend_resolve_error"
ROUTED_STATUS_TRANSPILE_ERROR = "transpile_error"
ROUTED_STATUS_ROUTING_CAPTURE_ERROR = "routing_capture_error"
ROUTED_STATUS_PST_COMPUTE_ERROR = "pst_compute_error"
ROUTED_STATUS_METRIC_COMPUTE_ERROR = "metric_compute_error"

ROUTED_STATUS_CODES = {
    ROUTED_STATUS_NOT_RUN,
    ROUTED_STATUS_SUCCESS,
    ROUTED_STATUS_CIRCUIT_LOAD_ERROR,
    ROUTED_STATUS_LAYOUT_BUILD_ERROR,
    ROUTED_STATUS_BACKEND_RESOLVE_ERROR,
    ROUTED_STATUS_TRANSPILE_ERROR,
    ROUTED_STATUS_ROUTING_CAPTURE_ERROR,
    ROUTED_STATUS_PST_COMPUTE_ERROR,
    ROUTED_STATUS_METRIC_COMPUTE_ERROR,
}


class RoutedEvalError(RuntimeError):
    """Base class for routed-evaluation failures."""


class CircuitLoadEvalError(RoutedEvalError):
    pass


class LayoutBuildEvalError(RoutedEvalError):
    pass


class BackendResolveEvalError(RoutedEvalError):
    pass


class TranspileEvalError(RoutedEvalError):
    pass


class RoutingCaptureEvalError(RoutedEvalError):
    pass


class PSTComputeEvalError(RoutedEvalError):
    pass


class MetricComputeEvalError(RoutedEvalError):
    pass


@dataclass
class RoutedEvalArtifacts:
    final_circuit: Any
    routing_stage_circuit: Any
    compile_time_s: float
    routing_method: str
    optimization_level: int


ROUTED_REAL_COLUMNS = [
    "routing_attempted",
    "routing_status",
    "routing_compile_time_s",
    "routing_total_eval_time_s",
    "real_pst_gate_readout",
    "swap_inserted_count",
    "original_2q_count",
    "routed_2q_count",
    "added_2q_ops",
    "original_depth",
    "routed_depth",
    "depth_increase_abs",
    "depth_increase_ratio",
]


def _routing_columns_enabled(eval_config: EvalConfig) -> bool:
    return bool(eval_config.route_final_eval or eval_config.include_routing_placeholders_in_csv)


def build_routing_placeholder_row() -> Dict[str, Any]:
    return {
        "routing_attempted": False,
        "routing_status": ROUTED_STATUS_NOT_RUN,
        "routing_compile_time_s": None,
        "routing_total_eval_time_s": None,
        "real_pst_gate_readout": None,
        "swap_inserted_count": None,
        "original_2q_count": None,
        "routed_2q_count": None,
        "added_2q_ops": None,
        "original_depth": None,
        "routed_depth": None,
        "depth_increase_abs": None,
        "depth_increase_ratio": None,
    }


def build_routed_failure_row(status: str) -> Dict[str, Any]:
    if status not in ROUTED_STATUS_CODES or status == ROUTED_STATUS_NOT_RUN:
        raise ValueError(f"Invalid routed failure status: {status}")

    row = build_routing_placeholder_row()
    row["routing_attempted"] = True
    row["routing_status"] = status
    for key in ROUTED_REAL_COLUMNS:
        if key in {"routing_attempted", "routing_status"}:
            continue
        row[key] = float("nan")
    return row


def build_routing_placeholder_summary() -> Dict[str, Any]:
    return {
        "implemented": False,
        "note": (
            "Routing-based downstream metrics are disabled for this run. "
            "Set route_final_eval=True to attach real routed-circuit metrics."
        ),
        "planned_fields": list(ROUTED_REAL_COLUMNS),
    }


def build_real_routing_summary(rows: Sequence[Dict[str, Any]], eval_config: EvalConfig) -> Dict[str, Any]:
    attempted_rows = [row for row in rows if row.get("routing_attempted") is True]
    status_breakdown: Dict[str, int] = {}
    for row in attempted_rows:
        status = str(row.get("routing_status", ROUTED_STATUS_NOT_RUN))
        status_breakdown[status] = status_breakdown.get(status, 0) + 1

    successful = sum(1 for row in attempted_rows if row.get("routing_status") == ROUTED_STATUS_SUCCESS)
    failures = sum(1 for row in attempted_rows if row.get("routing_status") != ROUTED_STATUS_SUCCESS)

    return {
        "implemented": True,
        "backend_name": eval_config.backend_name,
        "routing_method": eval_config.routing_method,
        "optimization_level": eval_config.transpile_optimization_level,
        "include_readout_in_pst": eval_config.include_readout_in_pst,
        "successful_routing_rows": successful,
        "routing_failure_rows": failures,
        "routing_status_breakdown": status_breakdown,
    }


def _get_project_root(eval_config: EvalConfig) -> Path:
    return Path(eval_config.project_root).resolve()


def _instruction_name(instruction: Any) -> str:
    return str(getattr(instruction, "name", instruction.__class__.__name__)).lower()


def _circuit_instruction_parts(item: Any) -> tuple[Any, Any]:
    """
    Support both modern CircuitInstruction objects and tuple-like old forms.
    """
    if hasattr(item, "operation") and hasattr(item, "qubits"):
        return item.operation, item.qubits
    return item[0], item[1]


def _qubit_index_in_circuit(circuit: Any, qubit: Any) -> int:
    try:
        return int(circuit.find_bit(qubit).index)
    except Exception:
        return int(getattr(qubit, "index"))


@lru_cache(maxsize=8)
def _resolve_backend_cached(backend_name: str) -> Any:
    from kmw.preprocessing.pipeline import resolve_backend
    return resolve_backend(backend_name=backend_name, num_qubits=27)


def load_original_circuit_for_sample(sample: Dict[str, Any], eval_config: EvalConfig) -> Any:
    qasm_relpath = get_meta_value(sample, "qasm_relpath", None)
    if qasm_relpath is None:
        raise CircuitLoadEvalError("Sample metadata does not contain qasm_relpath.")

    qasm_path = _get_project_root(eval_config) / str(qasm_relpath)
    if not qasm_path.exists():
        raise CircuitLoadEvalError(f"QASM file not found: {qasm_path}")

    try:
        from kmw.preprocessing.pipeline import load_quantum_circuit
        return load_quantum_circuit(qasm_path)
    except Exception as exc:
        raise CircuitLoadEvalError(f"Failed to load original circuit from {qasm_path}") from exc


def resolve_eval_backend(eval_config: EvalConfig) -> Any:
    try:
        return _resolve_backend_cached(str(eval_config.backend_name))
    except Exception as exc:
        raise BackendResolveEvalError(
            f"Failed to resolve backend '{eval_config.backend_name}'"
        ) from exc


def build_initial_layout_from_assignment(circuit: Any, M_map: torch.Tensor) -> Any:
    try:
        from qiskit.transpiler import Layout
    except Exception as exc:
        raise LayoutBuildEvalError("Qiskit Layout could not be imported.") from exc

    assignment = permutation_matrix_to_assignment_vector(M_map).detach().cpu().tolist()
    k = int(getattr(circuit, "num_qubits", 0))
    if k < 1:
        raise LayoutBuildEvalError("Circuit has no qubits.")
    if len(assignment) < k:
        raise LayoutBuildEvalError(
            f"Assignment length {len(assignment)} is smaller than circuit qubit count {k}."
        )

    physical = [int(x) for x in assignment[:k]]
    if len(set(physical)) != len(physical):
        raise LayoutBuildEvalError(
            f"Initial layout uses duplicate physical qubits: {physical}"
        )

    try:
        return Layout({circuit.qubits[u]: physical[u] for u in range(k)})
    except Exception as exc:
        raise LayoutBuildEvalError("Failed to build Qiskit Layout from hard mapping.") from exc


def count_two_qubit_ops(circuit: Any) -> int:
    count = 0
    for item in getattr(circuit, "data", []):
        instruction, qargs = _circuit_instruction_parts(item)
        if len(qargs) != 2:
            continue
        op_name = _instruction_name(instruction)
        if op_name in {"barrier", "delay", "measure", "reset"}:
            continue
        count += 1
    return count


def compute_circuit_depth_no_barrier(circuit: Any) -> int:
    try:
        return int(
            circuit.depth(
                filter_function=lambda instruction: _instruction_name(instruction.operation)
                not in {"barrier", "delay"}
            )
        )
    except Exception:
        return int(circuit.depth())


def extract_swap_inserted_count(routing_stage_circuit: Any) -> int:
    count = 0
    for item in getattr(routing_stage_circuit, "data", []):
        instruction, _ = _circuit_instruction_parts(item)
        if _instruction_name(instruction) == "swap":
            count += 1
    return count


def lookup_instruction_error(backend: Any, op_name: str, qargs: tuple[int, ...]) -> float:
    op_name = str(op_name).lower()
    target = getattr(backend, "target", None)
    if target is None:
        raise PSTComputeEvalError("Backend does not expose a target object.")

    props = None
    try:
        props = target[op_name][tuple(qargs)]
    except Exception:
        try:
            props = target[op_name].get(tuple(qargs))
        except Exception:
            props = None

    if props is None and op_name == "measure" and len(qargs) == 1:
        try:
            props = target["measure"][(int(qargs[0]),)]
        except Exception:
            props = None

    if props is None:
        raise PSTComputeEvalError(
            f"No backend instruction properties found for op={op_name}, qargs={qargs}."
        )

    error = getattr(props, "error", None)
    if error is None and isinstance(props, dict):
        error = props.get("error")

    if error is None:
        raise PSTComputeEvalError(
            f"Backend instruction properties for op={op_name}, qargs={qargs} do not define error."
        )

    error = float(error)
    if not math.isfinite(error):
        raise PSTComputeEvalError(
            f"Non-finite error for op={op_name}, qargs={qargs}: {error}"
        )
    return error


def estimate_real_pst_gate_readout(circuit: Any, backend: Any, include_readout: bool = True) -> float:
    success = 1.0

    for item in getattr(circuit, "data", []):
        instruction, qargs = _circuit_instruction_parts(item)
        op_name = _instruction_name(instruction)

        if op_name in {"barrier", "delay"}:
            continue
        if op_name == "measure" and not include_readout:
            continue

        qarg_indices = tuple(_qubit_index_in_circuit(circuit, q) for q in qargs)
        error = lookup_instruction_error(backend, op_name, qarg_indices)
        success *= max(0.0, min(1.0, 1.0 - error))

    return float(success)


def run_routed_transpile(circuit: Any, backend: Any, initial_layout: Any, eval_config: EvalConfig) -> RoutedEvalArtifacts:
    try:
        from qiskit import transpile
        from qiskit.converters import dag_to_circuit
    except Exception as exc:
        raise TranspileEvalError("Qiskit transpiler imports failed.") from exc

    routing_stage_circuit = None
    expected_routing_pass_name = {
        "sabre": "SabreSwap",
        "basic": "BasicSwap",
        "lookahead": "LookaheadSwap",
    }.get(str(eval_config.routing_method).lower())

    def callback(**kwargs: Any) -> None:
        nonlocal routing_stage_circuit
        if expected_routing_pass_name is None:
            return

        pass_obj = kwargs.get("pass_", None)
        dag = kwargs.get("dag", None)
        if pass_obj is None or dag is None:
            return

        if pass_obj.__class__.__name__ == expected_routing_pass_name:
            try:
                routing_stage_circuit = dag_to_circuit(dag)
            except Exception as exc:
                raise RoutingCaptureEvalError("Failed to convert routing-stage DAG to circuit.") from exc

    start = time.perf_counter()
    try:
        final_circuit = transpile(
            circuit,
            backend=backend,
            initial_layout=initial_layout,
            routing_method=eval_config.routing_method,
            optimization_level=eval_config.transpile_optimization_level,
            seed_transpiler=eval_config.seed_transpiler,
            callback=callback,
        )
    except RoutingCaptureEvalError:
        raise
    except Exception as exc:
        raise TranspileEvalError("Transpilation failed during routed evaluation.") from exc
    compile_time_s = time.perf_counter() - start

    # If no routing pass was captured, the layout may already satisfy connectivity.
    if routing_stage_circuit is None:
        routing_stage_circuit = final_circuit

    return RoutedEvalArtifacts(
        final_circuit=final_circuit,
        routing_stage_circuit=routing_stage_circuit,
        compile_time_s=float(compile_time_s),
        routing_method=str(eval_config.routing_method),
        optimization_level=int(eval_config.transpile_optimization_level),
    )


def compute_real_routed_metrics(
    original_circuit: Any,
    final_circuit: Any,
    routing_stage_circuit: Any,
    backend: Any,
    eval_config: EvalConfig,
    compile_time_s: float,
) -> Dict[str, Any]:
    try:
        original_2q_count = count_two_qubit_ops(original_circuit)
        routed_2q_count = count_two_qubit_ops(final_circuit)
        swap_inserted_count = extract_swap_inserted_count(routing_stage_circuit)
        original_depth = compute_circuit_depth_no_barrier(original_circuit)
        routed_depth = compute_circuit_depth_no_barrier(final_circuit)
    except Exception as exc:
        raise MetricComputeEvalError("Failed to compute routed circuit counts/depths.") from exc

    try:
        real_pst_gate_readout = estimate_real_pst_gate_readout(
            final_circuit,
            backend,
            include_readout=eval_config.include_readout_in_pst,
        )
    except Exception as exc:
        raise PSTComputeEvalError("Failed to compute real PST on routed circuit.") from exc

    depth_increase_abs = int(routed_depth - original_depth)
    depth_increase_ratio = float(routed_depth / max(original_depth, 1))
    added_2q_ops = int(routed_2q_count - original_2q_count)

    return {
        "routing_attempted": True,
        "routing_status": ROUTED_STATUS_SUCCESS,
        "routing_compile_time_s": float(compile_time_s),
        "real_pst_gate_readout": float(real_pst_gate_readout),
        "swap_inserted_count": int(swap_inserted_count),
        "original_2q_count": int(original_2q_count),
        "routed_2q_count": int(routed_2q_count),
        "added_2q_ops": int(added_2q_ops),
        "original_depth": int(original_depth),
        "routed_depth": int(routed_depth),
        "depth_increase_abs": int(depth_increase_abs),
        "depth_increase_ratio": float(depth_increase_ratio),
    }


def maybe_save_routed_artifacts(circuit_id: str, final_circuit: Any, eval_config: EvalConfig) -> None:
    if eval_config.save_routed_qpy_dir:
        try:
            from qiskit import qpy

            out_dir = Path(eval_config.save_routed_qpy_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{circuit_id}.qpy", "wb") as f:
                qpy.dump(final_circuit, f)
        except Exception:
            pass

    if eval_config.save_routed_qasm_dir:
        out_dir = Path(eval_config.save_routed_qasm_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{circuit_id}.qasm"
        dumped = None

        try:
            from qiskit import qasm2
            dumped = qasm2.dumps(final_circuit)
        except Exception:
            try:
                from qiskit import qasm3
                dumped = qasm3.dumps(final_circuit)
            except Exception:
                dumped = None

        if dumped is not None:
            out_path.write_text(dumped, encoding="utf-8")


# =============================================================================
# 10) One-sample inference pipeline
# =============================================================================

def run_single_sample_inference(
    model: KMWModel,
    sample: Dict[str, Any],
    device: torch.device,
    loss_config: Optional[LossConfig] = None,
    eval_config: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    """
    Run the full inference-time evaluation path for one sample.

    Structure
    ---------
    1. soft reindexer forward
    2. entropy diagnostics on R_L / R_H
    3. harden R_L / R_H with Hungarian
    4. reorder native tensors using hard reindexing
    5. build hardware tokens from hard-reordered hardware tensors
    6. mapper forward on hard-reordered latent tensors
    7. decode back to native frame using hard reindex matrices
    8. final Hungarian assignment on native logits
    9. compute proxy metrics / timings / validity
    10. if enabled, run real routed downstream evaluation
    """
    if loss_config is None:
        loss_config = LossConfig()
    if eval_config is None:
        eval_config = EvalConfig()

    sample = move_batch_to_device(sample, device)
    native = extract_native_batch(sample)

    circuit_id = infer_circuit_id(sample)
    source = infer_source(sample)
    k_logical = infer_k_logical(sample, native)
    num_1q = infer_num_1q(sample)
    num_2q = infer_num_2q(sample)

    t_total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # 1) Soft reindexer forward
    # -------------------------------------------------------------------------
    t_reindex_start = time.perf_counter()

    with torch.no_grad():
        soft_reidx = model.reindexer(
            A=native["A"],
            m=native["m"],
            Bmat=native["B"],
            c1=native["c1"],
            c2=native["c2"],
            D=native["D"],
            tau_r=1.0,
            freeze_hardware_reindex=eval_config.freeze_hardware_reindex,
        )

        R_L_soft = soft_reidx["R_L"]
        R_H_soft = soft_reidx["R_H"]

        RL_entropy = mean_row_entropy(R_L_soft, normalized=True)
        RH_entropy = mean_row_entropy(R_H_soft, normalized=True)

        R_L_hat = harden_soft_permutation(R_L_soft)
        R_H_hat = harden_soft_permutation(R_H_soft)

        hard_reordered = model.reindexer.reorder_all(
            A=native["A"],
            m=native["m"],
            Bmat=native["B"],
            c1=native["c1"],
            c2=native["c2"],
            D=native["D"],
            R_L=R_L_hat,
            R_H=R_H_hat,
        )

    reindex_time = time.perf_counter() - t_reindex_start

    # -------------------------------------------------------------------------
    # 2) Mapper forward on hard-reordered latent tensors
    # -------------------------------------------------------------------------
    t_mapper_start = time.perf_counter()

    with torch.no_grad():
        T_hw_star = model.token_encoder(
            B_star=hard_reordered["B_star"],
            c2_star=hard_reordered["c2_star"],
            c1_star=hard_reordered["c1_star"],
        )

        A_star_spatial = hard_reordered["A_star"].unsqueeze(1)
        S_star = model.mapper(A_star_spatial, T_hw_star)

        S_nat = decode_to_native(
            S_star=S_star,
            R_L=R_L_hat,
            R_H=R_H_hat,
        )

    mapper_forward_time = time.perf_counter() - t_mapper_start

    # -------------------------------------------------------------------------
    # 3) Final hard assignment in native frame
    # -------------------------------------------------------------------------
    t_hungarian_start = time.perf_counter()
    with torch.no_grad():
        M_map = hungarian_permutation_matrix(S_nat)
    hungarian_time = time.perf_counter() - t_hungarian_start

    inference_time_total = time.perf_counter() - t_total_start

    # -------------------------------------------------------------------------
    # 4) Validity checks
    # -------------------------------------------------------------------------
    mapping_valid = (
        is_valid_permutation_matrix(R_L_hat, tol=eval_config.perm_tol)
        and is_valid_permutation_matrix(R_H_hat, tol=eval_config.perm_tol)
        and is_valid_permutation_matrix(M_map, tol=eval_config.perm_tol)
    )

    # -------------------------------------------------------------------------
    # 5) Loss-style metrics under the hard final mapping
    # -------------------------------------------------------------------------
    with torch.no_grad():
        hard_task = compute_task_loss_from_assignment(
            A=native["A"],
            m=native["m"],
            Bmat=native["B"],
            D_raw=native["D_raw"],
            n1q=native["n1q"],
            nmeas=native["nmeas"],
            e1q=native["e1q"],
            ero=native["ero"],
            e2q=native["e2q"],
            P_map=M_map,
            config=loss_config,
        )

    # -------------------------------------------------------------------------
    # 6) Additional research metrics
    # -------------------------------------------------------------------------
    weighted_adj_hit_rate = compute_weighted_adj_hit_rate(
        A=native["A"],
        Bmat=native["B"],
        M_map=M_map,
    )

    avg_qubit_cost_used = compute_avg_qubit_cost_used(
        m=native["m"],
        c1=native["c1"],
        M_map=M_map,
    )

    avg_edge_cost_used = compute_avg_edge_cost_used(
        A=native["A"],
        Bmat=native["B"],
        c2=native["c2"],
        M_map=M_map,
    )

    row = {
        "circuit_id": circuit_id,
        "source": source,
        "k_logical": k_logical,
        "num_1q": num_1q,
        "num_2q": num_2q,
        "loss_1q": float(hard_task["L_1q"].item()),
        "loss_ro": float(hard_task["L_ro"].item()),
        "loss_2q": float(hard_task["L_2q"].item()),
        "loss_native": float(hard_task["L_native"].item()),
        "loss_route": float(hard_task["L_route"].item()),
        "task_loss": float(hard_task["L_task"].item()),
        "score_proxy_1q": float(hard_task["S_proxy_1q"].item()),
        "score_proxy_ro": float(hard_task["S_proxy_ro"].item()),
        "score_proxy_2q": float(hard_task["S_proxy_2q"].item()),
        "score_proxy_route": float(hard_task["S_proxy_route"].item()),
        "score_proxy_exec": float(hard_task["S_proxy_exec"].item()),
        "weighted_adj_hit_rate": float(weighted_adj_hit_rate),
        "avg_qubit_cost_used": float(avg_qubit_cost_used),
        "avg_edge_cost_used": float(avg_edge_cost_used),
        "inference_time_total": float(inference_time_total),
        "reindex_time": float(reindex_time),
        "mapper_forward_time": float(mapper_forward_time),
        "hungarian_time": float(hungarian_time),
        "mapping_valid": bool(mapping_valid),
        "RL_entropy": float(RL_entropy),
        "RH_entropy": float(RH_entropy),
        "error_type": None,
    }

    if not eval_config.route_final_eval:
        if _routing_columns_enabled(eval_config):
            row.update(build_routing_placeholder_row())
        return row

    routed_eval_start = time.perf_counter()
    try:
        original_circuit = load_original_circuit_for_sample(sample, eval_config)
        backend = resolve_eval_backend(eval_config)
        initial_layout = build_initial_layout_from_assignment(original_circuit, M_map)

        artifacts = run_routed_transpile(
            circuit=original_circuit,
            backend=backend,
            initial_layout=initial_layout,
            eval_config=eval_config,
        )

        routed_metrics = compute_real_routed_metrics(
            original_circuit=original_circuit,
            final_circuit=artifacts.final_circuit,
            routing_stage_circuit=artifacts.routing_stage_circuit,
            backend=backend,
            eval_config=eval_config,
            compile_time_s=artifacts.compile_time_s,
        )
        routed_metrics["routing_total_eval_time_s"] = float(time.perf_counter() - routed_eval_start)

        row.update(routed_metrics)
        maybe_save_routed_artifacts(circuit_id, artifacts.final_circuit, eval_config)
        return row

    except CircuitLoadEvalError as exc:
        row.update(build_routed_failure_row(ROUTED_STATUS_CIRCUIT_LOAD_ERROR))
        row["error_type"] = type(exc).__name__
        if eval_config.fail_fast:
            raise
        return row
    except BackendResolveEvalError as exc:
        row.update(build_routed_failure_row(ROUTED_STATUS_BACKEND_RESOLVE_ERROR))
        row["error_type"] = type(exc).__name__
        if eval_config.fail_fast:
            raise
        return row
    except LayoutBuildEvalError as exc:
        row.update(build_routed_failure_row(ROUTED_STATUS_LAYOUT_BUILD_ERROR))
        row["error_type"] = type(exc).__name__
        if eval_config.fail_fast:
            raise
        return row
    except RoutingCaptureEvalError as exc:
        row.update(build_routed_failure_row(ROUTED_STATUS_ROUTING_CAPTURE_ERROR))
        row["error_type"] = type(exc).__name__
        if eval_config.fail_fast:
            raise
        return row
    except TranspileEvalError as exc:
        row.update(build_routed_failure_row(ROUTED_STATUS_TRANSPILE_ERROR))
        row["error_type"] = type(exc).__name__
        if eval_config.fail_fast:
            raise
        return row
    except PSTComputeEvalError as exc:
        row.update(build_routed_failure_row(ROUTED_STATUS_PST_COMPUTE_ERROR))
        row["error_type"] = type(exc).__name__
        if eval_config.fail_fast:
            raise
        return row
    except MetricComputeEvalError as exc:
        row.update(build_routed_failure_row(ROUTED_STATUS_METRIC_COMPUTE_ERROR))
        row["error_type"] = type(exc).__name__
        if eval_config.fail_fast:
            raise
        return row


# =============================================================================
# 11) Failure-row helper
# =============================================================================

def build_failure_row(
    sample: Dict[str, Any],
    native: Optional[Dict[str, torch.Tensor]] = None,
    error: Optional[Exception] = None,
    eval_config: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    """Build a CSV-compatible failure row so evaluation can continue."""
    if eval_config is None:
        eval_config = EvalConfig()

    if native is None:
        native = {}

    circuit_id = infer_circuit_id(sample)
    source = infer_source(sample)

    if native and "m" in native:
        k_logical = infer_k_logical(sample, native)
    else:
        k_logical = get_meta_value(sample, "k_logical", None)

    row = {
        "circuit_id": circuit_id,
        "source": source,
        "k_logical": k_logical,
        "num_1q": infer_num_1q(sample),
        "num_2q": infer_num_2q(sample),
        "loss_1q": float("nan"),
        "loss_ro": float("nan"),
        "loss_2q": float("nan"),
        "loss_native": float("nan"),
        "loss_route": float("nan"),
        "task_loss": float("nan"),
        "score_proxy_1q": float("nan"),
        "score_proxy_ro": float("nan"),
        "score_proxy_2q": float("nan"),
        "score_proxy_route": float("nan"),
        "score_proxy_exec": float("nan"),
        "weighted_adj_hit_rate": float("nan"),
        "avg_qubit_cost_used": float("nan"),
        "avg_edge_cost_used": float("nan"),
        "inference_time_total": float("nan"),
        "reindex_time": float("nan"),
        "mapper_forward_time": float("nan"),
        "hungarian_time": float("nan"),
        "mapping_valid": False,
        "RL_entropy": float("nan"),
        "RH_entropy": float("nan"),
        "error_type": None if error is None else type(error).__name__,
    }

    if _routing_columns_enabled(eval_config):
        row.update(build_routing_placeholder_row())

    return row


# =============================================================================
# 12) CSV writer
# =============================================================================

MIN_RESEARCH_COLUMNS = [
    "circuit_id",
    "source",
    "k_logical",
    "num_1q",
    "num_2q",
    "loss_1q",
    "loss_ro",
    "loss_2q",
    "loss_native",
    "loss_route",
    "task_loss",
    "score_proxy_1q",
    "score_proxy_ro",
    "score_proxy_2q",
    "score_proxy_route",
    "score_proxy_exec",
    "weighted_adj_hit_rate",
    "avg_qubit_cost_used",
    "avg_edge_cost_used",
    "inference_time_total",
    "reindex_time",
    "mapper_forward_time",
    "hungarian_time",
    "mapping_valid",
    "RL_entropy",
    "RH_entropy",
]



def infer_csv_columns(rows: List[Dict[str, Any]], eval_config: EvalConfig) -> List[str]:
    """Build the CSV column order."""
    columns = list(MIN_RESEARCH_COLUMNS)

    if _routing_columns_enabled(eval_config):
        columns.extend(ROUTED_REAL_COLUMNS)

    seen = set(columns)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                columns.append(key)
                seen.add(key)

    return columns


def write_per_circuit_csv(
    rows: List[Dict[str, Any]],
    csv_path: str,
    eval_config: EvalConfig,
) -> str:
    """Write the per-circuit research CSV and return the path string."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = infer_csv_columns(rows, eval_config)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return str(path)


# =============================================================================
# 13) Summary statistics helpers
# =============================================================================

def is_finite_number(x: Any) -> bool:
    """True if x is an int/float and finite."""
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def finite_values_for_key(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    """Extract finite numeric values for one key across rows."""
    vals = []
    for row in rows:
        value = row.get(key, None)
        if is_finite_number(value):
            vals.append(float(value))
    return vals


def summarize_numeric_values(values: List[float]) -> Dict[str, Any]:
    """Compute mean / std / median / count for a list of numbers."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }

    if len(values) == 1:
        std = 0.0
    else:
        std = statistics.pstdev(values)

    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(std),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_rows_for_keys(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Summarize multiple numeric columns across rows."""
    return {key: summarize_numeric_values(finite_values_for_key(rows, key)) for key in keys}


def group_rows_by_key(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    """Group rows by a given dictionary key."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        group_value = row.get(key, None)
        group_key = str(group_value)
        groups.setdefault(group_key, []).append(row)
    return groups


def compute_failure_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute failure-related counts.
    """
    total = len(rows)
    invalid_mapping_count = sum(1 for r in rows if r.get("mapping_valid") is False)
    exception_rows = [r for r in rows if r.get("error_type") not in (None, "", "None")]
    exception_count = len(exception_rows)
    success_count = total - exception_count

    error_type_breakdown: Dict[str, int] = {}
    for row in exception_rows:
        err = str(row.get("error_type"))
        error_type_breakdown[err] = error_type_breakdown.get(err, 0) + 1

    return {
        "total": total,
        "success_count": success_count,
        "invalid_mapping_count": invalid_mapping_count,
        "exception_count": exception_count,
        "error_type_breakdown": error_type_breakdown,
    }


SUMMARY_NUMERIC_KEYS = [
    "loss_1q",
    "loss_ro",
    "loss_2q",
    "loss_native",
    "loss_route",
    "task_loss",
    "score_proxy_1q",
    "score_proxy_ro",
    "score_proxy_2q",
    "score_proxy_route",
    "score_proxy_exec",
    "weighted_adj_hit_rate",
    "avg_qubit_cost_used",
    "avg_edge_cost_used",
    "inference_time_total",
    "reindex_time",
    "mapper_forward_time",
    "hungarian_time",
    "RL_entropy",
    "RH_entropy",
    "routing_compile_time_s",
    "routing_total_eval_time_s",
    "real_pst_gate_readout",
    "swap_inserted_count",
    "original_2q_count",
    "routed_2q_count",
    "added_2q_ops",
    "original_depth",
    "routed_depth",
    "depth_increase_abs",
    "depth_increase_ratio",
]



def build_summary(rows: List[Dict[str, Any]], eval_config: Optional[EvalConfig] = None) -> Dict[str, Any]:
    """
    Build the summary JSON structure.
    """
    if eval_config is None:
        eval_config = EvalConfig()

    overall = summarize_rows_for_keys(rows, SUMMARY_NUMERIC_KEYS)

    by_source = {}
    for source, source_rows in group_rows_by_key(rows, "source").items():
        by_source[source] = {
            "count": len(source_rows),
            "metrics": summarize_rows_for_keys(source_rows, SUMMARY_NUMERIC_KEYS),
            "failures": compute_failure_counts(source_rows),
        }

    by_k = {}
    for k_value, k_rows in group_rows_by_key(rows, "k_logical").items():
        by_k[k_value] = {
            "count": len(k_rows),
            "metrics": summarize_rows_for_keys(k_rows, SUMMARY_NUMERIC_KEYS),
            "failures": compute_failure_counts(k_rows),
        }

    summary = {
        "eval_split_name": eval_config.eval_split_name,
        "num_rows": len(rows),
        "overall": overall,
        "by_source": by_source,
        "by_k_logical": by_k,
        "failure_counts": compute_failure_counts(rows),
        "routing": (
            build_real_routing_summary(rows, eval_config)
            if eval_config.route_final_eval
            else build_routing_placeholder_summary()
        ),
        "research_csv_columns_minimum": list(MIN_RESEARCH_COLUMNS),
    }

    return summary


def write_summary_json(summary: Dict[str, Any], json_path: str) -> str:
    """Write the summary JSON report."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return str(path)


# =============================================================================
# 14) Console summary
# =============================================================================

def _fmt_stat_block(metric_summary: Dict[str, Any]) -> str:
    """
    Format one metric summary line as:
        mean ± std | median | n
    """
    mean = metric_summary.get("mean", None)
    std = metric_summary.get("std", None)
    median = metric_summary.get("median", None)
    count = metric_summary.get("count", 0)

    def fmt(x: Any) -> str:
        if x is None:
            return "None"
        return f"{float(x):.6f}"

    return f"{fmt(mean)} ± {fmt(std)} | median={fmt(median)} | n={count}"


def print_console_summary(summary: Dict[str, Any]) -> None:
    """
    Print a compact console summary.
    """
    print("\n" + "=" * 80)
    print(f"KMW Evaluation Summary [{summary.get('eval_split_name', 'eval')}]")
    print("=" * 80)

    failures = summary["failure_counts"]
    print(
        f"Total={failures['total']} | "
        f"Success={failures['success_count']} | "
        f"InvalidMapping={failures['invalid_mapping_count']} | "
        f"Exceptions={failures['exception_count']}"
    )

    print("\nOverall metrics:")
    for key in [
        "task_loss",
        "score_proxy_exec",
        "loss_native",
        "loss_route",
        "weighted_adj_hit_rate",
        "avg_qubit_cost_used",
        "avg_edge_cost_used",
        "inference_time_total",
        "RL_entropy",
        "RH_entropy",
    ]:
        print(f"  {key:24s} {_fmt_stat_block(summary['overall'][key])}")

    routing_block = summary.get("routing", {})
    if routing_block.get("implemented", False):
        print("\nRouted downstream metrics:")
        for key in [
            "real_pst_gate_readout",
            "swap_inserted_count",
            "added_2q_ops",
            "routing_compile_time_s",
            "depth_increase_abs",
            "depth_increase_ratio",
        ]:
            if key in summary["overall"]:
                print(f"  {key:24s} {_fmt_stat_block(summary['overall'][key])}")

    print("\nPer-source breakdown:")
    for source, block in summary["by_source"].items():
        print(f"  [{source}] count={block['count']}")
        print(f"    task_loss              {_fmt_stat_block(block['metrics']['task_loss'])}")
        print(f"    score_proxy_exec       {_fmt_stat_block(block['metrics']['score_proxy_exec'])}")
        print(f"    weighted_adj_hit_rate  {_fmt_stat_block(block['metrics']['weighted_adj_hit_rate'])}")
        print(f"    inference_time_total   {_fmt_stat_block(block['metrics']['inference_time_total'])}")

    print("\nPer-K breakdown:")
    for k_value, block in sorted(summary["by_k_logical"].items(), key=lambda x: str(x[0])):
        print(f"  [K={k_value}] count={block['count']}")
        print(f"    task_loss              {_fmt_stat_block(block['metrics']['task_loss'])}")
        print(f"    score_proxy_exec       {_fmt_stat_block(block['metrics']['score_proxy_exec'])}")
        print(f"    weighted_adj_hit_rate  {_fmt_stat_block(block['metrics']['weighted_adj_hit_rate'])}")

    print("\nRouting:")
    print(f"  implemented = {routing_block.get('implemented', False)}")
    if routing_block.get("implemented", False):
        print(f"  backend_name = {routing_block.get('backend_name')}")
        print(f"  routing_method = {routing_block.get('routing_method')}")
        print(f"  optimization_level = {routing_block.get('optimization_level')}")
        print(f"  include_readout_in_pst = {routing_block.get('include_readout_in_pst')}")
        print(f"  successful_routing_rows = {routing_block.get('successful_routing_rows')}")
        print(f"  routing_failure_rows = {routing_block.get('routing_failure_rows')}")
        print(f"  routing_status_breakdown = {routing_block.get('routing_status_breakdown')}")
    else:
        print(f"  note = {routing_block.get('note')}")

    print("=" * 80 + "\n")


# =============================================================================
# 15) Main evaluation loop
# =============================================================================

def evaluate_loader(
    model: KMWModel,
    loader: Iterable[Dict[str, Any]],
    loss_config: Optional[LossConfig] = None,
    eval_config: Optional[EvalConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run full evaluation over a loader.
    """
    if loss_config is None:
        loss_config = LossConfig()
    if eval_config is None:
        eval_config = EvalConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    rows: List[Dict[str, Any]] = []

    for batch in loader:
        for sample in iter_samples_from_batch(batch):
            native = None
            try:
                sample = move_batch_to_device(sample, device)
                native = extract_native_batch(sample)

                row = run_single_sample_inference(
                    model=model,
                    sample=sample,
                    device=device,
                    loss_config=loss_config,
                    eval_config=eval_config,
                )
                rows.append(row)

            except Exception as e:
                if eval_config.fail_fast:
                    raise

                rows.append(
                    build_failure_row(
                        sample=sample,
                        native=native,
                        error=e,
                        eval_config=eval_config,
                    )
                )

    summary = build_summary(rows, eval_config=eval_config)

    per_circuit_csv_path = write_per_circuit_csv(
        rows,
        csv_path=eval_config.per_circuit_csv_path,
        eval_config=eval_config,
    )
    summary_json_path = write_summary_json(summary, eval_config.summary_json_path)

    if eval_config.print_console_summary:
        print_console_summary(summary)

    return {
        "rows": rows,
        "summary": summary,
        "per_circuit_csv_path": per_circuit_csv_path,
        "summary_json_path": summary_json_path,
    }


# =============================================================================
# 16) Convenience wrapper
# =============================================================================

def evaluate_model(
    model: KMWModel,
    loader: Iterable[Dict[str, Any]],
    device: Optional[torch.device] = None,
    loss_config: Optional[LossConfig] = None,
    eval_config: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    """
    Small convenience wrapper around evaluate_loader().
    """
    return evaluate_loader(
        model=model,
        loader=loader,
        device=device,
        loss_config=loss_config,
        eval_config=eval_config,
    )


# =============================================================================
# 17) Public exports
# =============================================================================

__all__ = [
    "EvalConfig",
    "evaluate_loader",
    "evaluate_model",
    "run_single_sample_inference",
    "harden_soft_permutation",
    "hungarian_permutation_matrix",
    "is_valid_permutation_matrix",
    "permutation_matrix_to_assignment_vector",
    "compute_weighted_adj_hit_rate",
    "compute_avg_qubit_cost_used",
    "compute_avg_edge_cost_used",
    "write_per_circuit_csv",
    "write_summary_json",
    "print_console_summary",
    "build_summary",
    "build_routing_placeholder_row",
    "build_routing_placeholder_summary",
    "build_real_routing_summary",
]



