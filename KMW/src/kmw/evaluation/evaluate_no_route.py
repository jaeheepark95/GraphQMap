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
7. write:
      - per-circuit CSV
      - summary JSON
      - console summary
8. keep a placeholder structure for future routing-based evaluation

Why this belongs here
---------------------
The combined design explicitly assigns evaluate.py to:
- inference
- hard reindexing
- dataset-level evaluation
- summary report generation

Also, the mapper backbone intentionally stops at latent logits S*.
Decode and assignment are outside the backbone, and Hungarian belongs on the
evaluation / inference side, not inside model.py. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

Important implementation notes
------------------------------
- Training uses soft assignment (Sinkhorn).
- Inference uses hard reindexing + hard final assignment.
- This file uses HARD permutations for:
      R_L_hat
      R_H_hat
      M_map
- The current project is still "initial mapping only".
  We therefore include a routing-placeholder structure now, so the next phase
  can attach real routed-circuit metrics later.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    """
    Evaluation-time configuration.

    Outputs
    -------
    per_circuit_csv_path:
        research CSV for later slicing / plotting

    summary_json_path:
        machine-readable summary report

    print_console_summary:
        whether to print a compact console summary table

    Routing placeholders
    --------------------
    The current project stage is still initial-mapping evaluation.
    But you asked to stub routing-related downstream metrics now so the next test
    phase can plug in the full routed-compiler procedure later.
    """

    per_circuit_csv_path: str = "artifacts/eval/per_circuit_metrics.csv"
    summary_json_path: str = "artifacts/eval/summary.json"
    print_console_summary: bool = True

    # Whether to add routing placeholder columns to the per-circuit CSV.
    include_routing_placeholders_in_csv: bool = True

    # If True, hard-fail on exceptions. Otherwise, record a failure row and continue.
    fail_fast: bool = False

    # Tolerance used when checking whether matrices are valid permutations.
    perm_tol: float = 1e-6

    # Optional tag describing the current evaluation split, e.g.:
    # "smoke_val", "test_qasmbench", "test_revlib", ...
    eval_split_name: str = "eval"


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


def extract_native_batch(sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Extract the native tensors needed by the model / losses.

    Required keys:
        A, m, B, c1, c2, D
    """
    required = ("A", "m", "B", "c1", "c2", "D")
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
    }

    for key, value in native.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Sample key '{key}' must be a tensor, got {type(value)}")
        _assert_finite_tensor(value, key)

    return native


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
    """
    Return a metadata dictionary if present, else {}.
    """
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
    """
    Best-effort source extraction from sample metadata.
    """
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
    """
    Try to recover num_1q from metadata.
    """
    value = get_meta_value(sample, "num_1q", None)
    return None if value is None else int(value)


def infer_num_2q(sample: Dict[str, Any]) -> Optional[int]:
    """
    Try to recover num_2q from metadata.
    """
    value = get_meta_value(sample, "num_2q", None)
    return None if value is None else int(value)


# =============================================================================
# 6) Hungarian helpers
# =============================================================================

def require_hungarian_available() -> None:
    """
    Fail with a clear error if SciPy's Hungarian solver is unavailable.
    """
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

    Input:
        R_soft : (1, N, N) or (N, N)

    Output:
        R_hat  : (1, N, N)

    Why Hungarian here?
    -------------------
    The design says inference replaces soft reindex matrices with hard permutations.
    So evaluate.py hardens R_L and R_H here before rerunning the latent-frame path. :contentReference[oaicite:4]{index=4}
    """
    return hungarian_permutation_matrix(R_soft)


# =============================================================================
# 7) Permutation / entropy / validity helpers
# =============================================================================

def mean_row_entropy(P: torch.Tensor, normalized: bool = True, eps: float = 1e-12) -> float:
    """
    Compute the mean row entropy of a soft permutation matrix.

    Input:
        P : (1, N, N) or (N, N)

    Output:
        Python float

    Interpretation
    --------------
    Lower = sharper / closer to one-hot
    Higher = more diffuse

    We default to normalized entropy:
        H(row) / log(N)
    so the result is roughly in [0, 1].
    """
    P = _ensure_batched_square_matrix(P, "P")
    if P.shape[0] != 1:
        raise ValueError("mean_row_entropy expects exactly one sample")

    probs = P[0].clamp(min=eps)
    H = -(probs * probs.log()).sum(dim=-1)  # row-wise entropy

    if normalized:
        N = probs.shape[-1]
        denom = math.log(max(N, 2))
        if denom > 0:
            H = H / denom

    return float(H.mean().item())


def is_valid_permutation_matrix(P: torch.Tensor, tol: float = 1e-6) -> bool:
    """
    Check whether a matrix is (approximately) a valid permutation matrix.

    Requirements:
    - square
    - entries close to 0/1
    - row sums close to 1
    - column sums close to 1
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

    # Optional strictness: because Hungarian constructs 0/1 matrices, this should hold.
    rounded = torch.round(M)
    if not torch.allclose(M, rounded, atol=tol, rtol=0.0):
        return False

    return True


def permutation_matrix_to_assignment_vector(P: torch.Tensor) -> torch.Tensor:
    """
    Convert a hard permutation matrix P[u, j] into an assignment vector:
        assign[u] = j

    Input:
        P : (1, N, N) or (N, N)

    Output:
        assign : (N,)
    """
    P = _ensure_batched_square_matrix(P, "P")
    if P.shape[0] != 1:
        raise ValueError("permutation_matrix_to_assignment_vector expects exactly one sample")

    return P[0].argmax(dim=-1)


# =============================================================================
# 8) Research metrics on a hard mapping
# =============================================================================

def _offdiag(A: torch.Tensor) -> torch.Tensor:
    """
    Zero the diagonal of a batched square matrix.
    """
    A = _ensure_batched_square_matrix(A, "A")
    _, N, _ = A.shape
    eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
    return A * (1.0 - eye)


def _safe_scalar_div(num: float, den: float) -> float:
    """
    Safe scalar division used for summary-style metrics.
    """
    if abs(den) < 1e-12:
        return 0.0
    return float(num / den)


def compute_weighted_adj_hit_rate(A: torch.Tensor, Bmat: torch.Tensor, M_map: torch.Tensor) -> float:
    """
    Weighted adjacency hit rate under the hard final mapping.

    Meaning
    -------
    Among the logical 2Q interaction mass in A_off, how much lands on an adjacent
    physical pair in B?

    Formula
    -------
    hit_rate =
        sum_{u,v} A_off[u,v] * B[ map(u), map(v) ]
        ------------------------------------------
              sum_{u,v} A_off[u,v]

    Returns:
        float in [0, 1] for well-formed binary B.
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

    Formula
    -------
    avg_qubit_cost_used =
        sum_u m[u] * c1[ map(u) ]
        ------------------------
             sum_u m[u]
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

    We use:
        C2_use = B * c2

    so non-edges do not contribute a fake edge cost.
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
# 9) Routing placeholder structure
# =============================================================================

def build_routing_placeholder_row() -> Dict[str, Any]:
    """
    Placeholder per-circuit routing fields for the next evaluation phase.

    Why include this now?
    ---------------------
    The current project stage evaluates the initial mapper only.
    But you explicitly asked to stub downstream routing metrics now so the next
    phase can attach real routed-circuit evaluation after convergence.

    These are placeholders only; they are NOT real routed metrics yet.
    """
    return {
        "routing_attempted": False,
        "routing_status": "not_run",
        "routing_swap_count": None,
        "routing_depth_after_routing": None,
        "routing_two_qubit_count_after_routing": None,
        "routing_valid": None,
        "routing_compile_time": None,
        "routing_total_time": None,
    }


def build_routing_placeholder_summary() -> Dict[str, Any]:
    """
    Placeholder summary structure for future routed-circuit metrics.
    """
    return {
        "implemented": False,
        "note": (
            "Routing-based downstream metrics are intentionally stubbed in the "
            "current phase. Attach real routed-circuit compilation metrics here "
            "after initial-mapper convergence is confirmed."
        ),
        "planned_fields": [
            "routing_swap_count",
            "routing_depth_after_routing",
            "routing_two_qubit_count_after_routing",
            "routing_valid",
            "routing_compile_time",
            "routing_total_time",
        ],
    }


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

    Locked inference-time structure
    -------------------------------
    1. soft reindexer forward
    2. compute entropy diagnostics on R_L / R_H
    3. harden R_L / R_H with Hungarian
    4. reorder native tensors using hard reindexing
    5. build hardware tokens from hard-reordered hardware tensors
    6. mapper forward on hard-reordered latent tensors
    7. decode back to native frame using hard reindex matrices
    8. final Hungarian assignment on native logits
    9. compute research metrics / timings / validity

    This matches the project decision that evaluate.py should perform hard
    reindexing and hard final assignment. :contentReference[oaicite:5]{index=5}
    """
    if loss_config is None:
        loss_config = LossConfig()
    if eval_config is None:
        eval_config = EvalConfig()

    sample = move_batch_to_device(sample, device)
    native = extract_native_batch(sample)

    # Metadata
    circuit_id = infer_circuit_id(sample)
    source = infer_source(sample)
    k_logical = infer_k_logical(sample, native)
    num_1q = infer_num_1q(sample)
    num_2q = infer_num_2q(sample)

    # Timing starts here
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
            tau_r=1.0,  # inference uses the current trained reindexer at its learned weights
        )

        R_L_soft = soft_reidx["R_L"]
        R_H_soft = soft_reidx["R_H"]

        # Entropy diagnostics computed BEFORE hardening
        RL_entropy = mean_row_entropy(R_L_soft, normalized=True)
        RH_entropy = mean_row_entropy(R_H_soft, normalized=True)

        # Hard reindexing
        R_L_hat = harden_soft_permutation(R_L_soft)
        R_H_hat = harden_soft_permutation(R_H_soft)

        # Reorder native tensors with HARD reindexing
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

    t_reindex_end = time.perf_counter()
    reindex_time = t_reindex_end - t_reindex_start

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

        # Decode back to native frame using HARD reindex matrices
        S_nat = decode_to_native(
            S_star=S_star,
            R_L=R_L_hat,
            R_H=R_H_hat,
        )

    t_mapper_end = time.perf_counter()
    mapper_forward_time = t_mapper_end - t_mapper_start

    # -------------------------------------------------------------------------
    # 3) Final hard assignment in native frame
    # -------------------------------------------------------------------------
    t_hungarian_start = time.perf_counter()

    with torch.no_grad():
        M_map = hungarian_permutation_matrix(S_nat)

    t_hungarian_end = time.perf_counter()
    hungarian_time = t_hungarian_end - t_hungarian_start

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
    # 5) Loss-style metrics computed under the HARD final mapping
    # -------------------------------------------------------------------------
    with torch.no_grad():
        hard_task = compute_task_loss_from_assignment(
            A=native["A"],
            m=native["m"],
            Bmat=native["B"],
            c1=native["c1"],
            c2=native["c2"],
            D=native["D"],
            P_map=M_map,  # hard assignment used as evaluation-time mapping
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
        # Required research identity columns
        "circuit_id": circuit_id,
        "source": source,
        "k_logical": k_logical,
        "num_1q": num_1q,
        "num_2q": num_2q,

        # Minimum requested loss/proxy columns
        "pst_proxy_1q": float(hard_task["L_pst_proxy_1q"].item()),
        "pst_proxy_2q": float(hard_task["L_pst_proxy_2q"].item()),
        "pst_proxy_total": float(hard_task["L_pst_proxy_total"].item()),
        "swap_proxy": float(hard_task["L_swap"].item()),
        "depth_proxy": float(hard_task["L_depth"].item()),
        "task_loss": float(hard_task["L_task"].item()),

        # Minimum requested research-analysis columns
        "weighted_adj_hit_rate": float(weighted_adj_hit_rate),
        "avg_qubit_cost_used": float(avg_qubit_cost_used),
        "avg_edge_cost_used": float(avg_edge_cost_used),

        # Minimum requested timing columns
        "inference_time_total": float(inference_time_total),
        "reindex_time": float(reindex_time),
        "mapper_forward_time": float(mapper_forward_time),
        "hungarian_time": float(hungarian_time),

        # Validity / sharpness diagnostics
        "mapping_valid": bool(mapping_valid),
        "RL_entropy": float(RL_entropy),
        "RH_entropy": float(RH_entropy),

        # Internal useful extras
        "error_type": None,
    }

    if eval_config.include_routing_placeholders_in_csv:
        row.update(build_routing_placeholder_row())

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
    """
    Build a CSV-compatible failure row so evaluation can continue even if one
    circuit fails.
    """
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

        "pst_proxy_1q": float("nan"),
        "pst_proxy_2q": float("nan"),
        "pst_proxy_total": float("nan"),
        "swap_proxy": float("nan"),
        "depth_proxy": float("nan"),
        "task_loss": float("nan"),

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

    if eval_config.include_routing_placeholders_in_csv:
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
    "pst_proxy_1q",
    "pst_proxy_2q",
    "pst_proxy_total",
    "swap_proxy",
    "depth_proxy",
    "task_loss",
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


ROUTING_PLACEHOLDER_COLUMNS = [
    "routing_attempted",
    "routing_status",
    "routing_swap_count",
    "routing_depth_after_routing",
    "routing_two_qubit_count_after_routing",
    "routing_valid",
    "routing_compile_time",
    "routing_total_time",
]


def infer_csv_columns(rows: List[Dict[str, Any]], include_routing_placeholders: bool) -> List[str]:
    """
    Build the CSV column order.

    Policy
    ------
    - start with the minimum requested research columns
    - optionally append routing placeholder columns
    - then append any extra keys seen in rows, preserving discovery order
    """
    columns = list(MIN_RESEARCH_COLUMNS)

    if include_routing_placeholders:
        columns.extend(ROUTING_PLACEHOLDER_COLUMNS)

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
    include_routing_placeholders: bool = True,
) -> str:
    """
    Write the per-circuit research CSV.

    Returns:
        path string
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = infer_csv_columns(rows, include_routing_placeholders)

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
    """
    True if x is an int/float and finite.
    """
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def finite_values_for_key(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    """
    Extract finite numeric values for one key across rows.
    """
    vals = []
    for row in rows:
        value = row.get(key, None)
        if is_finite_number(value):
            vals.append(float(value))
    return vals


def summarize_numeric_values(values: List[float]) -> Dict[str, Any]:
    """
    Compute mean / std / median / count for a list of numbers.
    """
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
    """
    Summarize multiple numeric columns across rows.
    """
    return {key: summarize_numeric_values(finite_values_for_key(rows, key)) for key in keys}


def group_rows_by_key(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group rows by a given dictionary key.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        group_value = row.get(key, None)
        group_key = str(group_value)
        groups.setdefault(group_key, []).append(row)
    return groups


def compute_failure_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute failure-related counts.

    Current counts:
    - total
    - success_count
    - invalid_mapping_count
    - exception_count
    - error_type_breakdown
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
    "pst_proxy_1q",
    "pst_proxy_2q",
    "pst_proxy_total",
    "swap_proxy",
    "depth_proxy",
    "task_loss",
    "weighted_adj_hit_rate",
    "avg_qubit_cost_used",
    "avg_edge_cost_used",
    "inference_time_total",
    "reindex_time",
    "mapper_forward_time",
    "hungarian_time",
    "RL_entropy",
    "RH_entropy",
]


def build_summary(rows: List[Dict[str, Any]], eval_config: Optional[EvalConfig] = None) -> Dict[str, Any]:
    """
    Build the summary JSON structure.

    You asked for:
    - overall mean/std/median
    - per-source breakdown
    - per-K breakdown
    - failure counts
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
        "routing_placeholder": build_routing_placeholder_summary(),
        "research_csv_columns_minimum": list(MIN_RESEARCH_COLUMNS),
    }

    return summary


def write_summary_json(summary: Dict[str, Any], json_path: str) -> str:
    """
    Write the summary JSON report.

    Returns:
        path string
    """
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

    This is intentionally simple plain-text output so it works anywhere.
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
        "pst_proxy_total",
        "swap_proxy",
        "depth_proxy",
        "weighted_adj_hit_rate",
        "avg_qubit_cost_used",
        "avg_edge_cost_used",
        "inference_time_total",
        "RL_entropy",
        "RH_entropy",
    ]:
        print(f"  {key:24s} { _fmt_stat_block(summary['overall'][key]) }")

    print("\nPer-source breakdown:")
    for source, block in summary["by_source"].items():
        print(f"  [{source}] count={block['count']}")
        print(f"    task_loss              { _fmt_stat_block(block['metrics']['task_loss']) }")
        print(f"    pst_proxy_total        { _fmt_stat_block(block['metrics']['pst_proxy_total']) }")
        print(f"    weighted_adj_hit_rate  { _fmt_stat_block(block['metrics']['weighted_adj_hit_rate']) }")
        print(f"    inference_time_total   { _fmt_stat_block(block['metrics']['inference_time_total']) }")

    print("\nPer-K breakdown:")
    for k_value, block in sorted(summary["by_k_logical"].items(), key=lambda x: str(x[0])):
        print(f"  [K={k_value}] count={block['count']}")
        print(f"    task_loss              { _fmt_stat_block(block['metrics']['task_loss']) }")
        print(f"    pst_proxy_total        { _fmt_stat_block(block['metrics']['pst_proxy_total']) }")
        print(f"    weighted_adj_hit_rate  { _fmt_stat_block(block['metrics']['weighted_adj_hit_rate']) }")

    print("\nRouting placeholder:")
    print(f"  implemented = {summary['routing_placeholder']['implemented']}")
    print(f"  note = {summary['routing_placeholder']['note']}")
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

    Returns:
        {
            "rows": [...],
            "summary": {...},
            "per_circuit_csv_path": "...",
            "summary_json_path": "...",
        }

    Behavior on errors
    ------------------
    - if fail_fast=True: raise immediately
    - else: record a failure row and continue
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
        include_routing_placeholders=eval_config.include_routing_placeholders_in_csv,
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
# 16) Convenience factory-style helper
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
]

# =============================================================================
# notes on implementation:
# =============================================================================

# I implemented hard reindexing at inference by hardening R_L and R_H with Hungarian, 
# then rerunning the latent-frame conditioning path with those hard permutations before doing the final Hungarian on S_nat.
# That matches the inference-side interpretation in the design.

# I defined weighted_adj_hit_rate as the A_off-weighted fraction of logical interaction mass that lands on adjacent physical pairs under the final hard mapping.
# That is a reasonable research metric for this project, but it is still a project-defined evaluation metric, not a standard benchmark metric from an external paper.

# I defined RL_entropy and RH_entropy as the normalized mean row entropy of the soft reindex matrices before hardening.
# That makes them interpretable on a roughly 0-to-1 scale.

# I used the hard final assignment M_map for the per-circuit proxy metrics in evaluation. 
# That keeps the reported CSV closer to actual inference behavior than reusing the training soft assignment.





