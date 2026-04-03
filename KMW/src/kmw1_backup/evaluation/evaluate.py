from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from kmw1.losses.loss import LossConfig, compute_task_loss_from_assignment, compute_task_loss_from_logits
from kmw1.models.model import KMWCanonicalModel, decode_canonical_to_native_logits
from kmw1.utils import ensure_dir, move_to_device, write_json

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


@dataclass
class EvalConfig:
    per_circuit_csv_path: str = "artifacts/kmw1/per_circuit_metrics.csv"
    summary_json_path: str = "artifacts/kmw1/summary.json"
    print_console_summary: bool = True
    fail_fast: bool = False
    eval_split_name: str = "eval"

    project_root: str = "."
    backend_name: str = "fake_toronto_v2"
    route_final_eval: bool = False
    routing_method: str = "sabre"
    transpile_optimization_level: int = 0
    seed_transpiler: int | None = None
    save_routed_qasm_dir: str | None = None
    save_routed_qpy_dir: str | None = None


def _assert_finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def _hungarian_assignment_single(S: torch.Tensor) -> torch.Tensor:
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for Hungarian assignment during evaluation.")
    cost = -S.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    M = torch.zeros_like(S)
    M[row_ind, col_ind] = 1.0
    return M


def _active_logical_count(m: torch.Tensor) -> int:
    return int((m > 0.5).sum().item())


def _hard_mapping_from_assignment(M_nat: torch.Tensor, m: torch.Tensor) -> list[int]:
    K = _active_logical_count(m)
    mapping: list[int] = []
    for u in range(K):
        phys = int(torch.argmax(M_nat[u]).item())
        mapping.append(phys)
    return mapping


def _load_circuit_from_qasm(qasm_path: str | Path):
    from qiskit import QuantumCircuit
    qasm_path = Path(qasm_path)
    try:
        return QuantumCircuit.from_qasm_file(str(qasm_path))
    except Exception:
        from qiskit.qasm3 import loads as qasm3_loads
        return qasm3_loads(qasm_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _resolve_backend_cached(backend_name: str):
    from kmw1.preprocessing.extractor import resolve_backend
    return resolve_backend(backend_name=backend_name)


def _run_routed_eval(
    *,
    qasm_path: str | Path,
    native_mapping: list[int],
    eval_config: EvalConfig,
) -> dict[str, Any]:
    from qiskit import transpile

    circuit = _load_circuit_from_qasm(qasm_path)
    backend = _resolve_backend_cached(eval_config.backend_name)

    initial_layout = {circuit.qubits[i]: native_mapping[i] for i in range(min(len(native_mapping), len(circuit.qubits)))}

    routed = transpile(
        circuit,
        backend=backend,
        routing_method=eval_config.routing_method,
        optimization_level=eval_config.transpile_optimization_level,
        seed_transpiler=eval_config.seed_transpiler,
        initial_layout=initial_layout,
    )

    orig_depth = int(circuit.depth() or 0)
    routed_depth = int(routed.depth() or 0)
    orig_ops = circuit.count_ops()
    routed_ops = routed.count_ops()

    swap_count = int(routed_ops.get("swap", 0))
    cx_count = int(routed_ops.get("cx", 0) + routed_ops.get("cz", 0) + routed_ops.get("ecr", 0))
    depth_increase = routed_depth - orig_depth

    if eval_config.save_routed_qasm_dir:
        out_dir = ensure_dir(eval_config.save_routed_qasm_dir)
        out_path = Path(out_dir) / (Path(qasm_path).stem + "_routed.qasm")
        out_path.write_text(routed.qasm(), encoding="utf-8")

    if eval_config.save_routed_qpy_dir:
        from qiskit import qpy
        out_dir = ensure_dir(eval_config.save_routed_qpy_dir)
        out_path = Path(out_dir) / (Path(qasm_path).stem + "_routed.qpy")
        with open(out_path, "wb") as f:
            qpy.dump(routed, f)

    return {
        "routed_depth": routed_depth,
        "orig_depth": orig_depth,
        "depth_increase": depth_increase,
        "swap_count": swap_count,
        "routed_2q_count": cx_count,
        "routed_success": True,
    }


def _write_per_circuit_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            pass
        return

    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_rows(rows: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "split": split_name,
        "num_samples": len(rows),
        "num_success": sum(1 for r in rows if not r.get("failed", False)),
        "num_failed": sum(1 for r in rows if r.get("failed", False)),
    }
    numeric_keys = set()
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (int, float)):
                numeric_keys.add(k)
    for key in sorted(numeric_keys):
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if vals:
            summary[f"mean_{key}"] = float(sum(vals) / len(vals))
            summary[f"median_{key}"] = float(statistics.median(vals))
    return summary


@torch.no_grad()
def evaluate_model(
    *,
    model: KMWCanonicalModel,
    loader,
    device: torch.device,
    eval_config: EvalConfig,
    loss_config: LossConfig | None = None,
) -> dict[str, Any]:
    loss_config = loss_config or LossConfig()
    model.eval()

    rows: list[dict[str, Any]] = []

    for batch in loader:
        batch = move_to_device(batch, device)
        Bsz = int(batch["A"].shape[0])

        try:
            outputs = model(
                A=batch["A"],
                B_can=batch["B_can"],
                c1_can=batch["c1_can"],
                c2_can=batch["c2_can"],
            )
            S_can = outputs["S_can"]
            _assert_finite(S_can, "S_can")

            soft_losses = compute_task_loss_from_logits(
                S_can=S_can,
                p=batch["p"],
                A=batch["A"],
                m=batch["m"],
                B_nat=batch["B_nat"],
                D_raw_nat=batch["D_raw_nat"],
                n1q=batch["n1q"],
                nmeas=batch["nmeas"],
                e1q_nat=batch["e1q_nat"],
                ero_nat=batch["ero_nat"],
                e2q_nat=batch["e2q_nat"],
                config=loss_config,
            )
            S_nat = soft_losses["S_nat"]

            for b in range(Bsz):
                metadata = batch["metadata"][b]
                S_can_b = S_can[b, 0]
                M_can_b = _hungarian_assignment_single(S_can_b)
                M_nat_b = torch.zeros_like(M_can_b)
                M_nat_b.scatter_(dim=-1, index=batch["p"][b].unsqueeze(0).expand(M_can_b.shape[0], -1), src=M_can_b)

                hard_losses = compute_task_loss_from_assignment(
                    A=batch["A"][b:b+1],
                    m=batch["m"][b:b+1],
                    Bmat=batch["B_nat"][b:b+1],
                    D_raw=batch["D_raw_nat"][b:b+1],
                    n1q=batch["n1q"][b:b+1],
                    nmeas=batch["nmeas"][b:b+1],
                    e1q=batch["e1q_nat"][b:b+1],
                    ero=batch["ero_nat"][b:b+1],
                    e2q=batch["e2q_nat"][b:b+1],
                    P_map=M_nat_b.unsqueeze(0),
                    config=loss_config,
                )

                native_mapping = _hard_mapping_from_assignment(M_nat_b, batch["m"][b])
                row = {
                    "id": metadata["id"],
                    "source": metadata["source"],
                    "split": metadata["split"],
                    "logical_qubits": _active_logical_count(batch["m"][b]),
                    "L_task_hard": float(hard_losses["L_task"].item()),
                    "L_native_hard": float(hard_losses["L_native"].item()),
                    "L_route_hard": float(hard_losses["L_route"].item()),
                    "S_proxy_exec_hard": float(hard_losses["S_proxy_exec"].item()),
                    "S_proxy_exec_soft": float(soft_losses["S_proxy_exec"][b].item() if soft_losses["S_proxy_exec"].ndim > 0 else soft_losses["S_proxy_exec"].item()),
                    "native_mapping": json.dumps(native_mapping),
                    "canonical_p": json.dumps(batch["p"][b].detach().cpu().tolist()),
                    "failed": False,
                }

                if eval_config.route_final_eval:
                    try:
                        qasm_relpath = metadata["qasm_relpath"]
                        qasm_path = Path(eval_config.project_root) / qasm_relpath
                        row.update(_run_routed_eval(
                            qasm_path=qasm_path,
                            native_mapping=native_mapping,
                            eval_config=eval_config,
                        ))
                    except Exception as exc:
                        if eval_config.fail_fast:
                            raise
                        row.update({
                            "routed_success": False,
                            "routing_error": f"{type(exc).__name__}: {exc}",
                        })

                rows.append(row)

        except Exception as exc:
            if eval_config.fail_fast:
                raise
            for metadata in batch["metadata"]:
                rows.append({
                    "id": metadata.get("id"),
                    "source": metadata.get("source"),
                    "split": metadata.get("split"),
                    "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    _write_per_circuit_csv(eval_config.per_circuit_csv_path, rows)
    summary = _summarize_rows(rows, split_name=eval_config.eval_split_name)
    write_json(eval_config.summary_json_path, summary)

    if eval_config.print_console_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary
