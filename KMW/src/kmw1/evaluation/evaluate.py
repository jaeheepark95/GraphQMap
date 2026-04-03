from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from kmw1.losses.loss import LossConfig, compute_task_loss_from_assignment, compute_task_loss_from_logits
from kmw1.models.model import KMWCanonicalModel, active_row_hungarian
from kmw1.preprocessing.pipeline import resolve_backend
from kmw1.utils import ensure_dir, move_to_device, write_json


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
    use_qiskit_default_mapper: bool = True
    routing_method: str | None = None
    transpile_optimization_level: int = 0
    seed_transpiler: int | None = None
    include_readout_in_pst: bool = True
    save_routed_qasm_dir: str | None = None
    save_routed_qpy_dir: str | None = None


def _assert_finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def _active_logical_count(m: torch.Tensor) -> int:
    return int((m > 0.5).sum().item())


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
    return resolve_backend(backend_name=backend_name)


def _instruction_name(instruction: Any) -> str:
    return str(getattr(instruction, "name", instruction.__class__.__name__)).lower()


def _circuit_instruction_parts(item: Any) -> tuple[Any, Any]:
    if hasattr(item, "operation") and hasattr(item, "qubits"):
        return item.operation, item.qubits
    return item[0], item[1]


def _qubit_index_in_circuit(circuit: Any, qubit: Any) -> int:
    try:
        return int(circuit.find_bit(qubit).index)
    except Exception:
        return int(getattr(qubit, "index"))


def _count_two_qubit_ops(circuit: Any) -> int:
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


def _compute_circuit_depth_no_barrier(circuit: Any) -> int:
    try:
        return int(circuit.depth(filter_function=lambda inst: _instruction_name(inst.operation) not in {"barrier", "delay"}))
    except Exception:
        return int(circuit.depth() or 0)


def _extract_swap_inserted_count(circuit: Any) -> int:
    count = 0
    for item in getattr(circuit, "data", []):
        instruction, _ = _circuit_instruction_parts(item)
        if _instruction_name(instruction) == "swap":
            count += 1
    return count


def _lookup_instruction_error(backend: Any, op_name: str, qargs: tuple[int, ...]) -> float:
    op_name = str(op_name).lower()
    target = getattr(backend, "target", None)
    if target is None:
        raise RuntimeError("Backend does not expose a target object.")
    props = None
    try:
        props = target[op_name][tuple(qargs)]
    except Exception:
        try:
            props = target[op_name].get(tuple(qargs))
        except Exception:
            props = None
    if props is None:
        raise RuntimeError(f"No backend instruction properties found for op={op_name}, qargs={qargs}")
    error = getattr(props, "error", None)
    if error is None and isinstance(props, dict):
        error = props.get("error")
    if error is None:
        raise RuntimeError(f"Backend instruction properties for op={op_name}, qargs={qargs} do not define error.")
    error = float(error)
    if not math.isfinite(error):
        raise RuntimeError(f"Non-finite error for op={op_name}, qargs={qargs}: {error}")
    return error


def _estimate_real_pst_gate_readout(circuit: Any, backend: Any, include_readout: bool = True) -> float:
    success = 1.0
    for item in getattr(circuit, "data", []):
        instruction, qargs = _circuit_instruction_parts(item)
        op_name = _instruction_name(instruction)
        if op_name in {"barrier", "delay"}:
            continue
        if op_name == "measure" and not include_readout:
            continue
        qarg_indices = tuple(_qubit_index_in_circuit(circuit, q) for q in qargs)
        error = _lookup_instruction_error(backend, op_name, qarg_indices)
        success *= max(0.0, min(1.0, 1.0 - error))
    return float(success)


def _run_routed_eval(*, qasm_path: str | Path, native_mapping: list[int], eval_config: EvalConfig) -> dict[str, Any]:
    from qiskit import transpile
    t_total_0 = time.perf_counter()
    circuit = _load_circuit_from_qasm(qasm_path)
    backend = _resolve_backend_cached(eval_config.backend_name)
    initial_layout = {circuit.qubits[i]: native_mapping[i] for i in range(min(len(native_mapping), len(circuit.qubits)))}
    transpile_kwargs = dict(
        backend=backend,
        initial_layout=initial_layout,
        optimization_level=eval_config.transpile_optimization_level,
        seed_transpiler=eval_config.seed_transpiler,
    )
    if not eval_config.use_qiskit_default_mapper and eval_config.routing_method is not None:
        transpile_kwargs["routing_method"] = eval_config.routing_method
    t0 = time.perf_counter()
    routed = transpile(circuit, **transpile_kwargs)
    compile_time_s = float(time.perf_counter() - t0)
    total_eval_time_s = float(time.perf_counter() - t_total_0)
    original_depth = _compute_circuit_depth_no_barrier(circuit)
    routed_depth = _compute_circuit_depth_no_barrier(routed)
    original_2q_count = _count_two_qubit_ops(circuit)
    routed_2q_count = _count_two_qubit_ops(routed)
    added_2q_ops = int(routed_2q_count - original_2q_count)
    swap_inserted_count = _extract_swap_inserted_count(routed)
    swap_overhead_ratio = float(swap_inserted_count / max(original_2q_count, 1))
    depth_increase_abs = int(routed_depth - original_depth)
    depth_increase_ratio = float(routed_depth / max(original_depth, 1))
    real_pst_gate_readout = _estimate_real_pst_gate_readout(routed, backend, include_readout=eval_config.include_readout_in_pst)
    if eval_config.save_routed_qasm_dir:
        out_dir = ensure_dir(eval_config.save_routed_qasm_dir)
        out_path = Path(out_dir) / (Path(qasm_path).stem + "_routed.qasm")
        dumped = None
        try:
            from qiskit import qasm2
            dumped = qasm2.dumps(routed)
        except Exception:
            try:
                from qiskit import qasm3
                dumped = qasm3.dumps(routed)
            except Exception:
                dumped = None
        if dumped is not None:
            out_path.write_text(dumped, encoding="utf-8")
    if eval_config.save_routed_qpy_dir:
        from qiskit import qpy
        out_dir = ensure_dir(eval_config.save_routed_qpy_dir)
        out_path = Path(out_dir) / (Path(qasm_path).stem + "_routed.qpy")
        with open(out_path, "wb") as f:
            qpy.dump(routed, f)
    return {
        "routed_success": True,
        "real_pst_gate_readout": float(real_pst_gate_readout),
        "swap_inserted_count": int(swap_inserted_count),
        "swap_overhead_ratio": float(swap_overhead_ratio),
        "added_2q_ops": int(added_2q_ops),
        "routing_compile_time_s": float(compile_time_s),
        "routing_total_eval_time_s": float(total_eval_time_s),
        "original_depth": int(original_depth),
        "routed_depth": int(routed_depth),
        "depth_increase_abs": int(depth_increase_abs),
        "depth_increase_ratio": float(depth_increase_ratio),
        "orig_2q_count": int(original_2q_count),
        "routed_2q_count": int(routed_2q_count),
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
def evaluate_model(*, model: KMWCanonicalModel, loader, device: torch.device, eval_config: EvalConfig,
                   loss_config: LossConfig | None = None) -> dict[str, Any]:
    loss_config = loss_config or LossConfig()
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        bsz = int(batch["A"].shape[0])
        try:
            outputs = model(A=batch["A"], B_can=batch["B_can"], c1_can=batch["c1_can"], c2_can=batch["c2_can"])
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
            for b in range(bsz):
                metadata = batch["metadata"][b]
                assign = active_row_hungarian(S_nat[b], batch["m"][b])
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
                    P_map=assign.M_nat.unsqueeze(0),
                    config=loss_config,
                )
                row = {
                    "id": metadata["id"],
                    "source": metadata["source"],
                    "split": metadata["split"],
                    "logical_qubits": _active_logical_count(batch["m"][b]),
                    "L_task_hard": float(hard_losses["L_task"].item()),
                    "L_native_hard": float(hard_losses["L_native"].item()),
                    "L_route_hard": float(hard_losses["L_route"].item()),
                    "S_proxy_exec_hard": float(hard_losses["S_proxy_exec"].item()),
                    "S_proxy_exec_soft": float(soft_losses["S_proxy_exec"].item()),
                    "native_mapping": json.dumps(assign.active_mapping),
                    "canonical_p": json.dumps(batch["p"][b].detach().cpu().tolist()),
                    "failed": False,
                }
                if eval_config.route_final_eval:
                    try:
                        qasm_relpath = metadata["qasm_relpath"]
                        qasm_path = Path(eval_config.project_root) / qasm_relpath
                        row.update(_run_routed_eval(qasm_path=qasm_path, native_mapping=assign.active_mapping, eval_config=eval_config))
                    except Exception as exc:
                        if eval_config.fail_fast:
                            raise
                        row.update({"routed_success": False, "routing_error": f"{type(exc).__name__}: {exc}"})
                rows.append(row)
        except Exception as exc:
            if eval_config.fail_fast:
                raise
            for metadata in batch["metadata"]:
                rows.append({"id": metadata.get("id"), "source": metadata.get("source"), "split": metadata.get("split"), "failed": True, "error": f"{type(exc).__name__}: {exc}"})
    _write_per_circuit_csv(eval_config.per_circuit_csv_path, rows)
    summary = _summarize_rows(rows, split_name=eval_config.eval_split_name)
    write_json(eval_config.summary_json_path, summary)
    if eval_config.print_console_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
