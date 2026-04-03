from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_NUM_QUBITS = 27
DEFAULT_EPS = 1e-8

_TWO_Q_WEIGHTS = {
    "cx": 1.0,
    "cz": 1.0,
    "ecr": 1.0,
    "iswap": 1.2,
    "swap": 3.0,
}


@dataclass(slots=True)
class CircuitFeatures:
    A: torch.Tensor
    m: torch.Tensor
    n1q: torch.Tensor
    nmeas: torch.Tensor
    metadata: dict[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "A": self.A,
            "m": self.m,
            "n1q": self.n1q,
            "nmeas": self.nmeas,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CircuitFeatures":
        return cls(**payload)


def _load_quantum_circuit(qasm_path: str | Path):
    try:
        from qiskit import QuantumCircuit
    except Exception as exc:
        raise ImportError("Qiskit is required to parse QASM circuits.") from exc

    qasm_path = Path(qasm_path)
    suffix = qasm_path.suffix.lower()
    if suffix == ".qpy":
        from qiskit import qpy
        with open(qasm_path, "rb") as f:
            circs = list(qpy.load(f))
        if len(circs) != 1:
            raise ValueError(f"Expected exactly one circuit in {qasm_path}")
        return circs[0]

    try:
        return QuantumCircuit.from_qasm_file(str(qasm_path))
    except Exception:
        # Some datasets contain qasm3-like content.
        try:
            from qiskit.qasm3 import loads as qasm3_loads
            return qasm3_loads(qasm_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not parse QASM file: {qasm_path}") from exc


def _normalize_A(A_raw: torch.Tensor, eps: float) -> torch.Tensor:
    denom = max(float(A_raw.max().item()), eps)
    return A_raw / denom


def _twoq_weight(op_name: str) -> float:
    return _TWO_Q_WEIGHTS.get(op_name.lower(), 1.0)


def featurize_circuit(
    circuit: Any,
    *,
    max_qubits: int = DEFAULT_NUM_QUBITS,
    eps: float = DEFAULT_EPS,
    allow_degenerate: bool = False,
) -> CircuitFeatures:
    qubits = list(circuit.qubits)
    K = len(qubits)
    if K > max_qubits:
        raise ValueError(f"Circuit uses {K} logical qubits, exceeds max_qubits={max_qubits}")

    qindex = {qb: idx for idx, qb in enumerate(qubits)}
    pair_weight_sum = torch.zeros((max_qubits, max_qubits), dtype=torch.float32)
    n1q = torch.zeros((max_qubits,), dtype=torch.float32)
    nmeas = torch.zeros((max_qubits,), dtype=torch.float32)

    num_ops_total = 0
    num_2q_total = 0

    for inst_context in circuit.data:
        try:
            operation = inst_context.operation
            qargs = list(inst_context.qubits)
        except AttributeError:
            operation, qargs, _ = inst_context  # older qiskit fallback

        name = operation.name.lower()
        num_ops_total += 1

        logical_ids = [qindex[q] for q in qargs if q in qindex]
        if name == "measure":
            for u in logical_ids:
                nmeas[u] += 1.0
            continue

        if len(logical_ids) == 1:
            if name not in {"barrier", "delay", "reset"}:
                n1q[logical_ids[0]] += 1.0
            continue

        if len(logical_ids) == 2:
            u, v = sorted(logical_ids)
            pair_weight_sum[u, v] += _twoq_weight(name)
            pair_weight_sum[v, u] += _twoq_weight(name)
            num_2q_total += 1

    A_raw = torch.zeros((max_qubits, max_qubits), dtype=torch.float32)
    if K > 0:
        active_slice = slice(0, K)
        # off-diagonal
        active_pairs = pair_weight_sum[active_slice, active_slice]
        mask_offdiag = 1.0 - torch.eye(K, dtype=torch.float32)
        A_raw[active_slice, active_slice] += torch.log1p(active_pairs) * mask_offdiag
        # diagonal
        diag_vals = torch.log1p(n1q[active_slice] + nmeas[active_slice])
        A_raw[active_slice, active_slice] += torch.diag(diag_vals)

    if num_2q_total == 0 and not allow_degenerate:
        raise ValueError("Degenerate circuit rejected: no meaningful 2Q structure.")

    A = _normalize_A(A_raw, eps=eps)
    m = torch.zeros((max_qubits,), dtype=torch.float32)
    m[:K] = 1.0

    if not torch.allclose(A, A.transpose(0, 1), atol=1e-6, rtol=0.0):
        raise ValueError("A must be symmetric after featurization.")
    if not torch.isfinite(A).all():
        raise ValueError("A contains NaN or Inf.")

    return CircuitFeatures(
        A=A,
        m=m,
        n1q=n1q,
        nmeas=nmeas,
        metadata={
            "logical_qubits": K,
            "num_ops_total": num_ops_total,
            "num_2q_total": num_2q_total,
            "allow_degenerate": allow_degenerate,
            "circuit_name": getattr(circuit, "name", None),
        },
    )


def featurize_circuit_from_qasm(
    qasm_path: str | Path,
    *,
    max_qubits: int = DEFAULT_NUM_QUBITS,
    eps: float = DEFAULT_EPS,
    allow_degenerate: bool = False,
) -> CircuitFeatures:
    circuit = _load_quantum_circuit(qasm_path)
    return featurize_circuit(
        circuit,
        max_qubits=max_qubits,
        eps=eps,
        allow_degenerate=allow_degenerate,
    )
