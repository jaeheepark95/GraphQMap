from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kmw1.utils import CACHE_SCHEMA_VERSION, ensure_dir

DEFAULT_NUM_QUBITS = 27
DEFAULT_EPS = 1e-8
UNREACHABLE_DISTANCE = DEFAULT_NUM_QUBITS + 1


@dataclass(slots=True)
class BackendTensors:
    backend_name: str
    B: torch.Tensor
    c1: torch.Tensor
    c2: torch.Tensor
    D: torch.Tensor
    B_raw: torch.Tensor
    c1_raw: torch.Tensor
    c2_raw: torch.Tensor
    D_raw: torch.Tensor
    e1q_raw: torch.Tensor
    ero_raw: torch.Tensor
    e2q_raw: torch.Tensor
    metadata: dict[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "backend_name": self.backend_name,
            "B": self.B,
            "c1": self.c1,
            "c2": self.c2,
            "D": self.D,
            "B_raw": self.B_raw,
            "c1_raw": self.c1_raw,
            "c2_raw": self.c2_raw,
            "D_raw": self.D_raw,
            "e1q_raw": self.e1q_raw,
            "ero_raw": self.ero_raw,
            "e2q_raw": self.e2q_raw,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BackendTensors":
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in payload.items() if k in allowed}
        return cls(**filtered)


@dataclass(slots=True)
class PreprocessedSample:
    A: torch.Tensor
    m: torch.Tensor

    B_nat: torch.Tensor
    c1_nat: torch.Tensor
    c2_nat: torch.Tensor
    D_nat: torch.Tensor

    B_can: torch.Tensor
    c1_can: torch.Tensor
    c2_can: torch.Tensor
    D_can: torch.Tensor

    D_raw_nat: torch.Tensor
    e1q_nat: torch.Tensor
    ero_nat: torch.Tensor
    e2q_nat: torch.Tensor

    p: torch.Tensor
    p_inv: torch.Tensor

    n1q: torch.Tensor
    nmeas: torch.Tensor

    circuit: dict[str, Any]
    backend: dict[str, Any]
    metadata: dict[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "A": self.A,
            "m": self.m,
            "B_nat": self.B_nat,
            "c1_nat": self.c1_nat,
            "c2_nat": self.c2_nat,
            "D_nat": self.D_nat,
            "B_can": self.B_can,
            "c1_can": self.c1_can,
            "c2_can": self.c2_can,
            "D_can": self.D_can,
            "D_raw_nat": self.D_raw_nat,
            "e1q_nat": self.e1q_nat,
            "ero_nat": self.ero_nat,
            "e2q_nat": self.e2q_nat,
            "p": self.p,
            "p_inv": self.p_inv,
            "n1q": self.n1q,
            "nmeas": self.nmeas,
            "circuit": self.circuit,
            "backend": self.backend,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PreprocessedSample":
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in payload.items() if k in allowed}
        return cls(**filtered)


def _assert_finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def _assert_symmetric(x: torch.Tensor, name: str, atol: float = 1e-6) -> None:
    if not torch.allclose(x, x.transpose(-1, -2), atol=atol, rtol=0.0):
        raise ValueError(f"{name} is not symmetric within atol={atol}.")


def _normalize_vector(x: torch.Tensor, eps: float) -> torch.Tensor:
    denom = max(float(x.max().item()), eps)
    return x / denom


def _normalize_matrix(x: torch.Tensor, eps: float) -> torch.Tensor:
    denom = max(float(x.max().item()), eps)
    return x / denom


def _normalize_distance(D_raw: torch.Tensor) -> torch.Tensor:
    denom = max(float(D_raw.max().item()), 1.0)
    return D_raw / denom


def resolve_backend(backend_name: str = "fake_toronto_v2", num_qubits: int = DEFAULT_NUM_QUBITS) -> Any:
    key = backend_name.lower().strip()
    if key in {"fake_toronto_v2", "fake_toronto", "toronto", "faketorontov2"}:
        try:
            from qiskit_ibm_runtime.fake_provider import FakeTorontoV2
            return FakeTorontoV2()
        except Exception as exc:
            raise ImportError("FakeTorontoV2 could not be imported. Install qiskit-ibm-runtime.") from exc
    if key in {"generic_backend_v2", "generic", "generic27"}:
        try:
            from qiskit.providers.fake_provider import GenericBackendV2
            return GenericBackendV2(num_qubits=num_qubits)
        except Exception as exc:
            raise ImportError("GenericBackendV2 could not be imported. Install qiskit.") from exc
    raise ValueError(
        f"Unsupported backend_name={backend_name!r}. Add a new branch in resolve_backend() to swap backends cleanly."
    )


def _target_get_instruction_error(target: Any, op_name: str, qargs: tuple[int, ...]) -> float | None:
    try:
        op_map = target[op_name]
    except Exception:
        return None
    try:
        props = op_map.get(qargs, None)
    except Exception:
        props = None
    if props is None:
        return None
    err = getattr(props, "error", None)
    if err is None:
        return None
    return float(err)


def _backend_get_readout_error(backend: Any, qubit: int) -> float:
    target = getattr(backend, "target", None)
    if target is not None:
        err = _target_get_instruction_error(target, "measure", (qubit,))
        if err is not None:
            return err
    props_method = getattr(backend, "properties", None)
    if callable(props_method):
        try:
            props = props_method()
            if hasattr(props, "readout_error"):
                return float(props.readout_error(qubit))
        except Exception:
            pass
    return 0.0


def _backend_get_mean_1q_error(backend: Any, qubit: int) -> float:
    target = getattr(backend, "target", None)
    vals: list[float] = []
    if target is not None:
        op_names = list(getattr(target, "operation_names", []))
        for name in op_names:
            if name in {"measure", "delay", "barrier", "reset"}:
                continue
            err = _target_get_instruction_error(target, name, (qubit,))
            if err is not None:
                vals.append(float(err))
    if vals:
        return float(sum(vals) / len(vals))
    props_method = getattr(backend, "properties", None)
    if callable(props_method):
        try:
            props = props_method()
            gate_errors: list[float] = []
            for gate in props.gates:
                if len(getattr(gate, "qubits", [])) == 1 and gate.qubits[0] == qubit:
                    for param in getattr(gate, "parameters", []):
                        if getattr(param, "name", "") == "gate_error":
                            gate_errors.append(float(param.value))
            if gate_errors:
                return float(sum(gate_errors) / len(gate_errors))
        except Exception:
            pass
    return 0.0


def _backend_get_directional_2q_errors(backend: Any, n: int) -> dict[tuple[int, int], float]:
    target = getattr(backend, "target", None)
    directed: dict[tuple[int, int], list[float]] = {}
    if target is not None:
        for name in getattr(target, "operation_names", []):
            try:
                qarg_items = list(target[name].items())
            except Exception:
                continue
            for qargs, props in qarg_items:
                if not isinstance(qargs, tuple) or len(qargs) != 2:
                    continue
                err = getattr(props, "error", None)
                if err is None:
                    continue
                directed.setdefault((int(qargs[0]), int(qargs[1])), []).append(float(err))
    if directed:
        return {k: min(v) for k, v in directed.items()}
    props_method = getattr(backend, "properties", None)
    if callable(props_method):
        try:
            props = props_method()
            for gate in props.gates:
                qubits = tuple(int(q) for q in getattr(gate, "qubits", []))
                if len(qubits) != 2:
                    continue
                for param in getattr(gate, "parameters", []):
                    if getattr(param, "name", "") == "gate_error":
                        directed.setdefault(qubits, []).append(float(param.value))
            return {k: min(v) for k, v in directed.items()}
        except Exception:
            pass
    return {}


def _backend_edges_from_target(backend: Any) -> list[tuple[int, int]]:
    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is not None:
        try:
            return [(int(a), int(b)) for a, b in coupling_map.get_edges()]
        except Exception:
            pass
    target = getattr(backend, "target", None)
    if target is not None:
        edges: set[tuple[int, int]] = set()
        for name in getattr(target, "operation_names", []):
            try:
                qarg_items = list(target[name].keys())
            except Exception:
                continue
            for qargs in qarg_items:
                if isinstance(qargs, tuple) and len(qargs) == 2:
                    edges.add((int(qargs[0]), int(qargs[1])))
        return sorted(edges)
    return []


def _build_shortest_path_distance(B_raw: torch.Tensor, unreachable_value: int = UNREACHABLE_DISTANCE) -> torch.Tensor:
    n = int(B_raw.shape[0])
    D = torch.full((n, n), float(unreachable_value), dtype=torch.float32)
    for src in range(n):
        D[src, src] = 0.0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            nbrs = torch.where(B_raw[u] > 0.5)[0].tolist()
            for v in nbrs:
                if D[src, v] > D[src, u] + 1:
                    D[src, v] = D[src, u] + 1
                    queue.append(int(v))
    return D


def _validate_backend_tensors(*, B_raw: torch.Tensor, c1_raw: torch.Tensor, c2_raw: torch.Tensor, D_raw: torch.Tensor,
                              B: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, D: torch.Tensor) -> None:
    if tuple(B_raw.shape) != (27, 27):
        raise ValueError(f"B_raw must have shape (27,27), got {tuple(B_raw.shape)}")
    if tuple(c1_raw.shape) != (27,):
        raise ValueError(f"c1_raw must have shape (27,), got {tuple(c1_raw.shape)}")
    if tuple(c2_raw.shape) != (27, 27):
        raise ValueError(f"c2_raw must have shape (27,27), got {tuple(c2_raw.shape)}")
    if tuple(D_raw.shape) != (27, 27):
        raise ValueError(f"D_raw must have shape (27,27), got {tuple(D_raw.shape)}")
    for name, tensor in [("B_raw", B_raw), ("c1_raw", c1_raw), ("c2_raw", c2_raw), ("D_raw", D_raw),
                         ("B", B), ("c1", c1), ("c2", c2), ("D", D)]:
        _assert_finite(tensor, name)
    _assert_symmetric(B_raw, "B_raw")
    _assert_symmetric(c2_raw, "c2_raw")
    _assert_symmetric(D_raw, "D_raw")
    _assert_symmetric(B, "B")
    _assert_symmetric(c2, "c2")
    _assert_symmetric(D, "D")
    if not torch.equal(B_raw, (B_raw > 0.5).float()):
        raise ValueError("B_raw must be binary.")
    if not torch.equal(B, (B > 0.5).float()):
        raise ValueError("B must be binary.")
    for name, tensor in [("B_raw", B_raw), ("c2_raw", c2_raw), ("D_raw", D_raw), ("B", B), ("c2", c2), ("D", D)]:
        if not torch.allclose(torch.diag(tensor), torch.zeros(27, dtype=tensor.dtype), atol=1e-6):
            raise ValueError(f"diag({name}) must be zero.")


def extract_backend_tensors(backend: Any, backend_name: str, expected_num_qubits: int = DEFAULT_NUM_QUBITS,
                            eps: float = DEFAULT_EPS) -> BackendTensors:
    n = int(getattr(getattr(backend, "target", None), "num_qubits", getattr(backend, "num_qubits", expected_num_qubits)))
    if n != expected_num_qubits:
        raise ValueError(f"Expected {expected_num_qubits} qubits, got {n} for backend {backend_name!r}")
    B_raw = torch.zeros((n, n), dtype=torch.float32)
    c2_raw = torch.zeros((n, n), dtype=torch.float32)
    e2q_raw = torch.zeros((n, n), dtype=torch.float32)
    c1_raw = torch.zeros((n,), dtype=torch.float32)
    e1q_raw = torch.zeros((n,), dtype=torch.float32)
    ero_raw = torch.zeros((n,), dtype=torch.float32)
    edges = _backend_edges_from_target(backend)
    dir_2q = _backend_get_directional_2q_errors(backend, n)
    for q in range(n):
        ero_raw[q] = _backend_get_readout_error(backend, q)
        e1q_raw[q] = _backend_get_mean_1q_error(backend, q)
        c1_raw[q] = 0.5 * ero_raw[q] + 0.5 * e1q_raw[q]
    undirected_pairs: set[tuple[int, int]] = set()
    for a, b in edges:
        if a != b:
            i, j = sorted((int(a), int(b)))
            undirected_pairs.add((i, j))
    for i, j in sorted(undirected_pairs):
        vals = []
        if (i, j) in dir_2q:
            vals.append(float(dir_2q[(i, j)]))
        if (j, i) in dir_2q:
            vals.append(float(dir_2q[(j, i)]))
        edge_err = min(vals) if vals else 0.0
        if vals:
            B_raw[i, j] = 1.0
            B_raw[j, i] = 1.0
        e2q_raw[i, j] = edge_err
        e2q_raw[j, i] = edge_err
        c2_raw[i, j] = edge_err
        c2_raw[j, i] = edge_err
    D_raw = _build_shortest_path_distance(B_raw, unreachable_value=expected_num_qubits + 1)
    B = B_raw.clone()
    c1 = _normalize_vector(c1_raw, eps=eps)
    c2 = _normalize_matrix(c2_raw, eps=eps)
    D = _normalize_distance(D_raw)
    _validate_backend_tensors(B_raw=B_raw, c1_raw=c1_raw, c2_raw=c2_raw, D_raw=D_raw, B=B, c1=c1, c2=c2, D=D)
    return BackendTensors(
        backend_name=backend_name,
        B=B,
        c1=c1,
        c2=c2,
        D=D,
        B_raw=B_raw,
        c1_raw=c1_raw,
        c2_raw=c2_raw,
        D_raw=D_raw,
        e1q_raw=e1q_raw,
        ero_raw=ero_raw,
        e2q_raw=e2q_raw,
        metadata={
            "backend_name": backend_name,
            "num_qubits": n,
            "num_edges_undirected": int((B_raw.sum().item()) / 2),
            "unreachable_value": expected_num_qubits + 1,
            "schema_version": CACHE_SCHEMA_VERSION,
        },
    )


def backend_cache_path(project_root: str | Path, backend_name: str) -> Path:
    return ensure_dir(Path(project_root) / "data" / "cache" / "backends") / f"{backend_name}.pt"


def load_or_build_backend_tensors(project_root: str | Path, backend_name: str = "fake_toronto_v2",
                                  force_recompute: bool = False, expected_num_qubits: int = DEFAULT_NUM_QUBITS,
                                  eps: float = DEFAULT_EPS) -> BackendTensors:
    cache_path = backend_cache_path(project_root, backend_name)
    if cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("schema_version") == CACHE_SCHEMA_VERSION:
            return BackendTensors.from_payload(payload)
    backend = resolve_backend(backend_name=backend_name, num_qubits=expected_num_qubits)
    backend_tensors = extract_backend_tensors(
        backend=backend,
        backend_name=backend_name,
        expected_num_qubits=expected_num_qubits,
        eps=eps,
    )
    ensure_dir(cache_path.parent)
    torch.save(backend_tensors.to_serializable(), cache_path)
    return backend_tensors


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
        try:
            from qiskit.qasm3 import loads as qasm3_loads
            return qasm3_loads(qasm_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not parse QASM file: {qasm_path}") from exc


def _normalize_A(A_raw: torch.Tensor, eps: float) -> torch.Tensor:
    denom = max(float(A_raw.max().item()), eps)
    return A_raw / denom


_TWO_Q_WEIGHTS = {"cx": 1.0, "cz": 1.0, "ecr": 1.0, "iswap": 1.2, "swap": 3.0}


def featurize_circuit(circuit: Any, *, max_qubits: int = DEFAULT_NUM_QUBITS, eps: float = DEFAULT_EPS,
                      allow_degenerate: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
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
            operation, qargs, _ = inst_context
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
            pair_weight_sum[u, v] += _TWO_Q_WEIGHTS.get(name, 1.0)
            pair_weight_sum[v, u] += _TWO_Q_WEIGHTS.get(name, 1.0)
            num_2q_total += 1
    A_raw = torch.zeros((max_qubits, max_qubits), dtype=torch.float32)
    if K > 0:
        active_pairs = pair_weight_sum[:K, :K]
        mask_offdiag = 1.0 - torch.eye(K, dtype=torch.float32)
        A_raw[:K, :K] += torch.log1p(active_pairs) * mask_offdiag
        diag_vals = torch.log1p(n1q[:K] + nmeas[:K])
        A_raw[:K, :K] += torch.diag(diag_vals)
    if num_2q_total == 0 and not allow_degenerate:
        raise ValueError("Degenerate circuit rejected: no meaningful 2Q structure.")
    A = _normalize_A(A_raw, eps=eps)
    m = torch.zeros((max_qubits,), dtype=torch.float32)
    m[:K] = 1.0
    if not torch.allclose(A, A.transpose(0, 1), atol=1e-6, rtol=0.0):
        raise ValueError("A must be symmetric after featurization.")
    _assert_finite(A, "A")
    return A, m, n1q, nmeas, {
        "logical_qubits": K,
        "num_ops_total": num_ops_total,
        "num_2q_total": num_2q_total,
        "allow_degenerate": allow_degenerate,
        "circuit_name": getattr(circuit, "name", None),
    }


def featurize_circuit_from_qasm(qasm_path: str | Path, *, max_qubits: int = DEFAULT_NUM_QUBITS,
                                eps: float = DEFAULT_EPS, allow_degenerate: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    circuit = _load_quantum_circuit(qasm_path)
    return featurize_circuit(circuit, max_qubits=max_qubits, eps=eps, allow_degenerate=allow_degenerate)


def _zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean()
    std = x.std(unbiased=False)
    return (x - mean) / max(float(std.item()), eps)


def build_canonical_hardware_index(B_nat: torch.Tensor, c1_nat: torch.Tensor, c2_nat: torch.Tensor,
                                   isolated_fallback: float | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if tuple(B_nat.shape) != (27, 27):
        raise ValueError(f"B_nat must have shape (27,27), got {tuple(B_nat.shape)}")
    adj = ((B_nat + B_nat.transpose(0, 1)) > 0).float()
    adj.fill_diagonal_(0.0)
    degree = adj.sum(dim=-1)
    if isolated_fallback is None:
        positive_edges = c2_nat[adj > 0.5]
        isolated_fallback = float(positive_edges.mean().item()) if positive_edges.numel() > 0 else 1.0
    mean_edge_cost = torch.full((27,), float(isolated_fallback), dtype=torch.float32)
    for i in range(27):
        nbrs = torch.where(adj[i] > 0.5)[0]
        if nbrs.numel() == 0:
            continue
        vals = torch.stack([torch.minimum(c2_nat[i, j], c2_nat[j, i]) for j in nbrs])
        mean_edge_cost[i] = vals.mean()
    qscore = _zscore(c1_nat) + _zscore(mean_edge_cost) - 0.3 * _zscore(degree)
    visited: set[int] = set()
    order: list[int] = []

    def pick_root() -> int:
        remaining = [i for i in range(27) if i not in visited]
        remaining.sort(key=lambda i: (float(qscore[i].item()), int(i)))
        return int(remaining[0])

    while len(visited) < 27:
        root = pick_root()
        queue = [root]
        if root not in visited:
            visited.add(root)
            order.append(root)
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            nbrs = [int(v) for v in torch.where(adj[u] > 0.5)[0].tolist() if int(v) not in visited]
            nbrs.sort(key=lambda v: (
                float(torch.minimum(c2_nat[u, v], c2_nat[v, u]).item()),
                float(c1_nat[v].item()),
                -float(degree[v].item()),
                int(v),
            ))
            for v in nbrs:
                if v in visited:
                    continue
                visited.add(v)
                order.append(v)
                queue.append(v)
    if len(order) != 27 or len(set(order)) != 27:
        raise ValueError(f"Invalid canonical order produced: {order}")
    p = torch.tensor(order, dtype=torch.long)
    p_inv = torch.empty_like(p)
    p_inv[p] = torch.arange(27, dtype=torch.long)
    if not torch.equal(p_inv[p], torch.arange(27, dtype=torch.long)):
        raise ValueError("Invalid canonical inverse permutation.")
    return p, p_inv, {
        "qscore": qscore.tolist(),
        "degree": degree.tolist(),
        "mean_edge_cost": mean_edge_cost.tolist(),
    }


def canonicalize_hardware_tensors(*, p: torch.Tensor, B: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, D: torch.Tensor,
                                  B_raw: torch.Tensor | None = None, c1_raw: torch.Tensor | None = None,
                                  c2_raw: torch.Tensor | None = None, D_raw: torch.Tensor | None = None,
                                  e1q_raw: torch.Tensor | None = None, ero_raw: torch.Tensor | None = None,
                                  e2q_raw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    out = {
        "B_can": B[p][:, p],
        "c1_can": c1[p],
        "c2_can": c2[p][:, p],
        "D_can": D[p][:, p],
    }
    if B_raw is not None:
        out["B_raw_can"] = B_raw[p][:, p]
    if c1_raw is not None:
        out["c1_raw_can"] = c1_raw[p]
    if c2_raw is not None:
        out["c2_raw_can"] = c2_raw[p][:, p]
    if D_raw is not None:
        out["D_raw_can"] = D_raw[p][:, p]
    if e1q_raw is not None:
        out["e1q_can"] = e1q_raw[p]
    if ero_raw is not None:
        out["ero_can"] = ero_raw[p]
    if e2q_raw is not None:
        out["e2q_can"] = e2q_raw[p][:, p]
    return out


def decode_canonical_to_native_logits(S_can: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if S_can.ndim == 4:
        if S_can.shape[1] != 1:
            raise ValueError("S_can with rank 4 must have channel dimension = 1")
        S_can = S_can[:, 0]
    if S_can.ndim != 3 or p.ndim != 2:
        raise ValueError("Expected S_can=(B,N,N), p=(B,N)")
    out = torch.zeros_like(S_can)
    out.scatter_(dim=-1, index=p.unsqueeze(1).expand(-1, S_can.shape[1], -1), src=S_can)
    _assert_finite(out, "S_nat")
    return out


def _validate_preprocessed_sample(sample: PreprocessedSample) -> None:
    tensor_names = [
        "A", "m", "B_nat", "c1_nat", "c2_nat", "D_nat", "B_can", "c1_can", "c2_can", "D_can",
        "D_raw_nat", "e1q_nat", "ero_nat", "e2q_nat", "p", "p_inv", "n1q", "nmeas",
    ]
    for name in tensor_names:
        tensor = getattr(sample, name)
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a tensor.")
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"{name} contains NaN or Inf.")
    if tuple(sample.A.shape) != (27, 27):
        raise ValueError(f"A must have shape (27,27), got {tuple(sample.A.shape)}")
    if tuple(sample.m.shape) != (27,):
        raise ValueError(f"m must have shape (27,), got {tuple(sample.m.shape)}")
    for name in ["B_nat", "c2_nat", "D_nat", "B_can", "c2_can", "D_can", "D_raw_nat", "e2q_nat"]:
        mat = getattr(sample, name)
        if tuple(mat.shape) != (27, 27):
            raise ValueError(f"{name} must have shape (27,27), got {tuple(mat.shape)}")
    for name in ["c1_nat", "c1_can", "e1q_nat", "ero_nat", "n1q", "nmeas", "p", "p_inv"]:
        vec = getattr(sample, name)
        if tuple(vec.shape) != (27,):
            raise ValueError(f"{name} must have shape (27,), got {tuple(vec.shape)}")
    if len(torch.unique(sample.p)) != 27:
        raise ValueError("p is not a valid permutation.")
    if not torch.equal(sample.p_inv[sample.p], torch.arange(27, dtype=sample.p.dtype)):
        raise ValueError("p_inv is not the inverse of p.")


def build_preprocessed_sample(*, qasm_path: str | Path, backend_tensors: BackendTensors, source: str, split: str,
                              cache_key: str, allow_degenerate: bool = False, max_qubits: int = 27,
                              eps: float = 1e-8) -> PreprocessedSample:
    A, m, n1q, nmeas, circuit_meta = featurize_circuit_from_qasm(
        qasm_path=qasm_path,
        max_qubits=max_qubits,
        eps=eps,
        allow_degenerate=allow_degenerate,
    )
    p, p_inv, canonical_meta = build_canonical_hardware_index(
        B_nat=backend_tensors.B,
        c1_nat=backend_tensors.c1,
        c2_nat=backend_tensors.c2,
    )
    can = canonicalize_hardware_tensors(
        p=p,
        B=backend_tensors.B,
        c1=backend_tensors.c1,
        c2=backend_tensors.c2,
        D=backend_tensors.D,
        B_raw=backend_tensors.B_raw,
        c1_raw=backend_tensors.c1_raw,
        c2_raw=backend_tensors.c2_raw,
        D_raw=backend_tensors.D_raw,
        e1q_raw=backend_tensors.e1q_raw,
        ero_raw=backend_tensors.ero_raw,
        e2q_raw=backend_tensors.e2q_raw,
    )
    sample = PreprocessedSample(
        A=A,
        m=m,
        B_nat=backend_tensors.B,
        c1_nat=backend_tensors.c1,
        c2_nat=backend_tensors.c2,
        D_nat=backend_tensors.D,
        B_can=can["B_can"],
        c1_can=can["c1_can"],
        c2_can=can["c2_can"],
        D_can=can["D_can"],
        D_raw_nat=backend_tensors.D_raw,
        e1q_nat=backend_tensors.e1q_raw,
        ero_nat=backend_tensors.ero_raw,
        e2q_nat=backend_tensors.e2q_raw,
        p=p,
        p_inv=p_inv,
        n1q=n1q,
        nmeas=nmeas,
        circuit=circuit_meta,
        backend=backend_tensors.metadata,
        metadata={
            "source": source,
            "split": split,
            "cache_key": cache_key,
            "qasm_path": str(qasm_path),
            "logical_qubits": int(circuit_meta["logical_qubits"]),
            "canonical_meta": canonical_meta,
            "schema_version": CACHE_SCHEMA_VERSION,
        },
    )
    _validate_preprocessed_sample(sample)
    return sample


def circuit_cache_path(project_root: str | Path, cache_key: str) -> Path:
    return ensure_dir(Path(project_root) / "data" / "cache" / "circuits") / f"{cache_key}.pt"


def load_or_build_preprocessed_sample(project_root: str | Path, *, qasm_path: str | Path,
                                      backend_tensors: BackendTensors, source: str, split: str, cache_key: str,
                                      force_recompute: bool = False, max_qubits: int = 27, eps: float = 1e-8,
                                      allow_degenerate: bool = False) -> PreprocessedSample:
    cache_path = circuit_cache_path(project_root, cache_key)
    if cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("schema_version") == CACHE_SCHEMA_VERSION:
            sample = PreprocessedSample.from_payload(payload)
            _validate_preprocessed_sample(sample)
            return sample
    sample = build_preprocessed_sample(
        qasm_path=qasm_path,
        backend_tensors=backend_tensors,
        source=source,
        split=split,
        cache_key=cache_key,
        allow_degenerate=allow_degenerate,
        max_qubits=max_qubits,
        eps=eps,
    )
    ensure_dir(cache_path.parent)
    torch.save(sample.to_serializable(), cache_path)
    return sample
