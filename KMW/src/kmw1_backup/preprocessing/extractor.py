from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from kmw1.utils import ensure_dir


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
        return cls(**payload)


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
    """
    Resolve a backend with an explicit factory so replacing the backend later is easy.

    Supported names out of the box:
    - fake_toronto_v2
    - generic_backend_v2
    """
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
        f"Unsupported backend_name={backend_name!r}. "
        "Add a new branch in resolve_backend() to swap backends cleanly."
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
                op_map = target[name]
                qarg_items = list(op_map.items())
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
            edges = coupling_map.get_edges()
            return [(int(a), int(b)) for a, b in edges]
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


def _validate_backend_tensors(
    *,
    B_raw: torch.Tensor,
    c1_raw: torch.Tensor,
    c2_raw: torch.Tensor,
    D_raw: torch.Tensor,
    B: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    D: torch.Tensor,
) -> None:
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


def extract_backend_tensors(
    backend: Any,
    backend_name: str,
    expected_num_qubits: int = DEFAULT_NUM_QUBITS,
    eps: float = DEFAULT_EPS,
) -> BackendTensors:
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
        if a == b:
            continue
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

    _validate_backend_tensors(
        B_raw=B_raw, c1_raw=c1_raw, c2_raw=c2_raw, D_raw=D_raw,
        B=B, c1=c1, c2=c2, D=D,
    )

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
        },
    )


def backend_cache_path(project_root: str | Path, backend_name: str) -> Path:
    return ensure_dir(Path(project_root) / "data" / "cache" / "backends") / f"{backend_name}.pt"


def load_or_build_backend_tensors(
    project_root: str | Path,
    backend_name: str = "fake_toronto_v2",
    force_recompute: bool = False,
    expected_num_qubits: int = DEFAULT_NUM_QUBITS,
    eps: float = DEFAULT_EPS,
) -> BackendTensors:
    cache_path = backend_cache_path(project_root, backend_name)
    if cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location="cpu")
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
