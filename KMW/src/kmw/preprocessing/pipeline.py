# =============================================================================
# UPDATE LOG (2026-03-30, v1.4.1)
# - Added raw backend error preservation for the loss/evaluation path:
#     * e1q_raw, ero_raw, e2q_raw, D_raw
# - Added circuit-side logical count tensors for the loss path:
#     * n1q, nmeas
# - Kept normalized c1 / c2 / D as the model-side representation.
# - The mapper/reindexer still consume normalized native tensors only.
# - The v1.4.1 loss/evaluation path now consumes raw errors + raw distance +
#   logical operation counts without changing the model input contract.
# =============================================================================
from __future__ import annotations
# from os import name # shadws the intended logic in resolve_backend; remove this line if it causes issues

"""
Preprocessing pipeline for the early KMW implementation.

This file is the *foundation* of the project because it defines how raw inputs
become the native-frame tensors consumed later by the model.

What this file does:
1. Resolve a backend object (for now: FakeTorontoV2 by default)
2. Extract backend-side tensors:
   - B  : hardware adjacency matrix
   - c1 : per-physical-qubit cost vector
   - c2 : per-physical-edge cost matrix
   - D  : shortest-path distance matrix on the hardware graph
3. Parse a QASM circuit
4. Build circuit-side tensors:
   - A : logical interaction matrix
   - m : occupancy / active-logical-qubit mask
5. Validate everything and cache results

Important design rule:
All outputs from this file are still in the *native* frame.
No reindexing happens here.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from kmw.utils import ensure_dir, stable_id

# -----------------------------
# Project-level constants
# -----------------------------
DEFAULT_NUM_QUBITS = 27
DEFAULT_EPS = 1e-8

# If two hardware qubits are disconnected, we fill that graph distance with a
# finite large value instead of infinity. This follows your stability rule:
# avoid Inf values in tensors and losses.
DEFAULT_UNREACHABLE_DISTANCE_FILL = DEFAULT_NUM_QUBITS + 1

# These operations are ignored when building logical or physical connectivity.
# They are not useful for the current interaction/cost view of the circuit.
IGNORED_CIRCUIT_OPS = {"barrier", "delay", "measure", "reset"}
IGNORED_TARGET_OPS = {"barrier", "delay", "measure", "reset"}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
# These are structured containers for the tensors and metadata we compute.
# Using dataclasses makes the code easier to read than passing around loose
# dictionaries everywhere.
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class BackendTensors:
    """Raw and normalized backend tensors for one BackendV2-like target."""

    backend_name: str
    num_qubits: int
    eps: float
    unreachable_distance_fill: int

    # Raw tensors = values before normalization.
    B_raw: torch.Tensor
    c1_raw: torch.Tensor
    c2_raw: torch.Tensor
    D_raw: torch.Tensor

    # Raw error tensors preserved for the v1.4.1 loss/evaluation path.
    e1q_raw: torch.Tensor
    ero_raw: torch.Tensor
    e2q_raw: torch.Tensor

    # Normalized tensors = values actually fed downstream to the model.
    B: torch.Tensor
    c1: torch.Tensor
    c2: torch.Tensor
    D: torch.Tensor

    metadata: dict[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for torch.save()."""
        return asdict(self)


@dataclass(slots=True)
class CircuitFeatures:
    """Raw and normalized circuit-side tensors plus derived metadata."""

    circuit_id: str
    qasm_path: str
    k_logical: int
    num_1q: int
    num_2q: int
    is_disconnected_logical_graph: bool
    offdiag_mass_raw: float
    A_raw: torch.Tensor
    A: torch.Tensor
    m: torch.Tensor
    n1q: torch.Tensor
    nmeas: torch.Tensor
    metadata: dict[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreprocessedSample:
    """One fully prepared sample in the native frame.

    This bundles the circuit tensors and backend tensors together so the dataset
    can load them in one shot.
    """

    sample_id: str
    source: str
    split: str | None
    qasm_path: str
    cache_key: str

    # Native-frame tensors returned to the dataset / model input pipeline.
    A: torch.Tensor
    m: torch.Tensor
    B: torch.Tensor
    c1: torch.Tensor
    c2: torch.Tensor
    D: torch.Tensor

    # Loss/evaluation-side raw tensors and logical count tensors.
    D_raw: torch.Tensor
    e1q: torch.Tensor
    ero: torch.Tensor
    e2q: torch.Tensor
    n1q: torch.Tensor
    nmeas: torch.Tensor

    # Rich metadata objects.
    circuit: CircuitFeatures
    backend: BackendTensors
    metadata: dict[str, Any]

    def tensor_dict(self) -> dict[str, torch.Tensor]:
        """Return only the tensors, useful for debugging or batch creation."""
        return {
            "A": self.A,
            "m": self.m,
            "B": self.B,
            "c1": self.c1,
            "c2": self.c2,
            "D": self.D,
            "D_raw": self.D_raw,
            "e1q": self.e1q,
            "ero": self.ero,
            "e2q": self.e2q,
            "n1q": self.n1q,
            "nmeas": self.nmeas,
        }

    def to_serializable(self) -> dict[str, Any]:
        """Return a dictionary that can be safely cached with torch.save()."""
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "split": self.split,
            "qasm_path": self.qasm_path,
            "cache_key": self.cache_key,
            "A": self.A,
            "m": self.m,
            "B": self.B,
            "c1": self.c1,
            "c2": self.c2,
            "D": self.D,
            "D_raw": self.D_raw,
            "e1q": self.e1q,
            "ero": self.ero,
            "e2q": self.e2q,
            "n1q": self.n1q,
            "nmeas": self.nmeas,
            "circuit": self.circuit.to_serializable(),
            "backend": self.backend.to_serializable(),
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# Backend resolution and circuit loading
# -----------------------------------------------------------------------------



# ---------------------------------------------------------

# def resolve_backend(
#     backend_name: str = "fake_toronto_v2",
#     num_qubits: int = DEFAULT_NUM_QUBITS,
# ) -> Any:
#     """Resolve a BackendV2-like object lazily.

#     Why lazy import?
#     Because this module should still import cleanly even in an environment where
#     Qiskit is not installed. The actual Qiskit requirement only appears when we
#     really need backend objects or QASM parsing.
#     """
#     # try:
#     #     from qiskit.providers.fake_provider import FakeTorontoV2, GenericBackendV2

#     try:
#         from qiskit_ibm_runtime.fake_provider import FakeTorontoV2
#     except ImportError:
#         FakeTorontoV2 = None

#     try:
#         from qiskit.providers.fake_provider import GenericBackendV2
#     except ImportError:
#         GenericBackendV2 = None


#     except Exception as exc:
#         raise ImportError(
#             "Qiskit fake backends could not be imported. Install Qiskit in the target environment."
#         ) from exc

#     key = backend_name.lower().strip()

#     # if key in {"fake_toronto_v2", "faketorontov2", "toronto", "fake_toronto"}:
#     #     return FakeTorontoV2()

#     # if key in {"generic_backend_v2", "generic", "generic27"}:
#     #     return GenericBackendV2(num_qubits=num_qubits)

#     if name in {"fake_toronto_v2", "fake_toronto", "toronto"}:
#         if FakeTorontoV2 is None:
#             raise ImportError(
#                 "FakeTorontoV2 is unavailable. Install qiskit-ibm-runtime in this environment."
#             )
#         return FakeTorontoV2()
#     if name in {"generic_backend_v2", "generic", "generic27"}:
#         if GenericBackendV2 is None:
#             raise ImportError(
#                 "GenericBackendV2 is unavailable in this Qiskit installation."
#             )
#         return GenericBackendV2(num_qubits=num_qubits)
    

#     raise ValueError(f"Unsupported backend name: {backend_name}")


def resolve_backend(
    backend_name: str = "fake_toronto_v2",
    num_qubits: int = DEFAULT_NUM_QUBITS,
) -> Any:
    """Resolve a BackendV2-like object lazily."""
    try:
        from qiskit_ibm_runtime.fake_provider import FakeTorontoV2
    except ImportError:
        FakeTorontoV2 = None

    try:
        from qiskit.providers.fake_provider import GenericBackendV2
    except ImportError:
        GenericBackendV2 = None

    key = backend_name.lower().strip()

    if key in {"fake_toronto_v2", "faketorontov2", "fake_toronto", "toronto"}:
        if FakeTorontoV2 is None:
            raise ImportError(
                "FakeTorontoV2 is unavailable. Install qiskit-ibm-runtime in this environment."
            )
        return FakeTorontoV2()

    if key in {"generic_backend_v2", "generic", "generic27"}:
        if GenericBackendV2 is None:
            raise ImportError(
                "GenericBackendV2 is unavailable in this Qiskit installation."
            )
        return GenericBackendV2(num_qubits=num_qubits)

    raise ValueError(f"Unsupported backend name: {backend_name}")



# ---------------------------------------------------------

def load_quantum_circuit(qasm_path: str | Path) -> Any:
    """Load a QASM circuit with conservative fallbacks.

    Qiskit has changed QASM loaders across versions, so we try several methods.
    This makes the code more robust across environments.
    """
    qasm_path = Path(qasm_path)

    try:
        from qiskit import QuantumCircuit, qasm2
    except Exception as exc:
        raise ImportError("Qiskit is required to parse circuit files.") from exc

    loaders = []

    # Try qasm2 first if available.
    if "qasm2" in locals() and hasattr(qasm2, "load"):
        loaders.append(qasm2.load)

    # Try qasm3 if installed / available.
    try:
        from qiskit import qasm3
        if hasattr(qasm3, "load"):
            loaders.append(qasm3.load)
    except Exception:
        pass

    # Final fallback: generic circuit loader.
    if hasattr(QuantumCircuit, "from_qasm_file"):
        loaders.append(QuantumCircuit.from_qasm_file)

    last_error: Exception | None = None
    for loader in loaders:
        try:
            return loader(str(qasm_path))
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to parse QASM file: {qasm_path}") from last_error


# -----------------------------------------------------------------------------
# Backend tensor extraction
# -----------------------------------------------------------------------------


def extract_backend_tensors(
    backend: Any,
    *,
    expected_num_qubits: int = DEFAULT_NUM_QUBITS,
    eps: float = DEFAULT_EPS,
    unreachable_distance_fill: int = DEFAULT_UNREACHABLE_DISTANCE_FILL,
) -> BackendTensors:
    """Extract backend tensors from a BackendV2-like object.

    Model path:
    - B, c1, c2, D (normalized)

    Loss/eval path:
    - D_raw, e1q_raw, ero_raw, e2q_raw
    """
    num_qubits = _get_backend_num_qubits(backend)
    if num_qubits != expected_num_qubits:
        raise ValueError(
            f"Expected backend with {expected_num_qubits} qubits, got {num_qubits}."
        )

    target = getattr(backend, "target", None)
    if target is None:
        raise ValueError("Backend does not expose a target object.")

    B_raw = _extract_binary_adjacency(target, num_qubits)
    e1q_raw, ero_raw = _extract_qubit_error_components(backend, target, num_qubits)
    e2q_raw = _extract_edge_costs(target, B_raw, num_qubits)

    c1_raw = 0.5 * ero_raw + 0.5 * e1q_raw
    c2_raw = e2q_raw.copy()
    D_raw = _all_pairs_shortest_paths(B_raw, unreachable_distance_fill)

    _validate_backend_raw_tensors(B_raw, c1_raw, c2_raw, D_raw, num_qubits)

    B = B_raw.copy()
    c1 = _zscore(c1_raw, eps=eps)
    c2 = _normalize_edge_costs(c2_raw, B_raw, eps=eps)
    D = _max_normalize(D_raw, eps=eps)

    _validate_backend_normalized_tensors(B, c1, c2, D, num_qubits)

    metadata = {
        "backend_name": getattr(backend, "name", backend.__class__.__name__),
        "num_qubits": num_qubits,
        "num_edges": int(B_raw.sum() // 2),
    }

    return BackendTensors(
        backend_name=str(metadata["backend_name"]),
        num_qubits=num_qubits,
        eps=eps,
        unreachable_distance_fill=unreachable_distance_fill,
        B_raw=torch.tensor(B_raw, dtype=torch.float32),
        c1_raw=torch.tensor(c1_raw, dtype=torch.float32),
        c2_raw=torch.tensor(c2_raw, dtype=torch.float32),
        D_raw=torch.tensor(D_raw, dtype=torch.float32),
        e1q_raw=torch.tensor(e1q_raw, dtype=torch.float32),
        ero_raw=torch.tensor(ero_raw, dtype=torch.float32),
        e2q_raw=torch.tensor(e2q_raw, dtype=torch.float32),
        B=torch.tensor(B, dtype=torch.float32),
        c1=torch.tensor(c1, dtype=torch.float32),
        c2=torch.tensor(c2, dtype=torch.float32),
        D=torch.tensor(D, dtype=torch.float32),
        metadata=metadata,
    )

def cache_backend_tensors(
    project_root: str | Path,
    backend_tensors: BackendTensors,
) -> Path:
    """Save backend tensors to the backend cache directory."""
    cache_dir = ensure_dir(Path(project_root) / "data" / "cache" / "backend")
    cache_path = cache_dir / f"{backend_tensors.backend_name.lower()}.pt"
    torch.save(backend_tensors.to_serializable(), cache_path)
    return cache_path



def load_backend_tensors_from_cache(cache_path: str | Path) -> BackendTensors:
    """Load cached backend tensors from disk."""
    payload = torch.load(cache_path, map_location="cpu")
    return BackendTensors(**payload)



def load_or_build_backend_tensors(
    project_root: str | Path,
    *,
    backend_name: str = "fake_toronto_v2",
    backend: Any | None = None,
    force_recompute: bool = False,
    expected_num_qubits: int = DEFAULT_NUM_QUBITS,
    eps: float = DEFAULT_EPS,
    unreachable_distance_fill: int = DEFAULT_UNREACHABLE_DISTANCE_FILL,
) -> BackendTensors:
    """Load cached backend tensors if possible, otherwise rebuild them."""
    cache_dir = ensure_dir(Path(project_root) / "data" / "cache" / "backend")
    cache_path = cache_dir / f"{backend_name.lower()}.pt"

    if cache_path.exists() and not force_recompute:
        try:
            return load_backend_tensors_from_cache(cache_path)
        except Exception:
            # Old backend cache missing the v1.4.1 tensors; rebuild it.
            pass

    backend_obj = backend if backend is not None else resolve_backend(backend_name, expected_num_qubits)
    tensors = extract_backend_tensors(
        backend_obj,
        expected_num_qubits=expected_num_qubits,
        eps=eps,
        unreachable_distance_fill=unreachable_distance_fill,
    )
    torch.save(tensors.to_serializable(), cache_path)
    return tensors


# -----------------------------------------------------------------------------
# Circuit featurization
# -----------------------------------------------------------------------------


def featurize_circuit(
    circuit: Any,
    *,
    qasm_path: str | Path,
    alpha_diag: float = 0.25,
    beta_diag: float = 1.0,
    max_qubits: int = DEFAULT_NUM_QUBITS,
    eps: float = DEFAULT_EPS,
) -> CircuitFeatures:
    """Convert a quantum circuit into logical-side tensors.

    Logical outputs:
    - A_raw / A : logical interaction matrix
    - m         : mask showing which logical slots are occupied

    Shape convention:
    Everything is padded to ``max_qubits`` so later code always sees fixed-size
    tensors compatible with the fixed 27-qubit backend phase.
    """
    k_logical = int(getattr(circuit, "num_qubits", 0))
    if k_logical > max_qubits:
        raise ValueError(f"Circuit has {k_logical} logical qubits, exceeds limit {max_qubits}.")
    if k_logical < 1:
        raise ValueError("Circuit must contain at least one logical qubit.")

    # A_raw is the unnormalized logical interaction matrix.
    A_raw = np.zeros((max_qubits, max_qubits), dtype=np.float32)

    # m marks which positions correspond to real logical qubits.
    m = np.zeros((max_qubits,), dtype=np.float32)
    m[:k_logical] = 1.0

    # We separately count 1Q activity, measurement activity, and 2Q participation.
    num_1q = np.zeros((max_qubits,), dtype=np.float32)
    num_meas = np.zeros((max_qubits,), dtype=np.float32)
    num_2q_part = np.zeros((max_qubits,), dtype=np.float32)

    # For 2-qubit interactions, we count pair weights.
    pair_weight: dict[tuple[int, int], float] = {}

    for op_name, qubit_indices in iter_circuit_operations(circuit):
        if op_name == "measure" and len(qubit_indices) == 1:
            num_meas[qubit_indices[0]] += 1.0
            continue

        if op_name in IGNORED_CIRCUIT_OPS:
            continue

        if len(qubit_indices) == 1:
            num_1q[qubit_indices[0]] += 1.0

        elif len(qubit_indices) == 2:
            # Sort so (u, v) and (v, u) count as the same logical pair.
            u, v = sorted(qubit_indices)
            pair_weight[(u, v)] = pair_weight.get((u, v), 0.0) + 1.0
            num_2q_part[u] += 1.0
            num_2q_part[v] += 1.0

    # Fill off-diagonal interaction strength using log(1 + count).
    for (u, v), weight in pair_weight.items():
        value = math.log1p(weight)
        A_raw[u, v] = value
        A_raw[v, u] = value

    # Fill the diagonal using a mix of 1q and 2q participation counts.
    for u in range(k_logical):
        A_raw[u, u] = math.log1p(alpha_diag * float(num_1q[u]) + beta_diag * float(num_2q_part[u]))

    # Force exact symmetry.
    A_raw = 0.5 * (A_raw + A_raw.T)

    # Check that there is at least some off-diagonal logical interaction mass.
    offdiag_mass_raw = float((A_raw - np.diag(np.diag(A_raw))).sum())
    if offdiag_mass_raw <= 0:
        raise ValueError("Degenerate circuit: zero off-diagonal interaction mass.")

    # Normalize the logical interaction matrix.
    A = _max_normalize(A_raw, eps=eps)

    graph_edges = list(pair_weight.keys())
    is_disconnected = _logical_graph_disconnected(k_logical, graph_edges)
    circuit_id = stable_id(Path(qasm_path).as_posix(), prefix="circ")

    metadata = {
        "alpha_diag": alpha_diag,
        "beta_diag": beta_diag,
        "num_pairs_with_2q": len(pair_weight),
    }

    return CircuitFeatures(
        circuit_id=circuit_id,
        qasm_path=str(qasm_path),
        k_logical=k_logical,
        num_1q=int(num_1q.sum()),
        num_2q=int(sum(pair_weight.values())),
        is_disconnected_logical_graph=is_disconnected,
        offdiag_mass_raw=offdiag_mass_raw,
        A_raw=torch.tensor(A_raw, dtype=torch.float32),
        A=torch.tensor(A, dtype=torch.float32),
        m=torch.tensor(m, dtype=torch.float32),
        n1q=torch.tensor(num_1q, dtype=torch.float32),
        nmeas=torch.tensor(num_meas, dtype=torch.float32),
        metadata=metadata,
    )



def inspect_circuit_file(
    qasm_path: str | Path,
    *,
    alpha_diag: float = 0.25,
    beta_diag: float = 1.0,
    max_qubits: int = DEFAULT_NUM_QUBITS,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    """Quickly inspect a circuit file and return manifest-friendly metadata.

    This is used by the manifest builder and intentionally returns a plain dict.
    """
    circuit = load_quantum_circuit(qasm_path)
    features = featurize_circuit(
        circuit,
        qasm_path=qasm_path,
        alpha_diag=alpha_diag,
        beta_diag=beta_diag,
        max_qubits=max_qubits,
        eps=eps,
    )
    return {
        "id": features.circuit_id,
        "qasm_path": str(qasm_path),
        "k_logical": features.k_logical,
        "num_1q": features.num_1q,
        "num_2q": features.num_2q,
        "is_disconnected_logical_graph": features.is_disconnected_logical_graph,
        "offdiag_mass_raw": features.offdiag_mass_raw,
    }


# -----------------------------------------------------------------------------
# Full sample preprocessing and caching
# -----------------------------------------------------------------------------


def preprocess_circuit_file(
    qasm_path: str | Path,
    *,
    backend_tensors: BackendTensors,
    source: str,
    split: str | None,
    cache_key: str | None = None,
    alpha_diag: float = 0.25,
    beta_diag: float = 1.0,
    max_qubits: int = DEFAULT_NUM_QUBITS,
    eps: float = DEFAULT_EPS,
) -> PreprocessedSample:
    """Build one complete preprocessed sample from a QASM file and backend tensors."""
    qasm_path = Path(qasm_path)
    circuit = load_quantum_circuit(qasm_path)

    circuit_features = featurize_circuit(
        circuit,
        qasm_path=qasm_path,
        alpha_diag=alpha_diag,
        beta_diag=beta_diag,
        max_qubits=max_qubits,
        eps=eps,
    )

    effective_cache_key = cache_key or stable_id(source, qasm_path.as_posix(), prefix="sample")
    sample_id = effective_cache_key

    metadata = {
        "source": source,
        "split": split,
        "backend_name": backend_tensors.backend_name,
    }

    # IMPORTANT: tensors returned here are still in the native frame.
    return PreprocessedSample(
        sample_id=sample_id,
        source=source,
        split=split,
        qasm_path=str(qasm_path),
        cache_key=effective_cache_key,
        A=circuit_features.A.clone(),
        m=circuit_features.m.clone(),
        B=backend_tensors.B.clone(),
        c1=backend_tensors.c1.clone(),
        c2=backend_tensors.c2.clone(),
        D=backend_tensors.D.clone(),
        D_raw=backend_tensors.D_raw.clone(),
        e1q=backend_tensors.e1q_raw.clone(),
        ero=backend_tensors.ero_raw.clone(),
        e2q=backend_tensors.e2q_raw.clone(),
        n1q=circuit_features.n1q.clone(),
        nmeas=circuit_features.nmeas.clone(),
        circuit=circuit_features,
        backend=backend_tensors,
        metadata=metadata,
    )



def cache_preprocessed_sample(project_root: str | Path, sample: PreprocessedSample) -> Path:
    """Save one preprocessed sample into the circuit cache."""
    cache_dir = ensure_dir(Path(project_root) / "data" / "cache" / "circuits")
    cache_path = cache_dir / f"{sample.cache_key}.pt"
    torch.save(sample.to_serializable(), cache_path)
    return cache_path



def load_preprocessed_sample(cache_path: str | Path) -> PreprocessedSample:
    """Load one cached preprocessed sample from disk."""
    payload = torch.load(cache_path, map_location="cpu")
    payload["circuit"] = CircuitFeatures(**payload["circuit"])
    payload["backend"] = BackendTensors(**payload["backend"])
    return PreprocessedSample(**payload)



def load_or_build_preprocessed_sample(
    project_root: str | Path,
    *,
    qasm_path: str | Path,
    backend_tensors: BackendTensors,
    source: str,
    split: str | None,
    cache_key: str,
    force_recompute: bool = False,
    alpha_diag: float = 0.25,
    beta_diag: float = 1.0,
    max_qubits: int = DEFAULT_NUM_QUBITS,
    eps: float = DEFAULT_EPS,
) -> PreprocessedSample:
    """Load a cached preprocessed sample if possible, otherwise rebuild it."""
    cache_dir = ensure_dir(Path(project_root) / "data" / "cache" / "circuits")
    cache_path = cache_dir / f"{cache_key}.pt"

    if cache_path.exists() and not force_recompute:
        try:
            return load_preprocessed_sample(cache_path)
        except Exception:
            # Old circuit cache missing the v1.4.1 tensors; rebuild it.
            pass

    sample = preprocess_circuit_file(
        qasm_path,
        backend_tensors=backend_tensors,
        source=source,
        split=split,
        cache_key=cache_key,
        alpha_diag=alpha_diag,
        beta_diag=beta_diag,
        max_qubits=max_qubits,
        eps=eps,
    )
    torch.save(sample.to_serializable(), cache_path)
    return sample


# -----------------------------------------------------------------------------
# Circuit / backend helper functions
# -----------------------------------------------------------------------------


def iter_circuit_operations(circuit: Any) -> Iterable[tuple[str, list[int]]]:
    """Yield circuit operations as (op_name, logical_qubit_indices).

    Why have this helper?
    Qiskit's internal representation has changed over time. This function hides
    those differences from the rest of the preprocessing code.
    """
    for item in getattr(circuit, "data", []):
        if hasattr(item, "operation"):
            # Newer Qiskit representation.
            operation = item.operation
            qubits = item.qubits
        else:
            # Older tuple-like representation.
            operation, qubits, _ = item

        op_name = str(getattr(operation, "name", operation.__class__.__name__)).lower()

        indices: list[int] = []
        for qubit in qubits:
            try:
                index = int(circuit.find_bit(qubit).index)
            except Exception:
                index = int(getattr(qubit, "index"))
            indices.append(index)

        yield op_name, indices



def _get_backend_num_qubits(backend: Any) -> int:
    """Read the backend qubit count robustly."""
    value = getattr(backend, "num_qubits", None)
    if value is None:
        raise ValueError("Backend does not expose num_qubits.")
    return int(value() if callable(value) else value)



def _iter_target_operations(target: Any) -> Iterable[tuple[str, tuple[int, ...], Any]]:
    """Iterate over target operations in a Qiskit-version-tolerant way.

    Yields
    ------
    (op_name, qargs, props)
        op_name : lowercased operation name
        qargs   : tuple of qubit indices
        props   : operation property object / dict / None
    """
    op_names = list(getattr(target, "operation_names", []) or [])

    for op_name in op_names:
        qargs_iter = None

        if hasattr(target, "qargs_for_operation_name"):
            try:
                qargs_iter = list(target.qargs_for_operation_name(op_name))
            except Exception:
                qargs_iter = None

        if qargs_iter is None:
            try:
                table = target[op_name]
                if hasattr(table, "keys"):
                    qargs_iter = list(table.keys())
            except Exception:
                qargs_iter = None

        if not qargs_iter:
            continue

        for qargs in qargs_iter:
            qargs = tuple(int(x) for x in qargs)
            props = None
            try:
                props = target[op_name][qargs]
            except Exception:
                try:
                    props = target[op_name].get(qargs)
                except Exception:
                    props = None
            yield str(op_name).lower(), qargs, props



def _extract_binary_adjacency(target: Any, num_qubits: int) -> np.ndarray:
    """Build the binary hardware adjacency matrix B_raw."""
    B = np.zeros((num_qubits, num_qubits), dtype=np.float32)
    for op_name, qargs, _ in _iter_target_operations(target):
        if op_name in IGNORED_TARGET_OPS or len(qargs) != 2:
            continue
        i, j = qargs
        B[i, j] = 1.0
        B[j, i] = 1.0

    np.fill_diagonal(B, 0.0)
    return np.maximum(B, B.T)


def _extract_qubit_error_components(backend: Any, target: Any, num_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Return raw per-qubit 1Q and readout error tensors for the loss path."""
    ero_raw = np.zeros((num_qubits,), dtype=np.float32)
    one_qubit_errors: list[list[float]] = [[] for _ in range(num_qubits)]

    for j in range(num_qubits):
        ero_raw[j] = float(_readout_error_for_qubit(backend, target, j))

    for op_name, qargs, props in _iter_target_operations(target):
        if op_name in IGNORED_TARGET_OPS or len(qargs) != 1:
            continue
        error = _extract_error(props)
        if error is None:
            continue
        one_qubit_errors[qargs[0]].append(float(error))

    e1q_raw = np.zeros((num_qubits,), dtype=np.float32)
    for j in range(num_qubits):
        e1q_raw[j] = float(np.mean(one_qubit_errors[j])) if one_qubit_errors[j] else 0.0

    return e1q_raw, ero_raw



def _extract_qubit_costs(backend: Any, target: Any, num_qubits: int) -> np.ndarray:
    """Build the per-physical-qubit raw cost vector c1_raw.

    Current heuristic:
    c1_raw[j] = 0.5 * readout_error(j) + 0.5 * mean_1q_gate_error(j)
    """
    readout = np.zeros((num_qubits,), dtype=np.float32)
    one_qubit_errors: list[list[float]] = [[] for _ in range(num_qubits)]

    for j in range(num_qubits):
        err = _readout_error_for_qubit(backend, target, j)
        readout[j] = float(err)

    for op_name, qargs, props in _iter_target_operations(target):
        if op_name in IGNORED_TARGET_OPS or len(qargs) != 1:
            continue
        error = _extract_error(props)
        if error is None:
            continue
        one_qubit_errors[qargs[0]].append(float(error))

    c1_raw = np.zeros((num_qubits,), dtype=np.float32)
    for j in range(num_qubits):
        e_1q = float(np.mean(one_qubit_errors[j])) if one_qubit_errors[j] else 0.0
        c1_raw[j] = 0.5 * float(readout[j]) + 0.5 * e_1q

    return c1_raw



def _extract_edge_costs(target: Any, B: np.ndarray, num_qubits: int) -> np.ndarray:
    """Build the per-physical-edge raw cost matrix c2_raw.

    For each connected pair (i, j), we look at directional 2-qubit gate errors
    and keep the best available value.
    """
    directional: dict[tuple[int, int], list[float]] = {}

    for op_name, qargs, props in _iter_target_operations(target):
        if op_name in IGNORED_TARGET_OPS or len(qargs) != 2:
            continue
        err = _extract_error(props)
        if err is None:
            continue
        directional.setdefault((qargs[0], qargs[1]), []).append(float(err))

    c2_raw = np.zeros((num_qubits, num_qubits), dtype=np.float32)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if B[i, j] == 0:
                # Non-edges stay at zero here. We do not use inf.
                cost = 0.0
            else:
                values: list[float] = []
                if (i, j) in directional:
                    values.append(min(directional[(i, j)]))
                if (j, i) in directional:
                    values.append(min(directional[(j, i)]))
                cost = min(values) if values else 0.0

            c2_raw[i, j] = cost
            c2_raw[j, i] = cost

    np.fill_diagonal(c2_raw, 0.0)
    return c2_raw



def _readout_error_for_qubit(backend: Any, target: Any, qubit_index: int) -> float:
    """Try a few ways to read the backend readout error for one qubit."""
    try:
        qprops = target.qubit_properties[qubit_index]
        if qprops is not None and hasattr(qprops, "readout_error"):
            value = getattr(qprops, "readout_error")
            if value is not None:
                return float(value)
    except Exception:
        pass

    try:
        if hasattr(backend, "qubit_properties"):
            qprops = backend.qubit_properties(qubit_index)
            if qprops is not None and hasattr(qprops, "readout_error"):
                value = getattr(qprops, "readout_error")
                if value is not None:
                    return float(value)
    except Exception:
        pass

    return 0.0



def _extract_error(props: Any) -> float | None:
    """Extract an error value from a Qiskit properties object or dict."""
    if props is None:
        return None
    if hasattr(props, "error") and getattr(props, "error") is not None:
        return float(getattr(props, "error"))
    if isinstance(props, dict) and props.get("error") is not None:
        return float(props["error"])
    return None



def _all_pairs_shortest_paths(B: np.ndarray, unreachable_fill: int) -> np.ndarray:
    """Compute all-pairs shortest-path distances on the hardware graph.

    Because the graph is unweighted, BFS from every node is enough.
    """
    n = B.shape[0]
    D = np.full((n, n), fill_value=float(unreachable_fill), dtype=np.float32)

    for i in range(n):
        D[i, i] = 0.0
        queue = [i]
        head = 0

        while head < len(queue):
            src = queue[head]
            head += 1
            neighbors = np.where(B[src] > 0)[0]

            for dst in neighbors:
                if D[i, dst] > D[i, src] + 1:
                    D[i, dst] = D[i, src] + 1
                    queue.append(int(dst))

    return D


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------


def _max_normalize(values: np.ndarray, *, eps: float) -> np.ndarray:
    """Divide by the maximum value so the largest entry becomes about 1.

    If the matrix is all zeros or has non-positive max, return it unchanged.
    """
    max_val = float(np.max(values))
    if max_val <= 0.0:
        return values.astype(np.float32, copy=True)
    return (values / (max_val + eps)).astype(np.float32)



def _zscore(values: np.ndarray, *, eps: float) -> np.ndarray:
    """Standardize values to roughly zero mean and unit variance."""
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    return ((values - mu) / (sigma + eps)).astype(np.float32)



def _normalize_edge_costs(c2_raw: np.ndarray, B: np.ndarray, *, eps: float) -> np.ndarray:
    """Z-score normalize c2 only on valid hardware edges.

    Non-edges remain zero by construction.
    This avoids contaminating the statistics with padded / invalid entries.
    """
    c2 = np.zeros_like(c2_raw, dtype=np.float32)
    valid_mask = B > 0

    if not np.any(valid_mask):
        return c2

    vals = c2_raw[valid_mask]
    mu = float(np.mean(vals))
    sigma = float(np.std(vals))
    c2[valid_mask] = (c2_raw[valid_mask] - mu) / (sigma + eps)
    np.fill_diagonal(c2, 0.0)
    c2[~valid_mask] = 0.0
    return c2.astype(np.float32)


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------
# These checks are important because they help us fail early instead of allowing
# a silent shape / NaN / symmetry bug to poison later training.
# -----------------------------------------------------------------------------


def _validate_backend_raw_tensors(
    B_raw: np.ndarray,
    c1_raw: np.ndarray,
    c2_raw: np.ndarray,
    D_raw: np.ndarray,
    num_qubits: int,
) -> None:
    if B_raw.shape != (num_qubits, num_qubits):
        raise ValueError(f"B_raw has wrong shape: {B_raw.shape}")
    if c1_raw.shape != (num_qubits,):
        raise ValueError(f"c1_raw has wrong shape: {c1_raw.shape}")
    if c2_raw.shape != (num_qubits, num_qubits):
        raise ValueError(f"c2_raw has wrong shape: {c2_raw.shape}")
    if D_raw.shape != (num_qubits, num_qubits):
        raise ValueError(f"D_raw has wrong shape: {D_raw.shape}")

    _assert_symmetric_binary(B_raw, name="B_raw")
    _assert_symmetric_finite(c2_raw, name="c2_raw")
    _assert_symmetric_finite(D_raw, name="D_raw")
    _assert_zero_diagonal(B_raw, name="B_raw")
    _assert_zero_diagonal(c2_raw, name="c2_raw")
    _assert_zero_diagonal(D_raw, name="D_raw")



def _validate_backend_normalized_tensors(
    B: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    D: np.ndarray,
    num_qubits: int,
) -> None:
    if B.shape != (num_qubits, num_qubits):
        raise ValueError(f"B has wrong shape: {B.shape}")
    if c1.shape != (num_qubits,):
        raise ValueError(f"c1 has wrong shape: {c1.shape}")
    if c2.shape != (num_qubits, num_qubits):
        raise ValueError(f"c2 has wrong shape: {c2.shape}")
    if D.shape != (num_qubits, num_qubits):
        raise ValueError(f"D has wrong shape: {D.shape}")

    _assert_symmetric_binary(B, name="B")
    _assert_finite(c1, name="c1")
    _assert_symmetric_finite(c2, name="c2")
    _assert_symmetric_finite(D, name="D")
    _assert_zero_diagonal(B, name="B")
    _assert_zero_diagonal(c2, name="c2")
    _assert_zero_diagonal(D, name="D")



def _assert_finite(values: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")



def _assert_symmetric_finite(values: np.ndarray, *, name: str) -> None:
    _assert_finite(values, name=name)
    if not np.allclose(values, values.T, atol=1e-6):
        raise ValueError(f"{name} is not symmetric.")



def _assert_symmetric_binary(values: np.ndarray, *, name: str) -> None:
    _assert_symmetric_finite(values, name=name)
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"{name} is not binary: {sorted(unique)}")



def _assert_zero_diagonal(values: np.ndarray, *, name: str) -> None:
    if not np.allclose(np.diag(values), 0.0, atol=1e-6):
        raise ValueError(f"{name} diagonal is not zero.")



def _logical_graph_disconnected(k_logical: int, edges: list[tuple[int, int]]) -> bool:
    """Return True if the logical interaction graph is disconnected."""
    if k_logical <= 1:
        return False

    adjacency = [[] for _ in range(k_logical)]
    for u, v in edges:
        if u >= k_logical or v >= k_logical:
            continue
        adjacency[u].append(v)
        adjacency[v].append(u)

    # No edges at all means disconnected for k_logical > 1.
    if not any(adjacency):
        return True

    # Standard DFS / graph reachability check.
    seen = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(nei for nei in adjacency[node] if nei not in seen)

    return len(seen) != k_logical
