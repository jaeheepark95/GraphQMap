from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kmw1.preprocessing.canonical_indexer import (
    CanonicalHardwareIndex,
    batched_decode_canonical_to_native_logits,
    build_canonical_hardware_index,
    canonicalize_hardware_tensors,
)
from kmw1.preprocessing.extractor import BackendTensors, load_or_build_backend_tensors
from kmw1.preprocessing.featurizer import CircuitFeatures, featurize_circuit_from_qasm
from kmw1.utils import ensure_dir

#==============================================================================
# @dataclass(slots=True)
# class PreprocessedSample:
#     A: torch.Tensor
#     m: torch.Tensor

#     B_nat: torch.Tensor
#     c1_nat: torch.Tensor
#     c2_nat: torch.Tensor
#     D_nat: torch.Tensor

#     B_can: torch.Tensor
#     c1_can: torch.Tensor
#     c2_can: torch.Tensor
#     D_can: torch.Tensor

#     D_raw_nat: torch.Tensor
#     e1q_nat: torch.Tensor
#     ero_nat: torch.Tensor
#     e2q_nat: torch.Tensor

#     p: torch.Tensor
#     p_inv: torch.Tensor

#     n1q: torch.Tensor
#     nmeas: torch.Tensor

#     circuit: dict[str, Any]
#     backend: dict[str, Any]
#     metadata: dict[str, Any]

#     def to_serializable(self) -> dict[str, Any]:
#         return {
#             "A": self.A,
#             "m": self.m,
#             "B_nat": self.B_nat,
#             "c1_nat": self.c1_nat,
#             "c2_nat": self.c2_nat,
#             "D_nat": self.D_nat,
#             "B_can": self.B_can,
#             "c1_can": self.c1_can,
#             "c2_can": self.c2_can,
#             "D_can": self.D_can,
#             "D_raw_nat": self.D_raw_nat,
#             "e1q_nat": self.e1q_nat,
#             "ero_nat": self.ero_nat,
#             "e2q_nat": self.e2q_nat,
#             "p": self.p,
#             "p_inv": self.p_inv,
#             "n1q": self.n1q,
#             "nmeas": self.nmeas,
#             "circuit": self.circuit,
#             "backend": self.backend,
#             "metadata": self.metadata,
#         }

#     @classmethod
#     def from_payload(cls, payload: dict[str, Any]) -> "PreprocessedSample":
#         return cls(**payload)
    


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
        # Ignore unknown keys from older cache versions.
        allowed_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in payload.items() if k in allowed_keys}
        return cls(**filtered)

#==============================================================================


def _validate_preprocessed_sample(sample: PreprocessedSample) -> None:
    for name in [
        "A", "m",
        "B_nat", "c1_nat", "c2_nat", "D_nat",
        "B_can", "c1_can", "c2_can", "D_can",
        "D_raw_nat", "e1q_nat", "ero_nat", "e2q_nat",
        "p", "p_inv", "n1q", "nmeas",
    ]:
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


def build_preprocessed_sample(
    *,
    qasm_path: str | Path,
    backend_tensors: BackendTensors,
    source: str,
    split: str,
    cache_key: str,
    allow_degenerate: bool = False,
    max_qubits: int = 27,
    eps: float = 1e-8,
) -> PreprocessedSample:
    circuit_features = featurize_circuit_from_qasm(
        qasm_path=qasm_path,
        max_qubits=max_qubits,
        eps=eps,
        allow_degenerate=allow_degenerate,
    )

    canonical_index = build_canonical_hardware_index(
        B_nat=backend_tensors.B,
        c1_nat=backend_tensors.c1,
        c2_nat=backend_tensors.c2,
    )
    can = canonicalize_hardware_tensors(
        p=canonical_index.p,
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
        A=circuit_features.A,
        m=circuit_features.m,
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
        p=canonical_index.p,
        p_inv=canonical_index.p_inv,
        n1q=circuit_features.n1q,
        nmeas=circuit_features.nmeas,
        circuit=circuit_features.metadata,
        backend=backend_tensors.metadata,
        metadata={
            "source": source,
            "split": split,
            "cache_key": cache_key,
            "qasm_path": str(qasm_path),
            "logical_qubits": int(circuit_features.metadata["logical_qubits"]),
            "canonical_qscore": canonical_index.qscore.tolist(),
        },
    )
    _validate_preprocessed_sample(sample)
    return sample


def circuit_cache_path(project_root: str | Path, cache_key: str) -> Path:
    return ensure_dir(Path(project_root) / "data" / "cache" / "circuits") / f"{cache_key}.pt"


def load_or_build_preprocessed_sample(
    project_root: str | Path,
    *,
    qasm_path: str | Path,
    backend_tensors: BackendTensors,
    source: str,
    split: str,
    cache_key: str,
    force_recompute: bool = False,
    max_qubits: int = 27,
    eps: float = 1e-8,
    allow_degenerate: bool = False,
) -> PreprocessedSample:
    cache_path = circuit_cache_path(project_root, cache_key)
    if cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location="cpu")
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


def load_or_build_backend_and_sample(
    project_root: str | Path,
    *,
    qasm_path: str | Path,
    backend_name: str,
    source: str,
    split: str,
    cache_key: str,
    force_recompute: bool = False,
    max_qubits: int = 27,
    eps: float = 1e-8,
    allow_degenerate: bool = False,
) -> tuple[BackendTensors, PreprocessedSample]:
    backend_tensors = load_or_build_backend_tensors(
        project_root=project_root,
        backend_name=backend_name,
        force_recompute=force_recompute,
        expected_num_qubits=max_qubits,
        eps=eps,
    )
    sample = load_or_build_preprocessed_sample(
        project_root=project_root,
        qasm_path=qasm_path,
        backend_tensors=backend_tensors,
        source=source,
        split=split,
        cache_key=cache_key,
        force_recompute=force_recompute,
        max_qubits=max_qubits,
        eps=eps,
        allow_degenerate=allow_degenerate,
    )
    return backend_tensors, sample
