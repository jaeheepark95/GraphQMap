from __future__ import annotations

"""
Dataset code for the early KMW pipeline.

Main responsibility of this file:
- read manifest rows
- load or build cached preprocessing results
- return native-frame tensors only

Important design decision:
This dataset does *not* do reindexing.
Reindexing is a model-side responsibility later in the pipeline.
"""

# from dataclasses import dataclass
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from kmw.preprocessing.pipeline import (
    BackendTensors,
    load_or_build_backend_tensors,
    load_or_build_preprocessed_sample,
)
from kmw.utils import ensure_dir, read_jsonl


#=============================================================================
#PREV MANIFEST ROW DEFINITION (before manifest_full changes)
# @dataclass(slots=True)
# class KMWManifestRow:
#     """One row from a JSONL manifest.

#     We mirror the agreed manifest schema here so the dataset can work with a
#     typed object instead of raw dictionaries everywhere.
#     """

#     id: str
#     source: str
#     split: str
#     qasm_relpath: str
#     k_logical: int | None = None
#     num_1q: int | None = None
#     num_2q: int | None = None
#     is_disconnected_logical_graph: bool | None = None
#     passed_parse: bool = True
#     passed_filter: bool = True
#     filter_tags: list[str] | None = None
#     include: bool = True
#     cache_key: str | None = None

#NEW MANIFEST ROW DEFINITION (after manifest_full changes)
@dataclass(slots=True)
class KMWManifestRow:
    id: str
    source: str
    split: str
    qasm_relpath: str
    k_logical: int | None = None
    num_1q: int | None = None
    num_2q: int | None = None
    is_disconnected_logical_graph: bool | None = None
    passed_parse: bool = True
    passed_filter: bool = True
    filter_tags: list[str] | None = None
    include: bool = True
    cache_key: str | None = None

    # new manifest_full fields
    dataset_version: str | None = None
    source_role: str | None = None


#=============================================================================

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KMWManifestRow":
        """Build a typed row object from a plain dictionary.

        We also fill in a few defaults so older or minimal manifests do not break
        immediately.
        """
        payload = dict(payload)
        payload.setdefault("filter_tags", [])
        payload.setdefault("include", True)
        payload.setdefault("cache_key", payload.get("id"))
        return cls(**payload)


class KMWDataset(Dataset):
    """Manifest-driven KMW dataset.

    What ``__getitem__`` returns:
    - A: logical interaction matrix (native frame)
    - m: logical occupancy / mask vector
    - B, c1, c2, D: backend tensors (native frame)
    - metadata: debugging / provenance information
    """

    def __init__(
        self,
        *,
        project_root: str | Path,
        manifest_path: str | Path,
        backend_name: str = "fake_toronto_v2",
        backend_tensors: BackendTensors | None = None,
        include_only: bool = True,
        force_recompute: bool = False,
        alpha_diag: float = 0.25,
        beta_diag: float = 1.0,
        max_qubits: int = 27,
        eps: float = 1e-8,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.include_only = include_only
        self.force_recompute = force_recompute
        self.alpha_diag = alpha_diag
        self.beta_diag = beta_diag
        self.max_qubits = max_qubits
        self.eps = eps

        # Read manifest rows first.
        self.rows = load_manifest_rows(self.manifest_path, include_only=include_only)

        # Build or load backend tensors once for the whole dataset.
        self.backend_tensors = (
            backend_tensors
            if backend_tensors is not None
            else load_or_build_backend_tensors(
                self.project_root,
                backend_name=backend_name,
                force_recompute=force_recompute,
                expected_num_qubits=max_qubits,
                eps=eps,
            )
        )

        # Make sure the circuit cache directory exists.
        ensure_dir(self.project_root / "data" / "cache" / "circuits")

    def __len__(self) -> int:
        """Return the number of usable manifest rows."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one sample.

        High-level flow:
        1. Read one manifest row.
        2. Resolve the QASM file path.
        3. Load cached preprocessing if it exists.
        4. Otherwise parse + featurize + cache the sample.
        5. Return native-frame tensors and metadata.
        """
        row = self.rows[index]
        qasm_path = self.project_root / row.qasm_relpath

        sample = load_or_build_preprocessed_sample(
            self.project_root,
            qasm_path=qasm_path,
            backend_tensors=self.backend_tensors,
            source=row.source,
            split=row.split,
            cache_key=row.cache_key or row.id,
            force_recompute=self.force_recompute,
            alpha_diag=self.alpha_diag,
            beta_diag=self.beta_diag,
            max_qubits=self.max_qubits,
            eps=self.eps,
        )

        metadata = {
            "id": row.id,
            "source": row.source,
            "split": row.split,
            "qasm_relpath": row.qasm_relpath,
            "cache_key": row.cache_key or row.id,
            # "manifest": row.__dict__, #KMWManifestRow is defined as a slotted dataclass, so it does not have a __dict__. Use asdict() to convert it to a regular dict.
            "manifest": asdict(row),
            "circuit": sample.circuit.metadata,
            "backend": sample.backend.metadata,
        }

        return {
            "A": sample.A,
            "m": sample.m,
            "B": sample.B,
            "c1": sample.c1,
            "c2": sample.c2,
            "D": sample.D,
            "metadata": metadata,
        }

    @property
    def backend_name(self) -> str:
        """Expose the resolved backend name as a convenience property."""
        return self.backend_tensors.backend_name



def load_manifest_rows(
    manifest_path: str | Path,
    *,
    include_only: bool = True,
) -> list[KMWManifestRow]:
    """Load manifest rows from a JSONL file.

    When ``include_only=True``, we keep only rows that:
    - are marked include=True
    - passed parsing
    - passed filtering
    """
    rows = [KMWManifestRow.from_dict(row) for row in read_jsonl(manifest_path)]
    if include_only:
        rows = [row for row in rows if row.include and row.passed_parse and row.passed_filter]
    return rows



def kmw_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for PyTorch DataLoader.

    PyTorch's default collate can handle many simple cases, but we use a custom
    one here so the intent is explicit and metadata stays as a list.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    tensor_keys = ["A", "m", "B", "c1", "c2", "D"]

    # Stack each tensor along a new batch dimension.
    collated = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}

    # Metadata is kept as a Python list because it is not a tensor.
    collated["metadata"] = [item["metadata"] for item in batch]
    return collated
