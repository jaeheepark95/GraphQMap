from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from kmw1.preprocessing.pipeline import load_or_build_backend_tensors, load_or_build_preprocessed_sample
from kmw1.utils import read_jsonl


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
    dataset_version: str | None = None
    source_role: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KMWManifestRow":
        payload = dict(payload)
        payload.setdefault("filter_tags", [])
        payload.setdefault("include", True)
        payload.setdefault("cache_key", payload.get("id"))
        payload.setdefault("source", payload.get("source") or payload.get("dataset") or payload.get("source_name"))
        return cls(**payload)


def load_manifest_rows(manifest_path: str | Path, include_only: bool = True) -> list[KMWManifestRow]:
    rows = [KMWManifestRow.from_dict(row) for row in read_jsonl(manifest_path)]
    if include_only:
        rows = [row for row in rows if row.include and row.passed_parse and row.passed_filter]
    return rows


def build_source_datasets(
    *,
    project_root: str | Path,
    source_manifest_root: str | Path,
    split: str,
    backend_name: str = "fake_toronto_v2",
    include_only: bool = True,
    force_recompute: bool = False,
    allow_degenerate: bool = False,
    max_qubits: int = 27,
    eps: float = 1e-8,
    required_sources: list[str] | None = None,
) -> dict[str, "KMW1Dataset"]:
    root = Path(source_manifest_root)
    if required_sources is None:
        required_sources = [p.name for p in root.iterdir() if p.is_dir()]
    out: dict[str, KMW1Dataset] = {}
    for source in required_sources:
        manifest_path = root / source / f"{split}.jsonl"
        if manifest_path.exists():
            out[source] = KMW1Dataset(
                project_root=project_root,
                manifest_path=manifest_path,
                backend_name=backend_name,
                include_only=include_only,
                force_recompute=force_recompute,
                allow_degenerate=allow_degenerate,
                max_qubits=max_qubits,
                eps=eps,
            )
    return out


class KMW1Dataset(Dataset):
    def __init__(
        self,
        *,
        project_root: str | Path,
        manifest_path: str | Path,
        backend_name: str = "fake_toronto_v2",
        include_only: bool = True,
        force_recompute: bool = False,
        allow_degenerate: bool = False,
        max_qubits: int = 27,
        eps: float = 1e-8,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.rows = load_manifest_rows(self.manifest_path, include_only=include_only)
        self.force_recompute = force_recompute
        self.allow_degenerate = allow_degenerate
        self.max_qubits = max_qubits
        self.eps = eps
        self.backend_name = backend_name
        self.backend_tensors = load_or_build_backend_tensors(
            project_root=self.project_root,
            backend_name=backend_name,
            force_recompute=force_recompute,
            expected_num_qubits=max_qubits,
            eps=eps,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        qasm_path = self.project_root / row.qasm_relpath
        sample = load_or_build_preprocessed_sample(
            project_root=self.project_root,
            qasm_path=qasm_path,
            backend_tensors=self.backend_tensors,
            source=row.source,
            split=row.split,
            cache_key=row.cache_key or row.id,
            force_recompute=self.force_recompute,
            max_qubits=self.max_qubits,
            eps=self.eps,
            allow_degenerate=self.allow_degenerate,
        )
        metadata = {
            "id": row.id,
            "source": row.source,
            "split": row.split,
            "qasm_relpath": row.qasm_relpath,
            "cache_key": row.cache_key or row.id,
            "manifest": asdict(row),
            "circuit": sample.circuit,
            "backend": sample.backend,
        }
        return {
            "A": sample.A,
            "m": sample.m,
            "B_can": sample.B_can,
            "c1_can": sample.c1_can,
            "c2_can": sample.c2_can,
            "D_can": sample.D_can,
            "B_nat": sample.B_nat,
            "c1_nat": sample.c1_nat,
            "c2_nat": sample.c2_nat,
            "D_nat": sample.D_nat,
            "D_raw_nat": sample.D_raw_nat,
            "e1q_nat": sample.e1q_nat,
            "ero_nat": sample.ero_nat,
            "e2q_nat": sample.e2q_nat,
            "p": sample.p,
            "p_inv": sample.p_inv,
            "n1q": sample.n1q,
            "nmeas": sample.nmeas,
            "metadata": metadata,
        }


def kmw1_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    tensor_keys = [
        "A", "m",
        "B_can", "c1_can", "c2_can", "D_can",
        "B_nat", "c1_nat", "c2_nat", "D_nat",
        "D_raw_nat", "e1q_nat", "ero_nat", "e2q_nat",
        "p", "p_inv", "n1q", "nmeas",
    ]
    out = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    out["metadata"] = [item["metadata"] for item in batch]
    return out
