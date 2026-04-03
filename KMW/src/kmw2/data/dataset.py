from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from qiskit import QuantumCircuit

from ..preprocessing.featurizer import CircuitFeaturizer
from ..preprocessing.pipeline import build_hardware_cache
from ..utils import load_structured_file, normalize_source_name


@dataclass
class CircuitRecord:
    circuit_id: str
    source: str
    qasm_path: Optional[str]
    logical_qubits: int
    circuit: QuantumCircuit


class QubitMappingDataset(Dataset):
    """Manifest-first dataset with direct-path fallback and cached backend preprocessing."""

    def __init__(
        self,
        backend,
        n: int = 27,
        manifest_path: Optional[str | Path] = None,
        qasm_paths: Optional[Sequence[str | Path]] = None,
        circuits: Optional[Sequence[QuantumCircuit]] = None,
        records: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        self.n = n
        self.backend = backend
        self.featurizer = CircuitFeaturizer(n=n)
        self.hardware = build_hardware_cache(backend, n=n)
        self.records = self._load_records(
            manifest_path=manifest_path,
            qasm_paths=qasm_paths,
            circuits=circuits,
            records=records,
        )
        if not self.records:
            raise ValueError('Dataset is empty.')

    @staticmethod
    def _load_qasm(path: str | Path) -> QuantumCircuit:
        return QuantumCircuit.from_qasm_file(str(path))

    def _normalize_record(self, idx: int, payload: Dict[str, Any]) -> CircuitRecord:
        qasm_path = payload.get('qasm_path') or payload.get('qasm_relpath')
        circuit = payload.get('circuit')
        if circuit is None:
            if not qasm_path:
                raise ValueError('Each dataset record must provide either circuit or qasm_path.')
            circuit = self._load_qasm(qasm_path)
        circuit_id = payload.get('circuit_id') or payload.get('id') or f'circuit_{idx:05d}'
        source = normalize_source_name(payload.get('source') or 'unknown')
        logical_qubits = payload.get('logical_qubits') or payload.get('k_logical') or circuit.num_qubits
        return CircuitRecord(
            circuit_id=str(circuit_id),
            source=str(source),
            qasm_path=str(qasm_path) if qasm_path is not None else None,
            logical_qubits=int(logical_qubits),
            circuit=circuit,
        )

    def _load_records(self, manifest_path=None, qasm_paths=None, circuits=None, records=None) -> List[CircuitRecord]:
        normalized: List[CircuitRecord] = []
        if manifest_path is not None:
            payload = load_structured_file(manifest_path)
            if isinstance(payload, list):
                items = payload
            else:
                items = payload.get('samples') or payload.get('items') or payload.get('records') or []
            for idx, item in enumerate(items):
                if item.get('include') is False:
                    continue
                normalized.append(self._normalize_record(idx, item))
            return normalized

        if records is not None:
            for idx, item in enumerate(records):
                if item.get('include') is False:
                    continue
                normalized.append(self._normalize_record(idx, item))
            return normalized

        if qasm_paths is not None:
            for idx, path in enumerate(qasm_paths):
                path = Path(path)
                normalized.append(
                    self._normalize_record(
                        idx,
                        {
                            'circuit_id': path.stem,
                            'source': path.parent.name,
                            'qasm_path': str(path),
                        },
                    )
                )
            return normalized

        if circuits is not None:
            for idx, circuit in enumerate(circuits):
                normalized.append(
                    self._normalize_record(
                        idx,
                        {
                            'circuit_id': f'in_memory_{idx:05d}',
                            'source': 'in_memory',
                            'circuit': circuit,
                        },
                    )
                )
            return normalized

        return normalized

    @classmethod
    def from_recipe(cls, recipe_path: str | Path, split: str, backend, n: int = 27):
        recipe = load_structured_file(recipe_path)
        split_manifests = recipe.get('split_manifests', {})
        manifest_path = split_manifests.get(split) or recipe.get(f'{split}_manifest')
        if not manifest_path:
            raise KeyError(f'Split {split!r} not found in recipe.')
        recipe_dir = Path(recipe_path).parent
        return cls(backend=backend, n=n, manifest_path=recipe_dir / manifest_path)

    def subset_by_sources(self, allowed_sources: Sequence[str]):
        allowed = {normalize_source_name(s) for s in allowed_sources}
        records = [
            {
                'circuit_id': rec.circuit_id,
                'source': rec.source,
                'qasm_path': rec.qasm_path,
                'logical_qubits': rec.logical_qubits,
            }
            for rec in self.records
            if rec.source in allowed
        ]
        return QubitMappingDataset(backend=self.backend, n=self.n, records=records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        W, m = self.featurizer.featurize(record.circuit)
        return {
            'W': torch.tensor(W, dtype=torch.float32),
            'm': torch.tensor(m, dtype=torch.float32),
            'A': torch.tensor(self.hardware['A'], dtype=torch.float32),
            'c1': torch.tensor(self.hardware['c1'], dtype=torch.float32),
            'c2': torch.tensor(self.hardware['c2'], dtype=torch.float32),
            'D': torch.tensor(self.hardware['D'], dtype=torch.float32),
            'p': torch.tensor(self.hardware['p'], dtype=torch.long),
            'p_inv': torch.tensor(self.hardware['p_inv'], dtype=torch.long),
            'circuit_id': record.circuit_id,
            'source': record.source,
            'qasm_path': record.qasm_path or '',
            'logical_qubits': record.logical_qubits,
        }
