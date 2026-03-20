"""PyG Dataset and DataLoader for GraphQMap.

Provides:
  - MappingDataset: stores (circuit_graph, hardware_graph, label) samples
  - BackendBucketSampler: groups samples by backend for uniform h within batch
  - create_dataloader: factory with dynamic batch sizing
  - load_split: build a MappingDataset from a splits/*.json file
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Batch, Data

from data.circuit_graph import build_circuit_graph, load_circuit
from data.hardware_graph import (
    build_hardware_graph,
    build_hardware_graph_from_synthetic,
    get_backend,
    get_hw_node_features,
    get_hw_node_features_synthetic,
    is_synthetic_backend,
    precompute_error_distance,
    precompute_error_distance_synthetic,
    precompute_hop_distance,
    precompute_hop_distance_synthetic,
)
from data.label_generation import layout_to_permutation_matrix

logger = logging.getLogger(__name__)


class MappingSample:
    """A single (circuit, hardware, label) training sample.

    Attributes:
        circuit_graph: PyG Data for the circuit.
        hardware_graph: PyG Data for the hardware.
        label_matrix: (h, h) permutation matrix (None for Stage 2).
        backend_name: Name of the hardware backend.
        num_logical: Number of logical qubits.
        num_physical: Number of physical qubits.
        layout: Raw layout list (optional).
        d_error: (h, h) error-weighted distance matrix (Stage 2).
        d_hw: (h, h) hardware distance matrix for separation loss (Stage 2).
        hw_node_features: (h, 5) quality features for QualityScore (Stage 2).
        circuit_edge_pairs: List of (i, j) logical qubit pairs with 2Q gates.
        cross_circuit_pairs: List of (i, j) cross-circuit pairs (multi-prog).
        qubit_importance: (l,) 2Q gate count per logical qubit.
    """

    def __init__(
        self,
        circuit_graph: Data,
        hardware_graph: Data,
        backend_name: str,
        num_logical: int,
        num_physical: int,
        label_matrix: np.ndarray | None = None,
        layout: list[int] | None = None,
        d_error: np.ndarray | None = None,
        d_hw: np.ndarray | None = None,
        hw_node_features: np.ndarray | None = None,
        circuit_edge_pairs: list[tuple[int, int]] | None = None,
        cross_circuit_pairs: list[tuple[int, int]] | None = None,
        qubit_importance: np.ndarray | None = None,
    ) -> None:
        self.circuit_graph = circuit_graph
        self.hardware_graph = hardware_graph
        self.backend_name = backend_name
        self.num_logical = num_logical
        self.num_physical = num_physical
        self.label_matrix = label_matrix
        self.layout = layout
        self.d_error = d_error
        self.d_hw = d_hw
        self.hw_node_features = hw_node_features
        self.circuit_edge_pairs = circuit_edge_pairs or []
        self.cross_circuit_pairs = cross_circuit_pairs or []
        self.qubit_importance = qubit_importance


class MappingDataset(Dataset):
    """Dataset of qubit mapping samples.

    Stores pre-built samples and provides indexing.
    Supports dynamic addition of samples.
    """

    def __init__(self) -> None:
        self.samples: list[MappingSample] = []
        self._backend_indices: dict[str, list[int]] = {}

    def add_sample(self, sample: MappingSample) -> None:
        """Add a sample to the dataset."""
        idx = len(self.samples)
        self.samples.append(sample)
        backend = sample.backend_name
        if backend not in self._backend_indices:
            self._backend_indices[backend] = []
        self._backend_indices[backend].append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MappingSample:
        return self.samples[idx]

    @property
    def backend_names(self) -> list[str]:
        """List of unique backend names in the dataset."""
        return list(self._backend_indices.keys())

    def indices_for_backend(self, backend_name: str) -> list[int]:
        """Get sample indices for a specific backend."""
        return self._backend_indices.get(backend_name, [])


class BackendBucketSampler(Sampler):
    """Sampler that groups samples by (backend, num_logical) for uniform tensor shapes.

    Ensures all samples in a mini-batch use the same hardware backend AND have
    the same number of logical qubits. This enables efficient batched computation:
    - Same h (physical qubits) from same backend
    - Same l (logical qubits) allows view-based reshaping in the model

    Args:
        dataset: MappingDataset instance.
        max_total_nodes: Maximum total physical qubits per batch.
        shuffle: Whether to shuffle within and across groups.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: MappingDataset,
        max_total_nodes: int = 512,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.max_total_nodes = max_total_nodes
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

    def _build_groups(self) -> dict[tuple[str, int], list[int]]:
        """Group sample indices by (backend_name, num_logical)."""
        groups: dict[tuple[str, int], list[int]] = {}
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            key = (sample.backend_name, sample.num_logical)
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        return groups

    def __iter__(self):
        groups = self._build_groups()
        if not groups:
            return iter([])

        all_batches = []

        for (backend, num_logical), indices in groups.items():
            indices = list(indices)
            if self.shuffle:
                self.rng.shuffle(indices)

            # Determine batch size based on physical qubit count
            sample = self.dataset[indices[0]]
            num_physical = sample.num_physical
            batch_size = max(1, self.max_total_nodes // num_physical)

            # Create batches
            for i in range(0, len(indices), batch_size):
                batch = indices[i : i + batch_size]
                all_batches.append(batch)

        # Shuffle batch order across groups
        if self.shuffle:
            self.rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        groups = self._build_groups()
        total = 0
        for (backend, num_logical), indices in groups.items():
            sample = self.dataset[indices[0]]
            batch_size = max(1, self.max_total_nodes // sample.num_physical)
            total += (len(indices) + batch_size - 1) // batch_size
        return total


def collate_mapping_samples(
    samples: list[MappingSample],
) -> dict[str, Any]:
    """Collate a list of MappingSample into batched tensors.

    All samples must share the same backend and num_logical.

    Returns:
        Dict with:
        - circuit_batch: Batched PyG Data
        - hardware_batch: Batched PyG Data
        - label_matrices: (batch, h, h) tensor or None
        - num_logical: list of per-sample logical qubit counts
        - num_physical: int
        - backend_name: str
        - batch_size: int
        Stage 2 fields (present if first sample has them):
        - d_error: (h, h) tensor
        - d_hw: (h, h) tensor
        - hw_node_features: (h, 5) tensor
        - circuit_edge_pairs: list of (i, j) tuples
        - cross_circuit_pairs: list of (i, j) tuples
        - qubit_importance: (l,) tensor
    """
    circuit_graphs = [s.circuit_graph for s in samples]
    hardware_graphs = [s.hardware_graph for s in samples]

    circuit_batch = Batch.from_data_list(circuit_graphs)
    hardware_batch = Batch.from_data_list(hardware_graphs)

    num_physical = samples[0].num_physical
    num_logical_list = [s.num_logical for s in samples]

    # Label matrices (Stage 1)
    if samples[0].label_matrix is not None:
        label_matrices = torch.tensor(
            np.stack([s.label_matrix for s in samples]),
            dtype=torch.float32,
        )
    else:
        label_matrices = None

    result: dict[str, Any] = {
        "circuit_batch": circuit_batch,
        "hardware_batch": hardware_batch,
        "label_matrices": label_matrices,
        "num_logical": num_logical_list,
        "num_physical": num_physical,
        "backend_name": samples[0].backend_name,
        "batch_size": len(samples),
    }

    # Stage 2 metadata — shared per backend, take from first sample
    s0 = samples[0]
    if s0.d_error is not None:
        result["d_error"] = torch.tensor(s0.d_error, dtype=torch.float32)
    if s0.d_hw is not None:
        result["d_hw"] = torch.tensor(s0.d_hw, dtype=torch.float32)
    if s0.hw_node_features is not None:
        result["hw_node_features"] = torch.tensor(
            s0.hw_node_features, dtype=torch.float32,
        )

    # Per-sample circuit metadata — use first sample (same num_logical in batch)
    result["circuit_edge_pairs"] = s0.circuit_edge_pairs
    result["cross_circuit_pairs"] = s0.cross_circuit_pairs
    if s0.qubit_importance is not None:
        result["qubit_importance"] = torch.tensor(
            s0.qubit_importance, dtype=torch.float32,
        )

    return result


def create_dataloader(
    dataset: MappingDataset,
    max_total_nodes: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader with backend-based bucketing.

    Args:
        dataset: MappingDataset instance.
        max_total_nodes: Max total physical qubits per batch.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        seed: Random seed.

    Returns:
        DataLoader that yields collated batch dicts.
    """
    sampler = BackendBucketSampler(
        dataset, max_total_nodes=max_total_nodes,
        shuffle=shuffle, seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_mapping_samples,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Split-based dataset loading
# ---------------------------------------------------------------------------

# Cache for hardware graphs and distance matrices (shared across samples)
_hw_graph_cache: dict[str, Data] = {}
_d_error_cache: dict[str, np.ndarray] = {}
_d_hw_cache: dict[str, np.ndarray] = {}
_hw_features_cache: dict[str, np.ndarray] = {}


def _get_hardware_graph(backend_name: str) -> Data:
    """Get hardware graph, using cache to avoid recomputation."""
    if backend_name not in _hw_graph_cache:
        if is_synthetic_backend(backend_name):
            _hw_graph_cache[backend_name] = build_hardware_graph_from_synthetic(backend_name)
        else:
            backend = get_backend(backend_name)
            _hw_graph_cache[backend_name] = build_hardware_graph(backend)
    return _hw_graph_cache[backend_name]


def _get_error_distance(backend_name: str) -> np.ndarray:
    """Get error distance matrix, using cache."""
    if backend_name not in _d_error_cache:
        if is_synthetic_backend(backend_name):
            _d_error_cache[backend_name] = precompute_error_distance_synthetic(backend_name)
        else:
            backend = get_backend(backend_name)
            _d_error_cache[backend_name] = precompute_error_distance(backend)
    return _d_error_cache[backend_name]


def _get_hop_distance(backend_name: str) -> np.ndarray:
    """Get hop distance matrix, using cache."""
    if backend_name not in _d_hw_cache:
        if is_synthetic_backend(backend_name):
            _d_hw_cache[backend_name] = precompute_hop_distance_synthetic(backend_name)
        else:
            backend = get_backend(backend_name)
            _d_hw_cache[backend_name] = precompute_hop_distance(backend)
    return _d_hw_cache[backend_name]


def _get_hw_node_features(backend_name: str) -> np.ndarray:
    """Get quality-score input features, using cache."""
    if backend_name not in _hw_features_cache:
        if is_synthetic_backend(backend_name):
            _hw_features_cache[backend_name] = get_hw_node_features_synthetic(backend_name)
        else:
            backend = get_backend(backend_name)
            _hw_features_cache[backend_name] = get_hw_node_features(backend)
    return _hw_features_cache[backend_name]


def load_split(
    split_path: str | Path,
    data_root: str | Path = "data/circuits",
    training_backends: list[str] | None = None,
    include_stage2_fields: bool = False,
) -> MappingDataset:
    """Build a MappingDataset from a splits JSON file.

    For supervised splits (stage1_supervised, val), each entry has
    source/file/backend and a matching label in labels/{source}/labels.json.

    For unsupervised splits (stage2_all), entries have source/file only.
    Each circuit is paired with a randomly assigned training backend.

    Args:
        split_path: Path to a splits/*.json file.
        data_root: Root of data/circuits/ directory.
        training_backends: List of backend names for unsupervised circuit assignment.
            Required if split contains entries without 'backend' field.
        include_stage2_fields: If True, populate d_error, qubit_importance, etc.

    Returns:
        MappingDataset with loaded samples.
    """
    data_root = Path(data_root)
    split_path = Path(split_path)

    with open(split_path) as f:
        entries = json.load(f)

    # Load label files
    labels_cache: dict[str, dict] = {}
    labels_dir = data_root / "labels"
    for label_dir in labels_dir.iterdir():
        if label_dir.is_dir():
            label_file = label_dir / "labels.json"
            if label_file.exists():
                with open(label_file) as f:
                    labels_cache[label_dir.name] = json.load(f)

    dataset = MappingDataset()
    rng = np.random.RandomState(42)
    loaded, skipped = 0, 0

    for entry in entries:
        source = entry["source"]
        filename = entry["file"]
        qasm_path = data_root / "qasm" / source / filename

        if not qasm_path.exists():
            skipped += 1
            continue

        # Determine backend
        backend_name = entry.get("backend")
        if backend_name is None:
            if training_backends is None:
                skipped += 1
                continue
            backend_name = rng.choice(training_backends)

        # Load circuit
        try:
            circuit = load_circuit(qasm_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", qasm_path, e)
            skipped += 1
            continue

        if circuit.num_qubits < 2:
            skipped += 1
            continue

        # Build graphs
        try:
            circuit_graph = build_circuit_graph(circuit)
            hw_graph = _get_hardware_graph(backend_name)
        except Exception as e:
            logger.warning("Failed to build graphs for %s on %s: %s", filename, backend_name, e)
            skipped += 1
            continue

        num_logical = circuit.num_qubits
        num_physical = hw_graph.num_qubits

        if num_logical > num_physical:
            skipped += 1
            continue

        # Label (if available)
        label_matrix = None
        layout = None
        source_labels = labels_cache.get(source, {})
        if filename in source_labels:
            label_info = source_labels[filename]
            if label_info["backend"] == backend_name:
                layout = label_info["layout"]
                label_matrix = layout_to_permutation_matrix(layout, num_physical)

        # Stage 2 fields
        d_error = None
        d_hw = None
        hw_feats = None
        qubit_importance = None
        circuit_edge_pairs = []
        if include_stage2_fields:
            d_error = _get_error_distance(backend_name)
            d_hw = _get_hop_distance(backend_name)
            hw_feats = _get_hw_node_features(backend_name)
            # Qubit importance: 2-qubit gate count per logical qubit
            from data.circuit_graph import extract_circuit_features
            feats = extract_circuit_features(circuit)
            qi = feats["node_features"][:, 1].numpy()  # two_qubit_gate_count
            qi_sum = qi.sum()
            qubit_importance = qi / qi_sum if qi_sum > 0 else np.ones(num_logical) / num_logical
            circuit_edge_pairs = feats["edge_list"]

        sample = MappingSample(
            circuit_graph=circuit_graph,
            hardware_graph=hw_graph,
            backend_name=backend_name,
            num_logical=num_logical,
            num_physical=num_physical,
            label_matrix=label_matrix,
            layout=layout,
            d_error=d_error,
            d_hw=d_hw,
            hw_node_features=hw_feats,
            circuit_edge_pairs=circuit_edge_pairs,
            qubit_importance=qubit_importance,
        )
        dataset.add_sample(sample)
        loaded += 1

    logger.info("Loaded %d samples from %s (%d skipped)", loaded, split_path.name, skipped)
    return dataset
