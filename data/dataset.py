"""PyG Dataset and DataLoader for GraphQMap.

Provides:
  - LazyMappingDataset: lazy-loading dataset from cached .pt files
  - MappingDataset: in-memory dataset (legacy, for small datasets/tests)
  - BackendBucketSampler: groups samples by backend for uniform h within batch
  - create_dataloader: factory with dynamic batch sizing
  - load_split: build a dataset from a splits/*.json file (lazy or eager)
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

from data.circuit_graph import (
    DEFAULT_NODE_FEATURES,
    build_circuit_graph,
    build_circuit_graph_from_raw,
    extract_circuit_features,
    load_circuit,
)
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
        d_hw: (h, h) hardware distance matrix (Stage 2).
        hw_node_features: (h, 7) quality features for QualityScore (Stage 2).
        circuit_edge_pairs: List of (i, j) logical qubit pairs with 2Q gates.
        circuit_edge_weights: Per-edge interaction count (parallel to circuit_edge_pairs).
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
        circuit_edge_weights: list[float] | None = None,
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
        self.circuit_edge_weights = circuit_edge_weights or []
        self.qubit_importance = qubit_importance


# ---------------------------------------------------------------------------
# Hardware graph cache (shared, lightweight — kept in memory)
# ---------------------------------------------------------------------------

_hw_graph_cache: dict[str, Data] = {}
_d_error_cache: dict[str, np.ndarray] = {}
_d_hw_cache: dict[str, np.ndarray] = {}
_hw_features_cache: dict[str, np.ndarray] = {}
_num_physical_cache: dict[str, int] = {}


def _get_hardware_graph(backend_name: str) -> Data:
    """Get hardware graph, using cache to avoid recomputation."""
    if backend_name not in _hw_graph_cache:
        if is_synthetic_backend(backend_name):
            _hw_graph_cache[backend_name] = build_hardware_graph_from_synthetic(backend_name)
        else:
            backend = get_backend(backend_name)
            _hw_graph_cache[backend_name] = build_hardware_graph(backend)
    return _hw_graph_cache[backend_name]


def _get_num_physical(backend_name: str) -> int:
    """Get number of physical qubits for a backend (cached)."""
    if backend_name not in _num_physical_cache:
        hw = _get_hardware_graph(backend_name)
        _num_physical_cache[backend_name] = hw.num_qubits
    return _num_physical_cache[backend_name]


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


# ---------------------------------------------------------------------------
# Lazy-loading dataset (reads cached .pt files on demand)
# ---------------------------------------------------------------------------

class _SampleMetadata:
    """Lightweight metadata for a single sample (kept in memory for grouping)."""
    __slots__ = ("cache_path", "backend_name", "num_logical", "num_physical",
                 "label_matrix", "layout", "include_stage2_fields")

    def __init__(
        self,
        cache_path: Path,
        backend_name: str,
        num_logical: int,
        num_physical: int,
        label_matrix: np.ndarray | None,
        layout: list[int] | None,
        include_stage2_fields: bool,
    ) -> None:
        self.cache_path = cache_path
        self.backend_name = backend_name
        self.num_logical = num_logical
        self.num_physical = num_physical
        self.label_matrix = label_matrix
        self.layout = layout
        self.include_stage2_fields = include_stage2_fields


class LazyMappingDataset(Dataset):
    """Lazy-loading dataset that reads cached .pt circuit files on demand.

    Only lightweight metadata (backend, num_logical, num_physical, label, cache_path)
    is kept in memory. Circuit graphs are loaded from disk in __getitem__.
    Hardware graphs and distance matrices are cached in memory (shared, small).

    Feature selection (node_feature_names, rwpe_k) is applied at load time,
    allowing experiments with different feature combinations without re-caching.
    """

    def __init__(
        self,
        node_feature_names: list[str] | None = None,
        rwpe_k: int = 0,
        edge_dim: int | None = None,
    ) -> None:
        self._entries: list[_SampleMetadata] = []
        self._backend_indices: dict[str, list[int]] = {}
        self.node_feature_names = node_feature_names or DEFAULT_NODE_FEATURES
        self.rwpe_k = rwpe_k
        self.edge_dim = edge_dim

    def add_entry(self, meta: _SampleMetadata) -> None:
        """Add a sample metadata entry."""
        idx = len(self._entries)
        self._entries.append(meta)
        backend = meta.backend_name
        if backend not in self._backend_indices:
            self._backend_indices[backend] = []
        self._backend_indices[backend].append(idx)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> MappingSample:
        meta = self._entries[idx]

        # Load circuit data from cached .pt file
        cache_data = torch.load(meta.cache_path, weights_only=False)

        # Build circuit graph from raw features (new format) or use pre-built (old format)
        if "node_features_dict" in cache_data:
            circuit_graph = build_circuit_graph_from_raw(
                node_features_dict=cache_data["node_features_dict"],
                edge_list=cache_data["edge_list"],
                edge_features=cache_data["edge_features"],
                num_qubits=cache_data["num_logical"],
                node_feature_names=self.node_feature_names,
                rwpe_k=self.rwpe_k,
                edge_dim=self.edge_dim,
            )
        else:
            # Legacy cache format: pre-built PyG Data
            circuit_graph = cache_data["circuit_graph"]

        # Hardware graph from memory cache
        hw_graph = _get_hardware_graph(meta.backend_name)

        # Stage 2 fields
        d_error = None
        d_hw = None
        hw_feats = None
        circuit_edge_pairs = cache_data.get("circuit_edge_pairs", [])
        circuit_edge_weights = cache_data.get("circuit_edge_weights", [])
        qubit_importance = cache_data.get("qubit_importance")

        if meta.include_stage2_fields:
            d_error = _get_error_distance(meta.backend_name)
            d_hw = _get_hop_distance(meta.backend_name)
            hw_feats = _get_hw_node_features(meta.backend_name)

        return MappingSample(
            circuit_graph=circuit_graph,
            hardware_graph=hw_graph,
            backend_name=meta.backend_name,
            num_logical=meta.num_logical,
            num_physical=meta.num_physical,
            label_matrix=meta.label_matrix,
            layout=meta.layout,
            d_error=d_error,
            d_hw=d_hw,
            hw_node_features=hw_feats,
            circuit_edge_pairs=circuit_edge_pairs,
            circuit_edge_weights=circuit_edge_weights,
            qubit_importance=qubit_importance,
        )

    @property
    def backend_names(self) -> list[str]:
        """List of unique backend names in the dataset."""
        return list(self._backend_indices.keys())

    def indices_for_backend(self, backend_name: str) -> list[int]:
        """Get sample indices for a specific backend."""
        return self._backend_indices.get(backend_name, [])


# ---------------------------------------------------------------------------
# In-memory dataset (legacy, for small datasets and tests)
# ---------------------------------------------------------------------------

class MappingDataset(Dataset):
    """In-memory dataset of qubit mapping samples.

    Stores pre-built samples in memory. Use for small datasets or testing.
    For large datasets, use LazyMappingDataset via load_split(lazy=True).
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


# ---------------------------------------------------------------------------
# Sampler and collation (shared by both dataset types)
# ---------------------------------------------------------------------------

class BackendBucketSampler(Sampler):
    """Sampler that groups samples by (backend, num_logical) for uniform tensor shapes.

    Works with both MappingDataset and LazyMappingDataset.
    For LazyMappingDataset, uses stored metadata instead of loading samples.
    """

    def __init__(
        self,
        dataset: MappingDataset | LazyMappingDataset,
        max_total_nodes: int = 512,
        shuffle: bool = True,
        seed: int = 42,
        large_backend_boost: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.max_total_nodes = max_total_nodes
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.large_backend_boost = large_backend_boost

    def _build_groups(self) -> dict[tuple[str, int], list[int]]:
        """Group sample indices by (backend_name, num_logical)."""
        groups: dict[tuple[str, int], list[int]] = {}

        if isinstance(self.dataset, LazyMappingDataset):
            for idx, meta in enumerate(self.dataset._entries):
                key = (meta.backend_name, meta.num_logical)
                if key not in groups:
                    groups[key] = []
                groups[key].append(idx)
        else:
            for idx in range(len(self.dataset)):
                sample = self.dataset[idx]
                key = (sample.backend_name, sample.num_logical)
                if key not in groups:
                    groups[key] = []
                groups[key].append(idx)
        return groups

    def _get_num_physical(self, idx: int) -> int:
        """Get num_physical for a sample without loading full data."""
        if isinstance(self.dataset, LazyMappingDataset):
            return self.dataset._entries[idx].num_physical
        return self.dataset[idx].num_physical

    def __iter__(self):
        groups = self._build_groups()
        if not groups:
            return iter([])

        all_batches = []

        for (backend, num_logical), indices in groups.items():
            indices = list(indices)
            if self.shuffle:
                self.rng.shuffle(indices)

            num_physical = self._get_num_physical(indices[0])
            batch_size = max(1, self.max_total_nodes // num_physical)

            # Oversample large backends (50Q+) to improve generalization
            if self.large_backend_boost > 1.0 and num_physical >= 50:
                repeat = int(self.large_backend_boost)
                remainder = self.large_backend_boost - repeat
                expanded = indices * repeat
                extra = int(len(indices) * remainder)
                if extra > 0:
                    expanded += list(self.rng.choice(indices, size=extra, replace=True))
                indices = expanded
                if self.shuffle:
                    self.rng.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                batch = indices[i : i + batch_size]
                all_batches.append(batch)

        if self.shuffle:
            self.rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        groups = self._build_groups()
        total = 0
        for (backend, num_logical), indices in groups.items():
            num_physical = self._get_num_physical(indices[0])
            batch_size = max(1, self.max_total_nodes // num_physical)
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
        - hw_node_features: (h, 7) tensor
        - circuit_edge_pairs: list of (i, j) tuples
        - circuit_edge_weights: list of interaction counts per edge
        - qubit_importance: (l,) tensor
    """
    circuit_graphs = [s.circuit_graph for s in samples]
    hardware_graphs = [s.hardware_graph for s in samples]

    circuit_batch = Batch.from_data_list(circuit_graphs)
    hardware_batch = Batch.from_data_list(hardware_graphs)

    num_physical = samples[0].num_physical
    num_logical_list = [s.num_logical for s in samples]

    # Label matrices (Stage 1) — only stack if ALL samples have labels with consistent shapes
    if all(s.label_matrix is not None for s in samples):
        shapes = {s.label_matrix.shape for s in samples}
        if len(shapes) == 1:
            label_matrices = torch.tensor(
                np.stack([s.label_matrix for s in samples]),
                dtype=torch.float32,
            )
        else:
            # Shape mismatch (e.g. label qubit count != circuit qubit count)
            label_matrices = None
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

    # Per-sample circuit metadata
    result["circuit_edge_pairs"] = s0.circuit_edge_pairs
    result["circuit_edge_weights"] = s0.circuit_edge_weights
    if s0.qubit_importance is not None:
        result["qubit_importance"] = torch.tensor(
            s0.qubit_importance, dtype=torch.float32,
        )

    return result


def create_dataloader(
    dataset: MappingDataset | LazyMappingDataset,
    max_total_nodes: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader with backend-based bucketing.

    Works with both MappingDataset and LazyMappingDataset.
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

def load_split(
    split_path: str | Path,
    data_root: str | Path = "data/circuits",
    training_backends: list[str] | None = None,
    include_stage2_fields: bool = False,
    lazy: bool = True,
    node_feature_names: list[str] | None = None,
    rwpe_k: int = 0,
    edge_dim: int | None = None,
) -> MappingDataset | LazyMappingDataset:
    """Build a dataset from a splits JSON file.

    When lazy=True (default), returns a LazyMappingDataset that reads cached
    .pt files on demand. Requires running scripts/preprocess_circuits.py first.

    When lazy=False, loads all circuits into memory (original behavior).

    Args:
        split_path: Path to a splits/*.json file.
        data_root: Root of data/circuits/ directory.
        training_backends: List of backend names for unsupervised circuit assignment.
        include_stage2_fields: If True, populate d_error, qubit_importance, etc.
        lazy: If True, use lazy loading from cached .pt files.
        node_feature_names: Circuit node features to use (None = defaults).
        rwpe_k: Number of RWPE steps (0 = disabled).
        edge_dim: Number of circuit edge feature dimensions (None = use all).

    Returns:
        MappingDataset or LazyMappingDataset.
    """
    if lazy:
        return _load_split_lazy(
            split_path, data_root, training_backends, include_stage2_fields,
            node_feature_names=node_feature_names, rwpe_k=rwpe_k,
            edge_dim=edge_dim,
        )
    return _load_split_eager(
        split_path, data_root, training_backends, include_stage2_fields,
        node_feature_names=node_feature_names, rwpe_k=rwpe_k,
    )


def _load_split_lazy(
    split_path: str | Path,
    data_root: str | Path,
    training_backends: list[str] | None,
    include_stage2_fields: bool,
    node_feature_names: list[str] | None = None,
    rwpe_k: int = 0,
    edge_dim: int | None = None,
) -> LazyMappingDataset:
    """Load split with lazy loading from cached .pt files.

    Uses metadata.json index to avoid reading .pt files during initialization.
    Only lightweight metadata is kept in memory.
    """
    data_root = Path(data_root)
    split_path = Path(split_path)
    cache_root = data_root / "cache"

    with open(split_path) as f:
        entries = json.load(f)

    # Load metadata index (source/filename.qasm -> num_logical)
    meta_index_path = cache_root / "metadata.json"
    if not meta_index_path.exists():
        raise FileNotFoundError(
            f"Cache metadata not found at {meta_index_path}. "
            "Run 'python scripts/preprocess_circuits.py' first."
        )
    with open(meta_index_path) as f:
        meta_index = json.load(f)

    # Load label files
    labels_cache: dict[str, dict] = {}
    labels_dir = data_root / "labels"
    if labels_dir.exists():
        for label_dir in labels_dir.iterdir():
            if label_dir.is_dir():
                label_file = label_dir / "labels.json"
                if label_file.exists():
                    with open(label_file) as f:
                        labels_cache[label_dir.name] = json.load(f)

    dataset = LazyMappingDataset(
        node_feature_names=node_feature_names,
        rwpe_k=rwpe_k,
        edge_dim=edge_dim,
    )
    rng = np.random.RandomState(42)
    loaded, skipped = 0, 0

    for entry in entries:
        source = entry["source"]
        filename = entry["file"]
        meta_key = f"{source}/{filename}"

        # Check metadata index (fast, no disk I/O per file)
        if meta_key not in meta_index:
            skipped += 1
            continue

        num_logical = meta_index[meta_key]
        stem = Path(filename).stem
        cache_path = cache_root / source / (stem + ".pt")

        # Determine backend
        backend_name = entry.get("backend")
        if backend_name is None:
            if training_backends is None:
                skipped += 1
                continue
            # Filter to backends large enough for this circuit
            valid_backends = [b for b in training_backends if _get_num_physical(b) >= num_logical]
            if not valid_backends:
                skipped += 1
                continue
            backend_name = rng.choice(valid_backends)

        num_physical = _get_num_physical(backend_name)

        # Label (if available)
        label_matrix = None
        layout = None
        source_labels = labels_cache.get(source, {})
        if filename in source_labels:
            label_info = source_labels[filename]
            if label_info["backend"] == backend_name:
                layout = label_info["layout"]
                label_matrix = layout_to_permutation_matrix(layout, num_physical)

        meta = _SampleMetadata(
            cache_path=cache_path,
            backend_name=backend_name,
            num_logical=num_logical,
            num_physical=num_physical,
            label_matrix=label_matrix,
            layout=layout,
            include_stage2_fields=include_stage2_fields,
        )
        dataset.add_entry(meta)
        loaded += 1

    logger.info("Loaded %d entries from %s (%d skipped) [lazy]", loaded, split_path.name, skipped)
    return dataset


def _load_split_eager(
    split_path: str | Path,
    data_root: str | Path,
    training_backends: list[str] | None,
    include_stage2_fields: bool,
    node_feature_names: list[str] | None = None,
    rwpe_k: int = 0,
) -> MappingDataset:
    """Load split eagerly into memory (original behavior)."""
    data_root = Path(data_root)
    split_path = Path(split_path)

    with open(split_path) as f:
        entries = json.load(f)

    # Load label files
    labels_cache: dict[str, dict] = {}
    labels_dir = data_root / "labels"
    if labels_dir.exists():
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

        backend_name = entry.get("backend")
        if backend_name is None:
            if training_backends is None:
                skipped += 1
                continue
            backend_name = rng.choice(training_backends)

        try:
            circuit = load_circuit(qasm_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", qasm_path, e)
            skipped += 1
            continue

        if circuit.num_qubits < 2:
            skipped += 1
            continue

        try:
            circuit_graph = build_circuit_graph(
                circuit,
                node_feature_names=node_feature_names,
                rwpe_k=rwpe_k,
            )
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

        label_matrix = None
        layout = None
        source_labels = labels_cache.get(source, {})
        if filename in source_labels:
            label_info = source_labels[filename]
            if label_info["backend"] == backend_name:
                layout = label_info["layout"]
                label_matrix = layout_to_permutation_matrix(layout, num_physical)

        d_error = None
        d_hw = None
        hw_feats = None
        qubit_importance = None
        circuit_edge_pairs = []
        circuit_edge_weights = []
        if include_stage2_fields:
            d_error = _get_error_distance(backend_name)
            d_hw = _get_hop_distance(backend_name)
            hw_feats = _get_hw_node_features(backend_name)
            feats = extract_circuit_features(circuit)
            qi = np.array(feats["node_features_dict"]["two_qubit_gate_count"])
            qi_sum = qi.sum()
            qubit_importance = qi / qi_sum if qi_sum > 0 else np.ones(num_logical) / num_logical
            circuit_edge_pairs = feats["edge_list"]
            circuit_edge_weights = feats["edge_features"][:, 0].tolist()

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
            circuit_edge_weights=circuit_edge_weights,
            qubit_importance=qubit_importance,
        )
        dataset.add_sample(sample)
        loaded += 1

    logger.info("Loaded %d samples from %s (%d skipped)", loaded, split_path.name, skipped)
    return dataset
