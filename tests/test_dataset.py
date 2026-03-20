"""Tests for Phase 4: label generation, dataset, dataloader, multi-prog sampler."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit

from data.circuit_graph import build_circuit_graph
from data.dataset import (
    BackendBucketSampler,
    MappingDataset,
    MappingSample,
    collate_mapping_samples,
    create_dataloader,
)
from data.hardware_graph import build_hardware_graph, get_backend
from data.label_generation import (
    count_additional_2q_gates,
    evaluate_layout,
    generate_candidate_layouts,
    generate_label,
    layout_to_permutation_matrix,
)
from data.multi_programming_sampler import sample_multi_programming_groups
from data.queko_loader import load_queko_layout


# ---- Label Generation ----

class TestLabelGeneration:
    @pytest.fixture
    def circuit_and_backend(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)
        backend = get_backend("manila")
        return qc, backend

    def test_count_additional_2q_gates_no_routing(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        # Layout [0,1] on Manila: direct connection, no SWAPs needed
        from qiskit import transpile
        backend = get_backend("manila")
        compiled = transpile(qc, backend=backend, initial_layout=[0, 1],
                            routing_method="sabre", optimization_level=1,
                            seed_transpiler=0)
        added = count_additional_2q_gates(qc, compiled, backend=backend)
        assert added == 0

    def test_generate_candidates_count(self, circuit_and_backend):
        qc, backend = circuit_and_backend
        candidates = generate_candidate_layouts(
            qc, backend, num_sabre_seeds=10, num_random=8, rng_seed=42,
        )
        # Should have ~20 candidates (10 sabre + 1 dense + 1 trivial + 8 random)
        assert len(candidates) >= 18  # allow some failures
        assert len(candidates) <= 20

    def test_generate_candidates_valid_layouts(self, circuit_and_backend):
        qc, backend = circuit_and_backend
        candidates = generate_candidate_layouts(
            qc, backend, num_sabre_seeds=2, num_random=2, rng_seed=42,
        )
        num_physical = backend.target.num_qubits
        for name, layout in candidates:
            assert len(layout) == qc.num_qubits
            assert all(0 <= p < num_physical for p in layout)
            assert len(set(layout)) == len(layout)  # no duplicates

    def test_evaluate_layout(self, circuit_and_backend):
        qc, backend = circuit_and_backend
        result = evaluate_layout(qc, backend, layout=[0, 1, 2])
        assert "swap_count" in result
        assert "depth" in result
        assert result["swap_count"] >= 0
        assert result["depth"] > 0

    def test_generate_label(self, circuit_and_backend):
        qc, backend = circuit_and_backend
        result = generate_label(
            qc, backend, num_sabre_seeds=3, num_random=2, rng_seed=42,
        )
        assert len(result.layout) == qc.num_qubits
        assert result.swap_count >= 0
        assert result.depth > 0
        assert result.num_candidates >= 4
        # Best should have minimum swap count
        min_swaps = min(r["swap_count"] for r in result.all_results)
        assert result.swap_count == min_swaps

    def test_layout_to_permutation_matrix(self):
        layout = [2, 0, 3]  # 3 logical -> 5 physical
        Y = layout_to_permutation_matrix(layout, num_physical=5)
        assert Y.shape == (5, 5)
        # Check logical qubit assignments
        assert Y[0, 2] == 1.0
        assert Y[1, 0] == 1.0
        assert Y[2, 3] == 1.0
        # Should be a valid permutation matrix
        assert np.allclose(Y.sum(axis=0), 1.0)
        assert np.allclose(Y.sum(axis=1), 1.0)

    def test_permutation_matrix_identity(self):
        layout = [0, 1, 2]
        Y = layout_to_permutation_matrix(layout, num_physical=3)
        np.testing.assert_array_equal(Y, np.eye(3))


# ---- QUEKO Loader ----

class TestQuekoLoader:
    def test_load_queko_layout(self, tmp_path):
        layout_file = tmp_path / "test.layout"
        layout_file.write_text("3\n1\n4\n")
        layout = load_queko_layout(layout_file)
        assert layout == [3, 1, 4]

    def test_load_queko_layout_with_comments(self, tmp_path):
        layout_file = tmp_path / "test.layout"
        layout_file.write_text("# optimal layout\n2\n0\n")
        layout = load_queko_layout(layout_file)
        assert layout == [2, 0]

    def test_load_queko_layout_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_queko_layout("/nonexistent/path.layout")


# ---- Dataset / DataLoader ----

def _make_sample(backend_name: str, num_logical: int, num_physical: int,
                 with_label: bool = True) -> MappingSample:
    """Create a dummy MappingSample for testing."""
    circuit_graph = torch.zeros(0)  # placeholder
    # Build minimal PyG data
    from torch_geometric.data import Data
    circuit_data = Data(
        x=torch.randn(num_logical, 8),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 3)),
    )
    hardware_data = Data(
        x=torch.randn(num_physical, 6),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 2)),
    )

    label = None
    layout = None
    if with_label:
        layout = list(range(num_logical))
        label = layout_to_permutation_matrix(layout, num_physical)

    return MappingSample(
        circuit_graph=circuit_data,
        hardware_graph=hardware_data,
        backend_name=backend_name,
        num_logical=num_logical,
        num_physical=num_physical,
        label_matrix=label,
        layout=layout,
    )


class TestMappingDataset:
    def test_add_and_index(self):
        ds = MappingDataset()
        s1 = _make_sample("manila", 3, 5)
        s2 = _make_sample("manila", 2, 5)
        ds.add_sample(s1)
        ds.add_sample(s2)
        assert len(ds) == 2
        assert ds[0].num_logical == 3
        assert ds[1].num_logical == 2

    def test_backend_indices(self):
        ds = MappingDataset()
        ds.add_sample(_make_sample("manila", 3, 5))
        ds.add_sample(_make_sample("guadalupe", 4, 16))
        ds.add_sample(_make_sample("manila", 2, 5))
        assert ds.indices_for_backend("manila") == [0, 2]
        assert ds.indices_for_backend("guadalupe") == [1]
        assert set(ds.backend_names) == {"manila", "guadalupe"}


class TestCollate:
    def test_collate_samples(self):
        samples = [
            _make_sample("manila", 3, 5),
            _make_sample("manila", 3, 5),
        ]
        batch = collate_mapping_samples(samples)
        assert batch["batch_size"] == 2
        assert batch["num_physical"] == 5
        assert batch["num_logical"] == [3, 3]
        assert batch["backend_name"] == "manila"
        assert batch["label_matrices"].shape == (2, 5, 5)

    def test_collate_without_labels(self):
        samples = [
            _make_sample("manila", 3, 5, with_label=False),
            _make_sample("manila", 2, 5, with_label=False),
        ]
        batch = collate_mapping_samples(samples)
        assert batch["label_matrices"] is None


class TestBackendBucketSampler:
    def test_batches_same_backend(self):
        ds = MappingDataset()
        for _ in range(10):
            ds.add_sample(_make_sample("manila", 3, 5))
        for _ in range(5):
            ds.add_sample(_make_sample("guadalupe", 4, 16))

        sampler = BackendBucketSampler(ds, max_total_nodes=512, shuffle=False)
        for batch_indices in sampler:
            backends = [ds[i].backend_name for i in batch_indices]
            # All samples in batch should be same backend
            assert len(set(backends)) == 1

    def test_dynamic_batch_size(self):
        ds = MappingDataset()
        # Manila: 5Q → batch_size = 512//5 = 102
        for _ in range(200):
            ds.add_sample(_make_sample("manila", 3, 5))

        sampler = BackendBucketSampler(ds, max_total_nodes=512, shuffle=False)
        batches = list(sampler)
        # Each batch should have at most 102 samples
        for batch in batches:
            assert len(batch) <= 102


class TestDataLoader:
    def test_create_and_iterate(self):
        ds = MappingDataset()
        for _ in range(5):
            ds.add_sample(_make_sample("manila", 3, 5))

        loader = create_dataloader(ds, max_total_nodes=512, shuffle=False)
        batches = list(loader)
        assert len(batches) >= 1
        for batch in batches:
            assert "circuit_batch" in batch
            assert "hardware_batch" in batch
            assert "label_matrices" in batch


# ---- Multi-Programming Sampler ----

class TestMultiProgrammingSampler:
    @pytest.fixture
    def circuits(self):
        pool = []
        for n in [2, 3, 4, 5]:
            for _ in range(10):
                qc = QuantumCircuit(n)
                if n >= 2:
                    qc.cx(0, 1)
                pool.append(qc)
        return pool

    def test_basic_sampling(self, circuits):
        groups = sample_multi_programming_groups(
            circuits, num_physical=16, num_samples=20, rng_seed=42,
        )
        assert len(groups) == 20

    def test_ratios(self, circuits):
        groups = sample_multi_programming_groups(
            circuits, num_physical=16, num_samples=100, rng_seed=42,
        )
        singles = sum(1 for g in groups if len(g) == 1)
        duals = sum(1 for g in groups if len(g) == 2)
        quads = sum(1 for g in groups if len(g) == 4)
        assert singles == 50
        assert duals == 30
        assert quads + singles + duals == 100  # rest are quads

    def test_occupancy_constraints(self, circuits):
        groups = sample_multi_programming_groups(
            circuits, num_physical=16,
            occupancy_min=0.3, occupancy_max=0.75,
            num_samples=50, rng_seed=42,
        )
        for group in groups:
            total_qubits = sum(circuits[i].num_qubits for i in group)
            assert total_qubits < 16  # strict less than
            assert total_qubits <= int(16 * 0.75)  # occupancy max

    def test_small_backend(self, circuits):
        """With a small backend, quad groups might not be possible."""
        groups = sample_multi_programming_groups(
            circuits, num_physical=5,
            occupancy_min=0.3, occupancy_max=0.75,
            num_samples=10, rng_seed=42,
        )
        # Should still produce some groups (at least singles)
        assert len(groups) > 0
