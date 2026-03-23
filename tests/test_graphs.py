"""Tests for graph construction modules."""

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit

from data.circuit_graph import build_circuit_graph, extract_circuit_features
from data.hardware_graph import (
    build_hardware_graph,
    extract_edge_properties,
    extract_qubit_properties,
    get_backend,
    precompute_error_distance,
)
from data.multi_programming import merge_circuits, validate_multi_programming
from data.normalization import zscore_normalize


# ---- Normalization ----

class TestZscoreNormalize:
    def test_basic(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normed = zscore_normalize(t, dim=0)
        # Mean should be ~0
        assert torch.allclose(normed.mean(dim=0), torch.zeros(2), atol=1e-6)

    def test_zero_std(self):
        t = torch.tensor([[3.0, 3.0], [3.0, 3.0]])
        normed = zscore_normalize(t, dim=0, eps=1e-8)
        # Should not produce NaN/Inf
        assert torch.isfinite(normed).all()

    def test_single_element(self):
        t = torch.tensor([[5.0, 10.0]])
        normed = zscore_normalize(t, dim=0)
        assert torch.isfinite(normed).all()


# ---- Hardware Graph ----

class TestHardwareGraph:
    @pytest.fixture
    def manila_backend(self):
        return get_backend("manila")

    def test_get_backend(self, manila_backend):
        assert manila_backend.target.num_qubits == 5

    def test_get_backend_invalid(self):
        with pytest.raises(ValueError):
            get_backend("nonexistent")

    def test_extract_qubit_properties(self, manila_backend):
        props = extract_qubit_properties(manila_backend)
        assert props["t1"].shape == (5,)
        assert props["t2"].shape == (5,)
        assert props["readout_error"].shape == (5,)
        assert props["single_qubit_error"].shape == (5,)
        assert props["degree"].shape == (5,)
        assert props["t1_cx_ratio"].shape == (5,)
        assert props["t2_cx_ratio"].shape == (5,)
        # T1, T2 should be positive; ratios non-negative
        assert (props["t1"] > 0).all()
        assert (props["t2"] > 0).all()
        assert (props["t1_cx_ratio"] >= 0).all()
        assert (props["t2_cx_ratio"] >= 0).all()

    def test_extract_edge_properties(self, manila_backend):
        edge_list, edge_feats = extract_edge_properties(manila_backend)
        # Manila has 4 undirected edges: 0-1, 1-2, 2-3, 3-4
        assert len(edge_list) == 4
        assert edge_feats.shape == (4, 1)  # cx_error only
        assert (edge_feats > 0).all()

    def test_build_hardware_graph_shape(self, manila_backend):
        data = build_hardware_graph(manila_backend)
        assert data.x.shape == (5, 7)  # 5 qubits, 7 features
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 8  # 4 undirected edges * 2 directions
        assert data.edge_attr.shape[0] == 8
        assert data.edge_attr.shape[1] == 1  # cx_error only
        assert data.num_qubits == 5

    def test_build_hardware_graph_normalized(self, manila_backend):
        data = build_hardware_graph(manila_backend)
        # Z-score normalized: mean ~0
        assert torch.allclose(data.x.mean(dim=0), torch.zeros(7), atol=0.1)

    def test_precompute_error_distance(self, manila_backend):
        d_error = precompute_error_distance(manila_backend)
        assert d_error.shape == (5, 5)
        # Diagonal should be 0
        np.testing.assert_array_almost_equal(np.diag(d_error), 0)
        # Symmetric
        np.testing.assert_array_almost_equal(d_error, d_error.T)
        # No infinities (connected graph)
        assert np.isfinite(d_error).all()
        # Triangle inequality
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    assert d_error[i, j] <= d_error[i, k] + d_error[k, j] + 1e-6


# ---- Circuit Graph ----

def _make_test_circuit() -> QuantumCircuit:
    """Create a small test circuit with 3 qubits."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.rz(0.5, 2)
    return qc


class TestCircuitGraph:
    @pytest.fixture
    def circuit(self):
        return _make_test_circuit()

    def test_extract_features_node_shape(self, circuit):
        feats = extract_circuit_features(circuit)
        assert feats["node_features"].shape == (3, 4)
        assert feats["num_qubits"] == 3

    def test_extract_features_edge_shape(self, circuit):
        feats = extract_circuit_features(circuit)
        # Edges: (0,1) and (1,2)
        assert len(feats["edge_list"]) == 2
        assert feats["edge_features"].shape == (2, 3)

    def test_extract_features_interaction_count(self, circuit):
        feats = extract_circuit_features(circuit)
        # (0,1) has 2 cx gates, (1,2) has 1 cx gate
        edge_dict = dict(zip(feats["edge_list"], feats["edge_features"].tolist()))
        assert edge_dict[(0, 1)][0] == 2  # interaction_count
        assert edge_dict[(1, 2)][0] == 1

    def test_build_circuit_graph_shape(self, circuit):
        data = build_circuit_graph(circuit)
        assert data.x.shape == (3, 4)  # 3 qubits, 4 features
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 4  # 2 edges * 2 directions
        assert data.edge_attr.shape == (4, 3)
        assert data.num_qubits == 3

    def test_build_circuit_graph_normalized(self, circuit):
        data = build_circuit_graph(circuit)
        # Z-score: mean should be ~0 along dim=0
        assert torch.allclose(data.x.mean(dim=0), torch.zeros(4), atol=0.1)

    def test_single_qubit_circuit(self):
        """Circuit with no 2-qubit gates should have no edges."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        data = build_circuit_graph(qc)
        assert data.x.shape == (2, 4)
        assert data.edge_index.shape[1] == 0
        assert data.edge_attr.shape[0] == 0


# ---- Multi-Programming ----

class TestMultiProgramming:
    def test_validate_valid(self):
        c1 = QuantumCircuit(3)
        c2 = QuantumCircuit(2)
        assert validate_multi_programming([c1, c2], num_physical_qubits=10) is True

    def test_validate_too_many_qubits(self):
        c1 = QuantumCircuit(3)
        c2 = QuantumCircuit(3)
        # total=6/6=100% > 75% occupancy limit
        assert validate_multi_programming([c1, c2], num_physical_qubits=6) is False

    def test_validate_occupancy_exceeded(self):
        c1 = QuantumCircuit(4)
        c2 = QuantumCircuit(4)
        # total=8/10=80% > 75%
        assert validate_multi_programming([c1, c2], num_physical_qubits=10) is False

    def test_merge_two_circuits(self):
        c1 = QuantumCircuit(2)
        c1.cx(0, 1)
        c2 = QuantumCircuit(3)
        c2.cx(0, 1)
        c2.cx(1, 2)

        merged = merge_circuits([c1, c2])
        assert merged.num_qubits == 5  # 2 + 3
        assert merged.x.shape[0] == 5
        assert merged.x.shape[1] == 4  # same 4-dim as single-circuit
        assert merged.circuit_sizes == [2, 3]
        assert merged.circuit_ids.shape == (5,)
        assert (merged.circuit_ids[:2] == 0).all()
        assert (merged.circuit_ids[2:] == 1).all()

    def test_merge_preserves_edges(self):
        c1 = QuantumCircuit(2)
        c1.cx(0, 1)
        c2 = QuantumCircuit(2)
        c2.cx(0, 1)

        merged = merge_circuits([c1, c2])
        # Each circuit has 1 undirected edge -> 2 directed edges each -> 4 total
        assert merged.edge_index.shape[1] == 4
        # No edges between circuits (disconnected)
        src = merged.edge_index[0]
        dst = merged.edge_index[1]
        for s, d in zip(src.tolist(), dst.tolist()):
            assert merged.circuit_ids[s] == merged.circuit_ids[d]

    def test_merge_single_circuit(self):
        """Merging a single circuit should produce the same result as single-circuit graph."""
        c1 = QuantumCircuit(3)
        c1.cx(0, 1)
        c1.cx(1, 2)

        merged = merge_circuits([c1])
        assert merged.num_qubits == 3
        assert merged.x.shape == (3, 4)
        assert merged.circuit_sizes == [3]
        assert (merged.circuit_ids == 0).all()
