"""Tests for evaluation components: PST, baselines, metrics."""

import pytest
from qiskit import QuantumCircuit

from data.hardware_graph import get_backend
from evaluation.baselines import (
    dense_layout,
    evaluate_baseline,
    naive_multi_programming_layout,
    random_layout,
    sabre_layout,
    trivial_layout,
)
from evaluation.metrics import EvalResult, aggregate_results, format_results_table
from evaluation.pst import measure_pst


# ---- PST ----

class TestPST:
    @pytest.fixture
    def bell_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        return qc

    @pytest.fixture
    def backend(self):
        return get_backend("manila")

    def test_measure_pst_returns_valid(self, bell_circuit, backend):
        result = measure_pst(bell_circuit, backend, layout=[0, 1], shots=1024)
        assert 0.0 <= result["pst"] <= 1.0
        assert result["swap_count"] >= 0
        assert result["depth"] > 0

    def test_measure_pst_good_layout_high_fidelity(self, bell_circuit, backend):
        """Bell state on adjacent qubits should have high PST."""
        result = measure_pst(bell_circuit, backend, layout=[0, 1], shots=4096)
        # Bell state on Manila qubits 0-1 (directly connected) should be decent
        assert result["pst"] > 0.4

    def test_measure_pst_with_measurements(self, backend):
        """Circuit that already has measurements."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        result = measure_pst(qc, backend, layout=[0, 1], shots=1024)
        assert 0.0 <= result["pst"] <= 1.0


# ---- Baselines ----

class TestBaselines:
    @pytest.fixture
    def circuit(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        return qc

    @pytest.fixture
    def backend(self):
        return get_backend("manila")

    def test_sabre_layout(self, circuit, backend):
        layout = sabre_layout(circuit, backend)
        assert len(layout) == 3
        assert len(set(layout)) == 3
        assert all(0 <= p < 5 for p in layout)

    def test_dense_layout(self, circuit, backend):
        layout = dense_layout(circuit, backend)
        assert len(layout) == 3
        assert len(set(layout)) == 3

    def test_trivial_layout(self, circuit, backend):
        layout = trivial_layout(circuit, backend)
        assert layout == [0, 1, 2]

    def test_random_layout(self, circuit, backend):
        layout = random_layout(circuit, backend, seed=42)
        assert len(layout) == 3
        assert len(set(layout)) == 3
        assert all(0 <= p < 5 for p in layout)

    def test_random_layout_different_seeds(self, circuit, backend):
        l1 = random_layout(circuit, backend, seed=0)
        l2 = random_layout(circuit, backend, seed=999)
        # Different seeds should (usually) give different layouts
        # Not guaranteed but very likely for 5 qubits
        # Just check they're valid
        assert len(l1) == 3
        assert len(l2) == 3

    def test_naive_multi_programming(self, backend):
        c1 = QuantumCircuit(2)
        c1.cx(0, 1)
        c2 = QuantumCircuit(2)
        c2.cx(0, 1)
        layout = naive_multi_programming_layout([c1, c2], backend)
        assert len(layout) == 4  # 2 + 2
        # All should be distinct (no conflicts)
        assert len(set(layout)) == 4

    def test_evaluate_baseline(self, circuit, backend):
        result = evaluate_baseline(circuit, backend, method="sabre", shots=1024)
        assert "pst" in result
        assert "method" in result
        assert result["method"] == "sabre"
        assert 0.0 <= result["pst"] <= 1.0


# ---- Metrics ----

class TestMetrics:
    def test_eval_result_stats(self):
        r = EvalResult(
            circuit_name="test", backend_name="manila", method="graphqmap",
            pst_values=[0.8, 0.85, 0.9],
            swap_counts=[2, 1, 2],
            depths=[10, 9, 11],
            inference_times=[0.001, 0.002, 0.001],
        )
        assert r.pst_mean == pytest.approx(0.85, abs=0.01)
        assert r.pst_std > 0
        assert r.swap_mean == pytest.approx(5 / 3, abs=0.1)
        assert r.depth_mean == pytest.approx(10.0, abs=0.1)

    def test_eval_result_summary(self):
        r = EvalResult(
            circuit_name="bell", backend_name="toronto", method="sabre",
            pst_values=[0.9], swap_counts=[0], depths=[5], inference_times=[0.01],
        )
        s = r.summary()
        assert s["circuit"] == "bell"
        assert s["backend"] == "toronto"
        assert s["method"] == "sabre"
        assert "0.9000" in s["pst"]

    def test_format_results_table(self):
        results = [
            EvalResult("c1", "manila", "graphqmap", [0.9], [1], [10], [0.001]),
            EvalResult("c1", "manila", "sabre", [0.85], [2], [12], [0.0]),
        ]
        table = format_results_table(results)
        assert "graphqmap" in table
        assert "sabre" in table
        assert "c1" in table

    def test_aggregate_results(self):
        results = [
            EvalResult("c1", "manila", "graphqmap", [0.9], [1], [10], [0.001]),
            EvalResult("c2", "manila", "graphqmap", [0.8], [2], [12], [0.002]),
        ]
        agg = aggregate_results(results)
        assert agg["num_circuits"] == 2
        assert agg["pst_mean"] == pytest.approx(0.85, abs=0.01)
        assert agg["pst_std"] > 0

    def test_empty_eval_result(self):
        r = EvalResult("empty", "manila", "test")
        assert r.pst_mean == 0.0
        assert r.pst_std == 0.0
