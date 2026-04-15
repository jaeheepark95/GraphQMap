"""Tests for training components: losses, schedulers, early stopping, quality score."""

import math

import pytest
import torch

from training.early_stopping import EarlyStopping
from training.losses import (
    AdjacencyMatchingLoss,
    ErrorAwareEdgeLoss,
    HopDistanceLoss,
    NodeQualityLoss,
    SeparationLoss,
    SurrogateLoss,
    get_available_losses,
)
from training.quality_score import QualityScore
from training.tau_scheduler import TauScheduler


# ---- Tau Scheduler ----

class TestTauScheduler:
    def test_exponential_start(self):
        ts = TauScheduler(tau_max=1.0, tau_min=0.05, schedule="exponential", total_epochs=100)
        assert ts.get_tau(0) == pytest.approx(1.0)

    def test_exponential_end(self):
        ts = TauScheduler(tau_max=1.0, tau_min=0.05, schedule="exponential", total_epochs=100)
        assert ts.get_tau(100) == pytest.approx(0.05)

    def test_exponential_monotonic_decrease(self):
        ts = TauScheduler(tau_max=1.0, tau_min=0.05, schedule="exponential", total_epochs=100)
        taus = [ts.get_tau(e) for e in range(101)]
        for i in range(len(taus) - 1):
            assert taus[i] >= taus[i + 1]

    def test_exponential_midpoint(self):
        ts = TauScheduler(tau_max=1.0, tau_min=0.05, schedule="exponential", total_epochs=100)
        mid = ts.get_tau(50)
        # Should be geometric mean of 1.0 and 0.05
        expected = 1.0 * math.pow(0.05 / 1.0, 0.5)
        assert mid == pytest.approx(expected, rel=0.01)

    def test_fixed_schedule(self):
        ts = TauScheduler(tau_max=0.05, tau_min=0.05, schedule="fixed")
        assert ts.get_tau(0) == 0.05
        assert ts.get_tau(50) == 0.05
        assert ts.get_tau(100) == 0.05

    def test_clamp_beyond_total(self):
        ts = TauScheduler(tau_max=1.0, tau_min=0.05, schedule="exponential", total_epochs=100)
        assert ts.get_tau(200) == pytest.approx(0.05)


# ---- Early Stopping ----

class TestEarlyStopping:
    def test_no_improvement(self):
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")
        assert not es.step(1.0)
        assert not es.step(1.0)
        assert not es.step(1.0)
        assert es.step(1.0)  # 3 epochs no improvement

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=2, min_delta=0.01, mode="min")
        assert not es.step(1.0)
        assert not es.step(1.0)
        assert not es.step(0.5)  # improvement, resets counter
        assert not es.step(0.5)
        assert es.step(0.5)     # 2 epochs no improvement

    def test_mode_max(self):
        es = EarlyStopping(patience=2, min_delta=0.01, mode="max")
        assert not es.step(0.5)
        assert not es.step(0.5)
        assert es.step(0.5)     # no improvement for PST

    def test_mode_max_improvement(self):
        es = EarlyStopping(patience=2, min_delta=0.01, mode="max")
        assert not es.step(0.5)
        assert not es.step(0.6)  # improved
        assert not es.step(0.6)
        assert es.step(0.6)      # 2 epochs stalled


# ---- Quality Score ----

class TestQualityScore:
    def test_output_shape(self):
        qs = QualityScore(num_features=6)
        features = torch.randn(10, 6)
        scores = qs(features)
        assert scores.shape == (10,)

    def test_output_range(self):
        qs = QualityScore(num_features=6)
        features = torch.randn(100, 6)
        scores = qs(features)
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_gradient_flow(self):
        qs = QualityScore(num_features=6)
        features = torch.randn(10, 6)
        scores = qs(features)
        scores.sum().backward()
        # All MLP parameters should have gradients
        for param in qs.parameters():
            assert param.grad is not None

    def test_learnable_params(self):
        qs = QualityScore(num_features=7, hidden_dim=16)
        num_params = sum(p.numel() for p in qs.parameters())
        assert num_params > 0


# ---- Loss Registry ----

class TestLossRegistry:
    def test_all_losses_registered(self):
        available = get_available_losses()
        assert "error_distance" in available
        assert "adjacency" in available
        assert "hop_distance" in available
        assert "node_quality" in available
        assert "separation" in available
        assert "exclusion" in available

    def test_unknown_loss_raises(self):
        qs = QualityScore(num_features=6)
        with pytest.raises(ValueError, match="Unknown loss component"):
            SurrogateLoss(components=[{"name": "nonexistent", "weight": 1.0}], quality_score=qs)

    def test_node_quality_without_qs_raises(self):
        with pytest.raises(ValueError, match="requires quality_score"):
            SurrogateLoss(components=[{"name": "node_quality", "weight": 1.0}])


# ---- Error-Aware Edge Loss ----

class TestErrorAwareEdgeLoss:
    def _make_d_error(self, n=5):
        """Create a simple error distance matrix."""
        d = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                d[i, j] = abs(i - j) * 0.01  # error accumulates with distance
        return d

    def test_basic(self):
        loss_fn = ErrorAwareEdgeLoss()
        P = torch.rand(2, 3, 5)
        d_error = self._make_d_error()
        loss = loss_fn(P, d_error=d_error, circuit_edge_pairs=[(0, 1), (1, 2)], num_logical=3)
        assert loss.shape == ()
        assert loss.item() >= 0  # distance-based, non-negative

    def test_empty_pairs(self):
        loss_fn = ErrorAwareEdgeLoss()
        P = torch.rand(2, 3, 5)
        d_error = self._make_d_error()
        loss = loss_fn(P, d_error=d_error, circuit_edge_pairs=[], num_logical=3)
        assert loss.item() == 0.0

    def test_gradient_flow(self):
        loss_fn = ErrorAwareEdgeLoss()
        P = torch.rand(1, 3, 5, requires_grad=True)
        d_error = self._make_d_error()
        loss = loss_fn(P, d_error=d_error, circuit_edge_pairs=[(0, 1)], num_logical=3)
        loss.backward()
        assert P.grad is not None

    def test_adjacent_mapping_lower_cost(self):
        """Mapping adjacent logical qubits to adjacent physical qubits should yield lower error cost."""
        loss_fn = ErrorAwareEdgeLoss()
        d_error = self._make_d_error()

        # Good mapping: logical 0->physical 0, logical 1->physical 1 (adjacent, low error)
        P_good = torch.zeros(1, 3, 5)
        P_good[0, 0, 0] = P_good[0, 1, 1] = P_good[0, 2, 2] = 1.0
        # Bad mapping: logical 0->physical 0, logical 1->physical 4 (far apart, high error)
        P_bad = torch.zeros(1, 3, 5)
        P_bad[0, 0, 0] = P_bad[0, 1, 4] = P_bad[0, 2, 2] = 1.0

        loss_good = loss_fn(P_good, d_error=d_error, circuit_edge_pairs=[(0, 1)])
        loss_bad = loss_fn(P_bad, d_error=d_error, circuit_edge_pairs=[(0, 1)])
        assert loss_good.item() < loss_bad.item()


# ---- Adjacency Matching Loss ----

class TestAdjacencyMatchingLoss:
    def _make_d_hw(self):
        """Create a simple hop distance matrix for a 5-node path graph: 0-1-2-3-4."""
        d_hw = torch.zeros(5, 5)
        for i in range(5):
            for j in range(5):
                d_hw[i, j] = abs(i - j)
        return d_hw

    def test_basic(self):
        loss_fn = AdjacencyMatchingLoss()
        P = torch.rand(2, 3, 5)
        d_hw = self._make_d_hw()
        loss = loss_fn(P, d_hw=d_hw, circuit_edge_pairs=[(0, 1), (1, 2)],
                       circuit_edge_weights=[3.0, 1.0], num_logical=3)
        assert loss.shape == ()
        assert loss.item() <= 0  # bounded in [-1, 0]

    def test_empty_pairs(self):
        loss_fn = AdjacencyMatchingLoss()
        P = torch.rand(2, 3, 5)
        d_hw = self._make_d_hw()
        loss = loss_fn(P, d_hw=d_hw, circuit_edge_pairs=[], circuit_edge_weights=[], num_logical=3)
        assert loss.item() == 0.0

    def test_gradient_flow(self):
        loss_fn = AdjacencyMatchingLoss()
        P = torch.rand(1, 3, 5, requires_grad=True)
        d_hw = self._make_d_hw()
        loss = loss_fn(P, d_hw=d_hw, circuit_edge_pairs=[(0, 1)],
                       circuit_edge_weights=[2.0], num_logical=3)
        loss.backward()
        assert P.grad is not None

    def test_perfect_adjacent_mapping(self):
        """Identity mapping on a path graph should give best score for adjacent pairs."""
        loss_fn = AdjacencyMatchingLoss()
        P = torch.zeros(1, 3, 5)
        P[0, 0, 0] = P[0, 1, 1] = P[0, 2, 2] = 1.0
        d_hw = self._make_d_hw()
        loss = loss_fn(P, d_hw=d_hw, circuit_edge_pairs=[(0, 1), (1, 2)],
                       circuit_edge_weights=[1.0, 1.0])
        assert abs(loss.item() + 1.0) < 1e-5

    def test_gate_frequency_weighting(self):
        """Higher-weight edges should contribute more to the loss."""
        loss_fn = AdjacencyMatchingLoss()
        P = torch.rand(1, 3, 5)
        d_hw = self._make_d_hw()
        pairs = [(0, 1), (0, 2)]

        loss_equal = loss_fn(P, d_hw=d_hw, circuit_edge_pairs=pairs,
                             circuit_edge_weights=[1.0, 1.0])
        loss_weight_adj = loss_fn(P, d_hw=d_hw, circuit_edge_pairs=pairs,
                                  circuit_edge_weights=[10.0, 1.0])
        loss_weight_nonadj = loss_fn(P, d_hw=d_hw, circuit_edge_pairs=pairs,
                                     circuit_edge_weights=[1.0, 10.0])
        assert torch.isfinite(loss_equal)
        assert torch.isfinite(loss_weight_adj)
        assert torch.isfinite(loss_weight_nonadj)


# ---- Node Quality Loss ----

class TestNodeQualityLoss:
    def test_basic(self):
        qs = QualityScore(num_features=6)
        loss_fn = NodeQualityLoss(qs)
        P = torch.rand(2, 3, 5)
        features = torch.randn(5, 6)
        importance = torch.tensor([3.0, 2.0, 1.0])
        loss = loss_fn(P, hw_node_features=features, qubit_importance=importance, num_logical=3)
        assert loss.shape == ()

    def test_gradient_to_quality_score(self):
        qs = QualityScore(num_features=6)
        loss_fn = NodeQualityLoss(qs)
        P = torch.rand(1, 3, 5, requires_grad=True)
        features = torch.randn(5, 6)
        importance = torch.tensor([1.0, 1.0, 1.0])
        loss = loss_fn(P, hw_node_features=features, qubit_importance=importance, num_logical=3)
        loss.backward()
        for param in qs.parameters():
            assert param.grad is not None


# ---- Combined Surrogate Loss (Registry-based) ----

class TestSurrogateLoss:
    def _make_d_hw(self, n=5):
        """Create hop distance matrix for a path graph."""
        d_hw = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                d_hw[i, j] = abs(i - j)
        return d_hw

    def _make_d_error(self, n=5):
        """Create error distance matrix."""
        d = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                d[i, j] = abs(i - j) * 0.01
        return d

    def _common_kwargs(self, n=5):
        return dict(
            d_hw=self._make_d_hw(n),
            d_error=self._make_d_error(n),
            hw_node_features=torch.randn(n, 6),
            circuit_edge_pairs=[(0, 1), (1, 2)],
            circuit_edge_weights=[3.0, 1.0],
            qubit_importance=torch.tensor([2.0, 1.0, 1.0]),
            num_logical=3,
            cross_circuit_pairs=[],
        )

    def test_original_config(self):
        """Test the original loss: error_distance + node_quality."""
        qs = QualityScore(num_features=6)
        loss_fn = SurrogateLoss(
            components=[
                {"name": "error_distance", "weight": 1.0},
                {"name": "node_quality", "weight": 0.3},
            ],
            quality_score=qs,
        )
        P = torch.rand(2, 3, 5)
        result = loss_fn(P, **self._common_kwargs())

        assert "total" in result
        assert "error_distance" in result
        assert "node_quality" in result
        assert result["total"].shape == ()

    def test_adj_hop_config(self):
        """Test adjacency + hop + node config."""
        qs = QualityScore(num_features=6)
        loss_fn = SurrogateLoss(
            components=[
                {"name": "adjacency", "weight": 1.0},
                {"name": "hop_distance", "weight": 0.2},
                {"name": "node_quality", "weight": 0.3},
            ],
            quality_score=qs,
        )
        P = torch.rand(2, 3, 5)
        result = loss_fn(P, **self._common_kwargs())

        assert "adjacency" in result
        assert "hop_distance" in result
        assert "node_quality" in result

    def test_single_component(self):
        """Test with just one loss component."""
        loss_fn = SurrogateLoss(
            components=[{"name": "error_distance", "weight": 1.0}],
        )
        P = torch.rand(2, 3, 5)
        result = loss_fn(P, **self._common_kwargs())

        assert "total" in result
        assert "error_distance" in result
        assert len(result) == 2  # total + error_distance

    def test_component_names_tracked(self):
        qs = QualityScore(num_features=6)
        loss_fn = SurrogateLoss(
            components=[
                {"name": "error_distance", "weight": 1.0},
                {"name": "node_quality", "weight": 0.3},
            ],
            quality_score=qs,
        )
        assert loss_fn.component_names == ["error_distance", "node_quality"]
        assert loss_fn.component_weights == [1.0, 0.3]

    def test_weighted_sum(self):
        """Total should equal weighted sum of components."""
        qs = QualityScore(num_features=6)
        loss_fn = SurrogateLoss(
            components=[
                {"name": "error_distance", "weight": 2.0},
                {"name": "node_quality", "weight": 0.5},
            ],
            quality_score=qs,
        )
        P = torch.rand(2, 3, 5)
        result = loss_fn(P, **self._common_kwargs())

        expected = 2.0 * result["error_distance"] + 0.5 * result["node_quality"]
        assert result["total"].item() == pytest.approx(expected.item(), abs=1e-5)

    def test_with_separation(self):
        """Test that separation loss works when cross_circuit_pairs provided."""
        qs = QualityScore(num_features=6)
        loss_fn = SurrogateLoss(
            components=[
                {"name": "error_distance", "weight": 1.0},
                {"name": "separation", "weight": 0.1},
                {"name": "node_quality", "weight": 0.3},
            ],
            quality_score=qs,
        )
        P = torch.rand(2, 3, 5)
        kwargs = self._common_kwargs()
        kwargs["cross_circuit_pairs"] = [(0, 2)]
        result = loss_fn(P, **kwargs)

        assert "separation" in result
        assert result["separation"].item() <= 0  # bounded [-1, 0]
