"""Tests for model architecture components."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from models.cross_attention import CrossAttentionModule
from models.gnn_encoder import GNNEncoder
from models.graphqmap import GraphQMap
from models.hungarian import hungarian_decode, hungarian_decode_batch
from models.score_head import ScoreHead
from models.sinkhorn import SinkhornLayer, SoftmaxNorm, log_sinkhorn


# ---- GNN Encoder ----

class TestGNNEncoder:
    @pytest.fixture
    def encoder(self):
        return GNNEncoder(
            node_input_dim=6, edge_input_dim=2, embedding_dim=64,
            num_layers=3, num_heads=4, dropout=0.0,
        )

    def test_output_shape(self, encoder):
        x = torch.randn(5, 6)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 2)
        out = encoder(x, edge_index, edge_attr)
        assert out.shape == (5, 64)

    def test_different_input_dims(self):
        enc = GNNEncoder(node_input_dim=4, edge_input_dim=3, embedding_dim=32,
                         num_layers=2, num_heads=4, dropout=0.0, residual_layers=[2])
        x = torch.randn(3, 4)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_attr = torch.randn(2, 3)
        out = enc(x, edge_index, edge_attr)
        assert out.shape == (3, 32)

    def test_no_edges(self):
        enc = GNNEncoder(node_input_dim=6, edge_input_dim=2, embedding_dim=64,
                         num_layers=3, num_heads=4, dropout=0.0)
        x = torch.randn(3, 6)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2))
        out = enc(x, edge_index, edge_attr)
        assert out.shape == (3, 64)

    def test_gradient_flow(self, encoder):
        x = torch.randn(5, 6, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 2)
        out = encoder(x, edge_index, edge_attr)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (5, 6)


# ---- Cross-Attention ----

class TestCrossAttention:
    @pytest.fixture
    def module(self):
        return CrossAttentionModule(d_model=64, num_layers=2, num_heads=4,
                                    d_ff=128, dropout=0.0)

    def test_output_shape(self, module):
        C = torch.randn(2, 5, 64)   # batch=2, l=5
        H = torch.randn(2, 16, 64)  # batch=2, h=16
        C_out, H_out = module(C, H)
        assert C_out.shape == (2, 5, 64)
        assert H_out.shape == (2, 16, 64)

    def test_single_sample(self, module):
        C = torch.randn(1, 3, 64)
        H = torch.randn(1, 5, 64)
        C_out, H_out = module(C, H)
        assert C_out.shape == (1, 3, 64)
        assert H_out.shape == (1, 5, 64)

    def test_gradient_flow(self, module):
        C = torch.randn(1, 3, 64, requires_grad=True)
        H = torch.randn(1, 5, 64, requires_grad=True)
        C_out, H_out = module(C, H)
        loss = C_out.sum() + H_out.sum()
        loss.backward()
        assert C.grad is not None
        assert H.grad is not None


# ---- Score Head ----

class TestScoreHead:
    def test_output_shape(self):
        head = ScoreHead(d_model=64, d_k=64)
        C = torch.randn(2, 5, 64)
        H = torch.randn(2, 16, 64)
        S = head(C, H)
        assert S.shape == (2, 5, 16)

    def test_gradient_flow(self):
        head = ScoreHead(d_model=64, d_k=64)
        C = torch.randn(1, 3, 64, requires_grad=True)
        H = torch.randn(1, 5, 64, requires_grad=True)
        S = head(C, H)
        S.sum().backward()
        assert C.grad is not None
        assert H.grad is not None


# ---- Score Normalization ----

class TestSoftmaxNorm:
    def test_row_stochastic(self):
        """Output P should be row-stochastic (rows sum to 1)."""
        layer = SoftmaxNorm()
        S = torch.randn(2, 3, 5)  # l=3, h=5
        P = layer(S, num_logical=3, num_physical=5, tau=0.1)
        assert P.shape == (2, 3, 5)
        row_sums = P.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_low_tau_stability(self):
        """Softmax should be stable at low τ values."""
        layer = SoftmaxNorm()
        S = torch.randn(1, 3, 5)
        P = layer(S, num_logical=3, num_physical=5, tau=0.05)
        assert torch.isfinite(P).all()
        assert (P >= 0).all()

    def test_gradient_flow(self):
        layer = SoftmaxNorm()
        S = torch.randn(1, 3, 5, requires_grad=True)
        P = layer(S, num_logical=3, num_physical=5, tau=0.5)
        P.sum().backward()
        assert S.grad is not None

    def test_square_case(self):
        """When l == h, should still work correctly."""
        layer = SoftmaxNorm()
        S = torch.randn(2, 5, 5)
        P = layer(S, num_logical=5, num_physical=5, tau=0.1)
        assert P.shape == (2, 5, 5)
        row_sums = P.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestSinkhorn:
    """Legacy Sinkhorn tests (kept for reference)."""

    def test_doubly_stochastic(self):
        """Output P should be doubly stochastic (rows and cols sum to 1)."""
        S = torch.randn(2, 5, 5)
        P = log_sinkhorn(S / 0.1, max_iter=100)
        # Row sums
        row_sums = P.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.05)
        # Column sums
        col_sums = P.sum(dim=-2)
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=0.05)

    def test_with_dummy_padding(self):
        """SinkhornLayer should handle l < h with dummy rows."""
        layer = SinkhornLayer(max_iter=100)
        S = torch.randn(2, 3, 5)  # l=3, h=5
        P = layer(S, num_logical=3, num_physical=5, tau=0.1)
        assert P.shape == (2, 5, 5)
        # Should still be doubly stochastic
        row_sums = P.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.05)


# ---- Hungarian ----

class TestHungarian:
    def test_basic_decode(self):
        """Near-identity P should produce identity mapping."""
        # (3, 5) row-stochastic with strong preference for first 3 physical qubits
        P = torch.zeros(3, 5)
        for i in range(3):
            P[i, i] = 0.8
            P[i] = P[i] + 0.04
        layout = hungarian_decode(P, num_logical=3)
        assert len(layout) == 3
        for i in range(3):
            assert layout[i] == i

    def test_unique_assignments(self):
        P = torch.softmax(torch.randn(3, 5), dim=-1)
        layout = hungarian_decode(P, num_logical=3)
        assert len(set(layout.values())) == 3

    def test_batch_decode(self):
        P = torch.softmax(torch.randn(4, 3, 5), dim=-1)
        layouts = hungarian_decode_batch(P, num_logical=3)
        assert len(layouts) == 4
        for layout in layouts:
            assert len(layout) == 3
            assert len(set(layout.values())) == 3


# ---- Full Model ----

class TestGraphQMap:
    @pytest.fixture
    def model(self):
        return GraphQMap(
            circuit_node_dim=4, circuit_edge_dim=3,
            hardware_node_dim=6, hardware_edge_dim=2,
            embedding_dim=32, gnn_layers=2, gnn_heads=4,
            gnn_dropout=0.0, cross_attn_layers=1, cross_attn_heads=4,
            cross_attn_ffn_dim=64, cross_attn_dropout=0.0,
            score_d_k=32, sinkhorn_max_iter=10,
        )

    def _make_batch(self, batch_size, num_logical, num_physical):
        """Create dummy batched PyG data."""
        circuit_graphs = []
        for _ in range(batch_size):
            x = torch.randn(num_logical, 4)
            # Simple chain topology
            src = list(range(num_logical - 1))
            dst = list(range(1, num_logical))
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
            edge_attr = torch.randn(len(src) * 2, 3)
            circuit_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        hw_graphs = []
        for _ in range(batch_size):
            x = torch.randn(num_physical, 6)
            src = list(range(num_physical - 1))
            dst = list(range(1, num_physical))
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
            edge_attr = torch.randn(len(src) * 2, 2)
            hw_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        circuit_batch = Batch.from_data_list(circuit_graphs)
        hw_batch = Batch.from_data_list(hw_graphs)
        return circuit_batch, hw_batch

    def test_forward_shape(self, model):
        circuit_batch, hw_batch = self._make_batch(2, 3, 5)
        P = model(circuit_batch, hw_batch, batch_size=2, num_logical=3,
                  num_physical=5, tau=0.5)
        assert P.shape == (2, 3, 5)

    def test_forward_row_stochastic(self, model):
        circuit_batch, hw_batch = self._make_batch(2, 3, 5)
        P = model(circuit_batch, hw_batch, batch_size=2, num_logical=3,
                  num_physical=5, tau=0.5)
        row_sums = P.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_predict(self, model):
        circuit_batch, hw_batch = self._make_batch(2, 3, 5)
        layouts = model.predict(circuit_batch, hw_batch, batch_size=2,
                                num_logical=3, num_physical=5, tau=0.5)
        assert len(layouts) == 2
        for layout in layouts:
            assert len(layout) == 3
            assert len(set(layout.values())) == 3
            for phys in layout.values():
                assert 0 <= phys < 5

    def test_gradient_flow(self, model):
        circuit_batch, hw_batch = self._make_batch(1, 3, 5)
        P = model(circuit_batch, hw_batch, batch_size=1, num_logical=3,
                  num_physical=5, tau=0.5)
        loss = P.sum()
        loss.backward()
        # Check that GNN params have gradients
        for param in model.circuit_gnn.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break

    def test_from_config(self):
        from configs.config_loader import load_config
        cfg = load_config("configs/stage1.yaml")
        model = GraphQMap.from_config(cfg)
        assert model.circuit_gnn.embedding_dim == 64
        assert model.hardware_gnn.embedding_dim == 64

    def test_no_shared_parameters(self, model):
        """Circuit and hardware GNNs must not share parameters."""
        circuit_params = set(id(p) for p in model.circuit_gnn.parameters())
        hw_params = set(id(p) for p in model.hardware_gnn.parameters())
        assert circuit_params.isdisjoint(hw_params)
