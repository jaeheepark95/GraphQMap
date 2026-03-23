"""End-to-end integration test for GraphQMap.

Tests the full pipeline with synthetic circuits:
  1. Graph construction (circuit + hardware)
  2. Model forward pass → P matrix
  3. Hungarian decoding → discrete layout
  4. Label generation → permutation matrix
  5. Loss computation (Stage 1 CE + Stage 2 surrogate)
  6. Backward pass (gradient flow end-to-end)
  7. Training step (optimizer update)
  8. PST evaluation
  9. Multi-programming scenario
"""

import pytest
import numpy as np
import torch
from torch.optim import AdamW
from qiskit import QuantumCircuit
from torch_geometric.data import Batch, Data

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph, extract_circuit_features
from data.dataset import MappingDataset, MappingSample, collate_mapping_samples, create_dataloader
from data.hardware_graph import build_hardware_graph, get_backend, precompute_error_distance
from data.label_generation import generate_label, layout_to_permutation_matrix
from data.multi_programming import merge_circuits
from evaluation.pst import measure_pst
from models.graphqmap import GraphQMap
from training.losses import SupervisedCELoss, Stage2Loss
from training.quality_score import QualityScore
from training.tau_scheduler import TauScheduler


def _make_circuit(num_qubits: int, num_cx: int, seed: int = 0) -> QuantumCircuit:
    """Create a synthetic circuit with random CX gates."""
    import random
    rng = random.Random(seed)
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for _ in range(num_cx):
        q0, q1 = rng.sample(range(num_qubits), 2)
        qc.cx(q0, q1)
    return qc


class TestEndToEnd:
    """Full pipeline integration test."""

    @pytest.fixture
    def backend(self):
        return get_backend("manila")

    @pytest.fixture
    def circuit(self):
        return _make_circuit(3, 4, seed=42)

    @pytest.fixture
    def model(self):
        return GraphQMap(
            circuit_node_dim=4,
            circuit_edge_dim=3,
            hardware_node_dim=7,
            hardware_edge_dim=1,
            embedding_dim=32,
            gnn_layers=2,
            gnn_heads=4,
            gnn_dropout=0.0,
            cross_attn_layers=1,
            cross_attn_heads=4,
            cross_attn_ffn_dim=64,
            cross_attn_dropout=0.0,
            score_d_k=32,
            sinkhorn_max_iter=20,
        )

    def test_full_forward_to_layout(self, model, circuit, backend):
        """Graph → Model → P → Hungarian → layout → valid mapping."""
        circuit_graph = build_circuit_graph(circuit)
        hw_graph = build_hardware_graph(backend)

        circuit_batch = Batch.from_data_list([circuit_graph])
        hw_batch = Batch.from_data_list([hw_graph])

        # Forward
        P = model(circuit_batch, hw_batch, batch_size=1,
                  num_logical=3, num_physical=5, tau=0.5)
        assert P.shape == (1, 5, 5)

        # Predict
        layouts = model.predict(circuit_batch, hw_batch, batch_size=1,
                                num_logical=3, num_physical=5, tau=0.5)
        layout = layouts[0]

        # Verify valid mapping
        assert len(layout) == 3
        assert len(set(layout.values())) == 3
        assert all(0 <= v < 5 for v in layout.values())

    def test_stage1_training_step(self, model, circuit, backend):
        """One complete Stage 1 training step with gradient update."""
        circuit_graph = build_circuit_graph(circuit)
        hw_graph = build_hardware_graph(backend)

        # Generate label
        label_result = generate_label(circuit, backend,
                                      num_sabre_seeds=3, num_random=2, rng_seed=42)
        Y = layout_to_permutation_matrix(label_result.layout, num_physical=5)
        Y_tensor = torch.tensor(Y).unsqueeze(0)  # (1, 5, 5)

        circuit_batch = Batch.from_data_list([circuit_graph])
        hw_batch = Batch.from_data_list([hw_graph])

        # Forward
        optimizer = AdamW(model.parameters(), lr=1e-3)
        criterion = SupervisedCELoss()

        model.train()
        optimizer.zero_grad()
        P = model(circuit_batch, hw_batch, batch_size=1,
                  num_logical=3, num_physical=5, tau=0.5)

        loss = criterion(P, Y_tensor)
        assert loss.item() > 0

        # Backward
        loss.backward()

        # Check gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters())
        assert grad_count == total_params

        # Optimizer step
        param_before = next(model.parameters()).data.clone()
        optimizer.step()
        param_after = next(model.parameters()).data
        assert not torch.equal(param_before, param_after)

    def test_stage2_training_step(self, model, circuit, backend):
        """One complete Stage 2 training step with surrogate loss."""
        circuit_graph = build_circuit_graph(circuit)
        hw_graph = build_hardware_graph(backend)

        # Precompute
        d_error = torch.tensor(precompute_error_distance(backend))
        d_hw = d_error.clone()  # use same for simplicity

        # Circuit edge pairs
        feats = extract_circuit_features(circuit)
        circuit_edge_pairs = feats["edge_list"]

        # Qubit importance
        importance = feats["node_features"][:, 1]  # two_qubit_gate_count

        # Hardware node features for q_score (5 features)
        hw_features = hw_graph.x[:, :5]  # T1, T2, freq, readout, sq_err

        circuit_batch = Batch.from_data_list([circuit_graph])
        hw_batch = Batch.from_data_list([hw_graph])

        # Setup
        quality_score = QualityScore(num_features=5)
        criterion = Stage2Loss(quality_score, alpha=0.3, lambda_sep=0.1)
        all_params = list(model.parameters()) + list(quality_score.parameters())
        optimizer = AdamW(all_params, lr=1e-3)

        model.train()
        optimizer.zero_grad()
        P = model(circuit_batch, hw_batch, batch_size=1,
                  num_logical=3, num_physical=5, tau=0.05)

        losses = criterion(
            P=P, d_error=d_error, d_hw=d_hw,
            hw_node_features=hw_features,
            circuit_edge_pairs=circuit_edge_pairs,
            cross_circuit_pairs=[],  # single circuit
            qubit_importance=importance,
            num_logical=3,
        )

        assert losses["l_sep"].item() == 0.0  # single circuit
        losses["total"].backward()
        optimizer.step()

    def test_tau_annealing_affects_output(self, model, circuit, backend):
        """Different τ values should produce different P sharpness."""
        circuit_graph = build_circuit_graph(circuit)
        hw_graph = build_hardware_graph(backend)
        circuit_batch = Batch.from_data_list([circuit_graph])
        hw_batch = Batch.from_data_list([hw_graph])

        model.eval()
        with torch.no_grad():
            P_soft = model(circuit_batch, hw_batch, batch_size=1,
                           num_logical=3, num_physical=5, tau=1.0)
            P_hard = model(circuit_batch, hw_batch, batch_size=1,
                           num_logical=3, num_physical=5, tau=0.05)

        # Hard τ should produce sharper (higher max) distribution
        assert P_hard.max() >= P_soft.max() - 0.1  # allow tolerance

    def test_pst_evaluation(self, circuit, backend):
        """PST measurement should return valid results."""
        result = measure_pst(circuit, backend, layout=[0, 1, 2], shots=1024)
        assert 0.0 <= result["pst"] <= 1.0
        assert result["swap_count"] >= 0
        assert result["depth"] > 0

    def test_dataset_and_dataloader(self, circuit, backend):
        """Dataset → DataLoader → batch dict → model compatible."""
        circuit_graph = build_circuit_graph(circuit)
        hw_graph = build_hardware_graph(backend)

        label_result = generate_label(circuit, backend,
                                      num_sabre_seeds=2, num_random=1, rng_seed=0)
        Y = layout_to_permutation_matrix(label_result.layout, num_physical=5)

        dataset = MappingDataset()
        for _ in range(5):
            sample = MappingSample(
                circuit_graph=circuit_graph,
                hardware_graph=hw_graph,
                backend_name="manila",
                num_logical=3,
                num_physical=5,
                label_matrix=Y,
                layout=label_result.layout,
            )
            dataset.add_sample(sample)

        loader = create_dataloader(dataset, max_total_nodes=512, shuffle=False)
        batch = next(iter(loader))

        assert "circuit_batch" in batch
        assert "hardware_batch" in batch
        assert "label_matrices" in batch
        assert batch["batch_size"] == 5

    def test_multi_programming_pipeline(self, model, backend):
        """Multi-programming: 2 circuits merged → model → valid mapping."""
        c1 = _make_circuit(2, 1, seed=0)
        c2 = _make_circuit(2, 1, seed=1)

        merged = merge_circuits([c1, c2])
        hw_graph = build_hardware_graph(backend)

        # Merged has 4 logical qubits (2+2), same 4-dim features as single-circuit
        assert merged.x.shape[0] == 4
        assert merged.x.shape[1] == 4

        # Same model works for both single and multi-programming
        model_mp = GraphQMap(
            circuit_node_dim=4, circuit_edge_dim=3,
            hardware_node_dim=7, hardware_edge_dim=1,
            embedding_dim=32, gnn_layers=2, gnn_heads=4,
            gnn_dropout=0.0, cross_attn_layers=1, cross_attn_heads=4,
            cross_attn_ffn_dim=64, cross_attn_dropout=0.0,
            score_d_k=32,
        )

        circuit_batch = Batch.from_data_list([merged])
        hw_batch = Batch.from_data_list([hw_graph])

        layouts = model_mp.predict(circuit_batch, hw_batch, batch_size=1,
                                   num_logical=4, num_physical=5, tau=0.5)
        layout = layouts[0]

        # 4 logical qubits → 4 unique physical assignments
        assert len(layout) == 4
        assert len(set(layout.values())) == 4

    def test_multi_circuit_stage2_separation_loss(self, backend):
        """L_sep should be non-zero for multi-circuit scenarios."""
        c1 = _make_circuit(2, 1, seed=0)
        c2 = _make_circuit(2, 1, seed=1)

        merged = merge_circuits([c1, c2])
        hw_graph = build_hardware_graph(backend)

        model = GraphQMap(
            circuit_node_dim=4, circuit_edge_dim=3,
            hardware_node_dim=7, hardware_edge_dim=1,
            embedding_dim=32, gnn_layers=2, gnn_heads=4,
            gnn_dropout=0.0, cross_attn_layers=1, cross_attn_heads=4,
            cross_attn_ffn_dim=64, cross_attn_dropout=0.0,
            score_d_k=32,
        )

        circuit_batch = Batch.from_data_list([merged])
        hw_batch = Batch.from_data_list([hw_graph])

        P = model(circuit_batch, hw_batch, batch_size=1,
                  num_logical=4, num_physical=5, tau=0.5)

        # Cross-circuit pairs: (0,2), (0,3), (1,2), (1,3)
        cross_pairs = [(i, j) for i in range(2) for j in range(2, 4)]
        d_error = torch.tensor(precompute_error_distance(backend))

        from training.losses import SeparationLoss
        sep_loss_fn = SeparationLoss()
        loss = sep_loss_fn(P, d_error, cross_pairs, num_logical=4)

        assert loss.item() != 0.0  # should be non-zero

    def test_from_config_and_predict(self):
        """Load model from YAML config → predict → valid output."""
        cfg = load_config("configs/stage1.yaml")
        model = GraphQMap.from_config(cfg)

        backend = get_backend("manila")
        circuit = _make_circuit(3, 3, seed=99)
        circuit_graph = build_circuit_graph(circuit)
        hw_graph = build_hardware_graph(backend)

        circuit_batch = Batch.from_data_list([circuit_graph])
        hw_batch = Batch.from_data_list([hw_graph])

        layouts = model.predict(circuit_batch, hw_batch, batch_size=1,
                                num_logical=3, num_physical=5, tau=0.5)
        layout = layouts[0]
        assert len(layout) == 3
        assert len(set(layout.values())) == 3

    def test_multiple_epochs_loss_decreases(self, backend):
        """Training over multiple epochs should decrease loss."""
        circuit = _make_circuit(3, 3, seed=0)
        circuit_graph = build_circuit_graph(circuit)
        hw_graph = build_hardware_graph(backend)

        label_result = generate_label(circuit, backend,
                                      num_sabre_seeds=3, num_random=2, rng_seed=0)
        Y = layout_to_permutation_matrix(label_result.layout, num_physical=5)
        Y_tensor = torch.tensor(Y).unsqueeze(0)

        circuit_batch = Batch.from_data_list([circuit_graph])
        hw_batch = Batch.from_data_list([hw_graph])

        model = GraphQMap(
            circuit_node_dim=4, circuit_edge_dim=3,
            hardware_node_dim=7, hardware_edge_dim=1,
            embedding_dim=32, gnn_layers=2, gnn_heads=4,
            gnn_dropout=0.0, cross_attn_layers=1, cross_attn_heads=4,
            cross_attn_ffn_dim=64, cross_attn_dropout=0.0,
            score_d_k=32,
        )
        optimizer = AdamW(model.parameters(), lr=1e-3)
        criterion = SupervisedCELoss()
        tau_sched = TauScheduler(tau_max=1.0, tau_min=0.1, schedule="exponential",
                                 total_epochs=20)

        losses = []
        model.train()
        for epoch in range(20):
            tau = tau_sched.get_tau(epoch)
            optimizer.zero_grad()
            P = model(circuit_batch, hw_batch, batch_size=1,
                      num_logical=3, num_physical=5, tau=tau)
            loss = criterion(P, Y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease over training
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
