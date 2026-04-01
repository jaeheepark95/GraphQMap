"""Inspect S (Score Head output) and P (Sinkhorn output) matrices from Stage 2 checkpoints.

Loads multiple checkpoints and prints S, P matrices for benchmark circuits
to verify that the model is learning meaningful assignment patterns.

Usage:
    python scripts/inspect_s_p_matrices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph, load_circuit
from data.hardware_graph import (
    extract_edge_properties,
    extract_qubit_properties,
    get_backend,
)
from data.normalization import zscore_normalize
from models.graphqmap import GraphQMap


# --- Configuration ---
CHECKPOINTS = {
    "scratch_default": {
        "config": "runs/stage2/20260329_235418_scratch_default/config.yaml",
        "ckpt": "runs/stage2/20260329_235418_scratch_default/checkpoints/best.pt",
    },
    "error_dist_adj": {
        "config": "runs/stage2/20260327_213036_ablation_error_dist_adj/config.yaml",
        "ckpt": "runs/stage2/20260327_213036_ablation_error_dist_adj/checkpoints/best.pt",
    },
    "error_dist_adj_v2": {
        "config": "runs/stage2/20260330_035101_scratch_error_dist_adj_v2/config.yaml",
        "ckpt": "runs/stage2/20260330_035101_scratch_error_dist_adj_v2/checkpoints/best.pt",
    },
}

# Small circuits for readable matrix output
TEST_CIRCUITS = [
    "data/circuits/qasm/benchmarks/3_17_13.qasm",        # 3Q
    "data/circuits/qasm/benchmarks/4mod5-v1_22.qasm",    # 5Q
    "data/circuits/qasm/benchmarks/alu-v0_27.qasm",      # 5Q
]

BACKEND_NAME = "toronto"  # 27Q test backend
TAU = 0.05
DEVICE = torch.device("cpu")


def build_hardware_graph_7feat(backend, eps: float = 1e-8) -> Data:
    """Build hardware graph with 7 node features (matches checkpoint training).

    Features: t1, t2, readout_error, single_qubit_error, degree, t1_cx_ratio, t2_cx_ratio
    """
    qubit_props = extract_qubit_properties(backend)
    edge_list, edge_feats = extract_edge_properties(backend)

    node_features = np.stack([
        qubit_props["t1"],
        qubit_props["t2"],
        qubit_props["readout_error"],
        qubit_props["single_qubit_error"],
        qubit_props["degree"],
        qubit_props["t1_cx_ratio"],
        qubit_props["t2_cx_ratio"],
    ], axis=1).astype(np.float32)

    x = torch.from_numpy(node_features)
    x = zscore_normalize(x, dim=0, eps=eps)

    if len(edge_list) > 0:
        src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
        dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.from_numpy(edge_feats)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_attr = zscore_normalize(edge_attr, dim=0, eps=eps)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)

    num_qubits = backend.target.num_qubits
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_qubits=num_qubits)


def load_model(config_path: str, checkpoint_path: str) -> GraphQMap:
    """Load model from config and checkpoint."""
    cfg = load_config(config_path)
    model = GraphQMap.from_config(cfg)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def forward_with_s_and_p(
    model: GraphQMap,
    circuit_batch: Batch,
    hw_batch: Batch,
    num_logical: int,
    num_physical: int,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass and return both S and P matrices."""
    with torch.no_grad():
        C, H = model.encode(circuit_batch, hw_batch, 1, num_logical, num_physical)
        C_prime, H_prime = model.cross_attention(C, H)
        S = model.score_head(C_prime, H_prime, None)        # (1, l, h)
        P = model.sinkhorn(S, num_logical, num_physical, tau)  # (1, h, h)
    return S.squeeze(0), P.squeeze(0)


def print_matrix_stats(name: str, M: torch.Tensor) -> None:
    """Print summary statistics for a matrix."""
    print(f"  {name}: shape={list(M.shape)}, "
          f"min={M.min().item():.4f}, max={M.max().item():.4f}, "
          f"mean={M.mean().item():.4f}, std={M.std().item():.4f}")


def print_matrix(name: str, M: torch.Tensor, max_cols: int = 27) -> None:
    """Print a matrix with row/column headers."""
    rows, cols = M.shape
    show_cols = min(cols, max_cols)

    header = "     " + " ".join(f"{j:6d}" for j in range(show_cols))
    if cols > show_cols:
        header += "  ..."
    print(f"  {name}:")
    print(f"  {header}")

    for i in range(rows):
        row_str = f"  {i:3d}: " + " ".join(f"{M[i, j].item():6.3f}" for j in range(show_cols))
        if cols > show_cols:
            row_str += "  ..."
        print(row_str)


def analyze_p_quality(P: torch.Tensor, num_logical: int) -> None:
    """Analyze P matrix quality: entropy, concentration."""
    h = P.shape[0]
    P_logical = P[:num_logical, :]  # (l, h)

    # Row/col sum check
    row_sums = P.sum(dim=1)
    col_sums = P.sum(dim=0)
    print(f"  Row sums (logical): {[f'{s:.4f}' for s in row_sums[:num_logical].tolist()]}")
    print(f"  Col sums range: [{col_sums.min().item():.4f}, {col_sums.max().item():.4f}]")

    # Entropy per logical qubit
    eps = 1e-10
    entropy = -(P_logical * (P_logical + eps).log()).sum(dim=1)
    max_entropy = np.log(h)
    print(f"  Entropy per logical qubit: {[f'{e:.2f}' for e in entropy.tolist()]}")
    print(f"    (max possible = {max_entropy:.2f}, uniform = {max_entropy:.2f}, lower = more confident)")

    # Top assignments
    topk_vals, topk_idx = P_logical.topk(min(3, h), dim=1)
    print(f"  Top-3 assignment per logical qubit:")
    for i in range(num_logical):
        top3 = ", ".join(f"p{topk_idx[i, k].item()}({topk_vals[i, k].item():.4f})" for k in range(min(3, h)))
        print(f"    q{i} -> {top3}  |  top-3 sum={topk_vals[i].sum().item():.4f}")


def analyze_s_quality(S: torch.Tensor) -> None:
    """Analyze S matrix: score differentiation."""
    row_ranges = S.max(dim=1).values - S.min(dim=1).values
    print(f"  Score range per logical qubit: {[f'{r:.4f}' for r in row_ranges.tolist()]}")
    print(f"  Overall dynamic range: {(S.max() - S.min()).item():.4f}")

    # Check if all rows are nearly identical (bad sign)
    if S.shape[0] > 1:
        row_diffs = []
        for i in range(1, S.shape[0]):
            row_diffs.append((S[i] - S[0]).abs().mean().item())
        print(f"  Row dissimilarity (vs row 0): {[f'{d:.4f}' for d in row_diffs]}")
        if all(d < 0.01 for d in row_diffs):
            print(f"  *** WARNING: All S rows are nearly identical! Model may not be differentiating logical qubits.")

    # Check if S is uniform (bad sign)
    if S.std().item() < 0.01:
        print(f"  *** WARNING: S is nearly uniform (std={S.std().item():.6f}). Model may not be learning.")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 80)
    print("S & P Matrix Inspection for Stage 2 Checkpoints")
    print("=" * 80)

    # Build hardware graph once (7 features)
    print(f"\nBackend: {BACKEND_NAME}")
    backend = get_backend(BACKEND_NAME)
    hw_graph = build_hardware_graph_7feat(backend)
    num_physical = backend.target.num_qubits
    print(f"Physical qubits: {num_physical}")
    print(f"HW graph node features: {hw_graph.x.shape}")

    # Load circuits
    circuits = []
    for path in TEST_CIRCUITS:
        circ = load_circuit(path)
        circuits.append((Path(path).stem, circ))
        print(f"Circuit: {Path(path).stem} ({circ.num_qubits}Q)")

    # Load models
    models = {}
    for name, paths in CHECKPOINTS.items():
        print(f"\nLoading checkpoint: {name}")
        print(f"  config: {paths['config']}")
        models[name] = load_model(paths["config"], paths["ckpt"])

    hw_batch = Batch.from_data_list([hw_graph])

    # Run analysis per circuit
    for circ_name, circuit in circuits:
        num_logical = circuit.num_qubits
        circuit_graph = build_circuit_graph(circuit)
        circuit_batch = Batch.from_data_list([circuit_graph])

        print("\n" + "=" * 80)
        print(f"Circuit: {circ_name} (l={num_logical}, h={num_physical})")
        print("=" * 80)

        for model_name, model in models.items():
            print(f"\n{'─' * 40}")
            print(f"  Checkpoint: {model_name}")
            print(f"{'─' * 40}")

            S, P = forward_with_s_and_p(
                model, circuit_batch, hw_batch, num_logical, num_physical, TAU,
            )

            print("\n[S matrix] Score Head output (l x h)")
            print_matrix_stats("S", S)
            analyze_s_quality(S)
            print_matrix("S", S)

            print(f"\n[P matrix] Sinkhorn output (showing logical rows only: l x h)")
            P_logical = P[:num_logical, :]
            print_matrix_stats("P (logical rows)", P_logical)
            analyze_p_quality(P, num_logical)
            print_matrix("P[:l]", P_logical)

            # Dummy rows
            if num_logical < num_physical:
                P_dummy = P[num_logical:, :]
                print(f"\n  Dummy rows ({num_physical - num_logical}): "
                      f"mean={P_dummy.mean().item():.6f}, std={P_dummy.std().item():.6f}")

    # Summary table
    print("\n" + "=" * 80)
    print("CROSS-CHECKPOINT COMPARISON")
    print("=" * 80)

    for circ_name, circuit in circuits:
        num_logical = circuit.num_qubits
        circuit_graph = build_circuit_graph(circuit)
        circuit_batch = Batch.from_data_list([circuit_graph])

        print(f"\n--- {circ_name} ({num_logical}Q) ---")
        print(f"{'Checkpoint':<25} {'S_range':>8} {'S_std':>8} {'S_row_sim':>10} "
              f"{'P_entropy':>10} {'P_top1_max':>11} {'Layout(argmax)':>25}")

        for model_name, model in models.items():
            S, P = forward_with_s_and_p(
                model, circuit_batch, hw_batch, num_logical, num_physical, TAU,
            )

            s_range = (S.max() - S.min()).item()
            s_std = S.std().item()

            # Row similarity: avg diff between consecutive rows
            if S.shape[0] > 1:
                row_sim = sum((S[i] - S[0]).abs().mean().item() for i in range(1, S.shape[0])) / (S.shape[0] - 1)
            else:
                row_sim = 0.0

            P_logical = P[:num_logical, :]
            eps = 1e-10
            entropy = -(P_logical * (P_logical + eps).log()).sum(dim=1).mean().item()
            top1_max = P_logical.max(dim=1).values.max().item()

            argmax = P_logical.argmax(dim=1).tolist()
            layout_str = str({i: argmax[i] for i in range(num_logical)})

            print(f"{model_name:<25} {s_range:8.4f} {s_std:8.4f} {row_sim:10.4f} "
                  f"{entropy:10.4f} {top1_max:11.4f} {layout_str:>25}")


if __name__ == "__main__":
    main()
