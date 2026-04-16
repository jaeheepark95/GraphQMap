"""Dump S (score) and P (permutation) matrices for a trained model on sample circuits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph
from data.hardware_graph import build_hardware_graph, get_backend
from evaluation.benchmark import load_benchmark_circuit
from models.graphqmap import GraphQMap


def dump(cfg_path: str, ckpt_path: str, backend_name: str, circuits: list[str]):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphQMap.from_config(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    bk = get_backend(backend_name[4:].lower() if backend_name.startswith("Fake") else backend_name)
    hw_graph = build_hardware_graph(bk)
    num_physical = bk.target.num_qubits
    hw_batch = Batch.from_data_list([hw_graph]).to(device)

    node_fnames = getattr(cfg.model.circuit_gnn, "node_features", None)
    rk = getattr(cfg.model.circuit_gnn, "rwpe_k", 0)
    edim = getattr(cfg.model.circuit_gnn, "edge_input_dim", None)
    tau = getattr(cfg.sinkhorn, "tau_min", 0.5)

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    for cname in circuits:
        circ = load_benchmark_circuit(cname, measure=False)
        l = circ.num_qubits
        if l > num_physical:
            print(f"[SKIP] {cname}: {l}Q > {num_physical}Q backend")
            continue
        c_graph = build_circuit_graph(circ, node_feature_names=node_fnames,
                                       rwpe_k=rk, edge_dim=edim)
        c_batch = Batch.from_data_list([c_graph]).to(device)

        with torch.no_grad():
            C, H = model.encode(c_batch, hw_batch, 1, l, num_physical)
            if model.cross_attention is not None:
                C_prime, H_prime = model.cross_attention(C, H)
            else:
                C_prime, H_prime = C, H
            S = model.score_head(C_prime, H_prime, None)  # (1, l, h)
            P = model.sinkhorn(S, l, num_physical, tau)
            if P.shape[1] != l:
                P = P[:, :l, :]

        S_np = S[0].cpu().numpy()
        P_np = P[0].cpu().numpy()

        print(f"\n{'='*80}")
        print(f"Circuit: {cname}  (l={l}, h={num_physical}, backend={backend_name}, tau={tau})")
        print(f"{'='*80}")
        print(f"S stats: mean={S_np.mean():.3f}, std={S_np.std():.3f}, "
              f"min={S_np.min():.3f}, max={S_np.max():.3f}")
        print(f"P row sums (should be ~1): {P_np.sum(axis=1)}")
        print(f"P col sums (max={P_np.sum(axis=0).max():.3f}, nonzero cols={np.sum(P_np.sum(axis=0) > 0.01)})")

        # Argmax layout (logical -> physical)
        layout = P_np.argmax(axis=1).tolist()
        print(f"Argmax layout (logical->physical): {layout}")
        print(f"P argmax values: {[f'{P_np[i, p]:.3f}' for i, p in enumerate(layout)]}")

        print(f"\n--- S matrix (l={l} x h={num_physical}) ---")
        print(S_np)
        print(f"\n--- P matrix (l={l} x h={num_physical}) ---")
        # Show only columns where P > 0.01 anywhere
        active_cols = np.where(P_np.max(axis=0) > 0.01)[0]
        print(f"Active physical qubits (P > 0.01): {active_cols.tolist()}")
        print(f"P[:, active_cols]:")
        print(P_np[:, active_cols])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backend", default="FakeToronto")
    ap.add_argument("--circuits", nargs="+", default=["bv_n3", "fredkin_3", "4mod5-v1_22"])
    args = ap.parse_args()
    dump(args.config, args.checkpoint, args.backend, args.circuits)
