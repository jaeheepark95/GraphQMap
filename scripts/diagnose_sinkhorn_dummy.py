"""Diagnose Sinkhorn dummy padding leakage.

For a trained checkpoint, measures how much probability mass from the
doubly-stochastic P (h x h) leaks into the dummy rows (rows l..h-1) that
are discarded during Hungarian decoding.

For each circuit in pozzi_train12 on multiple backends, reports:
  - logical_row_mass : sum of P rows 0..l-1 over all columns (= l exactly)
  - dummy_row_mass   : sum of P rows l..h-1 over all columns (= h - l exactly by doubly-stochastic)
  - col_mass_on_logical : per-column sum of P[0:l, :] — what each physical qubit
                          actually receives from logical qubits after Sinkhorn
  - col_leak_fraction : 1 - col_mass_on_logical, i.e. mass on column j taken by dummy rows.
  - Hungarian decode on P[:, :l, :] vs argmax to check assignment disturbance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch_geometric.data import Batch

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph
from data.hardware_graph import build_hardware_graph, get_backend
from evaluation.benchmark import load_benchmark_circuit
from models.graphqmap import GraphQMap


def run_diagnosis(cfg_path: str, ckpt_path: str, backends: list[str],
                   circuit_list: list[str], tau: float) -> None:
    cfg = load_config(cfg_path)
    device = torch.device("cpu")

    model = GraphQMap.from_config(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    node_fnames = getattr(cfg.model.circuit_gnn, "node_features", None)
    rk = getattr(cfg.model.circuit_gnn, "rwpe_k", 0)
    edge_dim = getattr(cfg.model.circuit_gnn, "edge_input_dim", None)

    print(f"{'backend':<16} {'circuit':<22} {'l':>3} {'h':>4} "
          f"{'dummy_mass':>11} {'leak_frac_mean':>16} "
          f"{'leak_top_hit':>13} {'argmax_collision':>17}")
    print("-" * 110)

    for bname in backends:
        backend = get_backend(bname)
        num_physical = backend.target.num_qubits
        hw_graph = build_hardware_graph(backend)
        hw_batch = Batch.from_data_list([hw_graph])

        for cname in circuit_list:
            try:
                circuit = load_benchmark_circuit(cname, measure=True)
            except Exception as e:
                print(f"  skip {cname}: {e}")
                continue
            num_logical = circuit.num_qubits
            if num_logical > num_physical or num_logical < 2:
                continue

            cg = build_circuit_graph(
                circuit, node_feature_names=node_fnames, rwpe_k=rk,
                edge_dim=edge_dim,
            )
            cb = Batch.from_data_list([cg])

            with torch.no_grad():
                C, H = model.encode(cb, hw_batch, 1, num_logical, num_physical)
                if model.cross_attention is not None:
                    C_p, H_p = model.cross_attention(C, H)
                else:
                    C_p, H_p = C, H
                S = model.score_head(C_p, H_p, None)  # (1, l, h)

                # Full doubly-stochastic P (1, h, h) before slicing
                batch_size, l, h = 1, num_logical, num_physical
                dummy_rows = h - l
                if dummy_rows > 0:
                    dummy = torch.zeros(1, dummy_rows, h, dtype=S.dtype)
                    S_pad = torch.cat([S, dummy], dim=1)
                else:
                    S_pad = S
                from models.sinkhorn import log_sinkhorn
                P_full = log_sinkhorn(S_pad / tau, max_iter=cfg.sinkhorn.max_iter,
                                       tol=cfg.sinkhorn.tolerance)  # (1, h, h)

            P_full = P_full[0]  # (h, h)
            P_logical = P_full[:l, :]  # (l, h) — what GraphQMap uses
            P_dummy = P_full[l:, :]  # (h-l, h)

            # Row sums
            logical_row_sums = P_logical.sum(dim=-1)  # each should be ~1 (doubly-stoch row)
            dummy_row_sums = P_dummy.sum(dim=-1)

            # Per-column mass on logical vs dummy
            col_mass_logical = P_logical.sum(dim=0)  # (h,)
            col_mass_dummy = P_dummy.sum(dim=0)  # (h,)
            # Since P is doubly stochastic, col sum = 1 → col_mass_dummy = 1 - col_mass_logical

            leak_frac_mean = col_mass_dummy.mean().item()

            # Check if top logical preferences are stolen by dummy
            # Find top-l physical qubits ranked by col_mass_logical (ideal case: these carry mass ~1)
            top_l_cols = torch.topk(col_mass_logical, l).indices
            leak_at_top = (1 - col_mass_logical[top_l_cols]).mean().item()

            # Argmax collision: does argmax of P_logical rows pick same physical qubit for different logical?
            argmax_map = P_logical.argmax(dim=-1)  # (l,)
            n_collisions = l - len(set(argmax_map.tolist()))

            print(f"{bname:<16} {cname:<22} {l:>3} {h:>4} "
                  f"{dummy_row_sums.sum().item():>11.3f} {leak_frac_mean:>16.4f} "
                  f"{leak_at_top:>13.4f} {n_collisions:>17d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backends", nargs="+",
                        default=["FakeToronto", "FakeRochester", "FakeWashington"])
    parser.add_argument("--circuits-json",
                        default="data/circuits/splits/pozzi_train12.json")
    parser.add_argument("--tau", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.circuits_json) as f:
        blob = json.load(f)
    # splits json is list of {"source":..., "file": "<name>.qasm"}
    circuits = []
    for entry in blob:
        if isinstance(entry, dict):
            circuits.append(entry["file"].replace(".qasm", ""))
        else:
            circuits.append(entry)

    run_diagnosis(args.config, args.checkpoint, args.backends, circuits, args.tau)
