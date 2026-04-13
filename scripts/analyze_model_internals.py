"""Analyze P matrix sharpness and S matrix row distinguishability.

For each checkpoint, runs benchmark circuits through the model and captures:
1. P matrix analysis: max per row, entropy, how one-hot-like
2. S matrix row analysis: pairwise cosine similarity, L2 distance
3. Embedding tracking through the full pipeline:
   - After GNN: C (l,d), H (h,d)
   - After cross-attention: C' (l,d), H' (h,d)
   - Score head projections: Q (l,d_k), K (h,d_k)
   - Score matrix S (l,h)
   - Final P (l,h)

Usage:
    python scripts/analyze_model_internals.py \
        --run-dir runs/stage2/20260413_150914_v5_perterm_s42 \
        --backend FakeToronto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph, extract_circuit_features
from data.hardware_graph import (
    build_hardware_graph,
    configure_hw_features,
    get_backend,
    precompute_c_eff,
)
from evaluation.benchmark import BENCHMARK_CIRCUITS, load_benchmark_circuit
from models.graphqmap import GraphQMap


def cosine_sim_matrix(X: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity for rows of X (n, d)."""
    X_norm = F.normalize(X, p=2, dim=-1)
    return X_norm @ X_norm.T


def analyze_p_matrix(P: torch.Tensor) -> dict:
    """Analyze how one-hot-like the P matrix is.

    P: (l, h) row-stochastic matrix.
    """
    l, h = P.shape
    # Per-row statistics
    max_vals = P.max(dim=-1).values  # (l,)
    # Entropy per row: -sum(p * log(p))
    log_P = torch.log(P.clamp(min=1e-12))
    entropy = -(P * log_P).sum(dim=-1)  # (l,)
    max_entropy = np.log(h)  # uniform distribution entropy

    # Top-2 gap: difference between max and second max
    topk = P.topk(min(2, h), dim=-1).values
    if h >= 2:
        top2_gap = topk[:, 0] - topk[:, 1]
    else:
        top2_gap = topk[:, 0]

    # How many columns have >0.01 probability (effective support)
    effective_support = (P > 0.01).float().sum(dim=-1)

    return {
        "max_mean": max_vals.mean().item(),
        "max_min": max_vals.min().item(),
        "max_max": max_vals.max().item(),
        "entropy_mean": entropy.mean().item(),
        "entropy_max_possible": max_entropy,
        "entropy_ratio": (entropy.mean().item() / max_entropy),
        "top2_gap_mean": top2_gap.mean().item(),
        "top2_gap_min": top2_gap.min().item(),
        "effective_support_mean": effective_support.mean().item(),
    }


def analyze_embedding_rows(emb: torch.Tensor, name: str) -> dict:
    """Analyze distinguishability of embedding rows.

    emb: (l, d) embeddings for logical qubits.
    """
    l, d = emb.shape
    if l < 2:
        return {f"{name}_num_rows": l, f"{name}_note": "single row, skip"}

    # Pairwise cosine similarity
    cos_sim = cosine_sim_matrix(emb)
    # Extract upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones(l, l, dtype=torch.bool), diagonal=1)
    pairwise_cos = cos_sim[mask]

    # Pairwise L2 distance
    diffs = emb.unsqueeze(0) - emb.unsqueeze(1)  # (l, l, d)
    l2_dist = diffs.norm(dim=-1)
    pairwise_l2 = l2_dist[mask]

    # Row norms
    row_norms = emb.norm(dim=-1)

    # Row-wise std (how much variance each dimension has across rows)
    row_std = emb.std(dim=0)  # (d,) std of each dimension across rows

    return {
        f"{name}_cos_sim_mean": pairwise_cos.mean().item(),
        f"{name}_cos_sim_max": pairwise_cos.max().item(),
        f"{name}_cos_sim_min": pairwise_cos.min().item(),
        f"{name}_cos_sim_std": pairwise_cos.std().item(),
        f"{name}_high_sim_ratio": (pairwise_cos > 0.95).float().mean().item(),
        f"{name}_l2_dist_mean": pairwise_l2.mean().item(),
        f"{name}_l2_dist_min": pairwise_l2.min().item(),
        f"{name}_row_norm_mean": row_norms.mean().item(),
        f"{name}_row_norm_std": row_norms.std().item(),
        f"{name}_dim_std_mean": row_std.mean().item(),
        f"{name}_dim_std_min": row_std.min().item(),
        f"{name}_dead_dims": (row_std < 1e-6).sum().item(),
    }


def run_analysis(
    model: GraphQMap,
    cfg,
    backend,
    hw_graph,
    device: torch.device,
    circuits: list[str],
) -> list[dict]:
    """Run analysis on a set of benchmark circuits."""
    num_physical = backend.target.num_qubits
    tau = getattr(cfg.sinkhorn, "tau_min", getattr(cfg.sinkhorn, "tau", 0.05))

    node_fnames = getattr(cfg.model.circuit_gnn, "node_features", None)
    rk = getattr(cfg.model.circuit_gnn, "rwpe_k", 0)
    edge_dim = getattr(cfg.model.circuit_gnn, "edge_input_dim", None)

    results = []

    for cname in circuits:
        try:
            circuit = load_benchmark_circuit(cname, measure=False)
        except Exception as e:
            print(f"  Skip {cname}: {e}")
            continue

        num_logical = circuit.num_qubits
        if num_logical > num_physical:
            print(f"  Skip {cname}: {num_logical}Q > {num_physical}Q backend")
            continue

        circuit_graph = build_circuit_graph(
            circuit, node_feature_names=node_fnames, rwpe_k=rk,
            edge_dim=edge_dim,
        )
        circuit_batch = Batch.from_data_list([circuit_graph]).to(device)
        hw_batch = Batch.from_data_list([hw_graph]).to(device)

        # Build C_eff and circuit_adj
        c_eff_np = precompute_c_eff(backend)
        c_eff = torch.tensor(c_eff_np, dtype=torch.float32).to(device)

        feats = extract_circuit_features(circuit)
        circuit_adj = torch.zeros(num_logical, num_logical, dtype=torch.float32)
        for (ci, cj), w in zip(
            feats["edge_list"],
            feats["edge_features"][:, 0].tolist(),
        ):
            circuit_adj[ci, cj] = w
            circuit_adj[cj, ci] = w
        circuit_adj = circuit_adj.to(device)

        row = {"circuit": cname, "num_logical": num_logical, "tau": tau}

        with torch.no_grad():
            # ===== Stage 1: GNN encoding =====
            C, H = model.encode(
                circuit_batch, hw_batch,
                batch_size=1,
                num_logical=num_logical,
                num_physical=num_physical,
            )
            # C: (1, l, d), H: (1, h, d)
            C_sq = C.squeeze(0)  # (l, d)
            H_sq = H.squeeze(0)  # (h, d)
            row.update(analyze_embedding_rows(C_sq, "gnn_C"))

            # ===== Stage 2: Cross-attention =====
            C_prime, H_prime = model.cross_attention(C, H)
            C_prime_sq = C_prime.squeeze(0)  # (l, d)
            H_prime_sq = H_prime.squeeze(0)  # (h, d)
            row.update(analyze_embedding_rows(C_prime_sq, "xattn_C"))

            # ===== Stage 3: Score head projections =====
            Q = model.score_head.W_q(C_prime_sq)  # (l, d_k)
            K = model.score_head.W_k(H_prime_sq)  # (h, d_k)
            row.update(analyze_embedding_rows(Q, "score_Q"))

            # ===== Stage 4: Score matrix S =====
            S = model.score_head(C_prime, H_prime)  # (1, l, h)
            S_sq = S.squeeze(0)  # (l, h)
            row.update(analyze_embedding_rows(S_sq, "S_rows"))
            row["S_mean"] = S_sq.mean().item()
            row["S_std"] = S_sq.std().item()
            row["S_range"] = (S_sq.max() - S_sq.min()).item()

            # ===== Stage 5: Iterative refinement (if applicable) =====
            if model.refine_iterations > 0:
                S_std_val = S_sq.std().clamp(min=1e-6)
                S_mean_val = S_sq.mean()
                S_norm = (S - S_mean_val) / S_std_val
                S_norm_sq = S_norm.squeeze(0)
                row.update(analyze_embedding_rows(S_norm_sq, "S_norm_rows"))

                # Track refinement iterations
                T = model.refine_iterations
                tau_t = tau / (model.refine_beta ** (T - 1))
                S_current = S_norm.clone()

                for t in range(T):
                    P_t = model.sinkhorn(S_current, num_logical, num_physical, tau_t)
                    if P_t.shape[1] != num_logical:
                        P_t = P_t[:, :num_logical, :]

                    Z = torch.matmul(P_t, c_eff)
                    feedback = torch.matmul(circuit_adj, Z)

                    S_current = S_norm - model.refine_lambda * feedback
                    tau_t = tau_t * model.refine_beta

                    # Analyze intermediate P
                    P_t_sq = P_t.squeeze(0)
                    p_stats = analyze_p_matrix(P_t_sq)
                    for k, v in p_stats.items():
                        row[f"refine_iter{t}_{k}"] = v

                    # Feedback magnitude
                    fb_sq = (model.refine_lambda * feedback).squeeze(0)
                    row[f"refine_iter{t}_feedback_mean"] = fb_sq.abs().mean().item()
                    row[f"refine_iter{t}_feedback_std"] = fb_sq.std().item()
                    row[f"refine_iter{t}_S_current_std"] = S_current.squeeze(0).std().item()

                # Final S after refinement
                S_final = S_current.squeeze(0)
                row.update(analyze_embedding_rows(S_final, "S_refined_rows"))

            # ===== Stage 6: Final P matrix =====
            P = model.forward(
                circuit_batch, hw_batch,
                batch_size=1,
                num_logical=num_logical,
                num_physical=num_physical,
                tau=tau,
                c_eff=c_eff,
                circuit_adj=circuit_adj,
            )
            P_sq = P.squeeze(0)  # (l, h)
            row.update(analyze_p_matrix(P_sq))

            # Also check P without refinement (raw Sinkhorn on S)
            P_raw = model.sinkhorn(S, num_logical, num_physical, tau)
            if P_raw.shape[1] != num_logical:
                P_raw = P_raw[:, :num_logical, :]
            P_raw_sq = P_raw.squeeze(0)
            p_raw_stats = analyze_p_matrix(P_raw_sq)
            for k, v in p_raw_stats.items():
                row[f"P_no_refine_{k}"] = v

        results.append(row)

    return results


def print_results(results: list[dict], label: str) -> None:
    """Pretty-print analysis results."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    if not results:
        print("  No results.")
        return

    # --- P Matrix Analysis ---
    print(f"\n--- P Matrix (Final) ---")
    print(f"{'Circuit':<25} {'nQ':>3} {'max_mean':>9} {'max_min':>8} {'entropy_ratio':>13} {'top2_gap':>9} {'eff_supp':>9}")
    print("-" * 80)
    for r in results:
        print(f"{r['circuit']:<25} {r['num_logical']:>3} "
              f"{r.get('max_mean', 0):>9.4f} {r.get('max_min', 0):>8.4f} "
              f"{r.get('entropy_ratio', 0):>13.4f} "
              f"{r.get('top2_gap_mean', 0):>9.4f} {r.get('effective_support_mean', 0):>9.1f}")

    # Averages
    avg_max = np.mean([r.get('max_mean', 0) for r in results])
    avg_entropy = np.mean([r.get('entropy_ratio', 0) for r in results])
    avg_gap = np.mean([r.get('top2_gap_mean', 0) for r in results])
    avg_supp = np.mean([r.get('effective_support_mean', 0) for r in results])
    print(f"{'AVERAGE':<25} {'':>3} {avg_max:>9.4f} {'':>8} {avg_entropy:>13.4f} {avg_gap:>9.4f} {avg_supp:>9.1f}")

    # --- P without refinement ---
    if "P_no_refine_max_mean" in results[0]:
        print(f"\n--- P Matrix (Without Refinement, raw Sinkhorn on S) ---")
        print(f"{'Circuit':<25} {'nQ':>3} {'max_mean':>9} {'max_min':>8} {'entropy_ratio':>13} {'top2_gap':>9} {'eff_supp':>9}")
        print("-" * 80)
        for r in results:
            print(f"{r['circuit']:<25} {r['num_logical']:>3} "
                  f"{r.get('P_no_refine_max_mean', 0):>9.4f} {r.get('P_no_refine_max_min', 0):>8.4f} "
                  f"{r.get('P_no_refine_entropy_ratio', 0):>13.4f} "
                  f"{r.get('P_no_refine_top2_gap_mean', 0):>9.4f} {r.get('P_no_refine_effective_support_mean', 0):>9.1f}")

    # --- S Matrix Row Analysis ---
    print(f"\n--- Embedding Distinguishability (cosine similarity between logical qubit rows) ---")
    stages = ["gnn_C", "xattn_C", "score_Q", "S_rows"]
    if "S_norm_rows_cos_sim_mean" in results[0]:
        stages += ["S_norm_rows", "S_refined_rows"]

    print(f"{'Circuit':<20} ", end="")
    for s in stages:
        short = s.replace("_rows", "").replace("_C", "")
        print(f"{'cos(' + short + ')':>12}", end="")
    print()
    print("-" * (20 + 12 * len(stages)))

    for r in results:
        print(f"{r['circuit']:<20} ", end="")
        for s in stages:
            val = r.get(f"{s}_cos_sim_mean", float("nan"))
            print(f"{val:>12.4f}", end="")
        print()

    # Averages
    print(f"{'AVERAGE':<20} ", end="")
    for s in stages:
        vals = [r.get(f"{s}_cos_sim_mean", float("nan")) for r in results]
        vals = [v for v in vals if not np.isnan(v)]
        avg = np.mean(vals) if vals else float("nan")
        print(f"{avg:>12.4f}", end="")
    print()

    # --- High similarity ratio ---
    print(f"\n--- High Similarity Ratio (fraction of pairs with cos_sim > 0.95) ---")
    print(f"{'Circuit':<20} ", end="")
    for s in stages:
        short = s.replace("_rows", "").replace("_C", "")
        print(f"{'hi(' + short + ')':>12}", end="")
    print()
    print("-" * (20 + 12 * len(stages)))

    for r in results:
        print(f"{r['circuit']:<20} ", end="")
        for s in stages:
            val = r.get(f"{s}_high_sim_ratio", float("nan"))
            print(f"{val:>12.3f}", end="")
        print()

    # --- L2 distances ---
    print(f"\n--- L2 Distance (mean pairwise) ---")
    print(f"{'Circuit':<20} ", end="")
    for s in stages:
        short = s.replace("_rows", "").replace("_C", "")
        print(f"{'L2(' + short + ')':>12}", end="")
    print()
    print("-" * (20 + 12 * len(stages)))

    for r in results:
        print(f"{r['circuit']:<20} ", end="")
        for s in stages:
            val = r.get(f"{s}_l2_dist_mean", float("nan"))
            print(f"{val:>12.4f}", end="")
        print()

    # --- S matrix statistics ---
    print(f"\n--- S Matrix Statistics ---")
    print(f"{'Circuit':<25} {'S_mean':>10} {'S_std':>10} {'S_range':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['circuit']:<25} {r.get('S_mean', 0):>10.3f} {r.get('S_std', 0):>10.3f} {r.get('S_range', 0):>10.3f}")

    # --- Refinement iteration tracking ---
    if "refine_iter0_max_mean" in results[0]:
        print(f"\n--- Refinement Iteration Tracking (P max_mean per iteration) ---")
        T = max(int(k.split("_")[1].replace("iter", ""))
                for k in results[0] if k.startswith("refine_iter") and "max_mean" in k) + 1
        print(f"{'Circuit':<20} ", end="")
        for t in range(T):
            print(f"{'iter' + str(t):>10}", end="")
        print(f"{'final':>10}")
        print("-" * (20 + 10 * (T + 1)))

        for r in results:
            print(f"{r['circuit']:<20} ", end="")
            for t in range(T):
                val = r.get(f"refine_iter{t}_max_mean", float("nan"))
                print(f"{val:>10.4f}", end="")
            print(f"{r.get('max_mean', 0):>10.4f}")

        # Feedback magnitude
        print(f"\n--- Refinement Feedback Magnitude (mean abs) ---")
        print(f"{'Circuit':<20} ", end="")
        for t in range(T):
            print(f"{'iter' + str(t):>10}", end="")
        print()
        print("-" * (20 + 10 * T))

        for r in results:
            print(f"{r['circuit']:<20} ", end="")
            for t in range(T):
                val = r.get(f"refine_iter{t}_feedback_mean", float("nan"))
                print(f"{val:>10.4f}", end="")
            print()

    # --- Dead dimensions ---
    print(f"\n--- Dead Dimensions (std < 1e-6 across rows) ---")
    print(f"{'Circuit':<20} ", end="")
    for s in stages:
        short = s.replace("_rows", "").replace("_C", "")
        print(f"{'dead(' + short + ')':>14}", end="")
    print()
    print("-" * (20 + 14 * len(stages)))

    for r in results:
        print(f"{r['circuit']:<20} ", end="")
        for s in stages:
            val = r.get(f"{s}_dead_dims", float("nan"))
            print(f"{val:>14.0f}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze P/S matrix internals")
    parser.add_argument("--run-dir", required=True, help="Stage 2 run directory")
    parser.add_argument("--backend", default="FakeToronto", help="Backend name")
    parser.add_argument("--circuits", nargs="*", default=None, help="Circuit names (default: all benchmarks)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"

    cfg = load_config(str(config_path))

    # Configure HW features
    hw_node_dim = cfg.model.hardware_gnn.node_input_dim
    configure_hw_features(
        include_t1_t2=(hw_node_dim >= 7),
        exclude_degree=getattr(cfg.model.hardware_gnn, "exclude_degree", False),
    )

    backend = get_backend(args.backend)
    hw_graph = build_hardware_graph(backend)

    device = torch.device("cpu")  # CPU for analysis (no GPU needed)

    circuits = args.circuits or BENCHMARK_CIRCUITS

    # Analyze both checkpoints
    for ckpt_name in ["best", "final"]:
        ckpt_path = run_dir / "checkpoints" / f"{ckpt_name}.pt"
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            continue

        model = GraphQMap.from_config(cfg)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        # Print refine_lambda if applicable
        if model.refine_lambda is not None:
            print(f"\n[{ckpt_name}] refine_lambda = {model.refine_lambda.item():.6f}")

        label = f"{run_dir.name} | {ckpt_name}.pt | {args.backend} | tau={getattr(cfg.sinkhorn, 'tau_min', 0.05)}"
        results = run_analysis(model, cfg, backend, hw_graph, device, circuits)
        print_results(results, label)


if __name__ == "__main__":
    main()
