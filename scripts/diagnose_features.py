"""Diagnose circuit node feature quality before training.

Measures feature degeneracy metrics (effective dimensionality, cosine similarity,
column correlation, dead columns) to verify that a feature configuration provides
sufficient qubit-level differentiation for the GNN.

Usage:
    # Diagnose current config features
    python scripts/diagnose_features.py --config configs/base.yaml

    # Compare specific feature sets
    python scripts/diagnose_features.py --features gate_count two_qubit_gate_count degree depth_participation
    python scripts/diagnose_features.py --features gate_count two_qubit_gate_count weighted_degree --rwpe-k 4

    # Sample more circuits for robust statistics
    python scripts/diagnose_features.py --config configs/base.yaml --num-samples 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.circuit_graph import (
    DEFAULT_NODE_FEATURES,
    build_circuit_graph,
    extract_circuit_features,
    load_circuit,
)


def diagnose_features(
    qasm_files: list[Path],
    node_feature_names: list[str],
    rwpe_k: int = 0,
) -> dict[str, float]:
    """Compute feature quality metrics across a set of circuits.

    Returns dict with:
        - effective_dim: average effective dimensionality (SVD)
        - total_dim: total feature dimensions
        - mean_cosine_sim: average pairwise cosine similarity between nodes
        - min_cosine_sim: minimum pairwise cosine similarity
        - dead_columns: average number of zero-variance columns
        - correlations: dict of column-pair correlation coefficients
    """
    total_dim = len(node_feature_names) + rwpe_k
    eff_dims = []
    cosine_sims = []
    min_cosine_sims = []
    dead_col_counts = []
    all_corrs: dict[str, list[float]] = {}

    for qf in qasm_files:
        try:
            circuit = load_circuit(qf)
            if circuit.num_qubits < 2:
                continue
            graph = build_circuit_graph(
                circuit, node_feature_names=node_feature_names, rwpe_k=rwpe_k,
            )
        except Exception:
            continue

        x = graph.x  # (n, d)
        n, d = x.shape

        # Dead columns (zero variance after z-score)
        col_std = x.std(dim=0)
        dead = (col_std < 1e-6).sum().item()
        dead_col_counts.append(dead)

        # Effective dimensionality via SVD
        if n > 1:
            U, S, V = torch.svd(x)
            threshold = S[0] * 0.01
            eff = (S > threshold).sum().item()
            eff_dims.append(eff)

            # Pairwise cosine similarity
            x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
            cos = x_norm @ x_norm.T
            mask = ~torch.eye(n, dtype=torch.bool)
            cosine_sims.append(cos[mask].mean().item())
            min_cosine_sims.append(cos[mask].min().item())

        # Column correlations (for named features, not RWPE)
        num_named = len(node_feature_names)
        if n > 2 and num_named > 1:
            for i in range(num_named):
                for j in range(i + 1, num_named):
                    pair_key = f"{node_feature_names[i]} ↔ {node_feature_names[j]}"
                    try:
                        corr = torch.corrcoef(torch.stack([x[:, i], x[:, j]]))[0, 1].item()
                        if not np.isnan(corr):
                            all_corrs.setdefault(pair_key, []).append(abs(corr))
                    except Exception:
                        pass

    results = {
        "total_dim": total_dim,
        "effective_dim": np.mean(eff_dims) if eff_dims else 0.0,
        "mean_cosine_sim": np.mean(cosine_sims) if cosine_sims else 0.0,
        "min_cosine_sim": np.mean(min_cosine_sims) if min_cosine_sims else 0.0,
        "dead_columns": np.mean(dead_col_counts) if dead_col_counts else 0.0,
        "num_circuits": len(eff_dims),
    }

    # Average correlations
    avg_corrs = {k: np.mean(v) for k, v in sorted(all_corrs.items())}
    results["correlations"] = avg_corrs

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose circuit node feature quality")
    parser.add_argument("--config", type=str, help="YAML config to read feature list from")
    parser.add_argument("--features", nargs="+", help="Feature names (overrides config)")
    parser.add_argument("--rwpe-k", type=int, default=None, help="RWPE steps (overrides config)")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of circuits to sample")
    parser.add_argument("--data-root", type=str, default="data/circuits", help="Data root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Determine feature config
    node_feature_names = None
    rwpe_k = 0

    if args.config:
        from configs.config_loader import load_config
        cfg = load_config(args.config)
        node_feature_names = getattr(cfg.model.circuit_gnn, "node_features", None)
        rwpe_k = getattr(cfg.model.circuit_gnn, "rwpe_k", 0)

    if args.features:
        node_feature_names = args.features
    if args.rwpe_k is not None:
        rwpe_k = args.rwpe_k

    if node_feature_names is None:
        node_feature_names = list(DEFAULT_NODE_FEATURES)

    # Collect QASM files
    data_root = Path(args.data_root)
    qasm_root = data_root / "qasm"
    all_qasm = []
    for source_dir in sorted(qasm_root.iterdir()):
        if source_dir.is_dir() and source_dir.name != "benchmarks":
            all_qasm.extend(sorted(source_dir.glob("*.qasm")))

    # Also include benchmarks separately
    benchmark_qasm = sorted((qasm_root / "benchmarks").glob("*.qasm")) if (qasm_root / "benchmarks").exists() else []

    # Sample
    rng = np.random.RandomState(args.seed)
    n_train = min(args.num_samples, len(all_qasm))
    sampled = list(rng.choice(all_qasm, size=n_train, replace=False))

    print(f"=== Circuit Node Feature Diagnostics ===")
    print(f"Features: {node_feature_names}")
    print(f"RWPE k: {rwpe_k}")
    print(f"Total dim: {len(node_feature_names) + rwpe_k}")
    print()

    # Diagnose training circuits
    print(f"--- Training circuits (sampled {n_train}/{len(all_qasm)}) ---")
    results = diagnose_features(sampled, node_feature_names, rwpe_k)
    _print_results(results, node_feature_names)

    # Diagnose benchmark circuits
    if benchmark_qasm:
        print(f"\n--- Benchmark circuits ({len(benchmark_qasm)}) ---")
        bench_results = diagnose_features(benchmark_qasm, node_feature_names, rwpe_k)
        _print_results(bench_results, node_feature_names)


def _print_results(results: dict, feature_names: list[str]) -> None:
    """Pretty-print diagnostic results."""
    print(f"  Circuits analyzed:     {results['num_circuits']}")
    print(f"  Effective dim:         {results['effective_dim']:.2f} / {results['total_dim']}")
    print(f"  Dead columns (avg):    {results['dead_columns']:.2f} / {results['total_dim']}")
    print(f"  Mean cosine sim:       {results['mean_cosine_sim']:.4f}")
    print(f"  Min cosine sim (avg):  {results['min_cosine_sim']:.4f}")

    corrs = results.get("correlations", {})
    if corrs:
        print(f"  Column correlations (|r| avg):")
        for pair, r in corrs.items():
            flag = " *** HIGH" if r > 0.9 else ""
            print(f"    {pair}: {r:.4f}{flag}")


if __name__ == "__main__":
    main()
