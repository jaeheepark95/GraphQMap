#!/usr/bin/env python3
"""Analyze QAP fidelity loss normalization across circuit sizes and backends.

Computes the raw scale of the edge term tr(Ã_c P C_eff P^T) and readout term
Σ ε_r(π(i)) for circuits of different sizes on different backends, and evaluates
whether (l + |E|) normalization is appropriate or if separate normalization
per term would be better.

Usage:
    python scripts/analyze_qap_normalization.py [--num-circuits N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.circuit_graph import extract_circuit_features, load_circuit
from data.hardware_graph import (
    get_backend,
    precompute_c_eff,
    precompute_grama_single_qubit_costs,
)


# ── Backend setup ──────────────────────────────────────────────────────────

TEST_BACKENDS = ["toronto", "rochester", "washington"]


def load_backend_data(backend_name: str) -> dict:
    """Load C_eff and readout errors for a backend.

    Returns dict with keys: c_eff, readout_error, num_qubits, and summary stats.
    """
    backend = get_backend(backend_name)
    c_eff = precompute_c_eff(backend)
    sq_costs = precompute_grama_single_qubit_costs(backend)
    h = c_eff.shape[0]

    # C_eff stats (exclude diagonal)
    mask = ~np.eye(h, dtype=bool)
    c_eff_offdiag = c_eff[mask]

    # Adjacent vs non-adjacent C_eff
    adjacent_mask = mask & (c_eff < 0.1) & (c_eff > 0)  # adjacent: raw ε₂ < 0.1
    non_adj_mask = mask & (c_eff >= 0.1)

    return {
        "c_eff": c_eff,
        "s_read": sq_costs["s_read"],
        "s_gate": sq_costs["s_gate"],
        "num_qubits": h,
        "c_eff_mean": float(np.mean(c_eff_offdiag)),
        "c_eff_median": float(np.median(c_eff_offdiag)),
        "c_eff_std": float(np.std(c_eff_offdiag)),
        "c_eff_min": float(np.min(c_eff_offdiag[c_eff_offdiag > 0])) if np.any(c_eff_offdiag > 0) else 0.0,
        "c_eff_max": float(np.max(c_eff_offdiag)),
        "c_eff_adj_mean": float(np.mean(c_eff[adjacent_mask])) if np.any(adjacent_mask) else 0.0,
        "c_eff_nonadj_mean": float(np.mean(c_eff[non_adj_mask])) if np.any(non_adj_mask) else 0.0,
        "readout_mean": float(np.mean(sq_costs["s_read"])),
        "readout_std": float(np.std(sq_costs["s_read"])),
        "readout_min": float(np.min(sq_costs["s_read"])),
        "readout_max": float(np.max(sq_costs["s_read"])),
    }


# ── Circuit analysis ───────────────────────────────────────────────────────

def analyze_circuit(qasm_path: Path) -> dict | None:
    """Extract circuit structure: l, |E|, total gate weight, edge weight distribution.

    Returns None if circuit fails to load.
    """
    try:
        circuit = load_circuit(qasm_path)
    except Exception:
        return None

    if circuit.num_qubits < 2:
        return None

    try:
        feats = extract_circuit_features(circuit)
    except Exception:
        return None

    l = feats["num_qubits"]
    edge_list = feats["edge_list"]
    edge_features = feats["edge_features"]  # (|E|, 5)
    num_edges = len(edge_list)

    if num_edges == 0:
        return None

    # Edge weights = interaction counts (column 0 of edge_features)
    edge_weights = edge_features[:, 0].numpy() if edge_features.shape[0] > 0 else np.array([])
    total_gate_weight = float(np.sum(edge_weights))
    mean_gate_weight = float(np.mean(edge_weights)) if len(edge_weights) > 0 else 0.0
    max_gate_weight = float(np.max(edge_weights)) if len(edge_weights) > 0 else 0.0

    return {
        "filename": qasm_path.name,
        "l": l,
        "num_edges": num_edges,
        "total_gate_weight": total_gate_weight,
        "mean_gate_weight": mean_gate_weight,
        "max_gate_weight": max_gate_weight,
        "edge_list": edge_list,
        "edge_weights": edge_weights,
    }


# ── Scale estimation ───────────────────────────────────────────────────────

def estimate_qap_scales(
    circuit_info: dict,
    backend_data: dict,
    rng: np.random.Generator,
) -> dict:
    """Estimate the raw magnitude of edge term and readout term.

    For the edge term, we compute:
        Σ_{(i,i') ∈ E_c} g_{ii'} · C_eff(π(i), π(i'))
    using a random mapping π (random subset of h qubits assigned to l logical qubits).

    For the readout term:
        Σ_i ε_r(π(i))

    We repeat with multiple random mappings and take the mean.

    Returns dict with per-sample estimates.
    """
    l = circuit_info["l"]
    h = backend_data["num_qubits"]
    c_eff = backend_data["c_eff"]
    s_read = backend_data["s_read"]
    edge_list = circuit_info["edge_list"]
    edge_weights = circuit_info["edge_weights"]
    num_edges = circuit_info["num_edges"]

    if l > h:
        return None

    num_trials = 50
    edge_terms = []
    readout_terms = []

    for _ in range(num_trials):
        # Random mapping: pick l distinct physical qubits
        mapping = rng.choice(h, size=l, replace=False)

        # Edge term: Σ g_{ii'} · C_eff(π(i), π(i'))
        edge_cost = 0.0
        for (i, j), w in zip(edge_list, edge_weights):
            edge_cost += w * c_eff[mapping[i], mapping[j]]
        edge_terms.append(edge_cost)

        # Readout term: Σ ε_r(π(i))
        readout_cost = float(np.sum(s_read[mapping]))
        readout_terms.append(readout_cost)

    edge_mean = float(np.mean(edge_terms))
    readout_mean = float(np.mean(readout_terms))

    # Also compute "optimal adjacent" edge term estimate:
    # If all circuit edges were mapped to adjacent HW edges
    adj_c_eff_mean = backend_data["c_eff_adj_mean"]
    edge_term_optimal = circuit_info["total_gate_weight"] * adj_c_eff_mean

    return {
        "edge_term_random": edge_mean,
        "edge_term_std": float(np.std(edge_terms)),
        "readout_term_random": readout_mean,
        "readout_term_std": float(np.std(readout_terms)),
        "edge_term_optimal_adj": edge_term_optimal,
        "ratio_edge_readout": edge_mean / max(readout_mean, 1e-12),
        # Normalization denominators
        "denom_l_plus_E": l + num_edges,
        "norm_combined": (edge_mean + readout_mean) / (l + num_edges),
        "norm_edge_only": edge_mean / max(num_edges, 1),
        "norm_readout_only": readout_mean / l,
        # Alternative: separate normalization then sum
        "norm_separate_sum": edge_mean / max(num_edges, 1) + readout_mean / l,
        # Alternative: normalize edge by total_gate_weight
        "norm_edge_by_gate_weight": edge_mean / max(circuit_info["total_gate_weight"], 1),
        # Alternative: normalize edge by |E| and readout by l, then weighted sum
        "norm_weighted_half": 0.5 * edge_mean / max(num_edges, 1) + 0.5 * readout_mean / l,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze QAP loss normalization")
    parser.add_argument("--num-circuits", type=int, default=200,
                        help="Number of circuits to sample (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── 1. Load backend data ──────────────────────────────────────────────
    print("=" * 80)
    print("SECTION 1: Backend C_eff and Readout Error Statistics")
    print("=" * 80)

    backend_data = {}
    for bname in TEST_BACKENDS:
        print(f"\nLoading {bname}...")
        bd = load_backend_data(bname)
        backend_data[bname] = bd
        print(f"  {bname}: h={bd['num_qubits']}")
        print(f"    C_eff (off-diag): mean={bd['c_eff_mean']:.6f}, "
              f"median={bd['c_eff_median']:.6f}, std={bd['c_eff_std']:.6f}")
        print(f"    C_eff range: [{bd['c_eff_min']:.6f}, {bd['c_eff_max']:.6f}]")
        print(f"    C_eff adjacent mean: {bd['c_eff_adj_mean']:.6f}")
        print(f"    C_eff non-adjacent mean: {bd['c_eff_nonadj_mean']:.6f}")
        print(f"    Readout error: mean={bd['readout_mean']:.6f}, "
              f"std={bd['readout_std']:.6f}, "
              f"range=[{bd['readout_min']:.6f}, {bd['readout_max']:.6f}]")

    # ── 2. Load circuits ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 2: Circuit Size Distribution")
    print("=" * 80)

    split_path = PROJECT_ROOT / "data" / "circuits" / "splits" / "stage2_all.json"
    with open(split_path) as f:
        split_data = json.load(f)

    # Sample circuits
    if args.num_circuits < len(split_data):
        indices = rng.choice(len(split_data), size=args.num_circuits, replace=False)
        sampled = [split_data[i] for i in indices]
    else:
        sampled = split_data

    qasm_root = PROJECT_ROOT / "data" / "circuits" / "qasm"
    circuits = []
    for entry in sampled:
        source = entry["source"]
        filename = entry["file"]
        qasm_path = qasm_root / source / filename
        if not qasm_path.exists():
            continue
        info = analyze_circuit(qasm_path)
        if info is not None:
            info["source"] = source
            circuits.append(info)

    print(f"\nLoaded {len(circuits)} circuits (from {len(sampled)} sampled)")

    # Size distribution
    ls = [c["l"] for c in circuits]
    edges = [c["num_edges"] for c in circuits]
    gate_weights = [c["total_gate_weight"] for c in circuits]

    print(f"\n  Logical qubits (l):")
    print(f"    min={min(ls)}, max={max(ls)}, mean={np.mean(ls):.1f}, "
          f"median={np.median(ls):.1f}, std={np.std(ls):.1f}")
    print(f"  Edges (|E|):")
    print(f"    min={min(edges)}, max={max(edges)}, mean={np.mean(edges):.1f}, "
          f"median={np.median(edges):.1f}, std={np.std(edges):.1f}")
    print(f"  Total gate weight (Σg):")
    print(f"    min={min(gate_weights):.0f}, max={max(gate_weights):.0f}, "
          f"mean={np.mean(gate_weights):.1f}, median={np.median(gate_weights):.1f}")
    print(f"  |E| / l ratio:")
    el_ratios = [e / l for e, l in zip(edges, ls)]
    print(f"    min={min(el_ratios):.2f}, max={max(el_ratios):.2f}, "
          f"mean={np.mean(el_ratios):.2f}, median={np.median(el_ratios):.2f}")
    print(f"  Σg / |E| ratio (mean interaction count per edge):")
    ge_ratios = [g / max(e, 1) for g, e in zip(gate_weights, edges)]
    print(f"    min={min(ge_ratios):.2f}, max={max(ge_ratios):.2f}, "
          f"mean={np.mean(ge_ratios):.2f}, median={np.median(ge_ratios):.2f}")

    # Size buckets
    print("\n  Size distribution by bucket:")
    buckets = {"tiny (2-4Q)": [], "small (5-10Q)": [], "medium (11-20Q)": [],
               "large (21-50Q)": [], "xlarge (51+Q)": []}
    for c in circuits:
        l = c["l"]
        if l <= 4:
            buckets["tiny (2-4Q)"].append(c)
        elif l <= 10:
            buckets["small (5-10Q)"].append(c)
        elif l <= 20:
            buckets["medium (11-20Q)"].append(c)
        elif l <= 50:
            buckets["large (21-50Q)"].append(c)
        else:
            buckets["xlarge (51+Q)"].append(c)

    for bkt, circs in buckets.items():
        if circs:
            ls_b = [c["l"] for c in circs]
            es_b = [c["num_edges"] for c in circs]
            gs_b = [c["total_gate_weight"] for c in circs]
            print(f"    {bkt}: n={len(circs)}, l=[{min(ls_b)}-{max(ls_b)}], "
                  f"|E|=[{min(es_b)}-{max(es_b)}], "
                  f"Σg=[{min(gs_b):.0f}-{max(gs_b):.0f}]")

    # ── 3. Compute raw term scales per circuit × backend ──────────────────
    print("\n" + "=" * 80)
    print("SECTION 3: Raw Term Scales (Edge vs Readout)")
    print("=" * 80)

    all_results = []
    for bname in TEST_BACKENDS:
        bd = backend_data[bname]
        print(f"\n--- Backend: {bname} (h={bd['num_qubits']}) ---")

        for c in circuits:
            if c["l"] > bd["num_qubits"]:
                continue  # Skip circuits that don't fit

            scales = estimate_qap_scales(c, bd, rng)
            if scales is None:
                continue

            row = {
                "backend": bname,
                "h": bd["num_qubits"],
                "filename": c["filename"],
                "source": c["source"],
                "l": c["l"],
                "num_edges": c["num_edges"],
                "total_gate_weight": c["total_gate_weight"],
                **scales,
            }
            all_results.append(row)

    df = pd.DataFrame(all_results)

    # ── 4. Summary tables ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 4: Scale Analysis by Circuit Size × Backend")
    print("=" * 80)

    # Assign size bucket
    def size_bucket(l):
        if l <= 4:
            return "tiny (2-4Q)"
        elif l <= 10:
            return "small (5-10Q)"
        elif l <= 20:
            return "medium (11-20Q)"
        elif l <= 50:
            return "large (21-50Q)"
        else:
            return "xlarge (51+Q)"

    df["size_bucket"] = df["l"].apply(size_bucket)

    # Group by backend × size_bucket
    for bname in TEST_BACKENDS:
        df_b = df[df["backend"] == bname]
        if df_b.empty:
            continue

        print(f"\n--- {bname} (h={backend_data[bname]['num_qubits']}) ---")
        print(f"{'Size Bucket':<18} {'n':>4} {'l_avg':>6} {'|E|_avg':>7} "
              f"{'EdgeTerm':>10} {'ReadTerm':>10} {'E/R ratio':>10} "
              f"{'(l+|E|)':>7} {'norm_comb':>10} {'norm_sep':>10}")
        print("-" * 105)

        for bkt in ["tiny (2-4Q)", "small (5-10Q)", "medium (11-20Q)",
                     "large (21-50Q)", "xlarge (51+Q)"]:
            df_bkt = df_b[df_b["size_bucket"] == bkt]
            if df_bkt.empty:
                continue

            print(f"{bkt:<18} {len(df_bkt):>4} "
                  f"{df_bkt['l'].mean():>6.1f} "
                  f"{df_bkt['num_edges'].mean():>7.1f} "
                  f"{df_bkt['edge_term_random'].mean():>10.4f} "
                  f"{df_bkt['readout_term_random'].mean():>10.4f} "
                  f"{df_bkt['ratio_edge_readout'].mean():>10.2f} "
                  f"{df_bkt['denom_l_plus_E'].mean():>7.1f} "
                  f"{df_bkt['norm_combined'].mean():>10.6f} "
                  f"{df_bkt['norm_separate_sum'].mean():>10.6f}")

    # ── 5. Normalization comparison ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 5: Normalization Strategy Comparison")
    print("=" * 80)
    print("\nGoal: normalized loss should have SIMILAR magnitude across circuit sizes")
    print("(so that small and large circuits contribute equally to gradient updates)")

    norm_cols = [
        ("norm_combined", "/(l+|E|): current"),
        ("norm_edge_only", "edge/|E|"),
        ("norm_readout_only", "readout/l"),
        ("norm_separate_sum", "edge/|E| + readout/l"),
        ("norm_edge_by_gate_weight", "edge/Σg"),
        ("norm_weighted_half", "0.5·edge/|E| + 0.5·readout/l"),
    ]

    for bname in TEST_BACKENDS:
        df_b = df[df["backend"] == bname]
        if df_b.empty:
            continue

        print(f"\n--- {bname} (h={backend_data[bname]['num_qubits']}) ---")

        # For each normalization, compute CoV across size buckets
        # (lower CoV = more size-invariant = better normalization)
        print(f"\n{'Normalization':<30} ", end="")
        for bkt in ["tiny", "small", "medium", "large", "xlarge"]:
            print(f"{bkt:>12}", end="")
        print(f"{'CoV':>10} {'range_ratio':>12}")
        print("-" * 110)

        bkt_labels = ["tiny (2-4Q)", "small (5-10Q)", "medium (11-20Q)",
                      "large (21-50Q)", "xlarge (51+Q)"]
        bkt_short = ["tiny", "small", "medium", "large", "xlarge"]

        for col, label in norm_cols:
            means = []
            print(f"{label:<30} ", end="")
            for bkt in bkt_labels:
                df_bkt = df_b[df_b["size_bucket"] == bkt]
                if df_bkt.empty:
                    print(f"{'—':>12}", end="")
                else:
                    m = df_bkt[col].mean()
                    means.append(m)
                    print(f"{m:>12.6f}", end="")

            if len(means) >= 2:
                means_arr = np.array(means)
                cov = np.std(means_arr) / np.mean(means_arr) if np.mean(means_arr) != 0 else float("inf")
                rr = max(means) / min(means) if min(means) > 0 else float("inf")
                print(f"{cov:>10.3f} {rr:>12.2f}")
            else:
                print(f"{'—':>10} {'—':>12}")

    # ── 6. Edge term decomposition ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 6: Edge Term Decomposition")
    print("=" * 80)
    print("\nEdge term = Σ g_{ii'} · C_eff(π(i), π(i'))")
    print("= (Σg) · mean(C_eff over mapped pairs weighted by g)")
    print("\nThis decomposes into: circuit_complexity × backend_cost")
    print("  circuit_complexity ~ Σg (total 2Q gate interactions)")
    print("  backend_cost ~ mean C_eff for the mapping")

    for bname in TEST_BACKENDS:
        df_b = df[df["backend"] == bname]
        if df_b.empty:
            continue

        print(f"\n--- {bname} ---")
        print(f"{'Size Bucket':<18} {'Σg_mean':>8} {'EdgeTerm':>10} "
              f"{'eff_cost':>10} {'ReadTerm':>10} "
              f"{'edge/Σg':>10} {'readout/l':>10}")
        print("-" * 80)

        for bkt in ["tiny (2-4Q)", "small (5-10Q)", "medium (11-20Q)",
                     "large (21-50Q)", "xlarge (51+Q)"]:
            df_bkt = df_b[df_b["size_bucket"] == bkt]
            if df_bkt.empty:
                continue

            # Effective per-interaction cost = edge_term / Σg
            eff_cost = df_bkt["edge_term_random"] / df_bkt["total_gate_weight"].clip(lower=1)
            print(f"{bkt:<18} "
                  f"{df_bkt['total_gate_weight'].mean():>8.1f} "
                  f"{df_bkt['edge_term_random'].mean():>10.4f} "
                  f"{eff_cost.mean():>10.6f} "
                  f"{df_bkt['readout_term_random'].mean():>10.4f} "
                  f"{df_bkt['norm_edge_by_gate_weight'].mean():>10.6f} "
                  f"{df_bkt['norm_readout_only'].mean():>10.6f}")

    # ── 7. Concrete recommendations ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 7: Recommendations")
    print("=" * 80)
    print("""
Analysis framework:
  - Edge term scales as O(|E| · mean_g · C_eff_pair) ≈ O(l² · C_eff) for dense circuits
  - Readout term scales as O(l · mean_ε_r)
  - Ratio edge/readout grows as O(l · mean_g · C_eff / mean_ε_r)

Key questions:
  1. Does (l + |E|) normalization equalize loss across circuit sizes?
     → Check CoV across size buckets (Section 5)
  2. Does the edge term dominate the readout term?
     → Check E/R ratio (Section 4)
  3. Would separate normalization per term be better?
     → Compare CoV of 'edge/|E| + readout/l' vs '/(l+|E|)' (Section 5)

Alternative normalizations to consider:
  A. Current: (edge + readout) / (l + |E|)
     Pro: single denominator, simple
     Con: mixes two terms with different scaling behaviors

  B. Separate: edge/|E| + readout/l
     Pro: each term normalized by its natural scale
     Con: relative weight of terms changes with circuit structure

  C. Edge by gate weight: edge/Σg + readout/l
     Pro: normalizes by actual circuit complexity, not just topology
     Con: circuits with high gate repetition on few edges get low weight

  D. Weighted separate: α·edge/|E| + (1-α)·readout/l
     Pro: explicit control over term balance
     Con: introduces another hyperparameter

Look at the CoV and range_ratio columns in Section 5 to determine which
normalization produces the most size-invariant loss values.
""")

    # ── 8. Save detailed results ──────────────────────────────────────────
    output_path = PROJECT_ROOT / "scripts" / "qap_normalization_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
