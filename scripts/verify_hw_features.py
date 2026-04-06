"""Verify hardware features are correctly loaded and carry independent information.

Uses the actual build_hardware_graph() / build_hardware_graph_from_synthetic()
code paths — same as training/evaluation.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from data.hardware_graph import (
    build_hardware_graph,
    build_hardware_graph_from_synthetic,
    get_backend,
    is_synthetic_backend,
    BACKEND_REGISTRY,
)

# All training + test backends
SYNTHETIC_BACKENDS = [
    "queko_aspen4", "queko_tokyo", "queko_rochester", "queko_sycamore", "mlqd_grid5x5",
]
TEST_BACKENDS = ["toronto", "brooklyn", "torino"]

NODE_FEATURE_NAMES = [
    "readout_err", "sq_err", "degree", "t1_cx_ratio", "t2_cx_ratio",  # z-scored
    "t2_t1_ratio",  # raw
]
EDGE_FEATURE_NAMES = [
    "2q_error",            # z-scored
    "edge_coherence",      # raw
]


def load_all_backends():
    """Load hardware graphs from all backends."""
    results = []

    # Qiskit FakeBackendV2
    for name in sorted(BACKEND_REGISTRY.keys()):
        try:
            backend = get_backend(name)
            data = build_hardware_graph(backend)
            results.append({"name": name, "data": data, "type": "qiskit",
                            "is_test": name in TEST_BACKENDS})
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")

    # Synthetic
    for name in SYNTHETIC_BACKENDS:
        try:
            data = build_hardware_graph_from_synthetic(name)
            results.append({"name": name, "data": data, "type": "synthetic",
                            "is_test": False})
        except Exception as e:
            print(f"  ERROR loading synthetic {name}: {e}")

    return results


def check_completeness(results):
    """Check for NaN, Inf, all-zero features across all backends."""
    print("=" * 90)
    print("1. DATA COMPLETENESS CHECK")
    print("=" * 90)

    issues = []

    for r in results:
        name = r["name"]
        data = r["data"]
        x = data.x  # (num_qubits, 6)
        ea = data.edge_attr  # (num_edges*2, 2)

        # Node features
        for col_idx, feat_name in enumerate(NODE_FEATURE_NAMES):
            col = x[:, col_idx]
            if torch.isnan(col).any():
                issues.append(f"  {name}: {feat_name} has NaN ({torch.isnan(col).sum()} qubits)")
            if torch.isinf(col).any():
                issues.append(f"  {name}: {feat_name} has Inf ({torch.isinf(col).sum()} qubits)")
            if (col == 0).all():
                issues.append(f"  {name}: {feat_name} ALL ZERO ({len(col)} qubits)")

        # Edge features
        for col_idx, feat_name in enumerate(EDGE_FEATURE_NAMES):
            col = ea[:, col_idx]
            if torch.isnan(col).any():
                issues.append(f"  {name}: {feat_name} has NaN ({torch.isnan(col).sum()} edges)")
            if torch.isinf(col).any():
                issues.append(f"  {name}: {feat_name} has Inf ({torch.isinf(col).sum()} edges)")
            if (col == 0).all():
                issues.append(f"  {name}: {feat_name} ALL ZERO ({len(col)} edges)")

    if issues:
        print(f"\n  Found {len(issues)} issues:\n")
        for issue in issues:
            print(issue)
    else:
        print(f"\n  All {len(results)} backends: NO issues (no NaN, Inf, or all-zero features)")

    # Summary table
    print(f"\n  {'Backend':<25} {'Type':<10} {'Q':>4} {'E':>4} "
          f"{'Node shape':>12} {'Edge shape':>12}")
    print("  " + "-" * 75)
    for r in results:
        d = r["data"]
        marker = " [TEST]" if r["is_test"] else ""
        print(f"  {r['name']:<25} {r['type']:<10} {d.x.shape[0]:>4} "
              f"{d.edge_attr.shape[0]:>4} "
              f"{str(tuple(d.x.shape)):>12} {str(tuple(d.edge_attr.shape)):>12}{marker}")


def check_distributions(results):
    """Check per-backend feature distributions."""
    print("\n" + "=" * 90)
    print("2. PER-BACKEND FEATURE DISTRIBUTIONS")
    print("=" * 90)

    # Node features
    print(f"\n── Node Features (std within backend, 0 = no variance) ──\n")
    print(f"  {'Backend':<22} {'Q':>3}", end="")
    for name in NODE_FEATURE_NAMES:
        print(f" {name[:11]:>12}", end="")
    print()
    print("  " + "-" * (27 + 13 * len(NODE_FEATURE_NAMES)))

    low_var_issues = []
    for r in results:
        x = r["data"].x
        print(f"  {r['name']:<22} {x.shape[0]:>3}", end="")
        for col_idx, feat_name in enumerate(NODE_FEATURE_NAMES):
            col = x[:, col_idx]
            std = col.std().item()
            # Flag if std is very low for z-scored features (should be ~1.0)
            if col_idx < 5 and abs(std - 1.0) > 0.3 and x.shape[0] > 2:
                marker = " !"
                low_var_issues.append(f"{r['name']}: {feat_name} std={std:.3f}")
            elif col_idx >= 5 and std < 0.01 and x.shape[0] > 2:
                marker = " !"
                low_var_issues.append(f"{r['name']}: {feat_name} std={std:.4f}")
            else:
                marker = "  "
            print(f" {std:>10.4f}{marker}", end="")
        print()

    if low_var_issues:
        print(f"\n  Flagged low-variance issues ({len(low_var_issues)}):")
        for issue in low_var_issues:
            print(f"    {issue}")

    # Edge features
    print(f"\n── Edge Features (std within backend) ──\n")
    print(f"  {'Backend':<22} {'E':>4}", end="")
    for name in EDGE_FEATURE_NAMES:
        print(f" {name[:14]:>15}", end="")
    print()
    print("  " + "-" * (28 + 16 * len(EDGE_FEATURE_NAMES)))

    for r in results:
        ea = r["data"].edge_attr
        print(f"  {r['name']:<22} {ea.shape[0]:>4}", end="")
        for col_idx, feat_name in enumerate(EDGE_FEATURE_NAMES):
            col = ea[:, col_idx]
            std = col.std().item()
            print(f" {std:>14.6f}", end="")
        print()


def check_correlations(results):
    """Check feature independence using pooled correlation analysis."""
    print("\n" + "=" * 90)
    print("3. FEATURE CORRELATIONS (pooled across all backends)")
    print("=" * 90)

    # Pool node features (already normalized per-backend by build_hardware_graph)
    all_node = []
    for r in results:
        all_node.append(r["data"].x.numpy())
    pooled_node = np.concatenate(all_node, axis=0)
    n_qubits = pooled_node.shape[0]

    print(f"\n── Node Feature Correlation Matrix (n={n_qubits} qubits) ──\n")

    # Header
    short = [n[:11] for n in NODE_FEATURE_NAMES]
    print(f"  {'':>14}", end="")
    for s in short:
        print(f" {s:>12}", end="")
    print()

    corr = np.corrcoef(pooled_node.T)
    for i, name in enumerate(NODE_FEATURE_NAMES):
        print(f"  {name[:14]:<14}", end="")
        for j in range(len(NODE_FEATURE_NAMES)):
            if i == j:
                print(f"     {'—':>7}", end="")
            else:
                val = corr[i, j]
                flag = " **" if abs(val) > 0.9 else " * " if abs(val) > 0.7 else "   "
                print(f" {val:>9.3f}{flag}", end="")
        print()

    print("\n  ** |r| > 0.9 (redundant)  * |r| > 0.7 (notable)")
    print("\n  High correlations (|r| > 0.7):")
    found = False
    for i in range(len(NODE_FEATURE_NAMES)):
        for j in range(i + 1, len(NODE_FEATURE_NAMES)):
            if abs(corr[i, j]) > 0.7:
                print(f"    {NODE_FEATURE_NAMES[i]} ↔ {NODE_FEATURE_NAMES[j]}: r = {corr[i, j]:.4f}")
                found = True
    if not found:
        print("    (none)")

    # Pool edge features
    all_edge = []
    for r in results:
        all_edge.append(r["data"].edge_attr.numpy())
    pooled_edge = np.concatenate(all_edge, axis=0)
    n_edges = pooled_edge.shape[0]

    print(f"\n── Edge Feature Correlation Matrix (n={n_edges} edges) ──\n")

    print(f"  {'':>18}", end="")
    for name in EDGE_FEATURE_NAMES:
        print(f" {name[:14]:>15}", end="")
    print()

    ecorr = np.corrcoef(pooled_edge.T)
    for i, name in enumerate(EDGE_FEATURE_NAMES):
        print(f"  {name[:18]:<18}", end="")
        for j in range(len(EDGE_FEATURE_NAMES)):
            if i == j:
                print(f"          {'—':>5}", end="")
            else:
                print(f" {ecorr[i, j]:>14.3f}", end="")
        print()

    print(f"\n  Correlation: r = {ecorr[0, 1]:.4f}")


def check_raw_feature_ranges(results):
    """Verify raw (non-z-scored) features have physically meaningful values."""
    print("\n" + "=" * 90)
    print("4. RAW FEATURE VALUE RANGES (non-z-scored features)")
    print("=" * 90)

    # T2/T1 ratio (node feature index 5)
    all_t2t1 = []
    for r in results:
        all_t2t1.append(r["data"].x[:, 5].numpy())
    all_t2t1 = np.concatenate(all_t2t1)

    print(f"\n  T2/T1 ratio (node, col 5):")
    print(f"    Range: [{all_t2t1.min():.4f}, {all_t2t1.max():.4f}]")
    print(f"    Mean:  {all_t2t1.mean():.4f}")
    print(f"    Std:   {all_t2t1.std():.4f}")
    print(f"    Theoretical: T2 ≤ 2*T1 → ratio ∈ (0, 2]")
    pct_valid = (all_t2t1 <= 2.0 + 1e-6).mean() * 100
    print(f"    % within (0, 2]: {pct_valid:.1f}%")

    # Edge coherence ratio (edge feature index 1)
    all_ecr = []
    for r in results:
        all_ecr.append(r["data"].edge_attr[:, 1].numpy())
    all_ecr = np.concatenate(all_ecr)

    print(f"\n  Edge coherence ratio (edge, col 1):")
    print(f"    Range: [{all_ecr.min():.6f}, {all_ecr.max():.6f}]")
    print(f"    Mean:  {all_ecr.mean():.6f}")
    print(f"    Std:   {all_ecr.std():.6f}")
    print(f"    Physical: cx_duration / min(T1,T2), smaller = better")
    print(f"    % zero: {(all_ecr == 0).mean() * 100:.1f}%")


def main():
    print("Loading all backends (Qiskit + Synthetic)...\n")
    results = load_all_backends()
    print(f"\nLoaded {len(results)} backends successfully.\n")

    check_completeness(results)
    check_distributions(results)
    check_correlations(results)
    check_raw_feature_ranges(results)

    print("\n" + "=" * 90)
    print("VERIFICATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
