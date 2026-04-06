"""Circuit graph feature analysis: completeness, distribution, correlation, normalization.

Applies the same 3-step framework used for hardware graph analysis
(verify_hw_features.py / analyze_edge_asymmetry.py) to circuit graph features.

7 phases:
  1. Data completeness (NaN, Inf, all-zero, constant)
  2. Raw statistics before z-score
  3. Within-circuit CoV (coefficient of variation)
  4. Correlation analysis (node, edge, cross)
  5. Normalization strategy comparison (all-zscore vs mixed)
  6. RWPE quality assessment
  7. Circuit size-dependent analysis

Usage:
    python scripts/analyze_circuit_features.py
    python scripts/analyze_circuit_features.py --num-samples 1000
    python scripts/analyze_circuit_features.py --full    # all circuits, no sampling
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.circuit_graph import compute_rwpe
from data.normalization import zscore_normalize

ALL_NODE_FEATURES = [
    "gate_count",
    "two_qubit_gate_count",
    "degree",
    "depth_participation",
    "weighted_degree",
    "single_qubit_gate_ratio",
    "critical_path_fraction",
    "interaction_entropy",
]

CURRENT_FEATURES = [
    "gate_count",
    "two_qubit_gate_count",
    "single_qubit_gate_ratio",
    "critical_path_fraction",
]

EDGE_FEATURE_NAMES = [
    "interaction_count",
    "earliest_interaction",
    "latest_interaction",
    "interaction_span",
    "interaction_density",
]

SIZE_BUCKETS = [
    ("tiny (2-3Q)", 2, 3),
    ("small (4-5Q)", 4, 5),
    ("medium (6-10Q)", 6, 10),
    ("large (11-20Q)", 11, 20),
    ("xlarge (21Q+)", 21, 9999),
]


def _get_size_bucket(nq: int) -> str:
    for label, lo, hi in SIZE_BUCKETS:
        if lo <= nq <= hi:
            return label
    return "unknown"


# ─────────────────────────────────────────────────────────────
# Data loading (from cache, same path as benchmark_feature_analysis.py)
# ─────────────────────────────────────────────────────────────

def _compute_derived_features(
    nfd: dict[str, list[float]],
    edge_list: list[tuple[int, int]],
    edge_features: torch.Tensor,
    num_qubits: int,
) -> tuple[dict[str, list[float]], torch.Tensor]:
    """Compute new features from cached raw data (backward compatible).

    Adds interaction_entropy (node) and interaction_span, interaction_density (edge)
    if not already present in the cached data.
    """
    import math

    # interaction_entropy: compute from edge_list + interaction_count
    if "interaction_entropy" not in nfd:
        # Reconstruct per-pair interaction counts from edge features col 0
        pair_counts: dict[tuple[int, int], int] = {}
        for i, (u, v) in enumerate(edge_list):
            pair_counts[(u, v)] = int(edge_features[i, 0].item()) if edge_features.shape[0] > 0 else 0

        # Build neighbor sets
        neighbors: list[set[int]] = [set() for _ in range(num_qubits)]
        for u, v in edge_list:
            neighbors[u].add(v)
            neighbors[v].add(u)

        entropy = [0.0] * num_qubits
        for qi in range(num_qubits):
            neighbor_counts = []
            for nj in neighbors[qi]:
                key = (min(qi, nj), max(qi, nj))
                neighbor_counts.append(pair_counts.get(key, 0))
            total = sum(neighbor_counts)
            if total > 0 and len(neighbor_counts) > 1:
                probs = [c / total for c in neighbor_counts]
                entropy[qi] = -sum(p * math.log(p) for p in probs if p > 0)
        nfd["interaction_entropy"] = entropy

    # interaction_span and interaction_density: compute from existing edge cols
    if edge_features.shape[1] < 5 and edge_features.shape[0] > 0:
        earliest = edge_features[:, 1]
        latest = edge_features[:, 2]
        span = latest - earliest
        count = edge_features[:, 0]
        density = count / (span + 1e-8)
        # Where span is ~0, density = count (single-layer interaction)
        density = torch.where(span > 1e-10, density, count)
        edge_features = torch.cat([
            edge_features, span.unsqueeze(1), density.unsqueeze(1),
        ], dim=1)
    elif edge_features.shape[0] == 0 and edge_features.shape[1] < 5:
        edge_features = torch.zeros((0, 5), dtype=torch.float32)

    return nfd, edge_features


def load_from_cache(cache_path: Path) -> dict | None:
    """Load raw features from a cached .pt file."""
    try:
        data = torch.load(cache_path, weights_only=False, map_location="cpu")
    except Exception:
        return None

    if not isinstance(data, dict) or "node_features_dict" not in data:
        return None

    nfd = data["node_features_dict"]
    num_qubits = data.get("num_logical", len(next(iter(nfd.values()))))
    edge_list = data.get("edge_list", [])
    edge_features = data.get("edge_features", torch.zeros((0, 3), dtype=torch.float32))
    if isinstance(edge_features, np.ndarray):
        edge_features = torch.tensor(edge_features, dtype=torch.float32)

    # Compute derived features from cached raw data
    nfd, edge_features = _compute_derived_features(nfd, edge_list, edge_features, num_qubits)

    return {
        "name": cache_path.stem,
        "source": cache_path.parent.name,
        "num_qubits": num_qubits,
        "node_features_dict": nfd,
        "edge_list": edge_list,
        "edge_features": edge_features,
    }


def collect_circuits(
    data_root: Path,
    num_samples: int | None,
    seed: int,
    include_benchmarks: bool = True,
) -> list[dict]:
    """Collect circuits from cache, optionally sampling."""
    cache_root = data_root / "cache"

    source_files: dict[str, list[Path]] = {}
    for source_dir in sorted(cache_root.iterdir()):
        if source_dir.is_dir() and source_dir.name != "benchmarks":
            files = sorted(source_dir.glob("*.pt"))
            if files:
                source_files[source_dir.name] = files

    total = sum(len(v) for v in source_files.values())
    print(f"Training circuits (from cache):")
    for src, files in source_files.items():
        print(f"  {src}: {len(files)}")
    print(f"  Total: {total}")

    # Sample proportionally or use all
    if num_samples is not None and num_samples < total:
        rng = np.random.RandomState(seed)
        sampled_paths: list[Path] = []
        for src, files in source_files.items():
            n_src = max(1, int(num_samples * len(files) / total))
            n_src = min(n_src, len(files))
            chosen = rng.choice(len(files), size=n_src, replace=False)
            for idx in chosen:
                sampled_paths.append(files[idx])
        print(f"Sampled {len(sampled_paths)}/{total} (seed={seed})")
    else:
        sampled_paths = [f for files in source_files.values() for f in files]
        print(f"Using all {len(sampled_paths)} circuits")

    # Load
    circuits = []
    errors = 0
    for i, cp in enumerate(sampled_paths):
        if (i + 1) % 200 == 0:
            print(f"  Loading {i+1}/{len(sampled_paths)}...", flush=True)
        data = load_from_cache(cp)
        if data is not None and data["num_qubits"] >= 2:
            data["size_bucket"] = _get_size_bucket(data["num_qubits"])
            circuits.append(data)
        else:
            errors += 1

    # Benchmarks
    if include_benchmarks:
        bench_dir = cache_root / "benchmarks"
        if bench_dir.exists():
            bench_files = sorted(bench_dir.glob("*.pt"))
            for bf in bench_files:
                data = load_from_cache(bf)
                if data is not None and data["num_qubits"] >= 2:
                    data["source"] = "benchmarks"
                    data["size_bucket"] = _get_size_bucket(data["num_qubits"])
                    circuits.append(data)

    print(f"Loaded {len(circuits)} circuits ({errors} errors)\n")
    return circuits


# ─────────────────────────────────────────────────────────────
# Phase 1: Data Completeness
# ─────────────────────────────────────────────────────────────

def phase1_completeness(circuits: list[dict]) -> None:
    print("=" * 90)
    print("PHASE 1: DATA COMPLETENESS")
    print("=" * 90)

    n = len(circuits)

    # Node features
    print(f"\n-- Node Features ({n} circuits) --\n")
    print(f"  {'Feature':<28s} {'NaN':>6s} {'Inf':>6s} {'AllZero':>8s} {'Constant':>9s} {'Const%':>7s}")
    print(f"  {'-'*66}")

    for fname in ALL_NODE_FEATURES:
        nan_count = 0
        inf_count = 0
        allzero_count = 0
        constant_count = 0
        for c in circuits:
            vals = np.array(c["node_features_dict"][fname])
            if np.any(np.isnan(vals)):
                nan_count += 1
            if np.any(np.isinf(vals)):
                inf_count += 1
            if np.all(vals == 0):
                allzero_count += 1
            if np.std(vals) < 1e-10:
                constant_count += 1
        const_pct = constant_count / n * 100
        flag = " <<<" if const_pct > 50 else " !" if const_pct > 30 else ""
        print(f"  {fname:<28s} {nan_count:6d} {inf_count:6d} {allzero_count:8d} "
              f"{constant_count:9d} {const_pct:6.1f}%{flag}")

    # Edge features
    print(f"\n-- Edge Features --\n")
    print(f"  {'Feature':<28s} {'NaN':>6s} {'Inf':>6s} {'AllZero':>8s} {'Constant':>9s} {'Const%':>7s} {'NoEdge':>7s}")
    print(f"  {'-'*73}")

    no_edge_count = sum(1 for c in circuits if len(c["edge_list"]) == 0)
    circuits_with_edges = [c for c in circuits if c["edge_features"].shape[0] > 0]
    ne = len(circuits_with_edges)

    for col_idx, fname in enumerate(EDGE_FEATURE_NAMES):
        nan_count = 0
        inf_count = 0
        allzero_count = 0
        constant_count = 0
        for c in circuits_with_edges:
            ef = c["edge_features"]
            vals = ef[:, col_idx].numpy()
            if np.any(np.isnan(vals)):
                nan_count += 1
            if np.any(np.isinf(vals)):
                inf_count += 1
            if np.all(vals == 0):
                allzero_count += 1
            if np.std(vals) < 1e-10:
                constant_count += 1
        const_pct = constant_count / max(ne, 1) * 100
        print(f"  {fname:<28s} {nan_count:6d} {inf_count:6d} {allzero_count:8d} "
              f"{constant_count:9d} {const_pct:6.1f}% {no_edge_count:7d}")


# ─────────────────────────────────────────────────────────────
# Phase 2: Raw Statistics (before z-score)
# ─────────────────────────────────────────────────────────────

def phase2_raw_statistics(circuits: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("PHASE 2: RAW STATISTICS (before z-score)")
    print("=" * 90)

    # Node features
    print(f"\n-- Node Features (global statistics across all qubits) --\n")
    print(f"  {'Feature':<28s} {'Min':>8s} {'P5':>8s} {'Median':>8s} {'Mean':>8s} "
          f"{'P95':>8s} {'Max':>8s} {'Std':>8s}")
    print(f"  {'-'*88}")

    for fname in ALL_NODE_FEATURES:
        all_vals = []
        for c in circuits:
            all_vals.extend(c["node_features_dict"][fname])
        arr = np.array(all_vals)
        print(f"  {fname:<28s} {arr.min():8.3f} {np.percentile(arr, 5):8.3f} "
              f"{np.median(arr):8.3f} {arr.mean():8.3f} {np.percentile(arr, 95):8.3f} "
              f"{arr.max():8.3f} {arr.std():8.3f}")

    # Within-circuit mean/std (averaged across circuits)
    print(f"\n-- Node Features (within-circuit statistics, averaged) --\n")
    print(f"  {'Feature':<28s} {'MeanOfMean':>11s} {'MeanOfStd':>10s} {'StdOfMean':>10s} {'StdOfStd':>10s}")
    print(f"  {'-'*71}")

    for fname in ALL_NODE_FEATURES:
        means = []
        stds = []
        for c in circuits:
            vals = np.array(c["node_features_dict"][fname])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means = np.array(means)
        stds = np.array(stds)
        print(f"  {fname:<28s} {means.mean():11.4f} {stds.mean():10.4f} "
              f"{means.std():10.4f} {stds.std():10.4f}")

    # Edge features
    circuits_with_edges = [c for c in circuits if c["edge_features"].shape[0] > 0]
    if circuits_with_edges:
        print(f"\n-- Edge Features (global statistics, {len(circuits_with_edges)} circuits with edges) --\n")
        print(f"  {'Feature':<28s} {'Min':>8s} {'P5':>8s} {'Median':>8s} {'Mean':>8s} "
              f"{'P95':>8s} {'Max':>8s} {'Std':>8s}")
        print(f"  {'-'*88}")

        for col_idx, fname in enumerate(EDGE_FEATURE_NAMES):
            all_vals = []
            for c in circuits_with_edges:
                all_vals.extend(c["edge_features"][:, col_idx].numpy())
            arr = np.array(all_vals)
            print(f"  {fname:<28s} {arr.min():8.3f} {np.percentile(arr, 5):8.3f} "
                  f"{np.median(arr):8.3f} {arr.mean():8.3f} {np.percentile(arr, 95):8.3f} "
                  f"{arr.max():8.3f} {arr.std():8.3f}")


# ─────────────────────────────────────────────────────────────
# Phase 3: Within-Circuit CoV
# ─────────────────────────────────────────────────────────────

def phase3_within_circuit_cov(circuits: list[dict]) -> dict[str, dict]:
    print("\n" + "=" * 90)
    print("PHASE 3: WITHIN-CIRCUIT CoV (Coefficient of Variation = std / |mean|)")
    print("=" * 90)

    n = len(circuits)
    node_cov_data: dict[str, list[float]] = {f: [] for f in ALL_NODE_FEATURES}

    for c in circuits:
        for fname in ALL_NODE_FEATURES:
            vals = np.array(c["node_features_dict"][fname])
            mean = np.mean(vals)
            std = np.std(vals)
            if abs(mean) > 1e-10:
                cov = std / abs(mean)
            elif std < 1e-10:
                cov = 0.0  # constant zero
            else:
                cov = float("inf")  # mean~0 but variance exists
            node_cov_data[fname].append(cov)

    print(f"\n-- Node Features ({n} circuits) --\n")
    print(f"  {'Feature':<28s} {'MedCoV':>8s} {'MeanCoV':>8s} {'<0.01':>7s} {'<0.1':>7s} {'<0.5':>7s}")
    print(f"  {'-'*66}")

    for fname in ALL_NODE_FEATURES:
        covs = np.array(node_cov_data[fname])
        finite = covs[np.isfinite(covs)]
        if len(finite) == 0:
            print(f"  {fname:<28s} {'n/a':>8s}")
            continue
        med = np.median(finite)
        mean = np.mean(finite)
        pct_001 = (finite < 0.01).sum() / n * 100
        pct_01 = (finite < 0.1).sum() / n * 100
        pct_05 = (finite < 0.5).sum() / n * 100
        flag = " <<<" if pct_01 > 50 else " !" if pct_01 > 30 else ""
        print(f"  {fname:<28s} {med:8.3f} {mean:8.3f} {pct_001:6.1f}% {pct_01:6.1f}% {pct_05:6.1f}%{flag}")

    # Edge features
    circuits_with_edges = [c for c in circuits if c["edge_features"].shape[0] > 1]
    edge_cov_data: dict[str, list[float]] = {f: [] for f in EDGE_FEATURE_NAMES}

    for c in circuits_with_edges:
        ef = c["edge_features"]
        for col_idx, fname in enumerate(EDGE_FEATURE_NAMES):
            vals = ef[:, col_idx].numpy()
            mean = np.mean(vals)
            std = np.std(vals)
            if abs(mean) > 1e-10:
                cov = std / abs(mean)
            elif std < 1e-10:
                cov = 0.0
            else:
                cov = float("inf")
            edge_cov_data[fname].append(cov)

    ne = len(circuits_with_edges)
    if ne > 0:
        print(f"\n-- Edge Features ({ne} circuits with >1 edge) --\n")
        print(f"  {'Feature':<28s} {'MedCoV':>8s} {'MeanCoV':>8s} {'<0.01':>7s} {'<0.1':>7s} {'<0.5':>7s}")
        print(f"  {'-'*66}")

        for fname in EDGE_FEATURE_NAMES:
            covs = np.array(edge_cov_data[fname])
            finite = covs[np.isfinite(covs)]
            if len(finite) == 0:
                print(f"  {fname:<28s} {'n/a':>8s}")
                continue
            med = np.median(finite)
            mean = np.mean(finite)
            pct_001 = (finite < 0.01).sum() / ne * 100
            pct_01 = (finite < 0.1).sum() / ne * 100
            pct_05 = (finite < 0.5).sum() / ne * 100
            print(f"  {fname:<28s} {med:8.3f} {mean:8.3f} {pct_001:6.1f}% {pct_01:6.1f}% {pct_05:6.1f}%")

    return {"node": node_cov_data, "edge": edge_cov_data}


# ─────────────────────────────────────────────────────────────
# Phase 4: Correlation Analysis
# ─────────────────────────────────────────────────────────────

def phase4_correlations(circuits: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("PHASE 4: CORRELATION ANALYSIS")
    print("=" * 90)

    n = len(circuits)
    num_feats = len(ALL_NODE_FEATURES)

    # Per-circuit correlation, then aggregate
    corr_accum = np.zeros((num_feats, num_feats))
    corr_high_count = np.zeros((num_feats, num_feats))  # |r| > 0.9
    corr_notable_count = np.zeros((num_feats, num_feats))  # |r| > 0.7
    valid_count = 0

    for c in circuits:
        nq = c["num_qubits"]
        if nq < 3:
            continue
        # Build feature matrix (raw, pre-zscore)
        mat = np.zeros((nq, num_feats))
        for j, fname in enumerate(ALL_NODE_FEATURES):
            mat[:, j] = c["node_features_dict"][fname]

        # Skip if any column is constant
        col_stds = mat.std(axis=0)
        if np.any(col_stds < 1e-10):
            continue

        corr = np.corrcoef(mat.T)
        if np.any(np.isnan(corr)):
            continue

        corr_accum += np.abs(corr)
        corr_high_count += (np.abs(corr) > 0.9).astype(float)
        corr_notable_count += (np.abs(corr) > 0.7).astype(float)
        valid_count += 1

    if valid_count == 0:
        print("  No valid circuits for correlation analysis.")
        return

    mean_corr = corr_accum / valid_count

    # Print node feature correlation matrix
    print(f"\n-- Node Feature Mean |r| ({valid_count} valid circuits) --\n")
    short = [f[:7] for f in ALL_NODE_FEATURES]
    print(f"  {'':>16s}", end="")
    for s in short:
        print(f" {s:>8s}", end="")
    print()

    for i, fname in enumerate(ALL_NODE_FEATURES):
        print(f"  {fname[:16]:<16s}", end="")
        for j in range(num_feats):
            if i == j:
                print(f"     {'--':>3s}", end="")
            else:
                val = mean_corr[i, j]
                flag = "**" if val > 0.9 else "* " if val > 0.7 else "  "
                print(f" {val:6.3f}{flag}", end="")
        print()

    print(f"\n  ** mean |r| > 0.9 (redundant)  * mean |r| > 0.7 (notable)")

    # Print high correlation frequency
    print(f"\n-- High Correlation Frequency (|r| > 0.9 per circuit) --\n")
    print(f"  {'':>16s}", end="")
    for s in short:
        print(f" {s:>8s}", end="")
    print()

    for i, fname in enumerate(ALL_NODE_FEATURES):
        print(f"  {fname[:16]:<16s}", end="")
        for j in range(num_feats):
            if i == j:
                print(f"     {'--':>3s}", end="")
            else:
                pct = corr_high_count[i, j] / valid_count * 100
                flag = "!!" if pct > 80 else "! " if pct > 50 else "  "
                print(f" {pct:5.1f}%{flag}", end="")
        print()

    # Flagged pairs
    print(f"\n  Pairs with |r| > 0.9 in >50% of circuits:")
    found = False
    for i in range(num_feats):
        for j in range(i + 1, num_feats):
            pct = corr_high_count[i, j] / valid_count * 100
            if pct > 50:
                print(f"    {ALL_NODE_FEATURES[i]} <-> {ALL_NODE_FEATURES[j]}: "
                      f"{pct:.1f}% (mean |r| = {mean_corr[i, j]:.3f})")
                found = True
    if not found:
        print(f"    (none)")

    print(f"\n  Pairs with |r| > 0.7 in >50% of circuits:")
    found = False
    for i in range(num_feats):
        for j in range(i + 1, num_feats):
            pct = corr_notable_count[i, j] / valid_count * 100
            if pct > 50 and corr_high_count[i, j] / valid_count * 100 <= 50:
                print(f"    {ALL_NODE_FEATURES[i]} <-> {ALL_NODE_FEATURES[j]}: "
                      f"{pct:.1f}% (mean |r| = {mean_corr[i, j]:.3f})")
                found = True
    if not found:
        print(f"    (none)")

    # Edge feature correlations
    circuits_with_edges = [c for c in circuits if c["edge_features"].shape[0] > 2]
    if circuits_with_edges:
        print(f"\n-- Edge Feature Correlations ({len(circuits_with_edges)} circuits) --\n")
        edge_corrs = defaultdict(list)
        for c in circuits_with_edges:
            ef = c["edge_features"].numpy()
            if ef.shape[0] < 3:
                continue
            col_stds = ef.std(axis=0)
            if np.any(col_stds < 1e-10):
                continue
            ecorr = np.corrcoef(ef.T)
            if np.any(np.isnan(ecorr)):
                continue
            for i in range(3):
                for j in range(i + 1, 3):
                    edge_corrs[f"{EDGE_FEATURE_NAMES[i]} <-> {EDGE_FEATURE_NAMES[j]}"].append(
                        abs(ecorr[i, j])
                    )

        for pair, vals in sorted(edge_corrs.items()):
            arr = np.array(vals)
            pct_high = (arr > 0.9).mean() * 100
            pct_notable = (arr > 0.7).mean() * 100
            print(f"  {pair}")
            print(f"    mean |r| = {arr.mean():.3f}, |r|>0.9: {pct_high:.1f}%, |r|>0.7: {pct_notable:.1f}%")

    # Node-Edge cross-correlation
    print(f"\n-- Node-Edge Cross-Correlation --")
    print(f"  (per-qubit node feature vs mean of adjacent edge features)\n")

    cross_corrs: dict[str, list[float]] = defaultdict(list)
    for c in circuits:
        if c["edge_features"].shape[0] == 0 or c["num_qubits"] < 3:
            continue
        nq = c["num_qubits"]
        ef = c["edge_features"]
        edge_list = c["edge_list"]

        # Compute per-qubit aggregated edge features
        qubit_edge_agg = {fname: np.zeros(nq) for fname in EDGE_FEATURE_NAMES}
        qubit_edge_count = np.zeros(nq)

        for edge_idx, (u, v) in enumerate(edge_list):
            for col_idx, fname in enumerate(EDGE_FEATURE_NAMES):
                val = ef[edge_idx, col_idx].item()
                qubit_edge_agg[fname][u] += val
                qubit_edge_agg[fname][v] += val
            qubit_edge_count[u] += 1
            qubit_edge_count[v] += 1

        # Average
        for fname in EDGE_FEATURE_NAMES:
            qubit_edge_agg[fname] = np.where(
                qubit_edge_count > 0,
                qubit_edge_agg[fname] / qubit_edge_count,
                0.0,
            )

        # Correlate each node feature with each aggregated edge feature
        for nfname in ALL_NODE_FEATURES:
            node_vals = np.array(c["node_features_dict"][nfname])
            if np.std(node_vals) < 1e-10:
                continue
            for efname in EDGE_FEATURE_NAMES:
                edge_vals = qubit_edge_agg[efname]
                if np.std(edge_vals) < 1e-10:
                    continue
                r = np.corrcoef(node_vals, edge_vals)[0, 1]
                if not np.isnan(r):
                    cross_corrs[f"{nfname[:16]} <-> edge_{efname[:12]}"].append(abs(r))

    if cross_corrs:
        print(f"  {'Pair':<42s} {'Mean|r|':>8s} {'>0.9':>6s} {'>0.7':>6s}")
        print(f"  {'-'*64}")
        for pair in sorted(cross_corrs.keys()):
            vals = np.array(cross_corrs[pair])
            mean_r = vals.mean()
            pct_09 = (vals > 0.9).mean() * 100
            pct_07 = (vals > 0.7).mean() * 100
            flag = " **" if mean_r > 0.9 else " * " if mean_r > 0.7 else ""
            print(f"  {pair:<42s} {mean_r:8.3f} {pct_09:5.1f}% {pct_07:5.1f}%{flag}")


# ─────────────────────────────────────────────────────────────
# Phase 5: Normalization Strategy Comparison
# ─────────────────────────────────────────────────────────────

def phase5_normalization(circuits: list[dict], rwpe_k: int = 2) -> None:
    print("\n" + "=" * 90)
    print("PHASE 5: NORMALIZATION STRATEGY COMPARISON")
    print("=" * 90)
    print(f"  Features: {CURRENT_FEATURES}")
    print(f"  RWPE k={rwpe_k}")
    print(f"  Strategy A: all z-score (current)")
    print(f"  Strategy B: count(gc, 2qc) z-score + ratio(sqr, cpf) raw")

    zscore_feats = {"gate_count", "two_qubit_gate_count"}
    raw_feats = {"single_qubit_gate_ratio", "critical_path_fraction"}

    results_a = []  # all z-score
    results_b = []  # mixed

    for c in circuits:
        nq = c["num_qubits"]
        if nq < 2:
            continue

        # Build feature matrices
        selected = [c["node_features_dict"][f] for f in CURRENT_FEATURES]
        raw_mat = torch.tensor(list(zip(*selected)), dtype=torch.float32)  # (nq, 4)

        # Strategy A: all z-score
        x_a = zscore_normalize(raw_mat, dim=0)

        # Strategy B: mixed (z-score counts, keep ratios raw)
        x_b_parts = []
        for col_idx, fname in enumerate(CURRENT_FEATURES):
            col = raw_mat[:, col_idx:col_idx+1]
            if fname in zscore_feats:
                x_b_parts.append(zscore_normalize(col, dim=0))
            else:
                x_b_parts.append(col)
        x_b = torch.cat(x_b_parts, dim=1)

        # Append RWPE
        if rwpe_k > 0:
            rwpe = compute_rwpe(c["edge_list"], nq, rwpe_k)
            x_a = torch.cat([x_a, rwpe], dim=1)
            x_b = torch.cat([x_b, rwpe], dim=1)

        # Compute metrics for both
        for x, results in [(x_a, results_a), (x_b, results_b)]:
            n, d = x.shape
            if n < 2:
                continue
            # Effective dim
            _, S, _ = torch.svd(x)
            threshold = S[0] * 0.01
            eff_dim = int((S > threshold).sum().item())

            # Indistinguishable pairs
            x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
            cos = x_norm @ x_norm.T
            upper = torch.triu(cos, diagonal=1)
            total_pairs = n * (n - 1) // 2
            indist = int((upper > 0.95).sum().item())
            indist_rate = indist / max(total_pairs, 1)

            # Dead columns
            dead = int((x.std(dim=0) < 1e-6).sum().item())

            results.append({
                "eff_dim": eff_dim,
                "total_dim": d,
                "indist_rate": indist_rate,
                "dead_cols": dead,
                "num_qubits": nq,
            })

    # Print comparison
    print(f"\n  {'Strategy':<20s} {'EffDim':>10s} {'Dead':>6s} {'Indist%':>8s} {'N':>6s}")
    print(f"  {'-'*52}")

    for label, results in [("A: all z-score", results_a), ("B: mixed", results_b)]:
        if not results:
            continue
        eff = np.mean([r["eff_dim"] for r in results])
        dim = results[0]["total_dim"]
        dead = np.mean([r["dead_cols"] for r in results])
        indist = np.mean([r["indist_rate"] for r in results]) * 100
        print(f"  {label:<20s} {eff:5.2f}/{dim:<3d} {dead:6.2f} {indist:7.2f}% {len(results):6d}")

    # Delta
    if results_a and results_b:
        delta_eff = np.mean([r["eff_dim"] for r in results_b]) - np.mean([r["eff_dim"] for r in results_a])
        delta_indist = (np.mean([r["indist_rate"] for r in results_b]) - np.mean([r["indist_rate"] for r in results_a])) * 100
        print(f"\n  Delta (B - A): eff_dim {delta_eff:+.3f}, indist_rate {delta_indist:+.2f}pp")
        if delta_eff > 0.3 and delta_indist < -2:
            print(f"  --> RECOMMENDATION: switch to mixed normalization")
        elif delta_eff > 0:
            print(f"  --> Mixed marginally better but below threshold (need +0.3 eff_dim & -2pp indist)")
        else:
            print(f"  --> Keep current all-z-score normalization")

    # Per-size-bucket comparison
    print(f"\n  Per-Size Comparison:")
    print(f"  {'Bucket':<18s} {'N':>5s} | {'A:EffD':>7s} {'A:Ind%':>7s} | {'B:EffD':>7s} {'B:Ind%':>7s} | {'dEff':>6s} {'dInd':>6s}")
    print(f"  {'-'*78}")

    for bucket_label, lo, hi in SIZE_BUCKETS:
        ra = [r for r in results_a if lo <= r["num_qubits"] <= hi]
        rb = [r for r in results_b if lo <= r["num_qubits"] <= hi]
        if not ra or not rb:
            continue
        eff_a = np.mean([r["eff_dim"] for r in ra])
        ind_a = np.mean([r["indist_rate"] for r in ra]) * 100
        eff_b = np.mean([r["eff_dim"] for r in rb])
        ind_b = np.mean([r["indist_rate"] for r in rb]) * 100
        de = eff_b - eff_a
        di = ind_b - ind_a
        print(f"  {bucket_label:<18s} {len(ra):5d} | {eff_a:7.2f} {ind_a:6.2f}% | {eff_b:7.2f} {ind_b:6.2f}% | {de:+5.2f} {di:+5.2f}")


# ─────────────────────────────────────────────────────────────
# Phase 6: RWPE Quality Assessment
# ─────────────────────────────────────────────────────────────

def phase6_rwpe(circuits: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("PHASE 6: RWPE QUALITY ASSESSMENT")
    print("=" * 90)

    for k in [2, 3, 4]:
        print(f"\n-- RWPE k={k} --")

        dead_dims_list = []
        eff_dim_with = []
        eff_dim_without = []
        rwpe_feature_corrs: dict[str, list[float]] = defaultdict(list)

        for c in circuits:
            nq = c["num_qubits"]
            if nq < 2 or len(c["edge_list"]) == 0:
                continue

            rwpe = compute_rwpe(c["edge_list"], nq, k)

            # Dead dims
            rwpe_std = rwpe.std(dim=0)
            dead = int((rwpe_std < 1e-6).sum().item())
            dead_dims_list.append(dead)

            # Build node features (current default, z-scored)
            selected = [c["node_features_dict"][f] for f in CURRENT_FEATURES]
            raw_mat = torch.tensor(list(zip(*selected)), dtype=torch.float32)
            x_no_rwpe = zscore_normalize(raw_mat, dim=0)
            x_with_rwpe = torch.cat([x_no_rwpe, rwpe], dim=1)

            # Eff dim comparison
            if nq > 1:
                _, S1, _ = torch.svd(x_no_rwpe)
                eff1 = int((S1 > S1[0] * 0.01).sum().item())
                eff_dim_without.append(eff1)

                _, S2, _ = torch.svd(x_with_rwpe)
                eff2 = int((S2 > S2[0] * 0.01).sum().item())
                eff_dim_with.append(eff2)

            # Correlation of RWPE dims with explicit features
            for rdim in range(k):
                rwpe_col = rwpe[:, rdim].numpy()
                if np.std(rwpe_col) < 1e-10:
                    continue
                for fname in CURRENT_FEATURES:
                    node_vals = np.array(c["node_features_dict"][fname])
                    if np.std(node_vals) < 1e-10:
                        continue
                    r = abs(np.corrcoef(rwpe_col, node_vals)[0, 1])
                    if not np.isnan(r):
                        rwpe_feature_corrs[f"RWPE[{rdim}] <-> {fname}"].append(r)

        n_valid = len(dead_dims_list)
        print(f"  Circuits analyzed: {n_valid}")
        if n_valid == 0:
            continue

        print(f"  Dead dims: mean={np.mean(dead_dims_list):.2f}/{k}, "
              f"all-dead: {sum(d == k for d in dead_dims_list)}/{n_valid} "
              f"({sum(d == k for d in dead_dims_list)/n_valid*100:.1f}%)")

        eff_delta = np.array(eff_dim_with) - np.array(eff_dim_without)
        print(f"  Eff dim contribution: mean={np.mean(eff_delta):+.3f} "
              f"(without: {np.mean(eff_dim_without):.2f}, with: {np.mean(eff_dim_with):.2f})")

        # RWPE-feature correlations
        if rwpe_feature_corrs:
            print(f"\n  RWPE-Feature Correlations:")
            print(f"  {'Pair':<35s} {'Mean|r|':>8s} {'>0.8':>6s} {'>0.5':>6s}")
            print(f"  {'-'*57}")
            for pair in sorted(rwpe_feature_corrs.keys()):
                vals = np.array(rwpe_feature_corrs[pair])
                mean_r = vals.mean()
                pct_08 = (vals > 0.8).mean() * 100
                pct_05 = (vals > 0.5).mean() * 100
                flag = " **" if mean_r > 0.8 else " * " if mean_r > 0.5 else ""
                print(f"  {pair:<35s} {mean_r:8.3f} {pct_08:5.1f}% {pct_05:5.1f}%{flag}")


# ─────────────────────────────────────────────────────────────
# Phase 7: Circuit Size-Dependent Analysis
# ─────────────────────────────────────────────────────────────

def phase7_size_analysis(circuits: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("PHASE 7: CIRCUIT SIZE-DEPENDENT ANALYSIS")
    print("=" * 90)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for c in circuits:
        buckets[c["size_bucket"]].append(c)

    # Constant rate by size
    print(f"\n-- Per-Feature Constant Rate by Size --\n")
    print(f"  {'Bucket':<18s} {'N':>5s} |", end="")
    for fname in ALL_NODE_FEATURES:
        print(f" {fname[:7]:>8s}", end="")
    print()
    print(f"  {'-'*25}+{'-' * (9 * len(ALL_NODE_FEATURES))}")

    for bucket_label, _, _ in SIZE_BUCKETS:
        group = buckets.get(bucket_label, [])
        if not group:
            continue
        ng = len(group)
        print(f"  {bucket_label:<18s} {ng:5d} |", end="")
        for fname in ALL_NODE_FEATURES:
            const_count = sum(
                1 for c in group
                if np.std(c["node_features_dict"][fname]) < 1e-10
            )
            pct = const_count / ng * 100
            flag = "!" if pct > 50 else " "
            print(f" {pct:6.1f}%{flag}", end="")
        print()

    # CoV by size
    print(f"\n-- Median CoV by Size (node features) --\n")
    print(f"  {'Bucket':<18s} {'N':>5s} |", end="")
    for fname in CURRENT_FEATURES:
        print(f" {fname[:7]:>8s}", end="")
    print()
    print(f"  {'-'*25}+{'-' * (9 * len(CURRENT_FEATURES))}")

    for bucket_label, _, _ in SIZE_BUCKETS:
        group = buckets.get(bucket_label, [])
        if not group:
            continue
        ng = len(group)
        print(f"  {bucket_label:<18s} {ng:5d} |", end="")
        for fname in CURRENT_FEATURES:
            covs = []
            for c in group:
                vals = np.array(c["node_features_dict"][fname])
                mean = np.mean(vals)
                std = np.std(vals)
                if abs(mean) > 1e-10:
                    covs.append(std / abs(mean))
                elif std < 1e-10:
                    covs.append(0.0)
            if covs:
                med = np.median(covs)
                print(f" {med:7.3f} ", end="")
            else:
                print(f"   {'n/a':>5s}", end="")
        print()

    # Eff dim and indist rate by size (current features + RWPE k=2)
    print(f"\n-- Eff Dim & Indist Rate by Size (current features + RWPE k=2) --\n")
    print(f"  {'Bucket':<18s} {'N':>5s} | {'EffDim':>8s} {'Dead':>6s} {'Indist%':>8s} {'MnCos':>7s}")
    print(f"  {'-'*58}")

    for bucket_label, _, _ in SIZE_BUCKETS:
        group = buckets.get(bucket_label, [])
        if not group:
            continue

        eff_dims = []
        dead_cols = []
        indist_rates = []
        cosine_sims = []

        for c in group:
            nq = c["num_qubits"]
            if nq < 2:
                continue
            selected = [c["node_features_dict"][f] for f in CURRENT_FEATURES]
            raw_mat = torch.tensor(list(zip(*selected)), dtype=torch.float32)
            x = zscore_normalize(raw_mat, dim=0)
            rwpe = compute_rwpe(c["edge_list"], nq, 2)
            x = torch.cat([x, rwpe], dim=1)

            n, d = x.shape
            dead = int((x.std(dim=0) < 1e-6).sum().item())
            dead_cols.append(dead)

            if n > 1:
                _, S, _ = torch.svd(x)
                eff = int((S > S[0] * 0.01).sum().item())
                eff_dims.append(eff)

                x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
                cos = x_norm @ x_norm.T
                mask = ~torch.eye(n, dtype=torch.bool)
                cosine_sims.append(cos[mask].mean().item())

                upper = torch.triu(cos, diagonal=1)
                total_pairs = n * (n - 1) // 2
                indist_rates.append((upper > 0.95).sum().item() / max(total_pairs, 1))

        ng = len(group)
        if eff_dims:
            print(f"  {bucket_label:<18s} {ng:5d} | {np.mean(eff_dims):5.2f}/6 "
                  f"{np.mean(dead_cols):6.2f} {np.mean(indist_rates)*100:7.2f}% "
                  f"{np.mean(cosine_sims):7.4f}")

    # Edge feature constant rate by size
    print(f"\n-- Edge Feature Constant Rate by Size --\n")
    print(f"  {'Bucket':<18s} {'N':>5s} {'NoEdge':>7s} |", end="")
    for fname in EDGE_FEATURE_NAMES:
        print(f" {fname[:12]:>13s}", end="")
    print()
    print(f"  {'-'*32}+{'-' * (14 * len(EDGE_FEATURE_NAMES))}")

    for bucket_label, _, _ in SIZE_BUCKETS:
        group = buckets.get(bucket_label, [])
        if not group:
            continue
        ng = len(group)
        no_edge = sum(1 for c in group if c["edge_features"].shape[0] == 0)
        with_edges = [c for c in group if c["edge_features"].shape[0] > 0]
        nwe = len(with_edges)
        print(f"  {bucket_label:<18s} {ng:5d} {no_edge:7d} |", end="")
        for col_idx, fname in enumerate(EDGE_FEATURE_NAMES):
            if nwe == 0:
                print(f"     {'n/a':>8s}", end="")
            else:
                const = sum(
                    1 for c in with_edges
                    if c["edge_features"][:, col_idx].std().item() < 1e-10
                )
                pct = const / nwe * 100
                print(f" {pct:11.1f}% ", end="")
        print()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Circuit graph feature analysis")
    parser.add_argument("--num-samples", type=int, default=500, help="Circuits to sample (default 500)")
    parser.add_argument("--full", action="store_true", help="Use all circuits (no sampling)")
    parser.add_argument("--data-root", type=str, default="data/circuits", help="Data root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rwpe-k", type=int, default=2, help="RWPE k for normalization comparison")
    args = parser.parse_args()

    num_samples = None if args.full else args.num_samples

    print("=" * 90)
    print("CIRCUIT GRAPH FEATURE ANALYSIS")
    print("=" * 90)
    print()

    circuits = collect_circuits(
        data_root=Path(args.data_root),
        num_samples=num_samples,
        seed=args.seed,
    )

    if not circuits:
        print("No circuits loaded. Check data_root path.")
        return

    phase1_completeness(circuits)
    phase2_raw_statistics(circuits)
    phase3_within_circuit_cov(circuits)
    phase4_correlations(circuits)
    phase5_normalization(circuits, rwpe_k=args.rwpe_k)
    phase6_rwpe(circuits)
    phase7_size_analysis(circuits)

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
