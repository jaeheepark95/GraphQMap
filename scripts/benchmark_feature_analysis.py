"""Per-circuit feature analysis for benchmark and training circuits.

For each circuit, computes and displays:
  - Raw feature values (all 7 features + RWPE)
  - Per-feature variance and dead column detection
  - Pairwise cosine similarity between qubits
  - Feature correlation matrix
  - Effective dimensionality (SVD)

Also produces aggregate comparison across feature set combinations.

Usage:
    # Benchmark circuits (23 circuits, per-circuit detail)
    python scripts/benchmark_feature_analysis.py
    python scripts/benchmark_feature_analysis.py --rwpe-k 4
    python scripts/benchmark_feature_analysis.py --verbose

    # Training circuits (all sources, grouped by source & size)
    python scripts/benchmark_feature_analysis.py --training
    python scripts/benchmark_feature_analysis.py --training --num-samples 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.circuit_graph import (
    build_circuit_graph,
    compute_rwpe,
    extract_circuit_features,
    load_circuit,
)
from data.normalization import zscore_normalize

ALL_FEATURES = [
    "gate_count",
    "two_qubit_gate_count",
    "degree",
    "depth_participation",
    "weighted_degree",
    "single_qubit_gate_ratio",
    "critical_path_fraction",
]

# Feature set combinations to compare
FEATURE_SETS = {
    "old_default": {
        "features": ["gate_count", "two_qubit_gate_count", "degree", "depth_participation"],
        "label": "Old default (gc, 2qc, deg, dp)",
    },
    "current_default": {
        "features": ["gate_count", "two_qubit_gate_count", "single_qubit_gate_ratio", "critical_path_fraction"],
        "label": "Current default (gc, 2qc, sqr, cpf)",
    },
    "minimal": {
        "features": ["two_qubit_gate_count", "single_qubit_gate_ratio"],
        "label": "Minimal (2qc, sqr)",
    },
    "no_cpf": {
        "features": ["gate_count", "two_qubit_gate_count", "single_qubit_gate_ratio", "degree"],
        "label": "No cpf (gc, 2qc, sqr, deg)",
    },
    "full_no_redundant": {
        "features": ["gate_count", "two_qubit_gate_count", "degree", "single_qubit_gate_ratio", "critical_path_fraction"],
        "label": "Full no-redundant (gc, 2qc, deg, sqr, cpf)",
    },
    "all_7": {
        "features": ALL_FEATURES,
        "label": "All 7 features",
    },
}


def analyze_from_cache(
    cache_path: Path,
    rwpe_k: int = 2,
) -> dict | None:
    """Analyze features from a cached .pt file (fast path, no QASM parsing).

    Returns a dict with per-circuit metrics, or None on error.
    """
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

    return _analyze_from_raw(
        name=cache_path.stem,
        nfd=nfd,
        num_qubits=num_qubits,
        edge_list=edge_list,
        edge_features=edge_features,
        rwpe_k=rwpe_k,
    )


def analyze_single_circuit(
    qasm_path: Path,
    rwpe_k: int = 2,
    verbose: bool = False,
) -> dict:
    """Analyze all features for a single circuit from QASM.

    Returns a dict with per-circuit metrics.
    """
    circuit = load_circuit(qasm_path)
    feats = extract_circuit_features(circuit)
    nfd = feats["node_features_dict"]
    num_qubits = feats["num_qubits"]
    edge_list = feats["edge_list"]

    return _analyze_from_raw(
        name=qasm_path.stem,
        nfd=nfd,
        num_qubits=num_qubits,
        edge_list=edge_list,
        edge_features=feats["edge_features"],
        rwpe_k=rwpe_k,
    )


def _analyze_from_raw(
    name: str,
    nfd: dict[str, list[float]],
    num_qubits: int,
    edge_list: list[tuple[int, int]],
    edge_features: torch.Tensor,
    rwpe_k: int = 2,
) -> dict:
    """Core analysis logic shared by QASM and cache paths."""
    from data.circuit_graph import build_circuit_graph_from_raw

    result = {
        "name": name,
        "num_qubits": num_qubits,
        "num_edges": len(edge_list),
    }

    # --- Raw feature values ---
    raw_values = {}
    for fname in ALL_FEATURES:
        vals = nfd[fname]
        raw_values[fname] = vals
        unique_ratio = len(set(f"{v:.6f}" for v in vals)) / max(len(vals), 1)
        result[f"{fname}_unique_ratio"] = unique_ratio
        result[f"{fname}_std"] = float(np.std(vals))
        result[f"{fname}_is_constant"] = unique_ratio <= 1.0 / max(num_qubits, 2)

    result["raw_values"] = raw_values

    # --- RWPE ---
    if rwpe_k > 0:
        rwpe = compute_rwpe(edge_list, num_qubits, rwpe_k)
        rwpe_std = rwpe.std(dim=0)
        result["rwpe_dead_dims"] = int((rwpe_std < 1e-6).sum().item())
        result["rwpe_values"] = rwpe.numpy()
    else:
        result["rwpe_dead_dims"] = 0
        result["rwpe_values"] = np.zeros((num_qubits, 0))

    # --- Per feature-set analysis ---
    set_results = {}
    for set_name, set_cfg in FEATURE_SETS.items():
        feat_names = set_cfg["features"]
        graph = build_circuit_graph_from_raw(
            node_features_dict=nfd,
            edge_list=edge_list,
            edge_features=edge_features,
            num_qubits=num_qubits,
            node_feature_names=feat_names,
            rwpe_k=rwpe_k,
        )
        x = graph.x  # (n, d)
        n, d = x.shape

        sr = {"dim": d, "label": set_cfg["label"]}

        # Dead columns
        col_std = x.std(dim=0)
        sr["dead_cols"] = int((col_std < 1e-6).sum().item())

        # Effective dim (SVD)
        if n > 1:
            _, S, _ = torch.svd(x)
            threshold = S[0] * 0.01
            sr["eff_dim"] = int((S > threshold).sum().item())

            # Pairwise cosine similarity
            x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
            cos = x_norm @ x_norm.T
            mask = ~torch.eye(n, dtype=torch.bool)
            sr["mean_cosine_sim"] = float(cos[mask].mean().item())
            sr["max_cosine_sim"] = float(cos[mask].max().item())
            sr["min_cosine_sim"] = float(cos[mask].min().item())

            # Count indistinguishable pairs (cosine > 0.95)
            upper = torch.triu(cos, diagonal=1)
            sr["indistinguishable_pairs"] = int((upper > 0.95).sum().item())
            total_pairs = n * (n - 1) // 2
            sr["total_pairs"] = total_pairs
        else:
            sr["eff_dim"] = 1
            sr["mean_cosine_sim"] = 0.0
            sr["max_cosine_sim"] = 0.0
            sr["min_cosine_sim"] = 0.0
            sr["indistinguishable_pairs"] = 0
            sr["total_pairs"] = 0

        # Feature correlations (named features only, no RWPE)
        num_named = len(feat_names)
        corrs = {}
        if n > 2 and num_named > 1:
            # Use raw (pre-zscore) features for correlation
            selected = [nfd[fn] for fn in feat_names]
            raw_x = torch.tensor(list(zip(*selected)), dtype=torch.float32)
            for i in range(num_named):
                for j in range(i + 1, num_named):
                    try:
                        corr = torch.corrcoef(torch.stack([raw_x[:, i], raw_x[:, j]]))[0, 1].item()
                        if not np.isnan(corr):
                            corrs[f"{feat_names[i]} ↔ {feat_names[j]}"] = corr
                    except Exception:
                        pass
        sr["correlations"] = corrs

        set_results[set_name] = sr

    result["feature_sets"] = set_results
    return result


def print_circuit_report(result: dict, verbose: bool = False) -> None:
    """Print detailed report for a single circuit."""
    name = result["name"]
    nq = result["num_qubits"]
    ne = result["num_edges"]

    print(f"\n{'='*70}")
    print(f"  {name}  ({nq}Q, {ne} edges)")
    print(f"{'='*70}")

    # Per-feature summary
    print(f"\n  Feature Quality (per qubit):")
    print(f"  {'Feature':<28s} {'Std':>7s} {'Unique%':>8s} {'Const?':>7s}")
    print(f"  {'-'*52}")
    for fname in ALL_FEATURES:
        std = result[f"{fname}_std"]
        uq = result[f"{fname}_unique_ratio"]
        const = result[f"{fname}_is_constant"]
        flag = " <<<" if const else ""
        print(f"  {fname:<28s} {std:7.4f} {uq*100:7.1f}% {'YES':>7s}{flag}" if const
              else f"  {fname:<28s} {std:7.4f} {uq*100:7.1f}% {'no':>7s}")

    # RWPE
    rdead = result["rwpe_dead_dims"]
    rwpe_vals = result["rwpe_values"]
    rwpe_k = rwpe_vals.shape[1] if rwpe_vals.ndim == 2 else 0
    if rwpe_k > 0:
        print(f"\n  RWPE (k={rwpe_k}): {rdead}/{rwpe_k} dead dims")

    # Verbose: print raw values
    if verbose:
        print(f"\n  Raw feature values:")
        raw = result["raw_values"]
        header = "  Qubit " + " ".join(f"{f[:6]:>8s}" for f in ALL_FEATURES)
        print(header)
        for qi in range(nq):
            row = f"  q{qi:<5d}" + " ".join(f"{raw[f][qi]:8.3f}" for f in ALL_FEATURES)
            print(row)
        if rwpe_k > 0:
            print(f"\n  RWPE values:")
            for qi in range(nq):
                row = f"  q{qi:<5d}" + " ".join(f"{rwpe_vals[qi, k]:8.4f}" for k in range(rwpe_k))
                print(row)

    # Feature set comparison
    print(f"\n  Feature Set Comparison:")
    print(f"  {'Set':<35s} {'Dim':>4s} {'EffD':>5s} {'Dead':>5s} {'MnCos':>7s} {'MxCos':>7s} {'Indist':>10s}")
    print(f"  {'-'*75}")
    for set_name, sr in result["feature_sets"].items():
        label = sr["label"][:34]
        indist_str = f"{sr['indistinguishable_pairs']}/{sr['total_pairs']}"
        print(
            f"  {label:<35s} {sr['dim']:4d} {sr['eff_dim']:5d} "
            f"{sr['dead_cols']:5d} {sr['mean_cosine_sim']:7.4f} "
            f"{sr['max_cosine_sim']:7.4f} {indist_str:>10s}"
        )

    # High correlations for current default
    curr = result["feature_sets"]["current_default"]
    high_corrs = {k: v for k, v in curr["correlations"].items() if abs(v) > 0.8}
    if high_corrs:
        print(f"\n  High correlations (current default, |r|>0.8):")
        for pair, r in sorted(high_corrs.items(), key=lambda x: -abs(x[1])):
            print(f"    {pair}: {r:+.4f}")


def print_aggregate_summary(all_results: list[dict], label: str = "benchmark circuits") -> None:
    """Print aggregate comparison across circuits."""
    print(f"\n{'#'*70}")
    print(f"  AGGREGATE SUMMARY ({len(all_results)} {label})")
    print(f"{'#'*70}")

    # --- Per-feature constant rate ---
    print(f"\n  Per-Feature Constant Rate (feature is same for all qubits):")
    print(f"  {'Feature':<28s} {'Const%':>8s} {'Circuits':>10s} {'MeanStd':>9s}")
    print(f"  {'-'*57}")
    for fname in ALL_FEATURES:
        const_count = sum(1 for r in all_results if r[f"{fname}_is_constant"])
        mean_std = np.mean([r[f"{fname}_std"] for r in all_results])
        print(f"  {fname:<28s} {const_count/len(all_results)*100:7.1f}% "
              f"{const_count:>5d}/{len(all_results):<4d} {mean_std:9.4f}")

    # --- Feature set aggregate ---
    print(f"\n  Feature Set Aggregate Comparison:")
    print(f"  {'Set':<35s} {'EffD':>7s} {'Dead':>6s} {'MnCos':>7s} {'Indist%':>8s}")
    print(f"  {'-'*65}")

    for set_name in FEATURE_SETS:
        eff_dims = []
        dead_cols = []
        cosine_sims = []
        indist_rates = []

        for r in all_results:
            sr = r["feature_sets"][set_name]
            eff_dims.append(sr["eff_dim"])
            dead_cols.append(sr["dead_cols"])
            cosine_sims.append(sr["mean_cosine_sim"])
            if sr["total_pairs"] > 0:
                indist_rates.append(sr["indistinguishable_pairs"] / sr["total_pairs"])

        dim = all_results[0]["feature_sets"][set_name]["dim"]
        label = FEATURE_SETS[set_name]["label"][:34]
        print(
            f"  {label:<35s} "
            f"{np.mean(eff_dims):4.1f}/{dim:<2d} "
            f"{np.mean(dead_cols):5.2f} "
            f"{np.mean(cosine_sims):7.4f} "
            f"{np.mean(indist_rates)*100:7.1f}%"
        )

    # --- Per-circuit ranking: which set has lowest indistinguishable rate ---
    print(f"\n  Best Feature Set Per Circuit (lowest indistinguishable pair ratio):")
    print(f"  {'Circuit':<25s} {'Best Set':<35s} {'Indist':>10s}")
    print(f"  {'-'*72}")
    set_wins = {s: 0 for s in FEATURE_SETS}
    for r in all_results:
        best_set = None
        best_rate = float("inf")
        for set_name in FEATURE_SETS:
            sr = r["feature_sets"][set_name]
            if sr["total_pairs"] > 0:
                rate = sr["indistinguishable_pairs"] / sr["total_pairs"]
            else:
                rate = 0.0
            if rate < best_rate:
                best_rate = rate
                best_set = set_name
        set_wins[best_set] += 1
        sr = r["feature_sets"][best_set]
        indist_str = f"{sr['indistinguishable_pairs']}/{sr['total_pairs']}"
        label = FEATURE_SETS[best_set]["label"][:34]
        print(f"  {r['name']:<25s} {label:<35s} {indist_str:>10s}")

    print(f"\n  Win counts:")
    for s, w in sorted(set_wins.items(), key=lambda x: -x[1]):
        print(f"    {FEATURE_SETS[s]['label']}: {w}")

    # --- Problematic circuits (high indistinguishable rate in current default) ---
    print(f"\n  Problematic Circuits (current default, indist > 30%):")
    for r in sorted(all_results, key=lambda x: -(
        x["feature_sets"]["current_default"]["indistinguishable_pairs"]
        / max(x["feature_sets"]["current_default"]["total_pairs"], 1)
    )):
        sr = r["feature_sets"]["current_default"]
        if sr["total_pairs"] > 0:
            rate = sr["indistinguishable_pairs"] / sr["total_pairs"]
            if rate > 0.3:
                print(f"    {r['name']:<25s} {sr['indistinguishable_pairs']}/{sr['total_pairs']} "
                      f"({rate*100:.0f}%)  eff_dim={sr['eff_dim']}/{sr['dim']}")


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


def print_grouped_summary(
    all_results: list[dict],
    group_key: str,
    group_fn,
) -> None:
    """Print aggregate stats grouped by a key function."""
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        groups[group_fn(r)].append(r)

    print(f"\n{'='*80}")
    print(f"  GROUPED BY {group_key.upper()}")
    print(f"{'='*80}")

    # Header for feature set comparison
    print(f"\n  {'Group':<22s} {'N':>5s} | ", end="")
    for set_name in FEATURE_SETS:
        short = set_name[:10]
        print(f" {short:>12s}", end="")
    print()
    print(f"  {'':<22s} {'':>5s} | ", end="")
    for _ in FEATURE_SETS:
        print(f" {'EffD Indst%':>12s}", end="")
    print()
    print(f"  {'-'*28}+{'-' * (13 * len(FEATURE_SETS))}")

    for group_label in sorted(groups.keys()):
        results = groups[group_label]
        n = len(results)
        print(f"  {group_label:<22s} {n:5d} | ", end="")

        for set_name in FEATURE_SETS:
            eff_dims = []
            indist_rates = []
            for r in results:
                sr = r["feature_sets"][set_name]
                eff_dims.append(sr["eff_dim"])
                if sr["total_pairs"] > 0:
                    indist_rates.append(
                        sr["indistinguishable_pairs"] / sr["total_pairs"]
                    )
            ed = np.mean(eff_dims) if eff_dims else 0
            ir = np.mean(indist_rates) * 100 if indist_rates else 0
            print(f" {ed:4.1f} {ir:5.1f}%", end="")
        print()

    # Per-feature constant rate by group
    print(f"\n  Per-Feature Constant Rate by {group_key}:")
    print(f"  {'Group':<22s} {'N':>5s} | ", end="")
    for fname in ALL_FEATURES:
        print(f" {fname[:6]:>7s}", end="")
    print()
    print(f"  {'-'*28}+{'-' * (8 * len(ALL_FEATURES))}")

    for group_label in sorted(groups.keys()):
        results = groups[group_label]
        n = len(results)
        print(f"  {group_label:<22s} {n:5d} | ", end="")
        for fname in ALL_FEATURES:
            const_count = sum(1 for r in results if r[f"{fname}_is_constant"])
            pct = const_count / n * 100
            print(f" {pct:6.1f}%", end="")
        print()

    # Correlation frequency by group (current default only)
    corr_pairs = [
        "gate_count ↔ two_qubit_gate_count",
        "gate_count ↔ single_qubit_gate_ratio",
        "gate_count ↔ critical_path_fraction",
        "two_qubit_gate_count ↔ single_qubit_gate_ratio",
        "two_qubit_gate_count ↔ critical_path_fraction",
        "single_qubit_gate_ratio ↔ critical_path_fraction",
    ]
    print(f"\n  High Correlation Frequency (current default, |r|>0.8) by {group_key}:")
    print(f"  {'Group':<22s} {'N':>5s} | ", end="")
    for cp in corr_pairs:
        parts = cp.split(" ↔ ")
        short = f"{parts[0][:3]}↔{parts[1][:3]}"
        print(f" {short:>8s}", end="")
    print()
    print(f"  {'-'*28}+{'-' * (9 * len(corr_pairs))}")

    for group_label in sorted(groups.keys()):
        results = groups[group_label]
        n = len(results)
        print(f"  {group_label:<22s} {n:5d} | ", end="")
        for cp in corr_pairs:
            count = 0
            for r in results:
                corrs = r["feature_sets"]["current_default"]["correlations"]
                if cp in corrs and abs(corrs[cp]) > 0.8:
                    count += 1
            pct = count / n * 100
            print(f" {pct:6.1f}%", end="")
        print()


def print_problematic_detail(all_results: list[dict], threshold: float = 0.3) -> None:
    """Print detail on circuits where current default has high indist rate."""
    problematic = []
    for r in all_results:
        sr = r["feature_sets"]["current_default"]
        if sr["total_pairs"] > 0:
            rate = sr["indistinguishable_pairs"] / sr["total_pairs"]
            if rate > threshold:
                problematic.append((r, rate))

    problematic.sort(key=lambda x: -x[1])
    print(f"\n  Problematic Circuits (current default, indist > {threshold*100:.0f}%): "
          f"{len(problematic)}/{len(all_results)}")

    if not problematic:
        print(f"    (none)")
        return

    print(f"  {'Circuit':<30s} {'NQ':>4s} {'Indist':>10s} {'Rate':>7s} "
          f"{'EffD':>5s} {'Dead':>5s} {'BestAlt':>12s} {'AltIndst':>10s}")
    print(f"  {'-'*90}")
    for r, rate in problematic[:30]:  # top 30
        sr = r["feature_sets"]["current_default"]
        indist_str = f"{sr['indistinguishable_pairs']}/{sr['total_pairs']}"

        # Find best alternative
        best_alt = None
        best_alt_rate = rate
        for set_name in FEATURE_SETS:
            if set_name == "current_default":
                continue
            asr = r["feature_sets"][set_name]
            if asr["total_pairs"] > 0:
                alt_rate = asr["indistinguishable_pairs"] / asr["total_pairs"]
                if alt_rate < best_alt_rate:
                    best_alt_rate = alt_rate
                    best_alt = set_name
        alt_label = best_alt[:12] if best_alt else "none"
        if best_alt:
            asr = r["feature_sets"][best_alt]
            alt_indist = f"{asr['indistinguishable_pairs']}/{asr['total_pairs']}"
        else:
            alt_indist = "-"

        print(
            f"  {r['name']:<30s} {r['num_qubits']:4d} {indist_str:>10s} "
            f"{rate*100:6.1f}% {sr['eff_dim']:5d} {sr['dead_cols']:5d} "
            f"{alt_label:>12s} {alt_indist:>10s}"
        )


def run_training_analysis(
    data_root: Path,
    rwpe_k: int,
    num_samples: int,
    seed: int,
) -> None:
    """Run feature analysis on training circuits grouped by source and size.

    Uses pre-computed cache files for speed (no QASM parsing).
    """
    cache_root = data_root / "cache"

    # Collect all cached circuits by source
    source_files: dict[str, list[Path]] = {}
    for source_dir in sorted(cache_root.iterdir()):
        if source_dir.is_dir() and source_dir.name != "benchmarks":
            files = sorted(source_dir.glob("*.pt"))
            if files:
                source_files[source_dir.name] = files

    total = sum(len(v) for v in source_files.values())
    print(f"Training circuits (from cache) by source:")
    for src, files in source_files.items():
        print(f"  {src}: {len(files)}")
    print(f"  Total: {total}")

    # Sample proportionally from each source
    rng = np.random.RandomState(seed)
    sampled: list[tuple[str, Path]] = []
    for src, files in source_files.items():
        n_src = max(1, int(num_samples * len(files) / total))
        n_src = min(n_src, len(files))
        chosen = rng.choice(len(files), size=n_src, replace=False)
        for idx in chosen:
            sampled.append((src, files[idx]))

    print(f"\nSampled {len(sampled)}/{total} circuits (seed={seed})")
    print(f"RWPE k={rwpe_k}")
    print(f"Analyzing (from cache, fast)...", flush=True)

    all_results = []
    errors = 0
    for i, (src, cf) in enumerate(sampled):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sampled)}...", flush=True)
        result = analyze_from_cache(cf, rwpe_k=rwpe_k)
        if result is not None:
            result["source"] = src
            result["size_bucket"] = _get_size_bucket(result["num_qubits"])
            all_results.append(result)
        else:
            errors += 1

    print(f"Analyzed {len(all_results)} circuits ({errors} errors)")

    # Overall aggregate
    print_aggregate_summary(all_results, label="training circuits (sampled)")

    # Grouped by source
    print_grouped_summary(all_results, "source", lambda r: r["source"])

    # Grouped by size bucket
    print_grouped_summary(all_results, "circuit size", lambda r: r["size_bucket"])

    # Grouped by source x size
    print_grouped_summary(
        all_results,
        "source × size",
        lambda r: f"{r['source'][:8]:<8s} {r['size_bucket']}",
    )

    # Problematic circuits detail
    print_problematic_detail(all_results, threshold=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Circuit feature analysis")
    parser.add_argument("--rwpe-k", type=int, default=2, help="RWPE steps (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print raw feature values")
    parser.add_argument("--data-root", type=str, default="data/circuits", help="Data root")
    parser.add_argument("--training", action="store_true", help="Analyze training circuits")
    parser.add_argument("--num-samples", type=int, default=300, help="Samples for training mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.training:
        run_training_analysis(
            data_root=Path(args.data_root),
            rwpe_k=args.rwpe_k,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    else:
        benchmark_dir = Path(args.data_root) / "qasm" / "benchmarks"
        qasm_files = sorted(benchmark_dir.glob("*.qasm"))
        print(f"Found {len(qasm_files)} benchmark circuits in {benchmark_dir}")
        print(f"RWPE k={args.rwpe_k}")

        all_results = []
        for qf in qasm_files:
            try:
                result = analyze_single_circuit(qf, rwpe_k=args.rwpe_k, verbose=args.verbose)
                all_results.append(result)
                print_circuit_report(result, verbose=args.verbose)
            except Exception as e:
                print(f"\n  ERROR processing {qf.stem}: {e}")

        if all_results:
            print_aggregate_summary(all_results)


if __name__ == "__main__":
    main()
