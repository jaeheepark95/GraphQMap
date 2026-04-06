"""Filter training circuits by feature indistinguishability.

Scans all cached circuits and identifies those where the current default
feature set cannot distinguish qubits (high indistinguishable pair rate).

Also flags MLQD circuits where single_qubit_gate_ratio is constant (all qubits
have the same ratio, meaning the feature is dead for that circuit).

Outputs:
  - Summary statistics by source and size
  - List of circuits to remove
  - Updated split JSON files (saved alongside originals with _filtered suffix)

Usage:
    # Dry run (report only, no file changes)
    python scripts/filter_indistinguishable.py

    # Apply filter: write new split files
    python scripts/filter_indistinguishable.py --apply

    # Custom threshold
    python scripts/filter_indistinguishable.py --indist-threshold 0.4 --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.circuit_graph import build_circuit_graph_from_raw, compute_rwpe
from data.normalization import zscore_normalize

CURRENT_FEATURES = [
    "gate_count",
    "two_qubit_gate_count",
    "single_qubit_gate_ratio",
    "critical_path_fraction",
]
RWPE_K = 2


def analyze_cache_file(cache_path: Path) -> dict | None:
    """Analyze a single cached circuit for indistinguishability.

    Returns dict with metrics, or None on error.
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

    if num_qubits < 2:
        return {
            "num_qubits": num_qubits,
            "indist_rate": 0.0,
            "indist_pairs": 0,
            "total_pairs": 0,
            "eff_dim": 1,
            "dim": len(CURRENT_FEATURES) + RWPE_K,
            "sqr_constant": False,
            "cpf_constant": False,
            "reason": None,
        }

    # Build graph with current default features
    graph = build_circuit_graph_from_raw(
        node_features_dict=nfd,
        edge_list=edge_list,
        edge_features=edge_features,
        num_qubits=num_qubits,
        node_feature_names=CURRENT_FEATURES,
        rwpe_k=RWPE_K,
    )
    x = graph.x
    n, d = x.shape

    # Effective dim
    _, S, _ = torch.svd(x)
    threshold = S[0] * 0.01
    eff_dim = int((S > threshold).sum().item())

    # Cosine similarity
    x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    cos = x_norm @ x_norm.T
    upper = torch.triu(cos, diagonal=1)
    total_pairs = n * (n - 1) // 2
    indist_pairs = int((upper > 0.95).sum().item())
    indist_rate = indist_pairs / total_pairs if total_pairs > 0 else 0.0

    # Check sqr and cpf constant
    sqr_vals = nfd["single_qubit_gate_ratio"]
    sqr_constant = len(set(f"{v:.6f}" for v in sqr_vals)) <= 1

    cpf_vals = nfd["critical_path_fraction"]
    cpf_constant = len(set(f"{v:.6f}" for v in cpf_vals)) <= 1

    # Determine removal reason
    reason = None
    # Will be set by caller based on threshold

    return {
        "num_qubits": num_qubits,
        "indist_rate": indist_rate,
        "indist_pairs": indist_pairs,
        "total_pairs": total_pairs,
        "eff_dim": eff_dim,
        "dim": d,
        "sqr_constant": sqr_constant,
        "cpf_constant": cpf_constant,
        "reason": reason,
    }


def get_size_bucket(nq: int) -> str:
    if nq <= 3:
        return "tiny (2-3Q)"
    elif nq <= 5:
        return "small (4-5Q)"
    elif nq <= 10:
        return "medium (6-10Q)"
    elif nq <= 20:
        return "large (11-20Q)"
    else:
        return "xlarge (21Q+)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter indistinguishable circuits")
    parser.add_argument(
        "--indist-threshold", type=float, default=0.5,
        help="Remove circuits with indist_rate > threshold (default: 0.5)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write filtered split files (default: dry run)",
    )
    parser.add_argument("--data-root", type=str, default="data/circuits")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_root = data_root / "cache"
    splits_dir = data_root / "splits"

    # ---- Phase 1: Analyze all cached circuits ----
    print(f"=== Indistinguishable Circuit Filter ===")
    print(f"Threshold: indist_rate > {args.indist_threshold}")
    print(f"Features: {CURRENT_FEATURES} + RWPE k={RWPE_K}")
    print()

    # Collect all cache files by source
    source_caches: dict[str, list[Path]] = {}
    for source_dir in sorted(cache_root.iterdir()):
        if source_dir.is_dir() and source_dir.name not in ("benchmarks",):
            files = sorted(source_dir.glob("*.pt"))
            if files:
                source_caches[source_dir.name] = files

    total = sum(len(v) for v in source_caches.values())
    print(f"Total cached circuits: {total}")

    # Analyze all
    results: dict[str, dict] = {}  # key: "source/stem" -> metrics
    errors = 0

    for src, files in source_caches.items():
        print(f"  Analyzing {src} ({len(files)} circuits)...", flush=True)
        for i, cf in enumerate(files):
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(files)}...", flush=True)
            metrics = analyze_cache_file(cf)
            if metrics is not None:
                key = f"{src}/{cf.stem}"
                results[key] = metrics
            else:
                errors += 1

    print(f"\nAnalyzed {len(results)} circuits ({errors} errors)")

    # ---- Phase 2: Classify circuits ----
    to_remove: dict[str, str] = {}  # key -> reason
    sqr_constant_count = 0
    cpf_constant_count = 0

    for key, m in results.items():
        if m["indist_rate"] > args.indist_threshold:
            to_remove[key] = f"indist={m['indist_rate']:.1%} ({m['indist_pairs']}/{m['total_pairs']})"

        if m["sqr_constant"]:
            sqr_constant_count += 1
        if m["cpf_constant"]:
            cpf_constant_count += 1

    # ---- Phase 3: Report ----
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")

    # Overall
    print(f"\n  Total circuits analyzed: {len(results)}")
    print(f"  Circuits to remove (indist > {args.indist_threshold}): {len(to_remove)}")
    print(f"  sqr constant circuits: {sqr_constant_count} ({sqr_constant_count/len(results)*100:.1f}%)")
    print(f"  cpf constant circuits: {cpf_constant_count} ({cpf_constant_count/len(results)*100:.1f}%)")

    # By source
    print(f"\n  Removal by source:")
    print(f"  {'Source':<15s} {'Total':>7s} {'Remove':>8s} {'Rate':>7s} {'sqr=C':>7s} {'cpf=C':>7s}")
    print(f"  {'-'*52}")

    for src in sorted(source_caches.keys()):
        src_keys = [k for k in results if k.startswith(f"{src}/")]
        src_remove = [k for k in src_keys if k in to_remove]
        src_sqr = sum(1 for k in src_keys if results[k]["sqr_constant"])
        src_cpf = sum(1 for k in src_keys if results[k]["cpf_constant"])
        n = len(src_keys)
        print(
            f"  {src:<15s} {n:7d} {len(src_remove):8d} "
            f"{len(src_remove)/max(n,1)*100:6.1f}% "
            f"{src_sqr:7d} {src_cpf:7d}"
        )

    # By size bucket
    print(f"\n  Removal by circuit size:")
    print(f"  {'Size':<22s} {'Total':>7s} {'Remove':>8s} {'Rate':>7s}")
    print(f"  {'-'*46}")

    size_groups: dict[str, list[str]] = defaultdict(list)
    for key, m in results.items():
        size_groups[get_size_bucket(m["num_qubits"])].append(key)

    for bucket in ["tiny (2-3Q)", "small (4-5Q)", "medium (6-10Q)", "large (11-20Q)", "xlarge (21Q+)"]:
        keys = size_groups.get(bucket, [])
        n = len(keys)
        removed = sum(1 for k in keys if k in to_remove)
        print(f"  {bucket:<22s} {n:7d} {removed:8d} {removed/max(n,1)*100:6.1f}%")

    # Indist rate distribution
    print(f"\n  Indist rate distribution (all circuits):")
    rates = [m["indist_rate"] for m in results.values() if m["total_pairs"] > 0]
    for lo, hi, label in [
        (0.0, 0.01, "  0-1%  (excellent)"),
        (0.01, 0.1, "  1-10% (good)    "),
        (0.1, 0.3, " 10-30% (moderate) "),
        (0.3, 0.5, " 30-50% (poor)     "),
        (0.5, 0.8, " 50-80% (bad)      "),
        (0.8, 1.01, " 80-100%(terrible) "),
    ]:
        count = sum(1 for r in rates if lo <= r < hi)
        bar = "#" * (count * 40 // max(len(rates), 1))
        print(f"  {label} {count:5d} ({count/len(rates)*100:5.1f}%) {bar}")

    # Sample removed circuits
    print(f"\n  Sample removed circuits (top 20 by indist rate):")
    sorted_remove = sorted(to_remove.items(), key=lambda x: -results[x[0]]["indist_rate"])
    for key, reason in sorted_remove[:20]:
        m = results[key]
        print(f"    {key:<65s} {m['num_qubits']:3d}Q {reason}")

    # ---- Phase 4: Update splits ----
    if not args.apply:
        print(f"\n  [DRY RUN] No files written. Use --apply to write filtered splits.")
        return

    # Map cache key back to split entry format
    # Split entry: {"source": "queko", "file": "16QBT_05CYC_TFL_0.qasm"}
    # Cache key: "queko/16QBT_05CYC_TFL_0"
    remove_set = set()
    for key in to_remove:
        src, stem = key.split("/", 1)
        remove_set.add((src, stem))

    split_files = [
        "stage2_all.json",
        "stage1_supervised.json",
        "stage1_unsupervised.json",
        "stage1_queko_only.json",
        "val.json",
        "val_queko_only.json",
    ]

    print(f"\n  Updating split files:")
    for split_name in split_files:
        split_path = splits_dir / split_name
        if not split_path.exists():
            continue

        with open(split_path) as f:
            entries = json.load(f)

        original_count = len(entries)
        filtered = []
        removed_count = 0

        for entry in entries:
            src = entry["source"]
            stem = Path(entry["file"]).stem
            if (src, stem) in remove_set:
                removed_count += 1
            else:
                filtered.append(entry)

        # Write filtered version
        filtered_path = splits_dir / split_name.replace(".json", "_filtered.json")
        with open(filtered_path, "w") as f:
            json.dump(filtered, f, indent=2)

        print(f"    {split_name}: {original_count} -> {len(filtered)} "
              f"(removed {removed_count})")
        print(f"      -> {filtered_path}")

    # Also write the removal list for reference
    removal_log = {
        "threshold": args.indist_threshold,
        "features": CURRENT_FEATURES,
        "rwpe_k": RWPE_K,
        "total_analyzed": len(results),
        "total_removed": len(to_remove),
        "removed_circuits": [
            {
                "key": key,
                "source": key.split("/")[0],
                "stem": key.split("/")[1],
                "num_qubits": results[key]["num_qubits"],
                "indist_rate": results[key]["indist_rate"],
                "indist_pairs": results[key]["indist_pairs"],
                "total_pairs": results[key]["total_pairs"],
                "reason": reason,
            }
            for key, reason in sorted(to_remove.items(), key=lambda x: -results[x[0]]["indist_rate"])
        ],
    }
    log_path = splits_dir / "filter_log.json"
    with open(log_path, "w") as f:
        json.dump(removal_log, f, indent=2)
    print(f"\n    Removal log: {log_path}")

    # Summary of sqr-constant circuits (not removed, just flagged)
    sqr_only = [k for k, m in results.items() if m["sqr_constant"] and k not in to_remove]
    print(f"\n  Note: {len(sqr_only)} circuits have constant sqr but were NOT removed "
          f"(indist_rate <= {args.indist_threshold})")


if __name__ == "__main__":
    main()
