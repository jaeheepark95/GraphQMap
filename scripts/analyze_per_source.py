#!/usr/bin/env python3
"""Per-source analysis of circuits in the training split."""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "circuits"


def load_split(split_path: Path) -> list[dict]:
    with open(split_path) as f:
        return json.load(f)


def load_cache(source: str, filename: str) -> dict | None:
    stem = Path(filename).stem
    cache_path = DATA_ROOT / "cache" / source / (stem + ".pt")
    if not cache_path.exists():
        return None
    return torch.load(cache_path, weights_only=False, map_location="cpu")


def extract_algorithm_family(source: str, filename: str) -> str:
    stem = Path(filename).stem

    if source == "queko":
        # Pattern: 16QBT_05CYC_TFL_0 -> TFL
        # Also: 20QBT_45CYC_.0D1_.1D2_0 -> BSS (bound-structure-substructure)
        m = re.match(r"\d+QBT_\d+CYC_(\w+?)_\d+$", stem)
        if m:
            return m.group(1)
        # BSS pattern with density params
        m = re.match(r"\d+QBT_\d+CYC_\.\d+D1_\.\d+D2_\d+$", stem)
        if m:
            return "BSS"
        return stem

    if source == "mlqd":
        # Patterns:
        # 1. Simple: algorithm_nN_backend (e.g., adder_n4_melbourne)
        # 2. melbourne_1qu_qu04_len22_basis_trotter_n4_transpiled_38
        # 3. melbourne_only2qu_qu06_twoqugatenum10_qram_n20_transpiled_9
        # 4. mlqd_grid5x5_1qu_qu11_len45_qft_n29_transpiled_7
        # 5. mlqd_grid5x5_only2qu_qu15_twoqugatenum13_wstate_n118_transpiled_3
        # 6. mlqd_grid5x5_qasmbench_multiply_n13_transpiled

        # Pattern 4/5: mlqd_grid5x5_ prefix
        m = re.match(r"mlqd_grid5x5_(?:1qu_qu\d+_len\d+|only2qu_qu\d+_twoqugatenum\d+)_(.+?)_n\d+_transpiled_\d+$", stem)
        if m:
            return m.group(1)
        # Pattern 6: mlqd_grid5x5_qasmbench_
        m = re.match(r"mlqd_grid5x5_qasmbench_(.+?)_n\d+_transpiled$", stem)
        if m:
            return m.group(1)
        # Pattern 2: melbourne/rochester_1qu_...
        m = re.match(r"(?:melbourne|rochester)_1qu_qu\d+_len\d+_(.+?)_n\d+_transpiled_\d+$", stem)
        if m:
            return m.group(1)
        # Pattern 3: melbourne/rochester_only2qu_...
        m = re.match(r"(?:melbourne|rochester)_only2qu_qu\d+_twoqugatenum\d+_(.+?)_n\d+_transpiled_\d+$", stem)
        if m:
            return m.group(1)
        # Pattern 7: melbourne/rochester_qasmbench_ or mlqd_grid5x5_qasmbench_
        m = re.match(r"(?:melbourne|rochester|mlqd_grid5x5)_qasmbench_(.+?)_n\d+_transpiled$", stem)
        if m:
            return m.group(1)
        # Pattern 1: simple algorithm_nN_backend
        backends = ("melbourne", "rochester", "grid5x5")
        parts = stem.split("_")
        algo_parts = []
        for p in parts:
            if re.match(r"n\d+$", p):
                break
            if p in backends:
                break
            algo_parts.append(p)
        return "_".join(algo_parts) if algo_parts else stem

    if source == "mqt_bench":
        # Patterns: ae_n2, bv_n10, qaoa_n10_seed0, bmw_quark_copula_n4
        # QAOA has _seed suffix: qaoa_n10_seed0 -> qaoa
        m = re.match(r"(.+?)_n\d+_seed\d+$", stem)
        if m:
            return m.group(1)
        # General: algo_n\d+
        m = re.match(r"(.+?)_n\d+$", stem)
        if m:
            return m.group(1)
        # Original MQT Bench format
        parts = stem.split("_")
        algo_parts = []
        for p in parts:
            if p in ("nativegates", "indep", "mapped"):
                break
            algo_parts.append(p)
        return "_".join(algo_parts) if algo_parts else stem

    if source == "qasmbench":
        # Patterns like: small_adder_n10, medium_qram_n20, large_knn_n115
        # Remove size prefix and _n\d+ suffix
        m = re.match(r"(?:small|medium|large)_(.+?)_n\d+", stem)
        if m:
            return m.group(1)
        m = re.match(r"(.+?)_n\d+", stem)
        if m:
            return m.group(1)
        return stem

    if source == "revlib":
        # Patterns like: 4gt11_82, hwb5_53, ...
        # Remove trailing _\d+ (RevLib benchmark ID)
        m = re.match(r"(.+?)_\d+$", stem)
        if m:
            return m.group(1)
        return stem

    return stem


def size_bucket(n: int) -> str:
    if n <= 4:
        return "2-4"
    if n <= 10:
        return "5-10"
    if n <= 20:
        return "11-20"
    if n <= 50:
        return "21-50"
    return "51+"


def density_bucket(d: float) -> str:
    if d < 1.0:
        return "<1.0"
    if d < 2.0:
        return "1.0-2.0"
    if d < 3.0:
        return "2.0-3.0"
    if d < 5.0:
        return "3.0-5.0"
    return "5.0+"


def stats(arr: list[float]) -> dict:
    a = np.array(arr)
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
    }


def main():
    split_path = DATA_ROOT / "splits" / "stage2_all.json"
    entries = load_split(split_path)
    print(f"Total circuits in split: {len(entries)}")

    # Group by source
    by_source: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_source[e["source"]].append(e)

    print(f"Sources: {list(by_source.keys())}")
    print()

    # Collect per-source data
    for source in ["queko", "mlqd", "mqt_bench", "qasmbench", "revlib"]:
        circuits = by_source.get(source, [])
        if not circuits:
            print(f"=== {source}: NO CIRCUITS ===\n")
            continue

        num_qubits_list = []
        num_edges_list = []
        total_2q_gates_list = []
        density_list = []
        algo_counter = Counter()
        missing_cache = 0

        for entry in circuits:
            filename = entry["file"]
            algo = extract_algorithm_family(source, filename)
            algo_counter[algo] += 1

            cache = load_cache(source, filename)
            if cache is None:
                missing_cache += 1
                continue

            # Extract num_qubits
            # Get num_qubits
            if "num_logical" in cache:
                nq = cache["num_logical"]
            elif "node_features_dict" in cache:
                nfd_tmp = cache["node_features_dict"]
                first_key = next(iter(nfd_tmp))
                nq = len(nfd_tmp[first_key])
            else:
                # Legacy PyG Data
                cg = cache.get("circuit_graph")
                if cg is not None:
                    nq = cg.num_nodes
                else:
                    continue

            # Get 2Q gate count from node_features_dict
            total_2q = 0.0
            if "node_features_dict" in cache:
                nfd = cache["node_features_dict"]
                if "two_qubit_gate_count" in nfd:
                    v = nfd["two_qubit_gate_count"]
                    if hasattr(v, 'sum'):
                        total_2q = float(v.sum())
                    else:
                        total_2q = float(sum(v))

            # Get edge count
            if "edge_list" in cache and cache["edge_list"] is not None:
                el = cache["edge_list"]
                if hasattr(el, '__len__'):
                    ne = len(el)  # edge_list is list of pairs (undirected unique)
                else:
                    ne = 0
            elif "edge_index" in cache:
                ne = cache["edge_index"].shape[1] // 2
            else:
                cg = cache.get("circuit_graph")
                if cg is not None and hasattr(cg, "edge_index"):
                    ne = cg.edge_index.shape[1] // 2
                else:
                    ne = 0

            # Also try circuit_edge_pairs for 2Q info
            if total_2q == 0.0 and "circuit_edge_weights" in cache:
                weights = cache["circuit_edge_weights"]
                if weights is not None and len(weights) > 0:
                    total_2q = float(sum(weights))

            num_qubits_list.append(nq)
            num_edges_list.append(ne)
            total_2q_gates_list.append(total_2q)
            if nq > 0:
                density_list.append(ne / nq)

        # Print report
        print(f"{'=' * 70}")
        print(f"  {source.upper()} — {len(circuits)} circuits ({missing_cache} missing cache)")
        print(f"{'=' * 70}")

        if not num_qubits_list:
            print("  No data available.\n")
            continue

        # 1. Size distribution
        s = stats(num_qubits_list)
        print(f"\n  NUM QUBITS: min={s['min']:.0f}, max={s['max']:.0f}, "
              f"mean={s['mean']:.1f}, median={s['median']:.0f}, std={s['std']:.1f}")

        bucket_counts = Counter(size_bucket(int(n)) for n in num_qubits_list)
        print("  Size buckets:")
        for b in ["2-4", "5-10", "11-20", "21-50", "51+"]:
            c = bucket_counts.get(b, 0)
            pct = 100 * c / len(num_qubits_list)
            bar = "#" * int(pct / 2)
            print(f"    {b:>5s}: {c:4d} ({pct:5.1f}%) {bar}")

        # 2. Edge count distribution
        s = stats(num_edges_list)
        print(f"\n  NUM EDGES: min={s['min']:.0f}, max={s['max']:.0f}, "
              f"mean={s['mean']:.1f}, median={s['median']:.0f}, std={s['std']:.1f}")

        # 3. Connectivity density
        s = stats(density_list)
        print(f"\n  DENSITY (edges/qubits): min={s['min']:.2f}, max={s['max']:.2f}, "
              f"mean={s['mean']:.2f}, median={s['median']:.2f}")

        dbucket_counts = Counter(density_bucket(d) for d in density_list)
        print("  Density buckets:")
        for b in ["<1.0", "1.0-2.0", "2.0-3.0", "3.0-5.0", "5.0+"]:
            c = dbucket_counts.get(b, 0)
            pct = 100 * c / len(density_list)
            bar = "#" * int(pct / 2)
            print(f"    {b:>7s}: {c:4d} ({pct:5.1f}%) {bar}")

        # 4. Total 2Q gate count
        s = stats(total_2q_gates_list)
        print(f"\n  TOTAL 2Q GATES: min={s['min']:.0f}, max={s['max']:.0f}, "
              f"mean={s['mean']:.1f}, median={s['median']:.0f}, std={s['std']:.1f}")

        # Percentile breakdown
        a = np.array(total_2q_gates_list)
        p25, p75, p90 = np.percentile(a, [25, 75, 90])
        print(f"  Percentiles: p25={p25:.0f}, p75={p75:.0f}, p90={p90:.0f}")

        # 5. Algorithm families
        print(f"\n  ALGORITHM FAMILIES ({len(algo_counter)} unique):")
        for algo, count in algo_counter.most_common(30):
            print(f"    {algo:40s}: {count:4d}")
        if len(algo_counter) > 30:
            remaining = sum(c for _, c in algo_counter.most_common()[30:])
            print(f"    {'... (' + str(len(algo_counter) - 30) + ' more)':40s}: {remaining:4d}")

        print()


    # Cross-source summary table
    print(f"\n{'=' * 90}")
    print(f"  CROSS-SOURCE SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Source':>12s} | {'Count':>5s} | {'Qubits':>16s} | {'Edges':>16s} | "
          f"{'Density':>10s} | {'2Q Gates':>16s} | {'Algos':>5s}")
    print(f"{'-'*12:>12s}-+-{'-'*5:>5s}-+-{'-'*16:>16s}-+-{'-'*16:>16s}-+-"
          f"{'-'*10:>10s}-+-{'-'*16:>16s}-+-{'-'*5:>5s}")

    all_qubits = []
    all_edges = []
    all_density = []
    all_2q = []

    for source in ["queko", "mlqd", "mqt_bench", "qasmbench", "revlib"]:
        circuits = by_source.get(source, [])
        if not circuits:
            continue

        nq_list = []
        ne_list = []
        dn_list = []
        tq_list = []
        algo_set = set()

        for entry in circuits:
            algo = extract_algorithm_family(source, entry["file"])
            algo_set.add(algo)
            cache = load_cache(source, entry["file"])
            if cache is None:
                continue

            if "num_logical" in cache:
                nq = cache["num_logical"]
            elif "node_features_dict" in cache:
                nfd_tmp = cache["node_features_dict"]
                first_key = next(iter(nfd_tmp))
                nq = len(nfd_tmp[first_key])
            else:
                continue

            if "edge_list" in cache and cache["edge_list"] is not None:
                ne = len(cache["edge_list"])
            else:
                ne = 0

            total_2q = 0.0
            if "node_features_dict" in cache:
                nfd = cache["node_features_dict"]
                if "two_qubit_gate_count" in nfd:
                    v = nfd["two_qubit_gate_count"]
                    total_2q = float(sum(v)) if not hasattr(v, 'sum') else float(v.sum())

            nq_list.append(nq)
            ne_list.append(ne)
            if nq > 0:
                dn_list.append(ne / nq)
            tq_list.append(total_2q)

        all_qubits.extend(nq_list)
        all_edges.extend(ne_list)
        all_density.extend(dn_list)
        all_2q.extend(tq_list)

        nqa = np.array(nq_list)
        nea = np.array(ne_list)
        dna = np.array(dn_list)
        tqa = np.array(tq_list)

        print(f"{source:>12s} | {len(circuits):5d} | "
              f"{np.median(nqa):5.0f} [{np.min(nqa):3.0f}-{np.max(nqa):3.0f}] | "
              f"{np.median(nea):5.0f} [{np.min(nea):3.0f}-{np.max(nea):3.0f}] | "
              f"{np.mean(dna):6.2f}    | "
              f"{np.median(tqa):7.0f} [{np.min(tqa):5.0f}-{np.max(tqa):.0f}] | "
              f"{len(algo_set):5d}")

    # Total row
    nqa = np.array(all_qubits)
    nea = np.array(all_edges)
    dna = np.array(all_density)
    tqa = np.array(all_2q)
    print(f"{'-'*12:>12s}-+-{'-'*5:>5s}-+-{'-'*16:>16s}-+-{'-'*16:>16s}-+-"
          f"{'-'*10:>10s}-+-{'-'*16:>16s}-+-{'-'*5:>5s}")
    print(f"{'TOTAL':>12s} | {len(all_qubits):5d} | "
          f"{np.median(nqa):5.0f} [{np.min(nqa):3.0f}-{np.max(nqa):3.0f}] | "
          f"{np.median(nea):5.0f} [{np.min(nea):3.0f}-{np.max(nea):3.0f}] | "
          f"{np.mean(dna):6.2f}    | "
          f"{np.median(tqa):7.0f} [{np.min(tqa):5.0f}-{np.max(tqa):.0f}] | "
          f"  n/a")

    # Overall size bucket distribution
    print(f"\n  OVERALL SIZE DISTRIBUTION:")
    bucket_counts = Counter(size_bucket(int(n)) for n in all_qubits)
    for b in ["2-4", "5-10", "11-20", "21-50", "51+"]:
        c = bucket_counts.get(b, 0)
        pct = 100 * c / len(all_qubits)
        bar = "#" * int(pct / 2)
        print(f"    {b:>5s}: {c:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
