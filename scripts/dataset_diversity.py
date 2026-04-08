"""Per-category circuit diversity analysis.

Goal: assess whether each dataset category contains *enough variety of
circuits* for the model to learn good qubit mappings — i.e., is the
model seeing many genuinely different patterns, or repetitions of a few?

For each category, compute:

  Algorithm diversity:
    unique_algos        — distinct algorithm names (parsed from filename)
    top_algo_pct        — % of circuits from the most common algorithm
    top3_algo_pct       — % from the top-3 algorithms (concentration)
    algo_entropy        — Shannon entropy of algorithm distribution (bits)
    algo_entropy_norm   — entropy / log2(unique_algos)  (0..1)

  Structural diversity:
    nq_unique           — distinct num_qubits values
    nq_entropy          — Shannon entropy of nq distribution (bits)
    nq_entropy_norm     — normalized
    topo_dominant       — most common topology pattern + %

  Near-duplicate clustering (structural fingerprint):
    fp = (num_qubits, num_edges, sorted_degree_sequence)
    unique_fp           — number of distinct fingerprints
    avg_cluster_size    — mean circuits per fingerprint (= total / unique_fp)
    p50_cluster_size    — median fingerprint cluster size
    max_cluster_size    — largest cluster
    singleton_pct       — % of circuits in singleton fingerprint clusters
    effective_unique    — exp(entropy of cluster distribution) — "effective
                          number of unique circuits" accounting for repetition

Active set only (post-filter, from stage2_all.json).

Output:
  - stdout markdown table
  - data/circuits/splits/dataset_diversity.md
  - data/circuits/splits/dataset_diversity.csv
  - per-category MLQD subcategory breakdown
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_ROOT = Path("data/circuits/cache")
QASM_ROOT = Path("data/circuits/qasm")
SPLITS_DIR = Path("data/circuits/splits")
DATASETS = ["queko", "mlqd", "mqt_bench", "qasmbench", "revlib"]

# ======================================================================
# Algorithm name parsers (one per dataset)
# ======================================================================

MLQD_BACKEND_TOKENS = {"melbourne", "mlqd", "grid5x5", "queko",
                       "aspen4", "rochester", "sycamore"}
MLQD_CONFIG_TOKENS = {"only", "1qu", "2qu", "only2qu", "qasmbench"}


def parse_algo(source: str, stem: str) -> str:
    """Return canonical algorithm name from filename stem (no .qasm)."""
    if source == "queko":
        # Pattern A: {nq}QBT_{cyc}CYC_{ALGO}_{idx}      (TFL/QSE)
        m = re.match(r"\d+QBT_\d+CYC_([A-Z]+)_\d+", stem)
        if m:
            return m.group(1)
        # Pattern B: {nq}QBT_{cyc}CYC_.{a}D1_.{b}D2_{idx}   (random)
        m = re.match(r"\d+QBT_\d+CYC_\.\d+D1_\.\d+D2_\d+", stem)
        if m:
            return "RAND_DENSITY"
        return "OTHER"

    if source == "mqt_bench":
        # {algo}_n{nq}
        m = re.match(r"(.+?)_n\d+$", stem)
        return m.group(1) if m else stem

    if source == "qasmbench":
        # {size}_{algo}_n{nq}_{algo}_n{nq} or variants
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0] in ("small", "medium", "large"):
            for i in range(1, len(parts)):
                if re.match(r"n\d+$", parts[i]):
                    return "_".join(parts[1:i])
            return "_".join(parts[1:])
        return stem

    if source == "revlib":
        # {func}_{idx}
        return re.sub(r"_\d+$", "", stem)

    if source == "mlqd":
        # Strip trailing index, _transpiled, _n{nq}
        name = re.sub(r"_\d+$", "", stem)
        name = re.sub(r"_transpiled$", "", name)
        name = re.sub(r"_n\d+$", "", name)
        tokens = name.split("_")
        keep = []
        for t in tokens:
            if t in MLQD_BACKEND_TOKENS or t in MLQD_CONFIG_TOKENS:
                continue
            if re.match(r"^(qu|len|twoqugatenum)\d+$", t):
                continue
            if re.match(r"^\d+$", t):
                continue
            keep.append(t)
        return "_".join(keep) if keep else "unknown"

    return stem


# ======================================================================
# Structural metrics from cache
# ======================================================================

def analyze_cache(cache_path: Path) -> dict | None:
    try:
        d = torch.load(cache_path, weights_only=False, map_location="cpu")
    except Exception:
        return None
    if not isinstance(d, dict) or "node_features_dict" not in d:
        return None

    nfd = d["node_features_dict"]
    nq = int(d.get("num_logical", len(next(iter(nfd.values())))))
    if nq < 2:
        return None

    edge_list = d.get("edge_list", [])
    num_edges = len(edge_list)

    # Degree per qubit (undirected)
    deg = [0] * nq
    for u, v in edge_list:
        if 0 <= u < nq:
            deg[u] += 1
        if 0 <= v < nq:
            deg[v] += 1
    deg_sorted = tuple(sorted(deg))

    return {
        "num_qubits": nq,
        "num_edges": num_edges,
        "deg_sorted": deg_sorted,
        "max_deg": max(deg),
        "min_deg": min(deg),
        "density": num_edges / (nq * (nq - 1) / 2) if nq >= 2 else 0.0,
    }


def classify_topology(rec: dict) -> str:
    nq = rec["num_qubits"]
    ne = rec["num_edges"]
    density = rec["density"]
    deg = sorted(rec["deg_sorted"])

    if ne == 0:
        return "no_2q"
    if density >= 0.95:
        return "complete"
    if density >= 0.5:
        return "dense"
    # Star: one node with deg = nq-1, others with low degree
    if deg[-1] == nq - 1 and all(d <= 2 for d in deg[:-1]):
        return "star"
    # Ring: all degrees = 2 and num_edges = nq
    if all(d == 2 for d in deg) and ne == nq:
        return "ring"
    # Line / path: 2 endpoints with deg=1, rest deg=2, num_edges = nq-1
    if (
        ne == nq - 1
        and deg.count(1) == 2
        and deg.count(2) == nq - 2
    ):
        return "line"
    if ne == nq - 1:
        return "tree"
    if density < 0.3:
        return "sparse"
    return "moderate"


def shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    stage2 = json.load(open(SPLITS_DIR / "stage2_all.json"))
    active = {(e["source"], Path(e["file"]).stem) for e in stage2}

    rows = []
    cat_results: dict[str, dict] = {}

    for src in DATASETS:
        cdir = CACHE_ROOT / src
        if not cdir.exists():
            continue
        files = sorted(cdir.glob("*.pt"))
        logger.info("Analyzing %s (%d files)...", src, len(files))

        algos: list[str] = []
        nq_list: list[int] = []
        topos: list[str] = []
        fingerprints: list[tuple] = []
        recs_active: list[dict] = []

        for f in files:
            if (src, f.stem) not in active:
                continue
            r = analyze_cache(f)
            if r is None:
                continue
            algos.append(parse_algo(src, f.stem))
            nq_list.append(r["num_qubits"])
            topos.append(classify_topology(r))
            fingerprints.append((r["num_qubits"], r["num_edges"], r["deg_sorted"]))
            recs_active.append(r)

        n = len(recs_active)
        if n == 0:
            continue

        # Algorithm diversity
        algo_counts = Counter(algos)
        unique_algos = len(algo_counts)
        top1 = algo_counts.most_common(1)[0]
        top3 = algo_counts.most_common(3)
        top_algo_pct = 100 * top1[1] / n
        top3_algo_pct = 100 * sum(c for _, c in top3) / n
        algo_entropy = shannon_entropy(list(algo_counts.values()))
        algo_entropy_norm = (algo_entropy / math.log2(unique_algos)
                             if unique_algos > 1 else 0.0)

        # nq diversity
        nq_counts = Counter(nq_list)
        nq_unique = len(nq_counts)
        nq_entropy = shannon_entropy(list(nq_counts.values()))
        nq_entropy_norm = (nq_entropy / math.log2(nq_unique)
                           if nq_unique > 1 else 0.0)

        # Topology
        topo_counts = Counter(topos)
        topo_dom = topo_counts.most_common(1)[0]
        topo_dom_pct = 100 * topo_dom[1] / n

        # Fingerprint clustering
        fp_counts = Counter(fingerprints)
        unique_fp = len(fp_counts)
        cluster_sizes = list(fp_counts.values())
        avg_cluster = n / unique_fp
        p50_cluster = float(np.median(cluster_sizes))
        max_cluster = max(cluster_sizes)
        singleton_pct = 100 * sum(1 for c in cluster_sizes if c == 1) / unique_fp
        # effective unique = exp(H) where H is entropy of fingerprint distribution
        fp_entropy = shannon_entropy(cluster_sizes)
        effective_unique = 2 ** fp_entropy

        cat_results[src] = {
            "n": n,
            "unique_algos": unique_algos,
            "top_algo": top1[0],
            "top_algo_pct": top_algo_pct,
            "top3_algo_pct": top3_algo_pct,
            "algo_entropy": algo_entropy,
            "algo_entropy_norm": algo_entropy_norm,
            "nq_unique": nq_unique,
            "nq_entropy": nq_entropy,
            "nq_entropy_norm": nq_entropy_norm,
            "topo_dom": topo_dom[0],
            "topo_dom_pct": topo_dom_pct,
            "topo_counts": dict(topo_counts),
            "unique_fp": unique_fp,
            "avg_cluster": avg_cluster,
            "p50_cluster": p50_cluster,
            "max_cluster": max_cluster,
            "singleton_pct": singleton_pct,
            "effective_unique": effective_unique,
            "top3_algos": top3,
            "algo_counts": algo_counts,
        }

    # ---- Build markdown table ----
    headers = [
        "dataset", "n", "unique_algos", "top_algo(%)", "top3(%)",
        "algo_H", "algo_H_norm",
        "nq_unique", "nq_H_norm",
        "topology(dom%)",
        "unique_fp", "avg_clust", "max_clust", "singletons%", "eff_unique",
    ]
    lines = []
    lines.append("# Dataset Circuit Diversity Summary\n")
    lines.append(
        "Active (post-filter) circuits only. "
        "**algo_H_norm**=algorithm entropy normalized to [0,1]; closer to 1 = more uniform. "
        "**eff_unique**=exp(H) of fingerprint cluster distribution — "
        "\"effective number of distinct circuits\" accounting for repetition. "
        "Fingerprint = (num_qubits, num_edges, sorted_degree_sequence).\n"
    )
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    csv_rows = [headers]
    for src in DATASETS:
        s = cat_results.get(src)
        if not s:
            continue
        row = [
            src,
            s["n"],
            s["unique_algos"],
            f"{s['top_algo']}({s['top_algo_pct']:.1f})",
            f"{s['top3_algo_pct']:.1f}",
            f"{s['algo_entropy']:.2f}",
            f"{s['algo_entropy_norm']:.2f}",
            s["nq_unique"],
            f"{s['nq_entropy_norm']:.2f}",
            f"{s['topo_dom']}({s['topo_dom_pct']:.0f})",
            s["unique_fp"],
            f"{s['avg_cluster']:.1f}",
            s["max_cluster"],
            f"{s['singleton_pct']:.0f}",
            f"{s['effective_unique']:.0f}",
        ]
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
        csv_rows.append(row)

    table = "\n".join(lines)
    print()
    print(table)
    print()

    # ---- Per-category top algorithms ----
    print("\n## Top algorithms per category\n")
    for src in DATASETS:
        s = cat_results.get(src)
        if not s:
            continue
        print(f"### {src}")
        for algo, c in s["algo_counts"].most_common(8):
            pct = 100 * c / s["n"]
            print(f"  {algo:30s} {c:5d}  ({pct:.1f}%)")
        if s["unique_algos"] > 8:
            print(f"  ... and {s['unique_algos']-8} more")
        print()

    # ---- Topology distribution per category ----
    print("\n## Topology distribution\n")
    print(f"{'dataset':12s} " + " ".join(
        f"{t:>10s}" for t in ["no_2q","line","ring","tree","star",
                              "sparse","moderate","dense","complete"]))
    for src in DATASETS:
        s = cat_results.get(src)
        if not s:
            continue
        tc = s["topo_counts"]
        cells = []
        for t in ["no_2q","line","ring","tree","star",
                  "sparse","moderate","dense","complete"]:
            v = tc.get(t, 0)
            pct = 100 * v / s["n"]
            cells.append(f"{pct:9.1f}%")
        print(f"{src:12s} " + " ".join(cells))
    print()

    # ---- Top fingerprint clusters per category ----
    print("\n## Top near-duplicate clusters (largest fingerprint groups)\n")
    for src in DATASETS:
        s = cat_results.get(src)
        if not s:
            continue
        # We need fingerprints again to show details — recompute lightly
        cdir = CACHE_ROOT / src
        files = sorted(cdir.glob("*.pt"))
        fp_to_files: dict[tuple, list[str]] = defaultdict(list)
        for f in files:
            if (src, f.stem) not in active:
                continue
            r = analyze_cache(f)
            if r is None:
                continue
            fp = (r["num_qubits"], r["num_edges"], r["deg_sorted"])
            fp_to_files[fp].append(f.stem)
        top = sorted(fp_to_files.items(), key=lambda kv: -len(kv[1]))[:3]
        print(f"### {src}")
        for fp, members in top:
            nq, ne, _ = fp
            print(f"  fp(nq={nq}, edges={ne}): {len(members)} circuits")
            for m in members[:3]:
                print(f"    - {m}")
            if len(members) > 3:
                print(f"    ... and {len(members)-3} more")
        print()

    # ---- Save outputs ----
    md_path = SPLITS_DIR / "dataset_diversity.md"
    csv_path = SPLITS_DIR / "dataset_diversity.csv"
    md_path.write_text(table + "\n")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    logger.info("Saved: %s, %s", md_path, csv_path)

    # ---- Diagnoses ----
    print("\n## Diagnoses\n")
    for src in DATASETS:
        s = cat_results.get(src)
        if not s:
            continue
        flags = []
        if s["top_algo_pct"] > 30:
            flags.append(f"single algo dominates ({s['top_algo']}={s['top_algo_pct']:.0f}%)")
        if s["top3_algo_pct"] > 60:
            flags.append(f"top-3 algos = {s['top3_algo_pct']:.0f}% of data")
        if s["algo_entropy_norm"] < 0.6:
            flags.append(f"low algo entropy ({s['algo_entropy_norm']:.2f})")
        if s["effective_unique"] / s["n"] < 0.3:
            flags.append(
                f"effective_unique/n = {s['effective_unique']/s['n']:.2f} "
                "(many duplicates)"
            )
        if s["max_cluster"] > 20:
            flags.append(f"largest cluster = {s['max_cluster']} circuits")
        if s["nq_unique"] < 10:
            flags.append(f"only {s['nq_unique']} distinct qubit sizes")
        if s["topo_dom_pct"] > 70:
            flags.append(f"{s['topo_dom']} topology = {s['topo_dom_pct']:.0f}%")
        if not flags:
            flags = ["diverse"]
        print(f"  {src}: " + "; ".join(flags))


if __name__ == "__main__":
    main()
