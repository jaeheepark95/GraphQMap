"""Curated dataset filter — select ~150-200 high-quality diverse circuits.

Strategy:
  - QASMBench: keep ALL (39 circuits, 29 algorithm families — highest per-circuit diversity)
  - RevLib: keep ALL (113 circuits, 88 algorithm families — unique reversible logic)
  - MQT Bench: stratified selection by (num_qubits, density_bucket), QAOA capped
  - MLQD: small quota for small-circuit coverage (2-15Q)
  - QUEKO: small quota for medium-large coverage (16-54Q)

Selection within each group: deterministic (alphabetical by stem).

Usage:
    python scripts/filter_curated.py                # dry run — show what would be selected
    python scripts/filter_curated.py --apply        # write new split file
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_ROOT = Path("data/circuits/cache")
SPLITS_DIR = Path("data/circuits/splits")

# Density buckets: edges / qubits ratio
DENSITY_BUCKETS = [
    ("sparse", 0.0, 0.5),
    ("low", 0.5, 1.0),
    ("medium", 1.0, 2.0),
    ("high", 2.0, 5.0),
    ("dense", 5.0, float("inf")),
]

SIZE_BUCKETS = [
    ("2-4Q", 2, 4),
    ("5-10Q", 5, 10),
    ("11-20Q", 11, 20),
    ("21-50Q", 21, 50),
    ("51Q+", 51, 9999),
]


def get_density_bucket(density: float) -> str:
    for name, lo, hi in DENSITY_BUCKETS:
        if lo <= density < hi:
            return name
    return "dense"


def get_size_bucket(nq: int) -> str:
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= nq <= hi:
            return name
    return "51Q+"


def load_circuit_info(cache_path: Path) -> dict | None:
    """Load basic circuit info from cache file."""
    try:
        d = torch.load(cache_path, weights_only=False, map_location="cpu")
    except Exception:
        return None
    if not isinstance(d, dict) or "node_features_dict" not in d:
        return None
    nfd = d["node_features_dict"]
    nq = int(d.get("num_logical", len(next(iter(nfd.values())))))
    edge_list = d.get("edge_list", [])
    ne = len(edge_list)
    density = ne / max(nq, 1)

    # Total 2Q gate count
    tqgc = nfd.get("two_qubit_gate_count", [0] * nq)
    total_2q = sum(tqgc) if hasattr(tqgc, '__iter__') else 0

    # Extract algorithm family from filename
    stem = cache_path.stem
    # Remove common suffixes like _n5, _indep_qiskit_5, etc.
    algo = stem.split("_")[0] if "_" in stem else stem

    return {
        "stem": stem,
        "nq": nq,
        "ne": ne,
        "density": density,
        "total_2q": total_2q,
        "density_bucket": get_density_bucket(density),
        "size_bucket": get_size_bucket(nq),
        "algo": algo,
    }


def select_stratified(
    circuits: list[dict],
    k_per_cell: int = 1,
    algo_cap: dict[str, int] | None = None,
    coarse: bool = False,
) -> list[dict]:
    """Select k circuits per (num_qubits, density_bucket) cell.

    Args:
        circuits: List of circuit info dicts.
        k_per_cell: Max circuits per cell.
        algo_cap: Optional per-algorithm cap {algo_prefix: max_count}.
        coarse: If True, use (size_bucket, density_bucket) instead of (nq, density_bucket).
    """
    # Sort deterministically
    circuits_sorted = sorted(circuits, key=lambda c: c["stem"])

    # Apply algorithm cap first
    if algo_cap:
        algo_counts: dict[str, int] = defaultdict(int)
        capped = []
        for c in circuits_sorted:
            algo = c["algo"].lower()
            cap = None
            for prefix, limit in algo_cap.items():
                if algo.startswith(prefix.lower()):
                    cap = limit
                    break
            if cap is not None and algo_counts[algo] >= cap:
                continue
            algo_counts[algo] += 1
            capped.append(c)
        circuits_sorted = capped

    # Stratify by cell key
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for c in circuits_sorted:
        if coarse:
            key = (c["size_bucket"], c["density_bucket"])
        else:
            key = (c["nq"], c["density_bucket"])
        cells[key].append(c)

    selected = []
    for key in sorted(cells.keys()):
        members = cells[key]
        # Pick up to k, preferring median complexity
        if len(members) <= k_per_cell:
            selected.extend(members)
        else:
            # Sort by total_2q and pick evenly spaced
            by_complexity = sorted(members, key=lambda c: c["total_2q"])
            if k_per_cell == 1:
                # Pick median
                selected.append(by_complexity[len(by_complexity) // 2])
            else:
                step = len(by_complexity) / k_per_cell
                for i in range(k_per_cell):
                    idx = int(i * step)
                    selected.append(by_complexity[idx])

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Curated dataset filter")
    parser.add_argument("--apply", action="store_true", help="Write new split file")
    parser.add_argument("--mqt-k", type=int, default=1, help="K per cell for MQT Bench")
    parser.add_argument("--mlqd-k", type=int, default=1, help="K per cell for MLQD")
    parser.add_argument("--queko-k", type=int, default=1, help="K per cell for QUEKO")
    parser.add_argument("--coarse", action="store_true",
                        help="Use (size_bucket, density_bucket) instead of (nq, density_bucket)")
    parser.add_argument("--qaoa-cap", type=int, default=5, help="Max QAOA circuits from MQT")
    args = parser.parse_args()

    # Load current split
    split_path = SPLITS_DIR / "stage2_all.json"
    stage2 = json.load(open(split_path))
    active = {}
    for e in stage2:
        key = (e["source"], Path(e["file"]).stem)
        active[key] = e

    # Load circuit info for all active circuits
    by_source: dict[str, list[dict]] = defaultdict(list)
    for (src, stem), entry in active.items():
        cache_path = CACHE_ROOT / src / f"{stem}.pt"
        info = load_circuit_info(cache_path)
        if info is None:
            continue
        info["source"] = src
        info["entry"] = entry
        by_source[src].append(info)

    # === Selection strategy ===
    selected: list[dict] = []

    # 1. QASMBench: keep ALL
    qasmbench = by_source.get("qasmbench", [])
    selected.extend(qasmbench)
    logger.info(f"QASMBench: {len(qasmbench)} → {len(qasmbench)} (keep all)")

    # 2. RevLib: keep ALL
    revlib = by_source.get("revlib", [])
    selected.extend(revlib)
    logger.info(f"RevLib: {len(revlib)} → {len(revlib)} (keep all)")

    # 3. MQT Bench: stratified, QAOA capped
    mqt = by_source.get("mqt_bench", [])
    mqt_selected = select_stratified(
        mqt,
        k_per_cell=args.mqt_k,
        algo_cap={"qaoa": args.qaoa_cap},
        coarse=args.coarse,
    )
    selected.extend(mqt_selected)
    logger.info(f"MQT Bench: {len(mqt)} → {len(mqt_selected)} (k={args.mqt_k}, QAOA cap={args.qaoa_cap}, coarse={args.coarse})")

    # 4. MLQD: stratified (always coarse — too many unique nq values for low-diversity source)
    mlqd = by_source.get("mlqd", [])
    mlqd_selected = select_stratified(mlqd, k_per_cell=args.mlqd_k, coarse=True)
    selected.extend(mlqd_selected)
    logger.info(f"MLQD: {len(mlqd)} → {len(mlqd_selected)} (k={args.mlqd_k}, coarse=True)")

    # 5. QUEKO: stratified (always coarse — only 3 algorithms)
    queko = by_source.get("queko", [])
    queko_selected = select_stratified(queko, k_per_cell=args.queko_k, coarse=True)
    selected.extend(queko_selected)
    logger.info(f"QUEKO: {len(queko)} → {len(queko_selected)} (k={args.queko_k}, coarse=True)")

    # === Report ===
    total = len(selected)
    logger.info(f"\nTotal selected: {total} circuits (from {len(stage2)})")

    # Per-source summary
    print("\n" + "=" * 70)
    print(f"CURATED DATASET: {total} circuits")
    print("=" * 70)

    source_counts = defaultdict(int)
    for c in selected:
        source_counts[c["source"]] += 1
    print("\nPer-source:")
    for src in ["qasmbench", "revlib", "mqt_bench", "mlqd", "queko"]:
        n = source_counts.get(src, 0)
        orig = len(by_source.get(src, []))
        print(f"  {src:12s}: {orig:4d} → {n:4d}")

    # Per-size summary
    print("\nPer-size bucket:")
    size_counts = defaultdict(int)
    for c in selected:
        size_counts[c["size_bucket"]] += 1
    for name, _, _ in SIZE_BUCKETS:
        print(f"  {name:8s}: {size_counts.get(name, 0):4d}")

    # Per-density summary
    print("\nPer-density bucket:")
    density_counts = defaultdict(int)
    for c in selected:
        density_counts[c["density_bucket"]] += 1
    for name, _, _ in DENSITY_BUCKETS:
        print(f"  {name:8s}: {density_counts.get(name, 0):4d}")

    # Size x density matrix
    print("\nSize × Density matrix:")
    header = f"{'':10s}" + "".join(f"{name:>8s}" for name, _, _ in DENSITY_BUCKETS) + f"{'Total':>8s}"
    print(header)
    for sz_name, sz_lo, sz_hi in SIZE_BUCKETS:
        row = f"{sz_name:10s}"
        row_total = 0
        for d_name, _, _ in DENSITY_BUCKETS:
            cnt = sum(1 for c in selected
                      if c["size_bucket"] == sz_name and c["density_bucket"] == d_name)
            row += f"{cnt:8d}"
            row_total += cnt
        row += f"{row_total:8d}"
        print(row)

    # Algorithm diversity
    algos = set(c["algo"] for c in selected)
    print(f"\nUnique algorithm families: {len(algos)}")

    # Qubit count coverage
    nqs = sorted(set(c["nq"] for c in selected))
    print(f"Unique qubit counts: {len(nqs)} ({min(nqs)}-{max(nqs)}Q)")

    if args.apply:
        # Build new split
        new_split = [c["entry"] for c in sorted(selected, key=lambda c: (c["source"], c["stem"]))]
        out_path = SPLITS_DIR / "stage2_curated.json"
        with open(out_path, "w") as f:
            json.dump(new_split, f, indent=2)
        logger.info(f"\nSaved curated split: {out_path} ({len(new_split)} circuits)")

        # Also create a matching validation split
        # Load current val.json and apply same source filter (keep all val circuits that exist)
        val_path = SPLITS_DIR / "val.json"
        if val_path.exists():
            val_data = json.load(open(val_path))
            logger.info(f"Validation split unchanged: {val_path} ({len(val_data)} circuits)")
    else:
        print("\n[DRY RUN] Use --apply to write the split file.")


if __name__ == "__main__":
    main()
