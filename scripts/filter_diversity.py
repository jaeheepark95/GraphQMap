"""Strong diversity filter — collapse near-duplicate circuits.

For each dataset category, two circuits are considered structural duplicates
if they share the same fingerprint:
    fingerprint = (num_qubits, num_edges, sorted_degree_sequence)

This catches the bulk of true duplicates produced by:
- QUEKO: random seed variants of identical (depth, density, backend) configs
- MLQD: small circuits (4-7Q) with identical interaction graphs across
        different algorithm names (basis_trotter / dnn / ising / vqe ...)
- MQT/RevLib: indexed instances of the same algorithm at the same size

Strategy: keep K representatives per fingerprint per source. K is configurable
per source (default K=1 = strongest possible filter).

Selection within a cluster: deterministic (alphabetical by stem) for
reproducibility.

Reads:
  data/circuits/cache/{source}/*.pt
  data/circuits/splits/train_all.json (and other splits)

Writes (when --apply):
  data/circuits/splits/{stage2_*, val*}.json   (in-place)
  data/circuits/splits/filter_log.json                   (extended)
  data/circuits/splits/diversity_filter_log.json         (per-cluster details)

Usage:
    python scripts/filter_diversity.py                # dry run
    python scripts/filter_diversity.py --apply        # commit
    python scripts/filter_diversity.py --k-mlqd 3     # custom per-source K
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
DATASETS = ["queko", "mlqd", "mqt_bench", "qasmbench", "revlib"]
SPLIT_FILES = [
    "train_all.json",
    "val.json",
    "val_queko_only.json",
]


def fingerprint(cache_path: Path) -> tuple | None:
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
    deg = [0] * nq
    for u, v in edge_list:
        if 0 <= u < nq:
            deg[u] += 1
        if 0 <= v < nq:
            deg[v] += 1
    return (nq, len(edge_list), tuple(sorted(deg)))


def select_keepers(active_files: list[Path], k: int) -> tuple[set[str], dict[tuple, list[str]]]:
    """Group files by fingerprint, return (kept_stems, fp_to_members)."""
    fp_to_files: dict[tuple, list[str]] = defaultdict(list)
    for f in active_files:
        fp = fingerprint(f)
        if fp is None:
            continue
        fp_to_files[fp].append(f.stem)

    kept: set[str] = set()
    for fp, members in fp_to_files.items():
        members_sorted = sorted(members)
        kept.update(members_sorted[:k])
    return kept, fp_to_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Strong diversity filter")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    parser.add_argument("--k-queko", type=int, default=1)
    parser.add_argument("--k-mlqd", type=int, default=1)
    parser.add_argument("--k-mqt-bench", type=int, default=1)
    parser.add_argument("--k-qasmbench", type=int, default=1)
    parser.add_argument("--k-revlib", type=int, default=1)
    args = parser.parse_args()

    K_PER_SOURCE = {
        "queko": args.k_queko,
        "mlqd": args.k_mlqd,
        "mqt_bench": args.k_mqt_bench,
        "qasmbench": args.k_qasmbench,
        "revlib": args.k_revlib,
    }

    # Load active set membership from current train_all.json
    stage2 = json.load(open(SPLITS_DIR / "train_all.json"))
    active = {(e["source"], Path(e["file"]).stem) for e in stage2}

    # For each dataset, find which active circuits to keep
    keepers: dict[str, set[str]] = {}
    fp_details: dict[str, dict] = {}
    for src in DATASETS:
        cdir = CACHE_ROOT / src
        if not cdir.exists():
            continue
        active_files = sorted([f for f in cdir.glob("*.pt") if (src, f.stem) in active])
        k = K_PER_SOURCE[src]
        kept, fp_to_files = select_keepers(active_files, k)
        keepers[src] = kept
        fp_details[src] = {
            "k": k,
            "before": len(active_files),
            "after": len(kept),
            "removed": len(active_files) - len(kept),
            "unique_fp": len(fp_to_files),
            "max_cluster": max((len(v) for v in fp_to_files.values()), default=0),
        }
        logger.info(
            "%s: K=%d, %d -> %d (removed %d, unique_fp=%d, max_cluster=%d)",
            src, k, len(active_files), len(kept),
            len(active_files) - len(kept),
            len(fp_to_files), fp_details[src]["max_cluster"],
        )

    total_before = sum(d["before"] for d in fp_details.values())
    total_after = sum(d["after"] for d in fp_details.values())
    logger.info("TOTAL: %d -> %d (removed %d, %.1f%% reduction)",
                total_before, total_after, total_before - total_after,
                100 * (total_before - total_after) / max(total_before, 1))

    # Compute per-split impact
    logger.info("\nPer-split impact:")
    split_results: dict[str, tuple] = {}
    for split_name in SPLIT_FILES:
        path = SPLITS_DIR / split_name
        if not path.exists():
            continue
        entries = json.load(open(path))
        kept_entries = []
        removed_entries = []
        for e in entries:
            src = e["source"]
            stem = Path(e["file"]).stem
            if src in keepers and stem not in keepers[src]:
                removed_entries.append(e)
            else:
                kept_entries.append(e)
        split_results[split_name] = (len(entries), len(kept_entries),
                                     kept_entries, removed_entries)
        logger.info("  %-25s %d -> %d (removed %d)",
                    split_name, len(entries), len(kept_entries), len(removed_entries))

    if not args.apply:
        logger.info("\n[DRY RUN] No files written. Use --apply to commit.")
        return

    # Write updated splits
    logger.info("\nWriting updated splits:")
    for split_name, (orig, new, kept_entries, removed_entries) in split_results.items():
        if orig == new:
            continue
        path = SPLITS_DIR / split_name
        with open(path, "w") as f:
            json.dump(kept_entries, f, indent=2)
        logger.info("  %s: %d -> %d", split_name, orig, new)

    # Extend filter_log.json
    flog_path = SPLITS_DIR / "filter_log.json"
    flog = json.load(open(flog_path))
    existing_keys = {(r["source"], r["stem"]) for r in flog["removed_circuits"]}
    appended = 0
    for src in DATASETS:
        cdir = CACHE_ROOT / src
        if not cdir.exists():
            continue
        for f in sorted(cdir.glob("*.pt")):
            if (src, f.stem) not in active:
                continue
            if f.stem in keepers.get(src, set()):
                continue
            if (src, f.stem) in existing_keys:
                continue
            flog["removed_circuits"].append({
                "key": f"{src}/{f.stem}",
                "source": src,
                "stem": f.stem,
                "reason": f"diversity_filter (k={K_PER_SOURCE[src]} per fingerprint)",
            })
            appended += 1
    flog["total_removed"] = len(flog["removed_circuits"])
    flog["diversity_filter_applied"] = True
    flog["diversity_filter_k"] = K_PER_SOURCE
    with open(flog_path, "w") as f:
        json.dump(flog, f, indent=2)
    logger.info("  filter_log.json: appended %d entries (total removed=%d)",
                appended, flog["total_removed"])

    # Write detailed diversity filter log
    div_log = {
        "k_per_source": K_PER_SOURCE,
        "per_source": fp_details,
        "totals": {
            "before": total_before,
            "after": total_after,
            "removed": total_before - total_after,
        },
    }
    div_path = SPLITS_DIR / "diversity_filter_log.json"
    with open(div_path, "w") as f:
        json.dump(div_log, f, indent=2)
    logger.info("  diversity_filter_log.json written")
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
