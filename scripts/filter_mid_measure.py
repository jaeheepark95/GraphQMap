"""Remove circuits with mid-circuit measurements from training/val splits.

A mid-circuit measurement is any non-measure/barrier op on a qubit that
occurs after that qubit has been measured. Such circuits cannot be modeled
correctly by the GraphQMap circuit graph (which represents each logical
qubit as a single node) and may carry inconsistent labels (most solvers
assume unitary-only circuits).

Reads:
  data/circuits/splits/mid_measure_log.json   (produced by check_mid_measure.py)

Writes (when --apply):
  data/circuits/splits/{stage1_*, stage2_*, val*}.json   (in-place update)
  data/circuits/splits/filter_log.json                   (extended with mid_measure entries)

Usage:
    python scripts/filter_mid_measure.py            # dry run
    python scripts/filter_mid_measure.py --apply    # write
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SPLITS_DIR = Path("data/circuits/splits")
SPLIT_FILES = [
    "stage1_supervised.json",
    "stage1_queko_only.json",
    "stage1_unsupervised.json",
    "stage2_all.json",
    "val.json",
    "val_queko_only.json",
]


def load_mid_measure_keys() -> dict[tuple[str, str], dict]:
    """Return {(source, stem): metadata} for all mid-measure circuits."""
    log_path = SPLITS_DIR / "mid_measure_log.json"
    if not log_path.exists():
        raise FileNotFoundError(
            f"{log_path} not found. Run scripts/check_mid_measure.py first."
        )
    log = json.load(open(log_path))
    keys: dict[tuple[str, str], dict] = {}
    for src, info in log["by_source"].items():
        for o in info["offenders"]:
            stem = Path(o["file"]).stem
            keys[(src, stem)] = {
                "num_qubits": o["num_qubits"],
                "offending_qubits": o["offending_qubits"],
            }
    return keys


def update_split(split_path: Path, remove_set: set[tuple[str, str]]) -> tuple[int, int, list[dict]]:
    """Return (original_count, new_count, removed_entries)."""
    entries = json.load(open(split_path))
    kept = []
    removed = []
    for e in entries:
        key = (e["source"], Path(e["file"]).stem)
        if key in remove_set:
            removed.append(e)
        else:
            kept.append(e)
    return len(entries), len(kept), removed, kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter mid-circuit measurement circuits from splits")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    args = parser.parse_args()

    mm_keys = load_mid_measure_keys()
    remove_set = set(mm_keys.keys())
    logger.info("Mid-measure circuits to remove: %d", len(remove_set))

    # Cross-reference with existing indist filter
    flog_path = SPLITS_DIR / "filter_log.json"
    flog = json.load(open(flog_path))
    already_filtered = {(r["source"], r["stem"]) for r in flog["removed_circuits"]}
    overlap = remove_set & already_filtered
    new_removals = remove_set - already_filtered
    logger.info("  Already removed by indist filter: %d", len(overlap))
    logger.info("  New removals (not in indist filter): %d", len(new_removals))

    # Per-split impact
    logger.info("\nPer-split impact:")
    split_changes: dict[str, tuple[int, int, list[dict]]] = {}
    for split_name in SPLIT_FILES:
        path = SPLITS_DIR / split_name
        if not path.exists():
            continue
        original_count, new_count, removed, kept = update_split(path, remove_set)
        split_changes[split_name] = (original_count, new_count, removed, kept)
        logger.info(
            "  %-25s %d -> %d (removed %d)",
            split_name, original_count, new_count, len(removed),
        )

    if not args.apply:
        logger.info("\n[DRY RUN] No files written. Use --apply to commit changes.")
        return

    # Write updated splits
    logger.info("\nWriting updated splits:")
    for split_name, (orig, new, removed, kept) in split_changes.items():
        if not removed:
            continue
        path = SPLITS_DIR / split_name
        with open(path, "w") as f:
            json.dump(kept, f, indent=2)
        logger.info("  %s: %d -> %d", split_name, orig, new)

    # Extend filter_log.json with mid_measure entries
    existing_keys = {(r["source"], r["stem"]) for r in flog["removed_circuits"]}
    appended = 0
    for (src, stem), meta in mm_keys.items():
        if (src, stem) in existing_keys:
            continue
        flog["removed_circuits"].append({
            "key": f"{src}/{stem}",
            "source": src,
            "stem": stem,
            "num_qubits": meta["num_qubits"],
            "indist_rate": None,
            "indist_pairs": None,
            "total_pairs": None,
            "reason": f"mid_measure (offending_qubits={meta['offending_qubits']})",
        })
        appended += 1
    flog["total_removed"] = len(flog["removed_circuits"])
    flog.setdefault("mid_measure_filter_applied", True)
    with open(flog_path, "w") as f:
        json.dump(flog, f, indent=2)
    logger.info("  filter_log.json: appended %d mid_measure entries (total removed=%d)",
                appended, flog["total_removed"])
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
