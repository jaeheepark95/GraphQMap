#!/usr/bin/env python3
from __future__ import annotations

"""
Build train / val / test JSONL manifest files for the KMW project.

This script scans circuit files on disk, extracts lightweight metadata from each
QASM file, applies simple filtering rules, and writes one JSONL file per split.

Important design choice:
The manifest files are the *authority* for which samples belong to each split.
That means later code should reuse these files instead of randomly re-splitting
circuits every time.
"""

import argparse
import random
from pathlib import Path
from typing import Any

from kmw.preprocessing.pipeline import inspect_circuit_file
from kmw.utils import stable_id, write_jsonl

# We only consider files with these suffixes as circuit files.
VALID_SUFFIXES = {".qasm", ".qasm2", ".qasm3"}



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Keeping argument parsing in its own function makes the script cleaner and
    easier to test.
    """
    parser = argparse.ArgumentParser(description="Build JSONL manifests for KMW datasets.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--circuits-root",
        type=Path,
        default=None,
        help="Override the circuit root. Defaults to <project-root>/data/circuits.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["mqt"],
        help="Dataset source folder names under data/circuits/.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-qubits", type=int, default=2)
    parser.add_argument("--max-qubits", type=int, default=27)
    parser.add_argument(
        "--allow-no-2q",
        action="store_true",
        help="If set, keep circuits even when they have no 2-qubit gate.",
    )
    return parser.parse_args()



def iter_qasm_files(root: Path) -> list[Path]:
    """Recursively collect all QASM-like files under a folder."""
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )



def build_manifest_row(
    *,
    project_root: Path,
    source: str,
    qasm_path: Path,
    min_qubits: int,
    max_qubits: int,
    require_two_qubit_gate: bool,
) -> dict[str, Any]:
    """Create one manifest row for one circuit file.

    We start by building a conservative default row that marks the sample as not
    yet parsed / not yet included. Then we try to inspect the circuit.

    If parsing fails, we still keep a row in the manifest so the failure is
    visible during debugging.
    """
    relpath = qasm_path.resolve().relative_to(project_root.resolve()).as_posix()
    row_id = stable_id(source, relpath, prefix=source)

    # Default row: assume failure until proven otherwise.
    row: dict[str, Any] = {
        "id": row_id,
        "source": source,
        "split": "",  # Filled in later after we assign splits.
        "qasm_relpath": relpath,
        "k_logical": None,
        "num_1q": None,
        "num_2q": None,
        "is_disconnected_logical_graph": None,
        "passed_parse": False,
        "passed_filter": False,
        "filter_tags": [],
        "include": False,
        "cache_key": row_id,
    }

    try:
        summary = inspect_circuit_file(qasm_path, max_qubits=max_qubits)
    except Exception as exc:
        # Keep the row, but record why it failed.
        row["filter_tags"] = [f"parse_or_feature_error:{exc.__class__.__name__}"]
        return row

    # If parsing worked, update the row with extracted metadata.
    row.update(
        {
            "id": summary["id"],
            "k_logical": summary["k_logical"],
            "num_1q": summary["num_1q"],
            "num_2q": summary["num_2q"],
            "is_disconnected_logical_graph": summary["is_disconnected_logical_graph"],
            "passed_parse": True,
            "cache_key": summary["id"],
        }
    )

    # Apply simple filtering rules.
    tags: list[str] = []
    if summary["k_logical"] < min_qubits:
        tags.append("too_few_qubits")
    if summary["k_logical"] > max_qubits:
        tags.append("too_many_qubits")
    if require_two_qubit_gate and summary["num_2q"] < 1:
        tags.append("no_two_qubit_gate")
    if summary["offdiag_mass_raw"] <= 0:
        tags.append("degenerate_zero_interaction")

    row["filter_tags"] = tags
    row["passed_filter"] = len(tags) == 0
    row["include"] = row["passed_parse"] and row["passed_filter"]
    return row



def assign_splits(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    """Split rows into train / val / test.

    We shuffle with a fixed seed so the split is reproducible.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    # Guard against rounding pushing us past n.
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    n_test = n - n_train - n_val

    split_map = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }

    # Write the split name back into every row.
    for split_name, split_rows in split_map.items():
        for row in split_rows:
            row["split"] = split_name

    return split_map



def main() -> None:
    """Main entry point for the manifest build script."""
    args = parse_args()

    project_root = args.project_root.resolve()
    circuits_root = (args.circuits_root or (project_root / "data" / "circuits")).resolve()
    output_dir = (args.output_dir or (project_root / "data" / "manifests")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    # Scan every requested source folder, e.g. data/circuits/mqt
    for source in args.sources:
        source_root = circuits_root / source
        if not source_root.exists():
            raise FileNotFoundError(f"Source root not found: {source_root}")

        qasm_files = iter_qasm_files(source_root)
        for qasm_path in qasm_files:
            row = build_manifest_row(
                project_root=project_root,
                source=source,
                qasm_path=qasm_path,
                min_qubits=args.min_qubits,
                max_qubits=args.max_qubits,
                require_two_qubit_gate=not args.allow_no_2q,
            )
            all_rows.append(row)

    split_map = assign_splits(
        all_rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # Save one JSONL file per split.
    for split_name, rows in split_map.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)

    included = sum(1 for row in all_rows if row["include"])
    print(f"Wrote manifests to {output_dir}")
    print(f"Discovered rows: {len(all_rows)}")
    print(f"Included rows:   {included}")
    for split_name, rows in split_map.items():
        split_included = sum(1 for row in rows if row["include"])
        print(f"  {split_name:<5} total={len(rows):>5} included={split_included:>5}")


if __name__ == "__main__":
    main()
