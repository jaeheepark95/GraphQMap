# scripts/build_manifest_full.py
#!/usr/bin/env python3
from __future__ import annotations

"""
UPDATE NOTE (2026-03-23)
- New file intended path: scripts/build_manifest_full.py
- Leaves the existing scripts/build_manifests.py untouched.
- Preserves the current circuit discovery, row-building, parsing, and filtering logic.
- Replaces global split assignment with deterministic per-source split assignment.
- Writes primitive source manifests first, then deterministic combined recipe manifests.
- Targets data/circuits_v2/qasm and does not depend on circuit_v2/splits.
"""

import argparse
import hashlib
import itertools
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from kmw.preprocessing.pipeline import inspect_circuit_file
from kmw.utils import stable_id, write_jsonl

# We only consider files with these suffixes as circuit files.
VALID_SUFFIXES = {".qasm", ".qasm2", ".qasm3"}
SPLIT_NAMES = ("train", "val", "test")

DEFAULT_TRAIN_SIDE_SOURCES = ["queko", "mlqd", "mqt_bench"]
DEFAULT_HELDOUT_SOURCES = ["qasmbench", "revlib"]
DEFAULT_BENCHMARK_SOURCES = ["benchmarks"]
ALL_CANONICAL_SOURCES = (
    DEFAULT_TRAIN_SIDE_SOURCES
    + DEFAULT_HELDOUT_SOURCES
    + DEFAULT_BENCHMARK_SOURCES
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full circuit_v2 manifest builder."""
    parser = argparse.ArgumentParser(
        description="Build per-source and recipe JSONL manifests for KMW circuit_v2 datasets."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--circuits-root",
        type=Path,
        default=None,
        help="Override the circuit root. Defaults to <project-root>/data/circuits_v2/qasm.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the manifest output root. Defaults to <project-root>/data/manifests/full.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(ALL_CANONICAL_SOURCES),
        help="Canonical source folder names under data/circuits_v2/qasm/.",
    )
    parser.add_argument(
        "--train-side-sources",
        nargs="+",
        default=list(DEFAULT_TRAIN_SIDE_SOURCES),
        help="Source names classified as train_side.",
    )
    parser.add_argument(
        "--heldout-sources",
        nargs="+",
        default=list(DEFAULT_HELDOUT_SOURCES),
        help="Source names classified as heldout.",
    )
    parser.add_argument(
        "--benchmark-sources",
        nargs="+",
        default=list(DEFAULT_BENCHMARK_SOURCES),
        help="Source names classified as benchmark.",
    )
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
    parser.add_argument(
        "--emit-train-subset-recipes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit all non-empty train-side source subset recipes.",
    )
    parser.add_argument(
        "--emit-heldout-recipes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit held-out evaluation recipes.",
    )
    parser.add_argument(
        "--emit-benchmark-recipes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit benchmark recipe manifests.",
    )
    parser.add_argument(
        "--emit-source-manifests-only",
        action="store_true",
        help="If set, write only primitive per-source manifests and skip combined recipes.",
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


def stable_int_hash(text: str) -> int:
    """Return a deterministic machine-independent integer hash for text."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def assign_splits_for_one_source(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    """Split one source's rows into train / val / test using current ratio semantics."""
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_source_manifests(
    source: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> Path:
    """Write all.jsonl plus split JSONL files for one primitive source manifest."""
    source_dir = output_dir / "source_manifests" / source
    source_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(source_dir / "all.jsonl", rows_by_split["all"])
    for split_name in SPLIT_NAMES:
        write_jsonl(source_dir / f"{split_name}.jsonl", rows_by_split[split_name])
    return source_dir


def concat_split_manifests(source_dirs: Sequence[Path], split_name: str) -> list[dict[str, Any]]:
    """Concatenate already-split source manifests in declared source order."""
    rows: list[dict[str, Any]] = []
    for source_dir in source_dirs:
        rows.extend(read_jsonl(source_dir / f"{split_name}.jsonl"))
    return rows


def write_recipe_manifests(
    recipe_name: str,
    source_names: Sequence[str],
    source_manifest_root: Path,
    recipe_root: Path,
) -> tuple[Path, dict[str, dict[str, int]]]:
    """Write one recipe directory by concatenating primitive source split manifests."""
    recipe_dir = recipe_root / recipe_name
    recipe_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = [source_manifest_root / source_name for source_name in source_names]
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        rows = concat_split_manifests(source_dirs, split_name)
        write_jsonl(recipe_dir / f"{split_name}.jsonl", rows)
        split_counts[split_name] = {
            "total": len(rows),
            "included": sum(1 for row in rows if row["include"]),
        }
    return recipe_dir, split_counts


def validate_source_lists(
    *,
    sources: Sequence[str],
    train_side_sources: Sequence[str],
    heldout_sources: Sequence[str],
    benchmark_sources: Sequence[str],
) -> dict[str, str]:
    """Validate source selection and return a source->role mapping."""

    def _ensure_unique(name: str, values: Sequence[str]) -> None:
        seen: set[str] = set()
        dups: list[str] = []
        for value in values:
            if value in seen and value not in dups:
                dups.append(value)
            seen.add(value)
        if dups:
            raise ValueError(f"Duplicate entries in {name}: {dups}")

    _ensure_unique("--sources", sources)
    _ensure_unique("--train-side-sources", train_side_sources)
    _ensure_unique("--heldout-sources", heldout_sources)
    _ensure_unique("--benchmark-sources", benchmark_sources)

    invalid_sources = [source for source in sources if source not in ALL_CANONICAL_SOURCES]
    if invalid_sources:
        raise ValueError(
            f"Unknown sources in --sources: {invalid_sources}. "
            f"Allowed canonical sources: {ALL_CANONICAL_SOURCES}"
        )

    source_set = set(sources)
    role_map: dict[str, str] = {}
    role_lists = {
        "train_side": list(train_side_sources),
        "heldout": list(heldout_sources),
        "benchmark": list(benchmark_sources),
    }

    for role_name, role_sources in role_lists.items():
        missing = [source for source in role_sources if source not in source_set]
        if missing:
            raise ValueError(f"{role_name} sources must be a subset of --sources. Missing: {missing}")
        for source in role_sources:
            if source in role_map:
                raise ValueError(
                    f"Source '{source}' appears in multiple role lists: "
                    f"'{role_map[source]}' and '{role_name}'"
                )
            role_map[source] = role_name

    unclassified = [source for source in sources if source not in role_map]
    if unclassified:
        raise ValueError(
            f"Every source in --sources must belong to exactly one role list. "
            f"Unclassified sources: {unclassified}"
        )

    return role_map


def build_catalog(
    *,
    script_name: str,
    generation_timestamp: str,
    project_root: Path,
    circuits_root: Path,
    output_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_qubits: int,
    max_qubits: int,
    allow_no_2q: bool,
    source_stats: dict[str, dict[str, Any]],
    recipe_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the reproducibility / inspection catalog payload."""
    return {
        "script": {
            "script_name": script_name,
            "generation_timestamp": generation_timestamp,
            "project_root": project_root.as_posix(),
            "circuits_root": circuits_root.as_posix(),
            "output_dir": output_dir.as_posix(),
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "min_qubits": min_qubits,
            "max_qubits": max_qubits,
            "allow_no_2q": allow_no_2q,
        },
        "sources": [source_stats[source] for source in source_stats],
        "recipes": [recipe_stats[recipe_name] for recipe_name in recipe_stats],
        "split_strategy": "per_source_random_v1",
    }


def _build_recipe_specs(
    *,
    sources: Sequence[str],
    train_side_sources: Sequence[str],
    heldout_sources: Sequence[str],
    benchmark_sources: Sequence[str],
    emit_train_subset_recipes: bool,
    emit_heldout_recipes: bool,
    emit_benchmark_recipes: bool,
) -> list[tuple[str, list[str]]]:
    """Construct the ordered list of recipe specs to emit."""
    source_set = set(sources)
    recipe_specs: list[tuple[str, list[str]]] = []

    if "mqt_bench" not in source_set:
        raise ValueError("smoke_mqt requires 'mqt_bench' to be present in --sources")
    recipe_specs.append(("smoke_mqt", ["mqt_bench"]))

    if emit_train_subset_recipes:
        ordered_train_sources = [source for source in train_side_sources if source in source_set]
        for subset_size in range(1, len(ordered_train_sources) + 1):
            for subset in itertools.combinations(ordered_train_sources, subset_size):
                recipe_specs.append((f"train_{'_'.join(subset)}", list(subset)))

    if emit_heldout_recipes:
        ordered_heldout_sources = [source for source in heldout_sources if source in source_set]
        for source in ordered_heldout_sources:
            recipe_specs.append((f"heldout_{source}", [source]))
        if len(ordered_heldout_sources) >= 2:
            recipe_specs.append(
                (
                    f"heldout_{'_'.join(ordered_heldout_sources)}",
                    ordered_heldout_sources,
                )
            )

    if emit_benchmark_recipes:
        ordered_benchmark_sources = [source for source in benchmark_sources if source in source_set]
        if ordered_benchmark_sources:
            recipe_specs.append(
                (
                    f"benchmark_{'_'.join(ordered_benchmark_sources)}",
                    ordered_benchmark_sources,
                )
            )

    return recipe_specs


def main() -> None:
    """Main entry point for the full manifest build script."""
    args = parse_args()

    project_root = args.project_root.resolve()
    circuits_root = (
        args.circuits_root or (project_root / "data" / "circuits_v2" / "qasm")
    ).resolve()
    output_dir = (args.output_dir or (project_root / "data" / "manifests" / "full")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    source_role_map = validate_source_lists(
        sources=args.sources,
        train_side_sources=args.train_side_sources,
        heldout_sources=args.heldout_sources,
        benchmark_sources=args.benchmark_sources,
    )

    source_roots: dict[str, Path] = {}
    for source in args.sources:
        source_root = circuits_root / source
        if not source_root.exists():
            raise FileNotFoundError(f"Source root not found: {source_root}")
        source_roots[source] = source_root

    source_stats: dict[str, dict[str, Any]] = {}
    source_manifest_root = output_dir / "source_manifests"
    recipe_root = output_dir / "recipes"

    # Phase 2: scan and build per-source rows.
    for source in args.sources:
        source_root = source_roots[source]
        qasm_files = iter_qasm_files(source_root)
        rows: list[dict[str, Any]] = []

        for qasm_path in qasm_files:
            row = build_manifest_row(
                project_root=project_root,
                source=source,
                qasm_path=qasm_path,
                min_qubits=args.min_qubits,
                max_qubits=args.max_qubits,
                require_two_qubit_gate=not args.allow_no_2q,
            )
            row["dataset_version"] = "circuit_v2"
            row["source_role"] = source_role_map[source]
            rows.append(row)

        # Phase 3: assign splits independently per source and write source manifests.
        source_seed = stable_int_hash(f"{args.seed}:{source}")
        split_rows = assign_splits_for_one_source(
            rows,
            seed=source_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        rows_by_split = {"all": rows, **split_rows}
        source_manifest_dir = write_source_manifests(source, rows_by_split, output_dir)

        source_stats[source] = {
            "source_name": source,
            "source_role": source_role_map[source],
            "source_root": source_root.as_posix(),
            "output_dir": source_manifest_dir.as_posix(),
            "discovered_row_count": len(rows),
            "included_row_count": sum(1 for row in rows if row["include"]),
            "split_counts": {
                split_name: {
                    "total": len(rows_by_split[split_name]),
                    "included": sum(1 for row in rows_by_split[split_name] if row["include"]),
                }
                for split_name in SPLIT_NAMES
            },
        }

    recipe_stats: dict[str, dict[str, Any]] = {}

    # Phase 4: emit recipes.
    if not args.emit_source_manifests_only:
        recipe_specs = _build_recipe_specs(
            sources=args.sources,
            train_side_sources=args.train_side_sources,
            heldout_sources=args.heldout_sources,
            benchmark_sources=args.benchmark_sources,
            emit_train_subset_recipes=args.emit_train_subset_recipes,
            emit_heldout_recipes=args.emit_heldout_recipes,
            emit_benchmark_recipes=args.emit_benchmark_recipes,
        )
        for recipe_name, recipe_sources in recipe_specs:
            recipe_dir, split_counts = write_recipe_manifests(
                recipe_name=recipe_name,
                source_names=recipe_sources,
                source_manifest_root=source_manifest_root,
                recipe_root=recipe_root,
            )
            recipe_stats[recipe_name] = {
                "recipe_name": recipe_name,
                "source_names": list(recipe_sources),
                "output_dir": recipe_dir.as_posix(),
                "split_counts": split_counts,
            }

    # Phase 5: write catalog.
    catalog = build_catalog(
        script_name=Path(__file__).name,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        project_root=project_root,
        circuits_root=circuits_root,
        output_dir=output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        allow_no_2q=args.allow_no_2q,
        source_stats=source_stats,
        recipe_stats=recipe_stats,
    )
    with (output_dir / "catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, sort_keys=False)
        f.write("\n")

    # Phase 6: print console summary.
    print(f"Wrote manifests to {output_dir}")
    print("Source manifests:")
    for source in args.sources:
        info = source_stats[source]
        split_counts = info["split_counts"]
        print(
            f"  {source:<12} role={info['source_role']:<10} "
            f"total={info['discovered_row_count']:>5} "
            f"included={info['included_row_count']:>5} "
            f"train={split_counts['train']['included']:>5} "
            f"val={split_counts['val']['included']:>5} "
            f"test={split_counts['test']['included']:>5}"
        )

    if recipe_stats:
        print("Recipe manifests:")
        for recipe_name, info in recipe_stats.items():
            split_counts = info["split_counts"]
            print(
                f"  {recipe_name:<30} "
                f"sources={','.join(info['source_names'])} "
                f"train={split_counts['train']['included']:>5} "
                f"val={split_counts['val']['included']:>5} "
                f"test={split_counts['test']['included']:>5}"
            )
    else:
        print("Recipe manifests: skipped (--emit-source-manifests-only)")


if __name__ == "__main__":
    main()


