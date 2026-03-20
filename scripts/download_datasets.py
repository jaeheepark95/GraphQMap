"""Download benchmark datasets for GraphQMap.

Downloads:
  - MQT Bench: .qasm circuits from GitHub
  - QUEKO: .qasm + .layout files from GitHub
  - QASMBench: .qasm circuits from GitHub

Usage:
    python scripts/download_datasets.py [--all | --mqt | --queko | --qasmbench]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "circuits"

REPOS = {
    "mqt_bench": {
        "url": "https://github.com/cda-tum/MQTBench.git",
        "sparse_paths": ["benchmarks"],
        "description": "MQT Bench (~70k circuits)",
    },
    "queko": {
        "url": "https://github.com/UCLA-VAST/QUEKO-benchmark.git",
        "sparse_paths": None,  # clone entire repo (small)
        "description": "QUEKO benchmark (optimal layouts)",
    },
    "qasmbench": {
        "url": "https://github.com/pnnl/QASMBench.git",
        "sparse_paths": None,
        "description": "QASMBench (53 circuits)",
    },
}


def run_cmd(cmd: list[str], cwd: str | None = None) -> int:
    """Run a shell command and return exit code."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
    return result.returncode


def clone_repo(name: str, info: dict, target_dir: Path) -> None:
    """Clone a repository into the target directory."""
    temp_dir = target_dir / f".{name}_clone"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print(f"\nCloning {info['description']}...")

    if info.get("sparse_paths"):
        # Sparse checkout for large repos
        run_cmd(["git", "clone", "--depth", "1", "--filter=blob:none",
                 "--sparse", info["url"], str(temp_dir)])
        for sp in info["sparse_paths"]:
            run_cmd(["git", "sparse-checkout", "add", sp], cwd=str(temp_dir))
    else:
        run_cmd(["git", "clone", "--depth", "1", info["url"], str(temp_dir)])

    # Collect .qasm and .layout files
    collect_files(temp_dir, target_dir / name, extensions=[".qasm", ".layout"])

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def collect_files(src: Path, dst: Path, extensions: list[str]) -> None:
    """Recursively collect files with given extensions from src to dst."""
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for ext in extensions:
        for f in src.rglob(f"*{ext}"):
            # Flatten directory structure but avoid name collisions
            rel = f.relative_to(src)
            flat_name = str(rel).replace("/", "_").replace("\\", "_")
            dest_file = dst / flat_name
            if not dest_file.exists():
                shutil.copy2(f, dest_file)
                count += 1
    print(f"  Collected {count} files → {dst}")


def filter_compatible_circuits(circuit_dir: Path, max_qubits: int = 27) -> int:
    """Remove circuits that exceed the max qubit count.

    Args:
        circuit_dir: Directory containing .qasm files.
        max_qubits: Maximum number of qubits to keep.

    Returns:
        Number of circuits remaining.
    """
    removed = 0
    remaining = 0
    for qasm_file in circuit_dir.glob("*.qasm"):
        try:
            with open(qasm_file) as f:
                content = f.read()
            # Quick parse: look for qreg declaration
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("qreg ") or line.startswith("qubit["):
                    # Extract qubit count
                    import re
                    match = re.search(r"\[(\d+)\]", line)
                    if match:
                        n = int(match.group(1))
                        if n > max_qubits:
                            qasm_file.unlink()
                            removed += 1
                            break
            else:
                remaining += 1
        except Exception:
            remaining += 1

    if removed > 0:
        print(f"  Removed {removed} circuits with > {max_qubits} qubits")
    print(f"  Remaining: {remaining} circuits")
    return remaining


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GraphQMap datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--mqt", action="store_true", help="Download MQT Bench")
    parser.add_argument("--queko", action="store_true", help="Download QUEKO")
    parser.add_argument("--qasmbench", action="store_true", help="Download QASMBench")
    parser.add_argument("--max-qubits", type=int, default=27,
                        help="Filter circuits exceeding this qubit count")
    args = parser.parse_args()

    if not (args.all or args.mqt or args.queko or args.qasmbench):
        args.all = True

    targets = []
    if args.all or args.mqt:
        targets.append("mqt_bench")
    if args.all or args.queko:
        targets.append("queko")
    if args.all or args.qasmbench:
        targets.append("qasmbench")

    for name in targets:
        info = REPOS[name]
        target = BASE_DIR / name
        if target.exists() and list(target.glob("*.qasm")):
            existing = len(list(target.glob("*.qasm")))
            print(f"\n{name}: {existing} .qasm files already exist, skipping.")
            print(f"  Delete {target} to re-download.")
            continue
        clone_repo(name, info, BASE_DIR)
        if name in ("mqt_bench", "qasmbench"):
            filter_compatible_circuits(target, max_qubits=args.max_qubits)

    # Summary
    print("\n=== Dataset Summary ===")
    for name in targets:
        target = BASE_DIR / name
        qasm_count = len(list(target.glob("*.qasm")))
        layout_count = len(list(target.glob("*.layout")))
        print(f"  {name}: {qasm_count} .qasm, {layout_count} .layout")


if __name__ == "__main__":
    main()
