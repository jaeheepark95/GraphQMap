"""Process MLQD dataset: copy qasm files and extract initial layouts from OLSQ2 results.

For each circuit:
1. Copy original .qasm to data/circuits/qasm/mlqd/
2. Extract initial layout by parsing measurement lines (final mapping)
   and reversing detected SWAP patterns (3-CNOT decomposition)
3. Save successfully extracted layouts to data/circuits/labels/mlqd/labels.json

Circuits where SWAP detection fails are still copied (usable for Stage 2 unsupervised).
"""

import json
import os
import re
from pathlib import Path


# MLQD backend name mapping
BACKEND_MAP = {
    "aspen4": "mlqd_aspen4",
    "grid 5x5": "mlqd_grid5x5",
    "ibmMelbourne": "mlqd_melbourne",
    "ibmRochester": "mlqd_rochester",
    "sycamore": "mlqd_sycamore",
}


def extract_layout(result_qasm_path: str, expected_swaps: int) -> dict[int, int] | None:
    """Extract initial layout from OLSQ2 result circuit.

    Returns logical->physical mapping or None on failure.
    """
    with open(result_qasm_path) as f:
        content = f.read()

    # 1. Final mapping from measurements
    measures = re.findall(r"measure q\[(\d+)\]->c\[(\d+)\]", content)
    if not measures:
        return None
    phys_to_log: dict[int, int] = {}
    for phys, log in measures:
        phys_to_log[int(phys)] = int(log)

    if expected_swaps == 0:
        return {v: k for k, v in phys_to_log.items()}

    # 2. Extract CX gates with moment info
    cx_gates: list[tuple[int, int, int]] = []
    current_moment = -1
    for line in content.split("\n"):
        m = re.match(r"// moment (\d+)", line.strip())
        if m:
            current_moment = int(m.group(1))
        cx_match = re.match(r"\s*cx q\[(\d+)\],\s*q\[(\d+)\];", line.strip())
        if cx_match:
            cx_gates.append(
                (current_moment, int(cx_match.group(1)), int(cx_match.group(2)))
            )

    # 3. Detect SWAP patterns: cx a,b; cx b,a; cx a,b
    swaps: list[tuple[int, int]] = []
    used: set[int] = set()
    for i in range(len(cx_gates)):
        if i in used:
            continue
        _, a1, b1 = cx_gates[i]
        for j in range(i + 1, min(i + 15, len(cx_gates))):
            if j in used:
                continue
            _, a2, b2 = cx_gates[j]
            if a2 == b1 and b2 == a1:
                for k in range(j + 1, min(j + 15, len(cx_gates))):
                    if k in used:
                        continue
                    _, a3, b3 = cx_gates[k]
                    if a3 == a1 and b3 == b1:
                        swaps.append((a1, b1))
                        used.update([i, j, k])
                        break
                break

    if len(swaps) != expected_swaps:
        return None

    # 4. Reverse SWAPs to get initial layout
    for a, b in reversed(swaps):
        if a in phys_to_log and b in phys_to_log:
            phys_to_log[a], phys_to_log[b] = phys_to_log[b], phys_to_log[a]
        elif a in phys_to_log:
            phys_to_log[b] = phys_to_log.pop(a)
        elif b in phys_to_log:
            phys_to_log[a] = phys_to_log.pop(b)

    return {v: k for k, v in phys_to_log.items()}


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    mlqd_src = project_root / "tmp_downloads" / "MLQD"
    qasm_dst = project_root / "data" / "circuits" / "qasm" / "mlqd"
    labels_dst = project_root / "data" / "circuits" / "labels" / "mlqd"

    qasm_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    labels: dict[str, dict] = {}
    stats = {"total": 0, "labeled": 0, "unlabeled": 0}

    for backend_dir, backend_name in BACKEND_MAP.items():
        bpath = mlqd_src / backend_dir
        if not bpath.exists():
            print(f"  Skipping {backend_dir} (not found)")
            continue

        b_total, b_labeled = 0, 0

        for sub in sorted(os.listdir(bpath)):
            subpath = bpath / sub
            if not subpath.is_dir():
                continue

            for circuit_dir in sorted(os.listdir(subpath)):
                circuit_path = subpath / circuit_dir
                if not circuit_path.is_dir():
                    continue

                # Source files
                qasm_file = circuit_path / f"{circuit_dir}.qasm"
                json_file = circuit_path / f"{circuit_dir}.json"
                result_file = circuit_path / "result" / "circuit_after_inserting_swaps.qasm"

                if not qasm_file.exists() or not json_file.exists() or not result_file.exists():
                    continue

                b_total += 1

                # Unique filename: {backend}_{subcategory}_{circuit_name}.qasm
                dst_name = f"{backend_name}_{sub}_{circuit_dir}.qasm"
                dst_path = qasm_dst / dst_name

                # Copy original qasm
                dst_path.write_text(qasm_file.read_text())

                # Try to extract layout
                with open(json_file) as f:
                    meta = json.load(f)

                layout = extract_layout(str(result_file), meta["swap_number"])
                if layout is not None:
                    # Convert keys to int for JSON (they already are)
                    labels[dst_name] = {
                        "backend": backend_name,
                        "layout": [layout[i] for i in range(len(layout))],
                    }
                    b_labeled += 1

        print(f"{backend_name}: {b_total} circuits, {b_labeled} labeled")
        stats["total"] += b_total
        stats["labeled"] += b_labeled

    stats["unlabeled"] = stats["total"] - stats["labeled"]

    # Save labels
    labels_path = labels_dst / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nTotal: {stats['total']} circuits copied, {stats['labeled']} labeled, {stats['unlabeled']} unlabeled")
    print(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    main()
