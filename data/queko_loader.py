"""QUEKO dataset label loader for GraphQMap.

QUEKO provides circuits with known optimal (zero-SWAP) initial layouts (τ⁻¹).
These layouts are used directly as ground truth labels — no self-generation needed.

Expected QUEKO directory structure:
  queko_dir/
    ├── <circuit_name>.qasm    # circuit file
    └── <circuit_name>.layout  # optimal layout (one physical qubit per line)
"""

from __future__ import annotations

from pathlib import Path


def load_queko_layout(layout_path: str | Path) -> list[int]:
    """Load an optimal layout from a QUEKO .layout file.

    The file contains one physical qubit index per line,
    where line i specifies the physical qubit for logical qubit i.

    Args:
        layout_path: Path to the .layout file.

    Returns:
        Layout as a list mapping logical qubit index -> physical qubit index.
    """
    path = Path(layout_path)
    if not path.exists():
        raise FileNotFoundError(f"Layout file not found: {path}")

    layout = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                layout.append(int(line))
    return layout
