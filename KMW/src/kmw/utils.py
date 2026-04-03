from __future__ import annotations

"""
Small utility helpers used across the early KMW implementation.

This file intentionally stays simple:
- create directories safely
- make stable IDs from text
- read/write JSONL files

Why keep these in a separate file?
Because many different modules need them, and repeating the same code in multiple
places makes bugs harder to fix later.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist.

    Parameters
    ----------
    path:
        The directory path we want to create.

    Returns
    -------
    Path
        The same path as a ``Path`` object.

    Notes
    -----
    ``parents=True`` means Python will also create any missing parent folders.
    ``exist_ok=True`` means it will not crash if the folder already exists.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def sha1_text(text: str) -> str:
    """Return the SHA1 hash of a string.

    We use this as a lightweight way to create short, stable IDs.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()



def stable_id(*parts: Any, prefix: str | None = None, length: int = 12) -> str:
    """Build a deterministic ID from several pieces of information.

    Example
    -------
    If we call:
        stable_id("mqt", "path/to/file.qasm", prefix="mqt")
    then the result will always be the same for the same input text.

    Why this is useful:
    - cache file names stay stable
    - manifest rows get consistent IDs
    - we do not rely on random numbering
    """
    joined = "::".join(str(p) for p in parts)
    digest = sha1_text(joined)[:length]
    if prefix:
        return f"{prefix}_{digest}"
    return digest



def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return it as a list of dictionaries.

    JSONL = JSON Lines.
    Each line is one independent JSON object.

    This format is convenient for manifests because:
    - it is easy to append to
    - it is easy to inspect line by line
    - each row is independent
    """
    rows: list[dict[str, Any]] = []
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                # Ignore blank lines quietly.
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc

    return rows



def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write dictionaries to a JSONL file.

    Each row becomes one line.
    We keep ``ensure_ascii=False`` so paths or comments with non-English text are
    preserved naturally.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return path
