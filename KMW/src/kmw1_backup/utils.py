from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = move_to_device(value, device)
        elif isinstance(value, list):
            out[key] = [move_to_device(v, device) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


def detach_scalar_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    x_cpu = x.detach().float().cpu()
    return {
        "shape": list(x_cpu.shape),
        "min": float(x_cpu.min().item()),
        "max": float(x_cpu.max().item()),
        "mean": float(x_cpu.mean().item()),
        "std": float(x_cpu.std(unbiased=False).item()) if x_cpu.numel() > 1 else 0.0,
        "finite": bool(torch.isfinite(x_cpu).all().item()),
    }


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def batched_index_select_cols(matrix: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    matrix: (B, N, N)
    index : (B, N) where output[..., k] = input[..., index[k]]
    """
    if matrix.ndim != 3 or index.ndim != 2:
        raise ValueError("Expected matrix=(B,N,N), index=(B,N)")
    return torch.gather(matrix, dim=-1, index=index.unsqueeze(1).expand(-1, matrix.shape[1], -1))


def batched_index_select_rows(matrix: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    matrix: (B, N, N)
    index : (B, N) where output[:, k, :] = input[:, index[k], :]
    """
    if matrix.ndim != 3 or index.ndim != 2:
        raise ValueError("Expected matrix=(B,N,N), index=(B,N)")
    return torch.gather(matrix, dim=1, index=index.unsqueeze(-1).expand(-1, -1, matrix.shape[-1]))
