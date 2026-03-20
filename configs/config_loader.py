"""Configuration loader for GraphQMap.

Loads YAML config files and provides typed access to hyperparameters.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """Dot-notation access wrapper around a nested dict.

    Example:
        cfg = Config({"model": {"embedding_dim": 64}})
        cfg.model.embedding_dim  # -> 64
    """

    def __init__(self, d: dict[str, Any]) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a plain dict."""
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return a Config object."""
    raw = load_yaml(path)
    return Config(raw)


def load_config_with_base(base_path: str | Path, override_path: str | Path) -> Config:
    """Load a base config and merge an override config on top."""
    base = load_yaml(base_path)
    override = load_yaml(override_path)
    merged = deep_merge(base, override)
    return Config(merged)


def parse_args_with_config() -> Config:
    """Parse CLI arguments and load the specified config file.

    Usage:
        python train.py --config configs/stage1.yaml
    """
    parser = argparse.ArgumentParser(description="GraphQMap")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values: key=value (dot notation, e.g. training.optimizer.lr=0.001)",
    )
    args = parser.parse_args()

    cfg_dict = load_yaml(args.config)

    # Apply CLI overrides
    for item in args.override:
        key, value = item.split("=", 1)
        keys = key.split(".")
        d = cfg_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Auto-cast value
        d[keys[-1]] = _auto_cast(value)

    return Config(cfg_dict)


def _auto_cast(value: str) -> Any:
    """Try to cast string value to int, float, bool, or leave as str."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
