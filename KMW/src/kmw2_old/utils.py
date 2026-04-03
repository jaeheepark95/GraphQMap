from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_structured_file(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        return json.loads(text)
    if path.suffix.lower() in {'.yaml', '.yml'}:
        if yaml is None:
            raise RuntimeError('PyYAML is required to load YAML files.')
        return yaml.safe_load(text)
    raise ValueError(f'Unsupported structured file type: {path}')


def save_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=indent, sort_keys=False), encoding='utf-8')
    return path


def save_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding='utf-8')
    return path


def resolve_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = load_structured_file(path)
    inherit_from = cfg.pop('inherit_from', None)
    if inherit_from:
        parent = resolve_config(path.parent / inherit_from)
        return dict(deep_update(parent, cfg))
    return cfg


def resolve_dotted_callable(dotted_path: Optional[str]):
    if not dotted_path:
        return None
    if ':' not in dotted_path:
        raise ValueError("Callable path must use 'module.submodule:function_name' format.")
    module_name, func_name = dotted_path.split(':', 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def tensor_only_batch(batch: Mapping[str, Any]) -> Dict[str, Any]:
    import torch
    return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}


def move_tensor_batch(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    import torch
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def backend_from_name(name: str):
    normalized = name.strip().lower()
    from qiskit_ibm_runtime import fake_provider

    aliases = {
        'faketorontov2': 'FakeTorontoV2',
        'fake_toronto_v2': 'FakeTorontoV2',
        'toronto27': 'FakeTorontoV2',
        'ibm_backendv2_fixed_27q': 'FakeTorontoV2',
    }
    class_name = aliases.get(normalized, name)
    if hasattr(fake_provider, class_name):
        return getattr(fake_provider, class_name)()
    raise ValueError(f'Unsupported backend identifier: {name}')
