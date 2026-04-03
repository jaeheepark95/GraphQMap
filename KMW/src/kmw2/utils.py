from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

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


def normalize_source_name(name: str) -> str:
    raw = (name or '').strip().lower()
    compact = re.sub(r'[^a-z0-9]+', '', raw)
    aliases = {
        'qasmbench': 'qasmbench',
        'qasm': 'qasmbench',
        'revlib': 'revlib',
        'rev': 'revlib',
        'queko': 'queko',
        'mqt': 'mqt',
        'mqtbench': 'mqt',
        'mlqd': 'mlqd',
    }
    if compact in aliases:
        return aliases[compact]
    if 'qasm' in compact and 'bench' in compact:
        return 'qasmbench'
    if 'rev' in compact:
        return 'revlib'
    if 'queko' in compact:
        return 'queko'
    if 'mqt' in compact:
        return 'mqt'
    return compact or 'unknown'


def load_structured_file(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    suffix = path.suffix.lower()
    if suffix == '.json':
        return json.loads(text)
    if suffix == '.jsonl':
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    if suffix in {'.yaml', '.yml'}:
        if yaml is None:
            raise RuntimeError('PyYAML is required to load YAML files.')
        return yaml.safe_load(text)
    raise ValueError(f'Unsupported structured file type: {path}')


def save_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=indent, sort_keys=False), encoding='utf-8')
    return path


def save_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=False))
            f.write('\n')
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


def load_merged_configs(paths: Sequence[str | Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for path in paths:
        if not path:
            continue
        merged = dict(deep_update(merged, resolve_config(path)))
    return merged


def set_nested(config: MutableMapping[str, Any], dotted_key: str, value: Any):
    parts = dotted_key.split('.')
    cur: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def apply_overrides(config: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        set_nested(config, dotted_key, value)
    return config
