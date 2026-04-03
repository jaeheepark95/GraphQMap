from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from qiskit import QuantumCircuit, qpy, transpile
from qiskit import qasm2

from ..data.dataset import QubitMappingDataset
from ..models.model import AssignmentHead, UNetMapping
from ..preprocessing.pipeline import build_model_inputs, canonical_mapping_to_native
from ..utils import backend_from_name, ensure_dir, load_merged_configs, move_tensor_batch, resolve_dotted_callable, save_json


def _load_model(checkpoint_path: str | Path, device, model_cfg: Optional[Dict[str, Any]] = None) -> UNetMapping:
    model_cfg = model_cfg or {}
    model = UNetMapping(
        in_channels=int(model_cfg.get('in_channels', 5)),
        token_dim=int(model_cfg.get('token_dim', 128)),
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _ordered_initial_layout(mapping: Dict[int, int]) -> list[int]:
    return [mapping[i] for i in sorted(mapping)]

#======================================================================================

# def _compute_pst(transpiled, backend, pst_callable_path: Optional[str]):
#     pst_fn = resolve_dotted_callable(pst_callable_path)
#     if pst_fn is None:
#         return None
#     try:
#         from qiskit_aer import AerSimulator
#     except Exception:
#         return None
#     noisy_sim = AerSimulator.from_backend(backend)
#     ideal_sim = AerSimulator.from_backend(backend, noise_model=None)
#     noisy_counts = noisy_sim.run(transpiled, shots=2048).result().get_counts()
#     ideal_counts = ideal_sim.run(transpiled, shots=2048).result().get_counts()
#     return pst_fn(noisy_counts, ideal_counts)

def _ensure_measured(circuit: QuantumCircuit) -> QuantumCircuit:
    measured = circuit.copy()
    if 'measure' not in measured.count_ops():
        measured.measure_all()
    return measured


def _build_simulators(backend):
    try:
        from qiskit_aer import AerSimulator
    except Exception:
        return None, None
    sim_config = {}
    noisy_sim = AerSimulator.from_backend(backend, **sim_config)
    ideal_sim = AerSimulator.from_backend(backend, noise_model=None, **sim_config)
    return noisy_sim, ideal_sim


def _compute_pst(transpiled, noisy_sim, ideal_sim, pst_callable_path: Optional[str], shots: int = 1024):
    pst_fn = resolve_dotted_callable(pst_callable_path)
    if pst_fn is None or noisy_sim is None:
        return None

    pst_circuit = _ensure_measured(transpiled)

    noisy_counts = noisy_sim.run(pst_circuit, shots=shots).result().get_counts()
    ideal_counts = ideal_sim.run(pst_circuit, shots=shots).result().get_counts()

    return float(pst_fn(noisy_counts, ideal_counts))

#======================================================================================

def _dump_qpy(path: str | Path, circuit: QuantumCircuit):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        qpy.dump(circuit, f)


def _save_routed_artifacts(transpiled: QuantumCircuit, output_stem: str, qasm_dir: Optional[str | Path], qpy_dir: Optional[str | Path]):
    saved = {}
    if qasm_dir:
        qasm_dir = ensure_dir(qasm_dir)
        qasm_path = qasm_dir / f'{output_stem}.qasm'
        qasm_path.write_text(qasm2.dumps(transpiled), encoding='utf-8')
        saved['routed_qasm_path'] = str(qasm_path)
    if qpy_dir:
        qpy_dir = ensure_dir(qpy_dir)
        qpy_path = qpy_dir / f'{output_stem}.qpy'
        _dump_qpy(qpy_path, transpiled)
        saved['routed_qpy_path'] = str(qpy_path)
    return saved


def _evaluate_sample(model, backend, batch: Dict[str, Any], device, pst_callable_path: Optional[str] = None, noisy_sim=None, ideal_sim=None, save_routed_qasm_dir=None, save_routed_qpy_dir=None):
    batch = move_tensor_batch(batch, device)
    X, Tlog_raw, Tphy_raw = build_model_inputs(batch['W'], batch['m'], batch['A'], batch['c1'], batch['c2'])
    with torch.no_grad():
        logits = model(X, Tlog_raw, Tphy_raw)
        M = AssignmentHead.hungarian(logits)

    logical_qubits = int(batch['logical_qubits'][0] if isinstance(batch['logical_qubits'], list) else batch['logical_qubits'])
    p = batch['p'][0] if isinstance(batch['p'], torch.Tensor) and batch['p'].ndim == 2 else batch['p']
    mapping = canonical_mapping_to_native(M, logical_qubits=logical_qubits, p=p)

    qasm_path = batch['qasm_path'][0] if isinstance(batch['qasm_path'], list) else batch['qasm_path']
    if not qasm_path:
        raise ValueError('Manifest/dataset evaluation requires qasm_path for routed metrics.')
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))

    original_depth = int(circuit.depth() or 0)
    original_count_ops = {str(k): int(v) for k, v in circuit.count_ops().items()}
    original_gate_count = int(sum(original_count_ops.values()))

    transpile_started = time.perf_counter()
    transpiled = transpile(circuit, backend=backend, initial_layout=_ordered_initial_layout(mapping), optimization_level=2)
    compile_seconds = time.perf_counter() - transpile_started

    routed_depth = int(transpiled.depth() or 0)
    routed_count_ops = {str(k): int(v) for k, v in transpiled.count_ops().items()}
    routed_gate_count = int(sum(routed_count_ops.values()))
    swap_overhead = int(routed_count_ops.get('swap', 0))
    depth_increase = int(routed_depth - original_depth)
    output_stem = Path(qasm_path).stem

    result = {
        'circuit_id': batch['circuit_id'][0] if isinstance(batch['circuit_id'], list) else batch['circuit_id'],
        'source': batch['source'][0] if isinstance(batch['source'], list) else batch['source'],
        'qasm_path': qasm_path,
        'logical_qubits': logical_qubits,
        'mapping': mapping,
        'pst': _compute_pst(transpiled, noisy_sim, ideal_sim, pst_callable_path),
        'compile_seconds': compile_seconds,
        'swap_overhead': swap_overhead,
        'depth_increase': depth_increase,
        'original_depth': original_depth,
        'routed_depth': routed_depth,
        'original_gate_count': original_gate_count,
        'routed_gate_count': routed_gate_count,
        'depth': routed_depth,
        'gate_count': routed_gate_count,
        'original_count_ops': original_count_ops,
        'count_ops': routed_count_ops,
    }
    result.update(_save_routed_artifacts(transpiled, output_stem, save_routed_qasm_dir, save_routed_qpy_dir))
    return result


def evaluate_one_circuit(
    checkpoint_path: str | Path,
    circuit_path: str | Path,
    backend='FakeTorontoV2',
    model_cfg: Optional[Dict[str, Any]] = None,
    pst_callable_path: Optional[str] = None,
    device_override: Optional[str] = None,
    save_routed_qasm_dir=None,
    save_routed_qpy_dir=None,
) -> Dict[str, Any]:
    device = torch.device(device_override or ('cuda' if torch.cuda.is_available() else 'cpu'))
    backend_obj = backend if not isinstance(backend, str) else backend_from_name(backend)
    model = _load_model(checkpoint_path, device=device, model_cfg=model_cfg)
    noisy_sim, ideal_sim = _build_simulators(backend_obj)
    dataset = QubitMappingDataset(backend=backend_obj, qasm_paths=[circuit_path])
    sample = dataset[0]
    return _evaluate_sample(model, backend_obj, sample, device, pst_callable_path=pst_callable_path, noisy_sim=noisy_sim, ideal_sim=ideal_sim, save_routed_qasm_dir=save_routed_qasm_dir, save_routed_qpy_dir=save_routed_qpy_dir)


def evaluate_manifest(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    backend='FakeTorontoV2',
    model_cfg: Optional[Dict[str, Any]] = None,
    pst_callable_path: Optional[str] = None,
    device_override: Optional[str] = None,
    per_circuit_csv_path: Optional[str | Path] = None,
    per_circuit_json_path: Optional[str | Path] = None,
    summary_json_path: Optional[str | Path] = None,
    save_routed_qasm_dir=None,
    save_routed_qpy_dir=None,
) -> Dict[str, Any]:
    device = torch.device(device_override or ('cuda' if torch.cuda.is_available() else 'cpu'))
    backend_obj = backend if not isinstance(backend, str) else backend_from_name(backend)
    model = _load_model(checkpoint_path, device=device, model_cfg=model_cfg)
    noisy_sim, ideal_sim = _build_simulators(backend_obj)
    dataset = QubitMappingDataset(backend=backend_obj, manifest_path=manifest_path)
    output_dir = ensure_dir(output_dir)

    rows = []
    for idx in range(len(dataset)):
        rows.append(_evaluate_sample(model, backend_obj, dataset[idx], device, pst_callable_path=pst_callable_path, noisy_sim=noisy_sim, ideal_sim=ideal_sim, save_routed_qasm_dir=save_routed_qasm_dir, save_routed_qpy_dir=save_routed_qpy_dir))

    csv_path = Path(per_circuit_csv_path) if per_circuit_csv_path else output_dir / 'per_circuit.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'circuit_id', 'source', 'qasm_path', 'logical_qubits',
                'pst', 'compile_seconds', 'swap_overhead', 'depth_increase',
                'original_depth', 'routed_depth', 'original_gate_count', 'routed_gate_count',
                'mapping', 'original_count_ops', 'count_ops', 'routed_qasm_path', 'routed_qpy_path',
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                'mapping': str(row['mapping']),
                'original_count_ops': str(row['original_count_ops']),
                'count_ops': str(row['count_ops']),
            })

    numeric_pst = [row['pst'] for row in rows if isinstance(row['pst'], (int, float))]
    summary = {
        'num_circuits': len(rows),
        'avg_pst': (sum(numeric_pst) / len(numeric_pst)) if numeric_pst else None,
        'avg_compile_seconds': (sum(row['compile_seconds'] for row in rows) / len(rows)) if rows else None,
        'avg_swap_overhead': (sum(row['swap_overhead'] for row in rows) / len(rows)) if rows else None,
        'avg_depth_increase': (sum(row['depth_increase'] for row in rows) / len(rows)) if rows else None,
        'avg_original_depth': (sum(row['original_depth'] for row in rows) / len(rows)) if rows else None,
        'avg_routed_depth': (sum(row['routed_depth'] for row in rows) / len(rows)) if rows else None,
        'avg_original_gate_count': (sum(row['original_gate_count'] for row in rows) / len(rows)) if rows else None,
        'avg_routed_gate_count': (sum(row['routed_gate_count'] for row in rows) / len(rows)) if rows else None,
        'per_circuit_csv': str(csv_path),
    }
    summary_path = Path(summary_json_path) if summary_json_path else output_dir / 'summary.json'
    rows_path = Path(per_circuit_json_path) if per_circuit_json_path else output_dir / 'per_circuit.json'
    save_json(summary_path, summary)
    save_json(rows_path, rows)
    return summary


def evaluate_from_config(config_paths, mode: str, circuit_path: Optional[str] = None):
    if isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]
    cfg = load_merged_configs(config_paths)
    eval_cfg = cfg.get('evaluation', {})
    paths_cfg = cfg.get('paths', {})
    backend = cfg.get('backend', 'FakeTorontoV2')
    runtime_cfg = cfg.get('runtime', {})
    checkpoint_path = paths_cfg.get('checkpoint_path') or (Path(paths_cfg.get('run_dir', 'runs/kmw2/train_main')) / paths_cfg.get('checkpoint_name', 'model_final.pth'))
    model_cfg = cfg.get('model', {})
    pst_callable_path = eval_cfg.get('pst_callable')

    if mode == 'eval-one':
        if not circuit_path:
            circuit_path = eval_cfg.get('single_circuit_path')
        if not circuit_path:
            raise ValueError('A circuit path is required for eval-one.')
        return evaluate_one_circuit(
            checkpoint_path=checkpoint_path,
            circuit_path=circuit_path,
            backend=backend,
            model_cfg=model_cfg,
            pst_callable_path=pst_callable_path,
            device_override=runtime_cfg.get('device'),
            save_routed_qasm_dir=paths_cfg.get('save_routed_qasm_dir'),
            save_routed_qpy_dir=paths_cfg.get('save_routed_qpy_dir'),
        )

    manifest_path = eval_cfg.get('manifest_path')
    if not manifest_path:
        recipe_path = cfg.get('dataset', {}).get('recipe_path')
        if recipe_path:
            from ..utils import load_structured_file
            recipe = load_structured_file(recipe_path)
            manifest_path = recipe.get('split_manifests', {}).get(eval_cfg.get('split', 'test'))
            if manifest_path:
                manifest_path = str((Path(recipe_path).parent / manifest_path).resolve())
    if not manifest_path:
        raise ValueError('Evaluation requires manifest_path or dataset.recipe_path.')

    output_dir = paths_cfg.get('eval_dir', 'runs/kmw2/eval')
    return evaluate_manifest(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        backend=backend,
        model_cfg=model_cfg,
        pst_callable_path=pst_callable_path,
        device_override=runtime_cfg.get('device'),
        per_circuit_csv_path=paths_cfg.get('per_circuit_csv'),
        per_circuit_json_path=paths_cfg.get('per_circuit_json'),
        summary_json_path=paths_cfg.get('summary_json'),
        save_routed_qasm_dir=paths_cfg.get('save_routed_qasm_dir'),
        save_routed_qpy_dir=paths_cfg.get('save_routed_qpy_dir'),
    )
