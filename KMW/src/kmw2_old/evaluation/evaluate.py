from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from qiskit import QuantumCircuit, transpile

from ..data.dataset import QubitMappingDataset
from ..models.model import AssignmentHead, UNetMapping
from ..preprocessing.pipeline import build_model_inputs, canonical_mapping_to_native
from ..utils import backend_from_name, ensure_dir, move_tensor_batch, resolve_config, resolve_dotted_callable, save_json


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


def _compute_pst(transpiled, backend, pst_callable_path: Optional[str]):
    pst_fn = resolve_dotted_callable(pst_callable_path)
    if pst_fn is None:
        return None
    try:
        from qiskit_aer import AerSimulator
    except Exception:
        return None
    noisy_sim = AerSimulator.from_backend(backend)
    ideal_sim = AerSimulator.from_backend(backend, noise_model=None)
    noisy_counts = noisy_sim.run(transpiled, shots=2048).result().get_counts()
    ideal_counts = ideal_sim.run(transpiled, shots=2048).result().get_counts()
    return pst_fn(noisy_counts, ideal_counts)


def _evaluate_sample(model, backend, batch: Dict[str, Any], device, pst_callable_path: Optional[str] = None):
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

    return {
        'circuit_id': batch['circuit_id'][0] if isinstance(batch['circuit_id'], list) else batch['circuit_id'],
        'source': batch['source'][0] if isinstance(batch['source'], list) else batch['source'],
        'qasm_path': qasm_path,
        'logical_qubits': logical_qubits,
        'mapping': mapping,
        'pst': _compute_pst(transpiled, backend, pst_callable_path),
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


def evaluate_one_circuit(
    checkpoint_path: str | Path,
    circuit_path: str | Path,
    backend='FakeTorontoV2',
    model_cfg: Optional[Dict[str, Any]] = None,
    pst_callable_path: Optional[str] = None,
) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backend_obj = backend if not isinstance(backend, str) else backend_from_name(backend)
    model = _load_model(checkpoint_path, device=device, model_cfg=model_cfg)
    dataset = QubitMappingDataset(backend=backend_obj, qasm_paths=[circuit_path])
    sample = dataset[0]
    return _evaluate_sample(model, backend_obj, sample, device, pst_callable_path=pst_callable_path)


def evaluate_manifest(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    backend='FakeTorontoV2',
    model_cfg: Optional[Dict[str, Any]] = None,
    pst_callable_path: Optional[str] = None,
) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backend_obj = backend if not isinstance(backend, str) else backend_from_name(backend)
    model = _load_model(checkpoint_path, device=device, model_cfg=model_cfg)
    dataset = QubitMappingDataset(backend=backend_obj, manifest_path=manifest_path)
    output_dir = ensure_dir(output_dir)

    rows = []
    for idx in range(len(dataset)):
        rows.append(_evaluate_sample(model, backend_obj, dataset[idx], device, pst_callable_path=pst_callable_path))

    csv_path = output_dir / 'per_circuit.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'circuit_id', 'source', 'qasm_path', 'logical_qubits',
                'pst', 'compile_seconds', 'swap_overhead', 'depth_increase',
                'original_depth', 'routed_depth', 'original_gate_count', 'routed_gate_count',
                'mapping', 'original_count_ops', 'count_ops',
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
    save_json(output_dir / 'summary.json', summary)
    save_json(output_dir / 'per_circuit.json', rows)
    return summary


def evaluate_from_config(config_path: str | Path, mode: str, circuit_path: Optional[str] = None):
    cfg = resolve_config(config_path)
    eval_cfg = cfg.get('evaluation', {})
    paths_cfg = cfg.get('paths', {})
    backend = cfg.get('backend', 'FakeTorontoV2')
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
    )
