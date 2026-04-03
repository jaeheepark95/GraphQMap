from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import apply_overrides, deep_update, load_merged_configs, load_structured_file


def _common_overrides(subparser: argparse.ArgumentParser, include_eval_outputs: bool = False):
    subparser.add_argument('--run-dir')
    subparser.add_argument('--checkpoint')
    subparser.add_argument('--checkpoint-name')
    subparser.add_argument('--trainer-config')
    subparser.add_argument('--loss-config')
    subparser.add_argument('--stage-config')
    subparser.add_argument('--eval-config')
    subparser.add_argument('--epochs', type=int)
    subparser.add_argument('--batch-size', type=int)
    subparser.add_argument('--num-workers', type=int)
    subparser.add_argument('--lr', type=float)
    subparser.add_argument('--weight-decay', type=float)
    subparser.add_argument('--backend-name')
    subparser.add_argument('--device')
    subparser.add_argument('--save-routed-qasm-dir')
    subparser.add_argument('--save-routed-qpy-dir')
    if include_eval_outputs:
        subparser.add_argument('--per-circuit-csv')
        subparser.add_argument('--per-circuit-json')
        subparser.add_argument('--summary-json')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='kmw2 unified CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_p = subparsers.add_parser('train', help='Train from config')
    train_p.add_argument('--config', required=True, action='append')
    _common_overrides(train_p)

    train_staged_p = subparsers.add_parser('train-staged', help='Run staged training from config')
    train_staged_p.add_argument('--config', required=True, action='append')
    _common_overrides(train_staged_p)

    eval_p = subparsers.add_parser('eval', help='Evaluate from manifest/recipe config')
    eval_p.add_argument('--config', required=True, action='append')
    _common_overrides(eval_p, include_eval_outputs=True)

    eval_one_p = subparsers.add_parser('eval-one', help='Evaluate a single circuit')
    eval_one_p.add_argument('--config', required=True, action='append')
    eval_one_p.add_argument('--circuit', default=None)
    _common_overrides(eval_one_p, include_eval_outputs=True)
    return parser


def _load_optional_config(path: str | None):
    return load_structured_file(path) if path else {}


def _materialize_config(args) -> dict:
    cfg = load_merged_configs(args.config)

    for opt in ('trainer_config', 'loss_config', 'stage_config', 'eval_config'):
        path = getattr(args, opt, None)
        if not path:
            continue
        extra = _load_optional_config(path)
        if opt == 'trainer_config':
            cfg.setdefault('training', {})
            cfg['training'] = dict(deep_update(dict(cfg['training']), extra))
        elif opt == 'loss_config':
            cfg.setdefault('loss', {})
            cfg['loss'] = dict(deep_update(dict(cfg['loss']), extra))
        elif opt == 'stage_config':
            if 'stages' in extra:
                cfg.setdefault('staged_training', {})
                cfg['staged_training']['stages'] = extra['stages']
            else:
                cfg.setdefault('staged_training', {})
                cfg['staged_training'] = dict(deep_update(dict(cfg['staged_training']), extra))
        elif opt == 'eval_config':
            cfg.setdefault('evaluation', {})
            cfg['evaluation'] = dict(deep_update(dict(cfg['evaluation']), extra))

    overrides = {
        'paths.run_dir': getattr(args, 'run_dir', None),
        'paths.checkpoint_path': getattr(args, 'checkpoint', None),
        'paths.checkpoint_name': getattr(args, 'checkpoint_name', None),
        'paths.per_circuit_csv': getattr(args, 'per_circuit_csv', None),
        'paths.per_circuit_json': getattr(args, 'per_circuit_json', None),
        'paths.summary_json': getattr(args, 'summary_json', None),
        'paths.save_routed_qasm_dir': getattr(args, 'save_routed_qasm_dir', None),
        'paths.save_routed_qpy_dir': getattr(args, 'save_routed_qpy_dir', None),
        'training.epochs': getattr(args, 'epochs', None),
        'training.batch_size': getattr(args, 'batch_size', None),
        'training.num_workers': getattr(args, 'num_workers', None),
        'training.lr': getattr(args, 'lr', None),
        'training.weight_decay': getattr(args, 'weight_decay', None),
        'backend': getattr(args, 'backend_name', None),
        'runtime.device': getattr(args, 'device', None),
    }
    cfg = dict(apply_overrides(cfg, overrides))

    if getattr(args, 'epochs', None) is not None and cfg.get('staged_training', {}).get('stages'):
        for stage in cfg['staged_training']['stages']:
            stage['epochs'] = args.epochs
    if getattr(args, 'batch_size', None) is not None and cfg.get('staged_training', {}).get('stages'):
        for stage in cfg['staged_training']['stages']:
            stage['batch_size'] = args.batch_size
    if getattr(args, 'num_workers', None) is not None and cfg.get('staged_training', {}).get('stages'):
        for stage in cfg['staged_training']['stages']:
            stage['num_workers'] = args.num_workers
    if getattr(args, 'lr', None) is not None and cfg.get('staged_training', {}).get('stages'):
        cfg['training']['lr'] = args.lr
        for stage in cfg['staged_training']['stages']:
            stage['lr'] = args.lr
    if getattr(args, 'weight_decay', None) is not None and cfg.get('staged_training', {}).get('stages'):
        cfg['training']['weight_decay'] = args.weight_decay
        for stage in cfg['staged_training']['stages']:
            stage['weight_decay'] = args.weight_decay

    return cfg


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _materialize_config(args)

    if args.command == 'train':
        from ..training.trainer import Trainer
        artifacts = Trainer(cfg).train()
        print(json.dumps({'checkpoint_path': str(artifacts.checkpoint_path), 'metrics_path': str(artifacts.metrics_path)}, indent=2))
        return 0

    if args.command == 'train-staged':
        from ..training.trainer import Trainer
        artifacts = Trainer(cfg).train_staged()
        print(json.dumps({'checkpoint_path': str(artifacts.checkpoint_path), 'metrics_path': str(artifacts.metrics_path)}, indent=2))
        return 0

    if args.command == 'eval':
        from ..evaluation.evaluate import evaluate_manifest
        backend = cfg.get('backend', 'FakeTorontoV2')
        checkpoint_path = cfg.get('paths', {}).get('checkpoint_path') or (Path(cfg.get('paths', {}).get('run_dir', 'runs/kmw2/train_main')) / cfg.get('paths', {}).get('checkpoint_name', 'model_final.pth'))
        result = evaluate_manifest(
            checkpoint_path=checkpoint_path,
            manifest_path=cfg['evaluation']['manifest_path'],
            output_dir=cfg.get('paths', {}).get('eval_dir', 'runs/kmw2/eval'),
            backend=backend,
            model_cfg=cfg.get('model', {}),
            pst_callable_path=cfg.get('evaluation', {}).get('pst_callable'),
            device_override=cfg.get('runtime', {}).get('device'),
            per_circuit_csv_path=cfg.get('paths', {}).get('per_circuit_csv'),
            per_circuit_json_path=cfg.get('paths', {}).get('per_circuit_json'),
            summary_json_path=cfg.get('paths', {}).get('summary_json'),
            save_routed_qasm_dir=cfg.get('paths', {}).get('save_routed_qasm_dir'),
            save_routed_qpy_dir=cfg.get('paths', {}).get('save_routed_qpy_dir'),
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == 'eval-one':
        from ..evaluation.evaluate import evaluate_one_circuit
        backend = cfg.get('backend', 'FakeTorontoV2')
        checkpoint_path = cfg.get('paths', {}).get('checkpoint_path') or (Path(cfg.get('paths', {}).get('run_dir', 'runs/kmw2/train_main')) / cfg.get('paths', {}).get('checkpoint_name', 'model_final.pth'))
        result = evaluate_one_circuit(
            checkpoint_path=checkpoint_path,
            circuit_path=args.circuit or cfg.get('evaluation', {}).get('single_circuit_path'),
            backend=backend,
            model_cfg=cfg.get('model', {}),
            pst_callable_path=cfg.get('evaluation', {}).get('pst_callable'),
            device_override=cfg.get('runtime', {}).get('device'),
            save_routed_qasm_dir=cfg.get('paths', {}).get('save_routed_qasm_dir'),
            save_routed_qpy_dir=cfg.get('paths', {}).get('save_routed_qpy_dir'),
        )
        print(json.dumps(result, indent=2))
        return 0

    parser.error(f'Unknown command: {args.command}')


if __name__ == '__main__':
    raise SystemExit(main())
