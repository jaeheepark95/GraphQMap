from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='kmw2 unified CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_p = subparsers.add_parser('train', help='Train from config')
    train_p.add_argument('--config', required=True)

    eval_p = subparsers.add_parser('eval', help='Evaluate from manifest/recipe config')
    eval_p.add_argument('--config', required=True)

    eval_one_p = subparsers.add_parser('eval-one', help='Evaluate a single circuit')
    eval_one_p.add_argument('--config', required=True)
    eval_one_p.add_argument('--circuit', default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'train':
        from ..training.trainer import train_from_config
        artifacts = train_from_config(args.config)
        print(json.dumps({'checkpoint_path': str(artifacts.checkpoint_path), 'metrics_path': str(artifacts.metrics_path)}, indent=2))
        return 0

    if args.command == 'eval':
        from ..evaluation.evaluate import evaluate_from_config
        result = evaluate_from_config(args.config, mode='eval')
        print(json.dumps(result, indent=2))
        return 0

    if args.command == 'eval-one':
        from ..evaluation.evaluate import evaluate_from_config
        result = evaluate_from_config(args.config, mode='eval-one', circuit_path=args.circuit)
        print(json.dumps(result, indent=2))
        return 0

    parser.error(f'Unknown command: {args.command}')


if __name__ == '__main__':
    raise SystemExit(main())
