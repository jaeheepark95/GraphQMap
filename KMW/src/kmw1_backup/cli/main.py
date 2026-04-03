from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from kmw1.data.dataset import KMW1Dataset, kmw1_collate_fn
from kmw1.evaluation.evaluate import EvalConfig, evaluate_model
from kmw1.losses.loss import LossConfig
from kmw1.models.model import KMWCanonicalModel
from kmw1.training.trainer import TrainerConfig, build_trainer, resolve_device
from kmw1.utils import read_json, set_global_seed


def _load_json_or_empty(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    return read_json(path)


def _update_dataclass(instance, payload: dict[str, Any]):
    for key, value in payload.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def _build_dataset(args) -> KMW1Dataset:
    return KMW1Dataset(
        project_root=args.project_root,
        manifest_path=args.manifest,
        backend_name=args.backend_name,
        include_only=not args.include_all,
        force_recompute=args.force_recompute,
        allow_degenerate=args.allow_degenerate,
    )


def _build_loader(dataset: KMW1Dataset, *, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=kmw1_collate_fn,
    )


def _build_model(args, loss_config: LossConfig) -> KMWCanonicalModel:
    return KMWCanonicalModel(
        n=27,
        token_dim=args.token_dim,
        sinkhorn_iters=loss_config.sinkhorn_iters,
        dropout=args.dropout,
    )


def _load_checkpoint_if_requested(model: KMWCanonicalModel, checkpoint_path: str | None, device: torch.device) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def cmd_train(args) -> None:
    set_global_seed(args.seed)

    trainer_cfg = _update_dataclass(TrainerConfig(), _load_json_or_empty(args.trainer_config))
    loss_cfg = _update_dataclass(LossConfig(), _load_json_or_empty(args.loss_config))
    eval_cfg = _update_dataclass(EvalConfig(), _load_json_or_empty(args.eval_config))

    trainer_cfg.run_dir = args.run_dir
    trainer_cfg.epochs = args.epochs
    trainer_cfg.batch_size = args.batch_size
    trainer_cfg.num_workers = args.num_workers
    trainer_cfg.device = args.device
    trainer_cfg.backend_name = args.backend_name

    loss_cfg.sinkhorn_tau = args.sinkhorn_tau
    loss_cfg.sinkhorn_iters = args.sinkhorn_iters

    eval_cfg.project_root = args.project_root
    eval_cfg.backend_name = args.backend_name
    eval_cfg.route_final_eval = args.route_final_eval

    train_ds = _build_dataset(args)
    train_loader = _build_loader(train_ds, batch_size=trainer_cfg.batch_size, num_workers=trainer_cfg.num_workers, shuffle=True)

    val_loader = None
    if args.val_manifest:
        val_args = argparse.Namespace(**vars(args))
        val_args.manifest = args.val_manifest
        val_ds = _build_dataset(val_args)
        val_loader = _build_loader(val_ds, batch_size=1, num_workers=trainer_cfg.num_workers, shuffle=False)

    model = _build_model(args, loss_cfg)
    trainer = build_trainer(model=model, trainer_config=trainer_cfg, loss_config=loss_cfg, eval_config=eval_cfg)

    if args.resume_checkpoint:
        _load_checkpoint_if_requested(model, args.resume_checkpoint, resolve_device(args.device))

    result = trainer.train(train_loader=train_loader, val_loader=val_loader)
    print(result)


def cmd_eval(args) -> None:
    set_global_seed(args.seed)

    loss_cfg = _update_dataclass(LossConfig(), _load_json_or_empty(args.loss_config))
    eval_cfg = _update_dataclass(EvalConfig(), _load_json_or_empty(args.eval_config))
    eval_cfg.project_root = args.project_root
    eval_cfg.backend_name = args.backend_name
    eval_cfg.route_final_eval = args.route_final_eval
    eval_cfg.per_circuit_csv_path = args.per_circuit_csv
    eval_cfg.summary_json_path = args.summary_json
    eval_cfg.eval_split_name = args.eval_split

    dataset = _build_dataset(args)
    loader = _build_loader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    model = _build_model(args, loss_cfg)
    device = resolve_device(args.device)
    _load_checkpoint_if_requested(model, args.checkpoint, device)
    model.to(device)

    summary = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        eval_config=eval_cfg,
        loss_config=loss_cfg,
    )
    print(summary)


def cmd_full_run(args) -> None:
    cmd_train(args)
    if args.eval_manifest is None:
        return
    eval_args = argparse.Namespace(**vars(args))
    eval_args.manifest = args.eval_manifest
    eval_args.checkpoint = str(Path(args.run_dir) / "checkpoints" / "best.pt")
    eval_args.per_circuit_csv = str(Path(args.run_dir) / "eval_per_circuit.csv")
    eval_args.summary_json = str(Path(args.run_dir) / "eval_summary.json")
    eval_args.eval_split = "eval"
    cmd_eval(eval_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="kmw1 canonical-hardware v1.4 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--project-root", type=str, default=".")
        p.add_argument("--manifest", type=str, required=True)
        p.add_argument("--backend-name", type=str, default="fake_toronto_v2")
        p.add_argument("--device", type=str, default="auto")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--num-workers", type=int, default=0)
        p.add_argument("--token-dim", type=int, default=128)
        p.add_argument("--dropout", type=float, default=0.1)
        p.add_argument("--sinkhorn-tau", type=float, default=0.5)
        p.add_argument("--sinkhorn-iters", type=int, default=30)
        p.add_argument("--loss-config", type=str, default=None)
        p.add_argument("--eval-config", type=str, default=None)
        p.add_argument("--include-all", action="store_true")
        p.add_argument("--force-recompute", action="store_true")
        p.add_argument("--allow-degenerate", action="store_true")
        p.add_argument("--route-final-eval", action="store_true")

    train_p = subparsers.add_parser("train", help="train the canonical mapper")
    add_common(train_p)
    train_p.add_argument("--run-dir", type=str, required=True)
    train_p.add_argument("--epochs", type=int, default=60)
    train_p.add_argument("--batch-size", type=int, default=1)
    train_p.add_argument("--trainer-config", type=str, default=None)
    train_p.add_argument("--val-manifest", type=str, default=None)
    train_p.add_argument("--resume-checkpoint", type=str, default=None)
    train_p.set_defaults(func=cmd_train)

    eval_p = subparsers.add_parser("eval", help="evaluate a trained canonical mapper")
    add_common(eval_p)
    eval_p.add_argument("--checkpoint", type=str, required=True)
    eval_p.add_argument("--per-circuit-csv", type=str, required=True)
    eval_p.add_argument("--summary-json", type=str, required=True)
    eval_p.add_argument("--eval-split", type=str, default="eval")
    eval_p.set_defaults(func=cmd_eval)

    full_p = subparsers.add_parser("full-run", help="train then evaluate from the same CLI")
    add_common(full_p)
    full_p.add_argument("--run-dir", type=str, required=True)
    full_p.add_argument("--epochs", type=int, default=60)
    full_p.add_argument("--batch-size", type=int, default=1)
    full_p.add_argument("--trainer-config", type=str, default=None)
    full_p.add_argument("--val-manifest", type=str, default=None)
    full_p.add_argument("--eval-manifest", type=str, default=None)
    full_p.add_argument("--resume-checkpoint", type=str, default=None)
    full_p.set_defaults(func=cmd_full_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
