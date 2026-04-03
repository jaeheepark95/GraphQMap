# =============================================================================
# UPDATE LOG (2026-03-25)
# - Added CLI support for staged curriculum training.
# - Added stage-length flags, tau_r schedule flags, and stage-specific aux-loss flags.
# - Updated TrainerConfig defaults to match the new trainer-side curriculum.
# - Added train-time validation for curriculum lengths.
# - Preserved backward compatibility: legacy training still works when
#   --use-staged-curriculum is not passed.
# =============================================================================

# src/kmw/cli/main.py

"""
Command-line entrypoint for the KMW project.

Design goal
-----------
Keep this file thin.

This CLI should NOT contain:
- model architecture logic
- loss formulas
- training-step logic
- evaluation metric definitions

Those belong in their respective modules.

This file should mainly:
1. parse command-line arguments
2. construct configs / paths
3. build datasets / loaders / model / trainer
4. dispatch to:
      - preprocessing / manifest generation
      - training
      - evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Core project imports
# -----------------------------------------------------------------------------

from kmw.models.model import KMWModel
from kmw.losses.loss import LossConfig
from kmw.training.trainer import TrainerConfig, build_trainer
from kmw.evaluation.evaluate import EvalConfig, evaluate_model

# -----------------------------------------------------------------------------
# Dataset / manifest imports
# -----------------------------------------------------------------------------

try:
    from kmw.data.dataset import KMWDataset, kmw_collate_fn
except Exception:
    KMWDataset = None
    kmw_collate_fn = None

try:
    from scripts.build_manifests import build_manifests
except Exception:
    build_manifests = None


# =============================================================================
# 1) Small config / JSON helpers
# =============================================================================

def load_json_file(path: Optional[str]) -> Dict[str, Any]:
    """Load a JSON file into a dict. If path is None, return {}."""
    if path is None:
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON config file does not exist: {p}")

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def update_dataclass_from_dict(instance, updates: Dict[str, Any]):
    """
    Update an existing dataclass-like config object from a dictionary.
    Unknown keys are ignored intentionally.
    """
    for key, value in updates.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def maybe_print_dict(title: str, d: Dict[str, Any]) -> None:
    """Pretty-print a small dictionary as JSON."""
    print(f"\n{title}")
    print(json.dumps(d, indent=2, ensure_ascii=False))


# =============================================================================
# 2) Device / reproducibility helpers
# =============================================================================

def resolve_device(device_arg: str) -> torch.device:
    """
    Resolve device from CLI string.

    Supported:
        "auto"  -> cuda if available else cpu
        "cpu"
        "cuda"
        "cuda:0", "cuda:1", ...
    """
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_global_seed(seed: int) -> None:
    """Set basic PyTorch random seeds."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 3) Dataset / DataLoader builders
# =============================================================================

def require_dataset_available() -> None:
    """Fail with a clear message if the dataset import is unavailable."""
    if KMWDataset is None:
        raise ImportError(
            "Could not import KMWDataset from kmw.data.dataset. "
            "Please make sure dataset.py defines KMWDataset, or update the CLI import."
        )


def build_dataset_from_args(args):
    require_dataset_available()

    dataset = KMWDataset(
        project_root=Path.cwd(),
        manifest_path=args.manifest,
        force_recompute=args.no_cache,
    )
    return dataset


def build_loader_from_args(dataset, args, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=kmw_collate_fn,
    )


# =============================================================================
# 4) Model / checkpoint helpers
# =============================================================================

def build_model_from_args(args) -> KMWModel:
    """Construct the KMW model using the currently locked dimensions."""
    model = KMWModel(
        n=args.n_qubits,
        d_r=args.d_r,
        d_tok=args.d_tok,
        sinkhorn_iters=args.sinkhorn_iters,
        dropout=args.dropout,
    )
    return model


def load_checkpoint_into_model(
    model: KMWModel,
    checkpoint_path: str,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """
    Load checkpoint weights into the model.
    Returns the loaded checkpoint dictionary so caller can also restore metadata.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)

    if "model" not in ckpt:
        raise KeyError("Checkpoint does not contain key 'model'")

    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


# =============================================================================
# 5) Manifest command
# =============================================================================

def cmd_build_manifests(args) -> int:
    """Build JSONL manifests for train / val / test."""
    if build_manifests is None:
        raise ImportError(
            "Could not import build_manifests from scripts.build_manifests. "
            "Please make sure scripts/build_manifests.py exposes a callable named build_manifests."
        )

    result = build_manifests(
        input_root=args.input_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        source=args.source,
    )

    print("\nManifest generation completed.")
    maybe_print_dict("Manifest build result", result if isinstance(result, dict) else {"result": str(result)})
    return 0


# =============================================================================
# 6) Train command
# =============================================================================

def build_loss_config_for_train(args) -> LossConfig:
    """
    Build LossConfig for training from:
    - defaults
    - optional JSON config
    - CLI overrides
    """
    cfg = LossConfig()

    cfg_updates = load_json_file(args.loss_config_json)
    cfg = update_dataclass_from_dict(cfg, cfg_updates)

    cfg.tau_m = args.tau_m
    cfg.sinkhorn_iters = args.sinkhorn_iters
    cfg.lambda_p = args.lambda_p
    cfg.lambda_s = args.lambda_s
    cfg.lambda_d = args.lambda_d
    cfg.kappa_depth = args.kappa_depth
    cfg.alpha_loc = args.alpha_loc
    cfg.beta_cons = args.beta_cons

    return cfg


def build_trainer_config_for_train(args) -> TrainerConfig:
    """
    Build TrainerConfig for training from:
    - defaults
    - optional JSON config
    - CLI overrides
    """
    cfg = TrainerConfig()

    cfg_updates = load_json_file(args.trainer_config_json)
    cfg = update_dataclass_from_dict(cfg, cfg_updates)

    # Legacy / always-relevant overrides
    cfg.mapper_lr = args.mapper_lr
    cfg.reindexer_lr = args.reindexer_lr
    cfg.weight_decay = args.weight_decay
    cfg.num_epochs = args.epochs
    cfg.grad_clip_norm = args.grad_clip_norm
    cfg.log_every_steps = args.log_every_steps
    cfg.save_every_epochs = args.save_every_epochs
    cfg.tau_r = args.tau_r
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.checkpoint_prefix = args.checkpoint_prefix

    # New staged curriculum overrides
    cfg.use_staged_curriculum = args.use_staged_curriculum
    cfg.warmup_epochs = args.warmup_epochs
    cfg.stage1_epochs = args.stage1_epochs
    cfg.stage2_epochs = args.stage2_epochs
    cfg.tau_r_start = args.tau_r_start
    cfg.tau_r_mid = args.tau_r_mid
    cfg.tau_r_end = args.tau_r_end
    cfg.tau_r_schedule = args.tau_r_schedule
    cfg.stage1_alpha_loc = args.stage1_alpha_loc
    cfg.stage1_beta_cons = args.stage1_beta_cons
    cfg.stage2_alpha_loc = args.stage2_alpha_loc
    cfg.stage2_beta_cons = args.stage2_beta_cons
    cfg.warmup_reindex_mode = args.warmup_reindex_mode

    return cfg


def validate_train_args(args, trainer_config: TrainerConfig) -> None:
    """
    Fail early on bad curriculum settings.
    """
    if not trainer_config.use_staged_curriculum:
        return

    if trainer_config.warmup_reindex_mode != "identity":
        raise ValueError(
            f"Unsupported --warmup-reindex-mode={trainer_config.warmup_reindex_mode!r}. "
            "Current implementation supports only 'identity'."
        )

    total_planned = (
        trainer_config.warmup_epochs
        + trainer_config.stage1_epochs
        + trainer_config.stage2_epochs
    )

    if total_planned > trainer_config.num_epochs:
        raise ValueError(
            "Curriculum phase lengths exceed total epochs: "
            f"{total_planned} > {trainer_config.num_epochs}. "
            "Increase --epochs or shorten the stage lengths."
        )


def cmd_train(args) -> int:
    """
    Run the full training pipeline.
    """
    set_global_seed(args.seed)
    device = resolve_device(args.device)

    dataset = build_dataset_from_args(args)
    loader = build_loader_from_args(dataset, args, shuffle=not args.no_shuffle)

    model = build_model_from_args(args)

    if args.resume_checkpoint is not None:
        load_checkpoint_into_model(model, args.resume_checkpoint, map_location="cpu")
        print(f"Loaded model checkpoint: {args.resume_checkpoint}")

    loss_config = build_loss_config_for_train(args)
    trainer_config = build_trainer_config_for_train(args)
    validate_train_args(args, trainer_config)

    trainer = build_trainer(
        model=model,
        loss_config=loss_config,
        trainer_config=trainer_config,
        device=device,
    )

    history = trainer.fit(
        train_loader=loader,
        num_epochs=args.epochs,
    )

    print("\nTraining finished.")
    print(f"Device used: {device}")
    print(f"Checkpoint dir: {trainer_config.checkpoint_dir}")

    if history:
        final_summary = history[-1]
        maybe_print_dict("Final epoch summary", final_summary)

    return 0


# =============================================================================
# 7) Eval command
# =============================================================================

def build_loss_config_for_eval(args) -> LossConfig:
    """
    Build LossConfig for evaluation.
    """
    cfg = LossConfig()

    cfg_updates = load_json_file(args.loss_config_json)
    cfg = update_dataclass_from_dict(cfg, cfg_updates)

    cfg.tau_m = args.tau_m
    cfg.sinkhorn_iters = args.sinkhorn_iters
    cfg.lambda_p = args.lambda_p
    cfg.lambda_s = args.lambda_s
    cfg.lambda_d = args.lambda_d
    cfg.kappa_depth = args.kappa_depth
    cfg.alpha_loc = args.alpha_loc
    cfg.beta_cons = args.beta_cons

    return cfg


def build_eval_config(args) -> EvalConfig:
    """Build EvalConfig from defaults + JSON + CLI overrides."""
    cfg = EvalConfig()

    cfg_updates = load_json_file(args.eval_config_json)
    cfg = update_dataclass_from_dict(cfg, cfg_updates)

    cfg.per_circuit_csv_path = args.per_circuit_csv
    cfg.summary_json_path = args.summary_json
    cfg.print_console_summary = not args.no_console_summary
    cfg.include_routing_placeholders_in_csv = not args.no_routing_placeholders
    cfg.fail_fast = args.fail_fast
    cfg.eval_split_name = args.eval_split_name
    cfg.route_final_eval = args.route_final_eval
    cfg.routing_method = args.routing_method
    cfg.transpile_optimization_level = args.transpile_optimization_level
    cfg.seed_transpiler = args.seed_transpiler
    cfg.include_readout_in_pst = args.include_readout_in_pst
    cfg.save_routed_qasm_dir = args.save_routed_qasm_dir
    cfg.save_routed_qpy_dir = args.save_routed_qpy_dir

    return cfg


def cmd_eval(args) -> int:
    """
    Run the evaluation pipeline.
    """
    set_global_seed(args.seed)
    device = resolve_device(args.device)

    dataset = build_dataset_from_args(args)
    loader = build_loader_from_args(dataset, args, shuffle=False)

    model = build_model_from_args(args)
    load_checkpoint_into_model(model, args.checkpoint, map_location="cpu")

    loss_config = build_loss_config_for_eval(args)
    eval_config = build_eval_config(args)
    eval_config.project_root = str(Path.cwd())
    eval_config.backend_name = dataset.backend_name

    result = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        loss_config=loss_config,
        eval_config=eval_config,
    )

    print("\nEvaluation finished.")
    print(f"Per-circuit CSV: {result['per_circuit_csv_path']}")
    print(f"Summary JSON:    {result['summary_json_path']}")

    return 0


# =============================================================================
# 8) Argument parser
# =============================================================================

def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by train/eval for model construction."""
    parser.add_argument("--n-qubits", type=int, default=27, help="Fixed hardware size. Current project phase uses n=27.")
    parser.add_argument("--d-r", type=int, default=128, help="Hidden size used in reindex branches.")
    parser.add_argument("--d-tok", type=int, default=128, help="Hardware token dimension.")
    parser.add_argument("--sinkhorn-iters", type=int, default=20, help="Number of Sinkhorn iterations.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in MLP-style components.")


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by train/eval for dataset construction."""
    parser.add_argument("--manifest", type=str, required=True, help="Path to the authoritative JSONL manifest for this split.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Stable default for current phase is 1.")
    parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader num_workers.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable pin_memory in DataLoader.")
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset/cache usage if dataset.py supports it.")


def add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by train/eval for runtime."""
    parser.add_argument("--device", type=str, default="auto", help='Device string: "auto", "cpu", "cuda", "cuda:0", ...')
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")


def add_common_loss_override_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by train/eval because both need loss-side formulas."""
    parser.add_argument("--loss-config-json", type=str, default=None, help="Optional JSON file for LossConfig overrides.")
    parser.add_argument("--tau-m", type=float, default=0.10, help="Training/eval Sinkhorn temperature for mapper assignment.")
    parser.add_argument("--lambda-p", type=float, default=1.0, help="Weight for PST proxy term.")
    parser.add_argument("--lambda-s", type=float, default=1.0, help="Weight for swap proxy term.")
    parser.add_argument("--lambda-d", type=float, default=0.25, help="Weight for depth proxy term.")
    parser.add_argument("--kappa-depth", type=float, default=1.0, help="Depth proxy scaling coefficient.")
    parser.add_argument("--alpha-loc", type=float, default=0.0, help="Base weight for locality auxiliary loss.")
    parser.add_argument("--beta-cons", type=float, default=0.0, help="Base weight for consistency auxiliary loss.")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="kmw",
        description="CLI for the KMW project (manifests, training, evaluation).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------------------------------------------------------------------------
    # build-manifests
    # -------------------------------------------------------------------------
    p_manifest = subparsers.add_parser(
        "build-manifests",
        help="Generate JSONL manifests for train/val/test.",
    )
    p_manifest.add_argument("--input-root", type=str, required=True, help="Root directory containing raw QASM files.")
    p_manifest.add_argument("--output-dir", type=str, required=True, help="Directory to write train/val/test JSONL manifests.")
    p_manifest.add_argument("--source", type=str, default="unknown", help="Source label to store in manifest rows, e.g. mqt.")
    p_manifest.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    p_manifest.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    p_manifest.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    p_manifest.add_argument("--seed", type=int, default=42, help="Random seed used for split generation.")
    p_manifest.set_defaults(func=cmd_build_manifests)

    # -------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------
    p_train = subparsers.add_parser(
        "train",
        help="Run the KMW training loop.",
    )
    add_common_model_args(p_train)
    add_common_data_args(p_train)
    add_common_runtime_args(p_train)
    add_common_loss_override_args(p_train)

    p_train.add_argument("--trainer-config-json", type=str, default=None, help="Optional JSON file for TrainerConfig overrides.")

    p_train.add_argument("--epochs", type=int, default=60, help="Number of total epochs.")
    p_train.add_argument("--mapper-lr", type=float, default=1e-4, help="Learning rate for mapper+token_encoder optimizer.")
    p_train.add_argument("--reindexer-lr", type=float, default=5e-5, help="Learning rate for reindexer optimizer.")
    p_train.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizers.")
    p_train.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping max norm.")
    p_train.add_argument("--log-every-steps", type=int, default=10, help="Console logging interval in steps.")
    p_train.add_argument("--save-every-epochs", type=int, default=1, help="Checkpoint save interval in epochs.")
    p_train.add_argument("--tau-r", type=float, default=1.0, help="Fixed reindexer Sinkhorn temperature used only when staged curriculum is disabled.")
    p_train.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for saved checkpoints.")
    p_train.add_argument("--checkpoint-prefix", type=str, default="kmw", help="Checkpoint filename prefix.")
    p_train.add_argument("--resume-checkpoint", type=str, default=None, help="Optional checkpoint path to initialize model weights.")
    p_train.add_argument("--no-shuffle", action="store_true", help="Disable DataLoader shuffling.")

    # New staged curriculum arguments
    p_train.add_argument("--use-staged-curriculum", action="store_true", help="Enable staged curriculum training.")
    p_train.add_argument("--warmup-epochs", type=int, default=10, help="Warmup epochs with identity reindex and Pass A only.")
    p_train.add_argument("--stage1-epochs", type=int, default=15, help="Stage-1 joint-training epochs.")
    p_train.add_argument("--stage2-epochs", type=int, default=35, help="Stage-2 joint-training epochs.")
    p_train.add_argument("--tau-r-start", type=float, default=1.0, help="Stage schedule start value for tau_r.")
    p_train.add_argument("--tau-r-mid", type=float, default=0.60, help="Stage schedule midpoint value for tau_r.")
    p_train.add_argument("--tau-r-end", type=float, default=0.15, help="Stage schedule end value for tau_r.")
    p_train.add_argument("--tau-r-schedule", type=str, choices=["cosine", "linear"], default="cosine", help="Interpolation schedule used for tau_r annealing.")
    p_train.add_argument("--stage1-alpha-loc", type=float, default=0.02, help="alpha_loc active during stage 1.")
    p_train.add_argument("--stage1-beta-cons", type=float, default=0.0, help="beta_cons active during stage 1.")
    p_train.add_argument("--stage2-alpha-loc", type=float, default=0.05, help="alpha_loc active during stage 2.")
    p_train.add_argument("--stage2-beta-cons", type=float, default=0.0, help="beta_cons active during stage 2.")
    p_train.add_argument("--warmup-reindex-mode", type=str, choices=["identity"], default="identity", help="Warmup reindex mode. Current implementation supports only identity.")
    p_train.set_defaults(func=cmd_train)

    # -------------------------------------------------------------------------
    # eval
    # -------------------------------------------------------------------------
    p_eval = subparsers.add_parser(
        "eval",
        help="Run the KMW evaluation pipeline.",
    )
    add_common_model_args(p_eval)
    add_common_data_args(p_eval)
    add_common_runtime_args(p_eval)
    add_common_loss_override_args(p_eval)

    p_eval.add_argument("--eval-config-json", type=str, default=None, help="Optional JSON file for EvalConfig overrides.")
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path containing trained model weights.")
    p_eval.add_argument("--per-circuit-csv", type=str, default="artifacts/eval/per_circuit_metrics.csv", help="Output CSV path.")
    p_eval.add_argument("--summary-json", type=str, default="artifacts/eval/summary.json", help="Output summary JSON path.")
    p_eval.add_argument("--eval-split-name", type=str, default="eval", help="Human-readable label for this evaluation split.")
    p_eval.add_argument("--no-console-summary", action="store_true", help="Disable console summary printing.")
    p_eval.add_argument("--no-routing-placeholders", action="store_true", help="When routed evaluation is disabled, omit not-run routing columns from the CSV.")
    p_eval.add_argument("--fail-fast", action="store_true", help="Raise immediately on first evaluation error instead of recording a failure row.")
    p_eval.add_argument("--route-final-eval", action="store_true", help="Run real routed downstream evaluation after the proxy inference path.")
    p_eval.add_argument("--routing-method", type=str, default="sabre", help="Qiskit routing method for real routed evaluation.")
    p_eval.add_argument("--transpile-optimization-level", type=int, default=0, help="Qiskit transpiler optimization level for real routed evaluation.")
    p_eval.add_argument("--seed-transpiler", type=int, default=None, help="Optional Qiskit transpiler seed for reproducible routed evaluation.")
    p_eval.add_argument(
        "--include-readout-in-pst",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the real PST metric should include readout error contributions.",
    )
    p_eval.add_argument("--save-routed-qasm-dir", type=str, default=None, help="Optional directory to save final routed circuits as QASM.")
    p_eval.add_argument("--save-routed-qpy-dir", type=str, default=None, help="Optional directory to save final routed circuits as QPY.")
    p_eval.set_defaults(func=cmd_eval)

    return parser


# =============================================================================
# 9) Main entrypoint
# =============================================================================

def main(argv=None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# notes on implementation:
# =============================================================================

# The CLI stays thin: all actual staged-training behavior lives in trainer.py.

# Legacy runs remain available:
# if --use-staged-curriculum is not passed, the trainer falls back to the old
# two-pass behavior with fixed --tau-r.

# The new recommended defaults are:
#   epochs            = 60
#   mapper_lr         = 1e-4
#   reindexer_lr      = 5e-5
#   weight_decay      = 1e-4
#   warmup/stage1/2   = 10 / 15 / 35
#   tau_r             = 1.0 -> 0.60 -> 0.15