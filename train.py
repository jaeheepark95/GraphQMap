"""GraphQMap training entry point.

Usage:
    python train.py --config configs/stage1.yaml
    python train.py --config configs/stage2.yaml
    python train.py --config configs/stage1.yaml --override training.optimizer.lr=0.0005
"""

from __future__ import annotations

import logging

import torch

from configs.config_loader import parse_args_with_config
from data.dataset import create_dataloader, load_split
from models.graphqmap import GraphQMap
from training.trainer import Stage1Trainer, Stage2Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy Qiskit logging
logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("stevedore").setLevel(logging.WARNING)


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_training_backends(cfg) -> list[str]:
    """Extract training backend names from config."""
    backends = []
    for name in cfg.backends.training:
        # Normalize: FakeAthens -> athens, queko_aspen4 stays as-is
        if name.startswith("Fake"):
            backends.append(name[4:].lower())
        else:
            backends.append(name)
    return backends


def main() -> None:
    cfg = parse_args_with_config()
    stage = cfg.training.stage
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    _set_seed(cfg.seed)

    logger.info("=== GraphQMap Training — Stage %d ===", stage)
    logger.info("Device: %s", device)
    logger.info("Run directory: %s", cfg.checkpoint_dir.rsplit("/checkpoints", 1)[0])

    data_root = cfg.data.data_root
    max_nodes = cfg.batching.max_total_nodes
    num_workers = cfg.num_workers
    training_backends = _get_training_backends(cfg)

    if stage == 1:
        model = GraphQMap.from_config(cfg)
        trainer = Stage1Trainer(model, cfg, device)

        # Load main supervised train/val datasets
        logger.info("Loading Stage 1 supervised training data...")
        train_ds = load_split(cfg.data.splits.train, data_root=data_root)
        val_ds = load_split(cfg.data.splits.val, data_root=data_root)
        logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

        train_loader = create_dataloader(
            train_ds, max_total_nodes=max_nodes, shuffle=True,
            num_workers=num_workers, seed=cfg.seed,
        )
        val_loader = create_dataloader(
            val_ds, max_total_nodes=max_nodes, shuffle=False,
            num_workers=num_workers, seed=cfg.seed,
        )

        # Load QUEKO-only data for fine-tuning phase
        queko_train_loader = None
        queko_val_loader = None
        queko_train_path = getattr(cfg.data.splits, "queko_train", None)
        queko_val_path = getattr(cfg.data.splits, "queko_val", None)
        if queko_train_path and queko_val_path:
            logger.info("Loading QUEKO fine-tuning data...")
            queko_train_ds = load_split(queko_train_path, data_root=data_root)
            queko_val_ds = load_split(queko_val_path, data_root=data_root)
            logger.info("QUEKO Train: %d, Val: %d", len(queko_train_ds), len(queko_val_ds))

            queko_train_loader = create_dataloader(
                queko_train_ds, max_total_nodes=max_nodes, shuffle=True,
                num_workers=num_workers, seed=cfg.seed,
            )
            queko_val_loader = create_dataloader(
                queko_val_ds, max_total_nodes=max_nodes, shuffle=False,
                num_workers=num_workers, seed=cfg.seed,
            )

        trainer.run(train_loader, val_loader, queko_train_loader, queko_val_loader)

    elif stage == 2:
        model = GraphQMap.from_config(cfg)

        # Load Stage 1 checkpoint
        pretrained = getattr(cfg, "pretrained_checkpoint", None)
        if pretrained:
            logger.info("Loading pretrained checkpoint: %s", pretrained)
            ckpt = torch.load(pretrained, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])

        trainer = Stage2Trainer(model, cfg, device)

        # Load all circuits for Stage 2 surrogate training
        logger.info("Loading Stage 2 training data (all circuits)...")
        train_ds = load_split(
            cfg.data.splits.train,
            data_root=data_root,
            training_backends=training_backends,
            include_stage2_fields=True,
        )
        logger.info("Train: %d samples", len(train_ds))

        train_loader = create_dataloader(
            train_ds, max_total_nodes=max_nodes, shuffle=True,
            num_workers=num_workers, seed=cfg.seed,
        )

        trainer.run(train_loader)

    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
