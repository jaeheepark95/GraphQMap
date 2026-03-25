"""Training loops for GraphQMap Stage 1 and Stage 2.

Stage 1: Supervised pre-training with CE loss, τ annealing.
         MQT Bench → QUEKO sequential phases with early stopping.
Stage 2: Noise-aware surrogate fine-tuning with configurable loss components.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.graphqmap import GraphQMap
from training.early_stopping import EarlyStopping
from training.losses import Stage2Loss, SupervisedCELoss
from training.quality_score import QualityScore
from training.tau_scheduler import TauScheduler

logger = logging.getLogger(__name__)


class Stage1Trainer:
    """Stage 1 supervised pre-training trainer.

    Handles MQT Bench → QUEKO transition with early stopping,
    τ annealing, and checkpoint saving.

    Args:
        model: GraphQMap model.
        cfg: Config object with training.* attributes.
        device: Torch device.
    """

    def __init__(
        self,
        model: GraphQMap,
        cfg: Any,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        self.criterion = SupervisedCELoss()
        self.tau_scheduler = TauScheduler(
            tau_max=cfg.sinkhorn.tau_max,
            tau_min=cfg.sinkhorn.tau_min,
            schedule=cfg.sinkhorn.schedule,
            total_epochs=cfg.training.mlqd_queko.max_epochs,
        )

        self._setup_optimizer(cfg.training.optimizer.lr)
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics CSV
        self.metrics_path = self.checkpoint_dir.parent / "metrics.csv"
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "phase", "tau", "lr", "train_loss", "val_loss"])

    def _log_metrics(self, epoch: int, phase: str, tau: float, lr: float,
                     train_loss: float, val_loss: float) -> None:
        """Append one row to metrics CSV."""
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, phase, f"{tau:.6f}", f"{lr:.8f}",
                             f"{train_loss:.6f}", f"{val_loss:.6f}"])

    def _setup_optimizer(self, lr: float) -> None:
        """Initialize optimizer and scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.cfg.training.optimizer.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.training.scheduler.T_max,
            eta_min=self.cfg.training.scheduler.eta_min,
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> float:
        """Train one epoch.

        Args:
            train_loader: DataLoader yielding collated batch dicts.
            epoch: Current epoch number.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        tau = self.tau_scheduler.get_tau(epoch)
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            circuit_batch = batch["circuit_batch"].to(self.device)
            hardware_batch = batch["hardware_batch"].to(self.device)
            label_matrices = batch["label_matrices"].to(self.device)
            num_logical = batch["num_logical"][0]  # same within backend bucket
            num_physical = batch["num_physical"]
            batch_size = batch["batch_size"]

            self.optimizer.zero_grad()

            P = self.model(
                circuit_batch, hardware_batch,
                batch_size=batch_size,
                num_logical=num_logical,
                num_physical=num_physical,
                tau=tau,
            )

            loss = self.criterion(P, label_matrices)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self.scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch} | τ={tau:.4f} | LR={self.scheduler.get_last_lr()[0]:.6f} | Loss={avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> float:
        """Validate and return CE loss.

        Args:
            val_loader: Validation DataLoader.
            epoch: Current epoch.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        tau = self.tau_scheduler.get_tau(epoch)
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            circuit_batch = batch["circuit_batch"].to(self.device)
            hardware_batch = batch["hardware_batch"].to(self.device)
            label_matrices = batch["label_matrices"].to(self.device)
            num_logical = batch["num_logical"][0]
            num_physical = batch["num_physical"]
            batch_size = batch["batch_size"]

            P = self.model(
                circuit_batch, hardware_batch,
                batch_size=batch_size,
                num_logical=num_logical,
                num_physical=num_physical,
                tau=tau,
            )

            loss = self.criterion(P, label_matrices)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch} | Val Loss={avg_loss:.6f}")
        return avg_loss

    def save_checkpoint(self, epoch: int, tag: str = "best") -> Path:
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"{tag}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved: {path}")
        return path

    def train_phase(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        patience: int,
        min_delta: float,
        tag: str,
    ) -> float:
        """Run a training phase with early stopping.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            max_epochs: Maximum epochs for this phase.
            patience: Early stopping patience.
            min_delta: Minimum improvement for early stopping.
            tag: Checkpoint tag (e.g. 'mqt_best', 'queko_best').

        Returns:
            Best validation loss.
        """
        early_stop = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")
        best_val_loss = float("inf")

        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)

            tau = self.tau_scheduler.get_tau(epoch)
            lr = self.scheduler.get_last_lr()[0]
            self._log_metrics(epoch, tag, tau, lr, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, tag=tag)

            if early_stop.step(val_loss):
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        return best_val_loss

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        queko_train_loader: DataLoader | None = None,
        queko_val_loader: DataLoader | None = None,
    ) -> None:
        """Run full Stage 1: MLQD+QUEKO → QUEKO fine-tuning.

        Args:
            train_loader: Main training DataLoader (MLQD + QUEKO labels).
            val_loader: Main validation DataLoader.
            queko_train_loader: Optional QUEKO-only training DataLoader for fine-tuning.
            queko_val_loader: Optional QUEKO-only validation DataLoader.
        """
        main_cfg = self.cfg.training.mlqd_queko
        logger.info("=== Stage 1 Phase 1: MLQD + QUEKO ===")
        self.train_phase(
            train_loader, val_loader,
            max_epochs=main_cfg.max_epochs,
            patience=main_cfg.early_stopping.patience,
            min_delta=main_cfg.early_stopping.min_delta,
            tag="mlqd_queko_best",
        )

        # Transition to QUEKO
        if queko_train_loader is not None and queko_val_loader is not None:
            logger.info("=== Stage 1 Phase 2: QUEKO Fine-tuning ===")
            queko_cfg = self.cfg.training.queko

            # Reduce LR
            new_lr = self.cfg.training.optimizer.lr * queko_cfg.lr_factor
            self._setup_optimizer(new_lr)
            logger.info(f"LR reduced to {new_lr}")

            self.train_phase(
                queko_train_loader, queko_val_loader,
                max_epochs=queko_cfg.max_epochs,
                patience=queko_cfg.early_stopping.patience,
                min_delta=queko_cfg.early_stopping.min_delta,
                tag="queko_best",
            )

        # Save final Stage 1 checkpoint
        self.save_checkpoint(0, tag="best")
        logger.info("Stage 1 complete.")


class Stage2Trainer:
    """Stage 2 noise-aware surrogate fine-tuning trainer.

    Uses configurable loss components from YAML config.

    Args:
        model: GraphQMap model (pre-trained from Stage 1).
        cfg: Config object.
        device: Torch device.
    """

    def __init__(
        self,
        model: GraphQMap,
        cfg: Any,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        # τ scheduler: supports both fixed and annealing modes
        self.tau_scheduler = TauScheduler(
            tau_max=getattr(cfg.sinkhorn, "tau_max", getattr(cfg.sinkhorn, "tau", 0.05)),
            tau_min=getattr(cfg.sinkhorn, "tau_min", getattr(cfg.sinkhorn, "tau", 0.05)),
            schedule=getattr(cfg.sinkhorn, "schedule", "fixed"),
            total_epochs=cfg.training.max_epochs,
        )

        # Learnable quality score (parameters trained jointly with model)
        self.quality_score = QualityScore().to(device)

        # Build loss from config components
        components = [
            {"name": c.name, "weight": c.weight}
            for c in cfg.loss.components
        ]
        self.criterion = Stage2Loss(
            components=components,
            quality_score=self.quality_score,
        )

        # Combine model + quality_score parameters
        all_params = list(self.model.parameters()) + list(self.quality_score.parameters())
        self.optimizer = AdamW(
            all_params,
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.scheduler.T_max,
            eta_min=cfg.training.scheduler.eta_min,
        )

        self.warmup_epochs = getattr(cfg.training, "warmup_epochs", 0)
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics CSV — columns derived from active loss components
        self.loss_names = self.criterion.component_names
        self.metrics_path = self.checkpoint_dir.parent / "metrics.csv"
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "tau", "lr", "l_total"] + self.loss_names + ["val_pst"])

    def _warmup_lr(self, epoch: int) -> None:
        """Apply linear warmup to learning rate."""
        if epoch < self.warmup_epochs and self.warmup_epochs > 0:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.cfg.training.optimizer.lr * warmup_factor

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train one epoch.

        Args:
            train_loader: DataLoader yielding batch dicts with additional
                fields: d_hw, hw_node_features, circuit_edge_pairs,
                circuit_edge_weights, qubit_importance.
            epoch: Current epoch.

        Returns:
            Dict of average losses: total, l_adj, l_hop, l_node.
        """
        self.model.train()
        self._warmup_lr(epoch)
        tau = self.tau_scheduler.get_tau(epoch)

        accum: dict[str, float] = {"total": 0.0}
        for name in self.loss_names:
            accum[name] = 0.0
        num_batches = 0

        for batch in train_loader:
            circuit_batch = batch["circuit_batch"].to(self.device)
            hardware_batch = batch["hardware_batch"].to(self.device)
            num_logical = batch["num_logical"][0]
            num_physical = batch["num_physical"]
            batch_size = batch["batch_size"]

            self.optimizer.zero_grad()

            hw_feats = batch.get("hw_node_features")
            if hw_feats is not None:
                hw_feats = hw_feats.to(self.device)

            P = self.model(
                circuit_batch, hardware_batch,
                batch_size=batch_size,
                num_logical=num_logical,
                num_physical=num_physical,
                tau=tau,
                hw_node_features=hw_feats,
            )

            # Build kwargs for loss components — pass all available fields
            loss_kwargs: dict[str, Any] = {
                "num_logical": num_logical,
                "circuit_edge_pairs": batch["circuit_edge_pairs"],
                "circuit_edge_weights": batch.get("circuit_edge_weights", []),
                "qubit_importance": batch["qubit_importance"].to(self.device),
                "hw_node_features": batch["hw_node_features"].to(self.device),
                "cross_circuit_pairs": batch.get("cross_circuit_pairs", []),
            }
            if "d_hw" in batch:
                loss_kwargs["d_hw"] = batch["d_hw"].to(self.device)
            if "d_error" in batch:
                loss_kwargs["d_error"] = batch["d_error"].to(self.device)

            losses = self.criterion(P, **loss_kwargs)

            # Skip nan/inf losses to avoid corrupting weights
            if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
                continue

            losses["total"].backward()
            grad_clip = getattr(self.cfg.training, 'grad_clip_norm', 1.0)
            all_params = list(self.model.parameters()) + list(self.quality_score.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
            self.optimizer.step()

            for k in accum:
                if k in losses:
                    accum[k] += losses[k].item()
            num_batches += 1

        if epoch >= self.warmup_epochs:
            self.scheduler.step()

        avg = {k: v / max(num_batches, 1) for k, v in accum.items()}
        lr = self.optimizer.param_groups[0]['lr']

        parts = [f"Epoch {epoch} | τ={tau:.4f} | LR={lr:.6f} | L_total={avg['total']:.6f}"]
        for name in self.loss_names:
            parts.append(f"{name}={avg[name]:.6f}")
        logger.info(" | ".join(parts))

        # Store metrics for CSV (PST filled in by run() after validation)
        self._last_metrics = [epoch, f"{tau:.6f}", f"{lr:.8f}", f"{avg['total']:.6f}"]
        for name in self.loss_names:
            self._last_metrics.append(f"{avg[name]:.6f}")

        return avg

    def _write_metrics_row(self, val_pst: str = "") -> None:
        """Append the last training metrics row with optional PST value."""
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._last_metrics + [val_pst])

    def save_checkpoint(self, epoch: int, tag: str = "best") -> Path:
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"{tag}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "quality_score_state_dict": self.quality_score.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved: {path}")
        return path

    def run(
        self,
        train_loader: DataLoader,
        val_pst_fn: Any | None = None,
    ) -> None:
        """Run Stage 2 training.

        Args:
            train_loader: Training DataLoader.
            val_pst_fn: Optional callable(model, epoch) -> float that measures
                actual validation PST. Used for early stopping.
                If None, trains for max_epochs without PST-based stopping.
        """
        max_epochs = self.cfg.training.max_epochs
        pst_cfg = getattr(self.cfg.training, "pst_validation", None)

        early_stop = None
        if val_pst_fn is not None and pst_cfg is not None:
            early_stop = EarlyStopping(
                patience=self.cfg.training.early_stopping.patience,
                min_delta=self.cfg.training.early_stopping.min_delta,
                mode="max",  # PST: higher is better
            )

        best_pst = -1.0

        for epoch in range(max_epochs):
            losses = self.train_epoch(train_loader, epoch)

            # PST validation at intervals
            if (val_pst_fn is not None and pst_cfg is not None
                    and (epoch + 1) % pst_cfg.interval == 0):
                pst = val_pst_fn(self.model, epoch)
                logger.info(f"Epoch {epoch} | Val PST={pst:.4f}")
                self._write_metrics_row(val_pst=f"{pst:.4f}")

                if pst > best_pst:
                    best_pst = pst
                    self.save_checkpoint(epoch, tag="best")

                if early_stop is not None and early_stop.step(pst):
                    logger.info(f"Early stopping at epoch {epoch} (PST)")
                    break
            else:
                self._write_metrics_row()

        # Save final
        self.save_checkpoint(0, tag="final")
        logger.info(f"Stage 2 complete. Best PST={best_pst:.4f}")
