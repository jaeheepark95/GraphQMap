"""Training loop for GraphQMap.

Noise-aware surrogate fine-tuning with configurable loss components.
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
from training.losses import SurrogateLoss
from training.quality_score import QualityScore
from training.tau_scheduler import TauScheduler

logger = logging.getLogger(__name__)


class Trainer:
    """Noise-aware surrogate fine-tuning trainer.

    Uses configurable loss components from YAML config.

    Args:
        model: GraphQMap model.
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
        components = []
        for c in cfg.loss.components:
            comp = {"name": c.name, "weight": c.weight}
            if hasattr(c, "params"):
                p = c.params
                comp["params"] = {k: v for k, v in p.__dict__.items()
                                  if not k.startswith("_")}
            components.append(comp)
        self.criterion = SurrogateLoss(
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

            # C_eff and circuit_adj for iterative refinement and QAP loss
            c_eff = batch.get("c_eff")
            if c_eff is not None:
                c_eff = c_eff.to(self.device)
            circuit_adj = batch.get("circuit_adj")
            if circuit_adj is not None:
                circuit_adj = circuit_adj.to(self.device)

            P = self.model(
                circuit_batch, hardware_batch,
                batch_size=batch_size,
                num_logical=num_logical,
                num_physical=num_physical,
                tau=tau,
                hw_node_features=hw_feats,
                c_eff=c_eff,
                circuit_adj=circuit_adj,
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
            if c_eff is not None:
                loss_kwargs["c_eff"] = c_eff
            if circuit_adj is not None:
                loss_kwargs["circuit_adj"] = circuit_adj
            if "grama_W" in batch:
                loss_kwargs["grama_W"] = batch["grama_W"].to(self.device)
            if "grama_s_read" in batch:
                loss_kwargs["grama_s_read"] = batch["grama_s_read"].to(self.device)
            if "grama_s_gate" in batch:
                loss_kwargs["grama_s_gate"] = batch["grama_s_gate"].to(self.device)
            if "grama_g_single" in batch:
                loss_kwargs["grama_g_single"] = batch["grama_g_single"].to(self.device)

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
        """Append the last training metrics row with optional PST."""
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
        """Run training.

        Args:
            train_loader: Training DataLoader.
            val_pst_fn: Optional callable(model, epoch) -> float that measures
                actual validation PST. Used for best checkpoint selection.
        """
        max_epochs = self.cfg.training.max_epochs
        pst_cfg = getattr(self.cfg.training, "pst_validation", None)

        best_pst = -1.0
        best_loss = float("inf")

        for epoch in range(max_epochs):
            losses = self.train_epoch(train_loader, epoch)

            # Best loss checkpoint — track lowest total training loss
            total_loss = losses.get("total", float("inf"))
            if total_loss < best_loss:
                best_loss = total_loss
                self.save_checkpoint(epoch, tag="best_loss")

            # PST validation at intervals — best checkpoint selected by PST
            pst_str = ""
            if (val_pst_fn is not None and pst_cfg is not None
                    and (epoch + 1) % pst_cfg.interval == 0):
                pst = val_pst_fn(self.model, epoch)
                logger.info(f"Epoch {epoch} | Val PST={pst:.4f}")
                pst_str = f"{pst:.4f}"
                if pst > best_pst:
                    best_pst = pst
                    self.save_checkpoint(epoch, tag="best")

            self._write_metrics_row(val_pst=pst_str)

        # Save final
        self.save_checkpoint(0, tag="final")
        logger.info(f"Training complete. Best PST={best_pst:.4f} | Best loss={best_loss:.6f}")
