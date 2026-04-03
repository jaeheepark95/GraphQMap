# src/kmw/training/trainer.py

"""
Trainer for the KMW project.

What this file does
-------------------
This file implements the locked two-pass training procedure.

Pass A: mapper update
---------------------
Goal:
    update mapper parameters only

Rule:
    - run reindexer
    - DETACH reindex outputs before feeding mapper
    - compute mapper task loss
    - backprop only through mapper

Pass B: reindexer update
------------------------
Goal:
    update reindexer parameters only

Rule:
    - run reindexer again without detach
    - freeze mapper parameters
    - keep mapper forward differentiable (NO torch.no_grad())
    - mapper acts as a frozen operator
    - compute reindex objective:
          L_reindex = L_task + alpha_loc * L_loc + beta_cons * L_cons
    - backprop only through reindexer

Important architectural note
----------------------------
The mapper freeze policy belongs here, not in model.py.
That was a deliberate implementation decision:
- architecture stays in model.py
- training policy stays in trainer.py

This file is intentionally explicit and verbose so that a reader can follow the
two-pass logic without guessing where gradients are allowed to flow.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from kmw.losses.loss import (
    LossConfig,
    compute_reindex_objective,
    compute_task_loss_from_logits,
)
from kmw.models.model import KMWModel, maybe_detach_reindex_outputs


# =============================================================================
# 1) Trainer configuration
# =============================================================================

@dataclass
class TrainerConfig:
    """
    Training-related configuration.

    These are trainer-side settings, not model-architecture settings.
    """

    # Optimization
    mapper_lr: float = 1e-4
    reindexer_lr: float = 1e-4
    weight_decay: float = 0.0

    # Epoch / checkpoint behavior
    num_epochs: int = 50
    grad_clip_norm: Optional[float] = 1.0
    log_every_steps: int = 10
    save_every_epochs: int = 1

    # Reindexer / mapper temperatures
    tau_r: float = 1.0

    # Fail-fast behavior
    fail_on_nonfinite_loss: bool = True
    fail_on_nonfinite_grad: bool = True

    # Output
    checkpoint_dir: str = "checkpoints"
    checkpoint_prefix: str = "kmw"

    # If True, save a "latest.pt" checkpoint every epoch as well.
    save_latest: bool = True

    # Human-readable JSONL log file
    metrics_jsonl_name: str = "train_metrics.jsonl"


# =============================================================================
# 2) Low-level helpers
# =============================================================================

def _assert_finite_tensor(x: torch.Tensor, name: str) -> None:
    """
    Fail loudly if a tensor contains NaN or Inf.
    """
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def _assert_required_batch_keys(batch: Dict, required_keys: Iterable[str]) -> None:
    """
    Check that the batch dictionary contains the required keys.
    """
    missing = [k for k in required_keys if k not in batch]
    if missing:
        raise KeyError(f"Batch is missing required keys: {missing}")


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """
    Move every tensor value in a batch dict to the target device.

    Non-tensor values are passed through unchanged.
    """
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def extract_native_batch(batch: Dict) -> Dict[str, torch.Tensor]:
    """
    Extract just the native-frame tensors the model/losses expect.

    Required keys:
        A, m, B, c1, c2, D

    We intentionally keep this helper small so trainer code is readable.
    """
    required = ("A", "m", "B", "c1", "c2", "D")
    _assert_required_batch_keys(batch, required)

    native = {k: batch[k] for k in required}

    for key, value in native.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Batch key '{key}' must contain a tensor, got {type(value)}")
        _assert_finite_tensor(value, key)

    return native


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    Enable or disable gradients for every parameter in a module.

    This does NOT affect whether forward() is differentiable.
    It only controls whether the module's parameters receive gradients.
    """
    for param in module.parameters():
        param.requires_grad = requires_grad


def zero_optimizer_grads(*optimizers: torch.optim.Optimizer) -> None:
    """
    Zero gradients for one or more optimizers.
    """
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def clip_gradients_if_needed(module: nn.Module, max_norm: Optional[float]) -> Optional[float]:
    """
    Clip gradients if requested.

    Returns:
        total_norm reported by torch, or None if clipping is disabled.
    """
    if max_norm is None:
        return None

    params = [p for p in module.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return None

    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    return float(total_norm)


def assert_module_gradients_finite(module: nn.Module, module_name: str) -> None:
    """
    Fail loudly if any gradient inside a module is NaN/Inf.
    """
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise ValueError(f"Non-finite gradient detected in {module_name}.{name}")


def detach_scalar_dict(metrics: Dict) -> Dict:
    """
    Convert tensor scalars to plain Python numbers for easy logging.

    Non-scalar tensors are skipped.
    Non-tensor values are copied through unchanged if JSON-friendly.
    """
    out = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


def count_parameters(module: nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# =============================================================================
# 3) Optimizer builders
# =============================================================================

def build_mapper_optimizer(model: KMWModel, config: TrainerConfig) -> torch.optim.Optimizer:
    """
    Build optimizer for mapper parameters only.
    """
    return torch.optim.Adam(
        model.mapper.parameters(),
        lr=config.mapper_lr,
        weight_decay=config.weight_decay,
    )


def build_reindexer_optimizer(model: KMWModel, config: TrainerConfig) -> torch.optim.Optimizer:
    """
    Build optimizer for reindexer + token encoder parameters.

    Why token encoder is included here
    ----------------------------------
    The token encoder is part of the conditioning path attached to the reindexed
    hardware tensors. In the current project structure, it is most natural to
    update it together with the reindexer-side path.
    """
    params = list(model.reindexer.parameters()) + list(model.token_encoder.parameters())

    return torch.optim.Adam(
        params,
        lr=config.reindexer_lr,
        weight_decay=config.weight_decay,
    )


# =============================================================================
# 4) Main trainer class
# =============================================================================

class KMWTrainer:
    """
    Main trainer implementing the locked two-pass update scheme.
    """

    def __init__(
        self,
        model: KMWModel,
        loss_config: Optional[LossConfig] = None,
        trainer_config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.loss_config = loss_config if loss_config is not None else LossConfig()
        self.trainer_config = trainer_config if trainer_config is not None else TrainerConfig()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model.to(self.device)

        self.mapper_optimizer = build_mapper_optimizer(self.model, self.trainer_config)
        self.reindexer_optimizer = build_reindexer_optimizer(self.model, self.trainer_config)

        # Keep basic running state
        self.global_step = 0
        self.current_epoch = 0

        # Output paths
        self.checkpoint_dir = Path(self.trainer_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_jsonl_path = self.checkpoint_dir / self.trainer_config.metrics_jsonl_name

    # -------------------------------------------------------------------------
    # 4.1) Logging / checkpoint helpers
    # -------------------------------------------------------------------------

    def log_metrics(self, metrics: Dict) -> None:
        """
        Append one JSON object to the metrics JSONL log.

        This makes it easy to inspect training progress later.
        """
        record = detach_scalar_dict(metrics)
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def state_dict(self) -> Dict:
        """
        Trainer state for checkpoints.
        """
        return {
            "model": self.model.state_dict(),
            "mapper_optimizer": self.mapper_optimizer.state_dict(),
            "reindexer_optimizer": self.reindexer_optimizer.state_dict(),
            "loss_config": asdict(self.loss_config),
            "trainer_config": asdict(self.trainer_config),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "device": str(self.device),
        }

    def save_checkpoint(self, filename: str) -> str:
        """
        Save a checkpoint file and return the path as string.
        """
        path = self.checkpoint_dir / filename
        torch.save(self.state_dict(), path)
        return str(path)

    def save_epoch_checkpoint(self, epoch: int) -> Dict[str, str]:
        """
        Save standard epoch checkpoints.
        """
        saved = {}
        name = f"{self.trainer_config.checkpoint_prefix}_epoch_{epoch:04d}.pt"
        saved["epoch"] = self.save_checkpoint(name)

        if self.trainer_config.save_latest:
            saved["latest"] = self.save_checkpoint("latest.pt")

        return saved

    # -------------------------------------------------------------------------
    # 4.2) Core forward helpers
    # -------------------------------------------------------------------------

    def run_pass_a_mapper_forward(
        self,
        native: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Pass A: mapper update path.

        Locked policy:
        --------------
        - reindexer is run first
        - reindex outputs are detached before mapper usage
        - mapper sees detached reordered tensors / tokens
        - mapper loss updates mapper only

        Implementation note
        -------------------
        We run the reindexer directly rather than calling model.forward(),
        because detach policy belongs here in trainer.py.
        """
        # Reindex first
        reidx = self.model.reindexer(
            A=native["A"],
            m=native["m"],
            Bmat=native["B"],
            c1=native["c1"],
            c2=native["c2"],
            D=native["D"],
            tau_r=self.trainer_config.tau_r,
        )

        # Detach here by trainer policy
        reidx_detached = maybe_detach_reindex_outputs(reidx, detach=True)

        # Build hardware tokens from detached reordered hardware tensors
        T_hw_star = self.model.token_encoder(
            B_star=reidx_detached["B_star"],
            c2_star=reidx_detached["c2_star"],
            c1_star=reidx_detached["c1_star"],
        )

        # Mapper sees detached A_star as spatial input
        A_star_spatial = reidx_detached["A_star"].unsqueeze(1)
        S_star = self.model.mapper(A_star_spatial, T_hw_star)

        return {
            **reidx_detached,
            "T_hw_star": T_hw_star,
            "S_star": S_star,
        }

    def run_pass_b_reindex_forward(
        self,
        native: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Pass B: reindexer update path.

        Locked policy:
        --------------
        - reindexer outputs are NOT detached
        - mapper parameters are frozen
        - mapper forward remains differentiable
        - DO NOT use torch.no_grad() around mapper forward

        Why?
        ----
        We need gradients to flow:
            reindexer -> reordered tensors/tokens -> frozen mapper -> S_star -> losses

        But we do NOT want mapper parameters themselves updated in Pass B.
        """
        reidx = self.model.reindexer(
            A=native["A"],
            m=native["m"],
            Bmat=native["B"],
            c1=native["c1"],
            c2=native["c2"],
            D=native["D"],
            tau_r=self.trainer_config.tau_r,
        )

        # No detach here
        T_hw_star = self.model.token_encoder(
            B_star=reidx["B_star"],
            c2_star=reidx["c2_star"],
            c1_star=reidx["c1_star"],
        )

        A_star_spatial = reidx["A_star"].unsqueeze(1)
        S_star = self.model.mapper(A_star_spatial, T_hw_star)

        return {
            **reidx,
            "T_hw_star": T_hw_star,
            "S_star": S_star,
        }

    # -------------------------------------------------------------------------
    # 4.3) Pass A / Pass B step functions
    # -------------------------------------------------------------------------

    def pass_a_mapper_step(self, native: Dict[str, torch.Tensor]) -> Dict:
        """
        One Pass-A update:
            mapper only
        """
        # -------------------------------------------------------------
        # Freeze policy for Pass A
        # -------------------------------------------------------------
        self.model.train()

        # Mapper should receive gradients
        set_module_requires_grad(self.model.mapper, True)

        # Reindexer + token encoder are treated as fixed providers for Pass A
        set_module_requires_grad(self.model.reindexer, False)
        set_module_requires_grad(self.model.token_encoder, False)

        zero_optimizer_grads(self.mapper_optimizer, self.reindexer_optimizer)

        # -------------------------------------------------------------
        # Forward
        # -------------------------------------------------------------
        outputs = self.run_pass_a_mapper_forward(native)

        task_losses = compute_task_loss_from_logits(
            S_star=outputs["S_star"],
            R_L=outputs["R_L"],
            R_H=outputs["R_H"],
            A=native["A"],
            m=native["m"],
            Bmat=native["B"],
            c1=native["c1"],
            c2=native["c2"],
            D=native["D"],
            config=self.loss_config,
        )

        loss = task_losses["L_task"]
        _assert_finite_tensor(loss, "Pass-A L_task")

        if self.trainer_config.fail_on_nonfinite_loss and not torch.isfinite(loss):
            raise ValueError("Pass-A task loss is non-finite.")

        # -------------------------------------------------------------
        # Backward
        # -------------------------------------------------------------
        loss.backward()

        if self.trainer_config.fail_on_nonfinite_grad:
            assert_module_gradients_finite(self.model.mapper, "mapper")

        grad_norm = clip_gradients_if_needed(self.model.mapper, self.trainer_config.grad_clip_norm)
        self.mapper_optimizer.step()

        metrics = {
            "pass": "A",
            "L_task": task_losses["L_task"],
            "L_pst_proxy_1q": task_losses["L_pst_proxy_1q"],
            "L_pst_proxy_2q": task_losses["L_pst_proxy_2q"],
            "L_pst_proxy_total": task_losses["L_pst_proxy_total"],
            "L_swap": task_losses["L_swap"],
            "L_depth": task_losses["L_depth"],
        }

        if grad_norm is not None:
            metrics["mapper_grad_norm"] = grad_norm

        return metrics

    def pass_b_reindex_step(self, native: Dict[str, torch.Tensor]) -> Dict:
        """
        One Pass-B update:
            reindexer + token encoder only

        Locked training policy:
        -----------------------
        - mapper parameters frozen
        - mapper set to eval() for deterministic module behavior if relevant
        - mapper still used in a differentiable forward
        """
        # -------------------------------------------------------------
        # Freeze policy for Pass B
        # -------------------------------------------------------------
        self.model.train()

        # Reindexer-side path should receive gradients
        set_module_requires_grad(self.model.reindexer, True)
        set_module_requires_grad(self.model.token_encoder, True)

        # Mapper must be frozen
        set_module_requires_grad(self.model.mapper, False)

        # Optional but recommended by design discussion:
        # keep mapper in eval mode during Pass B.
        self.model.mapper.eval()

        zero_optimizer_grads(self.mapper_optimizer, self.reindexer_optimizer)

        # -------------------------------------------------------------
        # Forward
        # -------------------------------------------------------------
        # IMPORTANT:
        # We do NOT use torch.no_grad() here, because gradients still need to
        # flow through the frozen mapper into reindexer outputs.
        outputs = self.run_pass_b_reindex_forward(native)

        reindex_losses = compute_reindex_objective(
            reindexer=self.model.reindexer,
            base_reindex_outputs=outputs,
            S_star=outputs["S_star"],
            A=native["A"],
            m=native["m"],
            Bmat=native["B"],
            c1=native["c1"],
            c2=native["c2"],
            D=native["D"],
            tau_r=self.trainer_config.tau_r,
            config=self.loss_config,
            generator=None,
        )

        loss = reindex_losses["L_reindex"]
        _assert_finite_tensor(loss, "Pass-B L_reindex")

        if self.trainer_config.fail_on_nonfinite_loss and not torch.isfinite(loss):
            raise ValueError("Pass-B reindex loss is non-finite.")

        # -------------------------------------------------------------
        # Backward
        # -------------------------------------------------------------
        loss.backward()

        if self.trainer_config.fail_on_nonfinite_grad:
            assert_module_gradients_finite(self.model.reindexer, "reindexer")
            assert_module_gradients_finite(self.model.token_encoder, "token_encoder")

        grad_norm_reindex = clip_gradients_if_needed(
            self.model.reindexer, self.trainer_config.grad_clip_norm
        )
        grad_norm_tok = clip_gradients_if_needed(
            self.model.token_encoder, self.trainer_config.grad_clip_norm
        )

        self.reindexer_optimizer.step()

        metrics = {
            "pass": "B",
            "L_task": reindex_losses["L_task"],
            "L_loc": reindex_losses["L_loc"],
            "L_cons": reindex_losses["L_cons"],
            "L_reindex": reindex_losses["L_reindex"],
            "L_pst_proxy_1q": reindex_losses["L_pst_proxy_1q"],
            "L_pst_proxy_2q": reindex_losses["L_pst_proxy_2q"],
            "L_pst_proxy_total": reindex_losses["L_pst_proxy_total"],
            "L_swap": reindex_losses["L_swap"],
            "L_depth": reindex_losses["L_depth"],
        }

        if grad_norm_reindex is not None:
            metrics["reindexer_grad_norm"] = grad_norm_reindex
        if grad_norm_tok is not None:
            metrics["token_encoder_grad_norm"] = grad_norm_tok

        return metrics

    # -------------------------------------------------------------------------
    # 4.4) Public step / epoch methods
    # -------------------------------------------------------------------------

    def train_one_batch(self, batch: Dict) -> Dict:
        """
        Run both Pass A and Pass B for a single batch.

        This is the core unit of training for the current project.
        """
        batch = move_batch_to_device(batch, self.device)
        native = extract_native_batch(batch)

        start_time = time.time()

        pass_a_metrics = self.pass_a_mapper_step(native)
        pass_b_metrics = self.pass_b_reindex_step(native)

        self.global_step += 1
        elapsed = time.time() - start_time

        metrics = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "step_time_sec": elapsed,

            # Pass A
            "passA_L_task": pass_a_metrics["L_task"],
            "passA_L_pst_proxy_1q": pass_a_metrics["L_pst_proxy_1q"],
            "passA_L_pst_proxy_2q": pass_a_metrics["L_pst_proxy_2q"],
            "passA_L_pst_proxy_total": pass_a_metrics["L_pst_proxy_total"],
            "passA_L_swap": pass_a_metrics["L_swap"],
            "passA_L_depth": pass_a_metrics["L_depth"],

            # Pass B
            "passB_L_task": pass_b_metrics["L_task"],
            "passB_L_loc": pass_b_metrics["L_loc"],
            "passB_L_cons": pass_b_metrics["L_cons"],
            "passB_L_reindex": pass_b_metrics["L_reindex"],
            "passB_L_pst_proxy_1q": pass_b_metrics["L_pst_proxy_1q"],
            "passB_L_pst_proxy_2q": pass_b_metrics["L_pst_proxy_2q"],
            "passB_L_pst_proxy_total": pass_b_metrics["L_pst_proxy_total"],
            "passB_L_swap": pass_b_metrics["L_swap"],
            "passB_L_depth": pass_b_metrics["L_depth"],
        }

        if "mapper_grad_norm" in pass_a_metrics:
            metrics["mapper_grad_norm"] = pass_a_metrics["mapper_grad_norm"]
        if "reindexer_grad_norm" in pass_b_metrics:
            metrics["reindexer_grad_norm"] = pass_b_metrics["reindexer_grad_norm"]
        if "token_encoder_grad_norm" in pass_b_metrics:
            metrics["token_encoder_grad_norm"] = pass_b_metrics["token_encoder_grad_norm"]

        self.log_metrics(metrics)
        return metrics

    def train_one_epoch(self, loader: Iterable[Dict]) -> Dict:
        """
        Train for one epoch over the loader.

        Returns mean metrics across the epoch.
        """
        self.current_epoch += 1
        self.model.train()

        running: List[Dict] = []

        for step_idx, batch in enumerate(loader, start=1):
            metrics = self.train_one_batch(batch)
            running.append(detach_scalar_dict(metrics))

            if step_idx % self.trainer_config.log_every_steps == 0:
                print(
                    f"[Epoch {self.current_epoch:03d} | Step {step_idx:04d}] "
                    f"PassA L_task={metrics['passA_L_task']:.6f} | "
                    f"PassB L_reindex={metrics['passB_L_reindex']:.6f}"
                )

        epoch_summary = average_metric_dicts(running)
        epoch_summary["epoch"] = self.current_epoch

        if self.current_epoch % self.trainer_config.save_every_epochs == 0:
            saved = self.save_epoch_checkpoint(self.current_epoch)
            epoch_summary["checkpoint_epoch"] = saved.get("epoch")
            if "latest" in saved:
                epoch_summary["checkpoint_latest"] = saved["latest"]

        return epoch_summary

    def fit(
        self,
        train_loader: Iterable[Dict],
        num_epochs: Optional[int] = None,
    ) -> List[Dict]:
        """
        Full training loop.

        Returns:
            list of epoch summaries
        """
        if num_epochs is None:
            num_epochs = self.trainer_config.num_epochs

        history = []

        print("Starting KMW training")
        print(f"Device: {self.device}")
        print(f"Mapper trainable params: {count_parameters(self.model.mapper):,}")
        print(
            "Reindexer+token_encoder trainable params: "
            f"{count_parameters(self.model.reindexer) + count_parameters(self.model.token_encoder):,}"
        )

        for _ in range(num_epochs):
            summary = self.train_one_epoch(train_loader)
            history.append(summary)

            print(
                f"[Epoch {summary['epoch']:03d} done] "
                f"PassA L_task={summary.get('passA_L_task', float('nan')):.6f} | "
                f"PassB L_reindex={summary.get('passB_L_reindex', float('nan')):.6f}"
            )

        return history

    # -------------------------------------------------------------------------
    # 4.5) Lightweight validation helper
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_mapper_task_only(self, loader: Iterable[Dict]) -> Dict:
        """
        Lightweight evaluation helper.

        Purpose
        -------
        This is NOT the final full evaluation pipeline.
        It is only a quick validation helper for training-time monitoring.

        It runs the model forward and computes the mapper task loss path.
        No hard Hungarian inference is used here; that belongs later in evaluation/.
        """
        self.model.eval()

        rows = []

        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            native = extract_native_batch(batch)

            outputs = self.model(
                A=native["A"],
                m=native["m"],
                Bmat=native["B"],
                c1=native["c1"],
                c2=native["c2"],
                D=native["D"],
                tau_r=self.trainer_config.tau_r,
            )

            task_losses = compute_task_loss_from_logits(
                S_star=outputs["S_star"],
                R_L=outputs["R_L"],
                R_H=outputs["R_H"],
                A=native["A"],
                m=native["m"],
                Bmat=native["B"],
                c1=native["c1"],
                c2=native["c2"],
                D=native["D"],
                config=self.loss_config,
            )

            rows.append(
                detach_scalar_dict(
                    {
                        "L_task": task_losses["L_task"],
                        "L_pst_proxy_1q": task_losses["L_pst_proxy_1q"],
                        "L_pst_proxy_2q": task_losses["L_pst_proxy_2q"],
                        "L_pst_proxy_total": task_losses["L_pst_proxy_total"],
                        "L_swap": task_losses["L_swap"],
                        "L_depth": task_losses["L_depth"],
                    }
                )
            )

        return average_metric_dicts(rows)


# =============================================================================
# 5) Metric aggregation helper
# =============================================================================

def average_metric_dicts(rows: List[Dict]) -> Dict:
    """
    Average numeric fields across a list of dicts.

    Non-numeric keys are ignored.
    """
    if not rows:
        return {}

    sums = {}
    counts = {}

    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1

    return {key: sums[key] / counts[key] for key in sums}


# =============================================================================
# 6) Convenience factory
# =============================================================================

def build_trainer(
    model: KMWModel,
    loss_config: Optional[LossConfig] = None,
    trainer_config: Optional[TrainerConfig] = None,
    device: Optional[torch.device] = None,
) -> KMWTrainer:
    """
    Small convenience constructor.
    """
    return KMWTrainer(
        model=model,
        loss_config=loss_config,
        trainer_config=trainer_config,
        device=device,
    )


# =============================================================================
# 7) Public exports
# =============================================================================

__all__ = [
    "TrainerConfig",
    "KMWTrainer",
    "move_batch_to_device",
    "extract_native_batch",
    "set_module_requires_grad",
    "average_metric_dicts",
    "build_trainer",
]


# =============================================================================
# notes on implementation:
# =============================================================================

# I grouped token_encoder with the reindexer-side optimizer, 
# because in this architecture it is downstream of reordered hardware tensors and part of the reindex-conditioned path.
#  If you want, I can keep it this way unless we later see a training-stability reason to split it.

# evaluate_mapper_task_only(...) is just a lightweight validation helper.
#  The real hard-assignment evaluation still belongs in src/kmw/evaluation/evaluate.py, as you already decided.

# I did not use torch.no_grad() in Pass B around the mapper forward,
#  because that would break gradient flow into the reindexer path.


