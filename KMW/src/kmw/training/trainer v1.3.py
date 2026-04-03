# =============================================================================
# UPDATE LOG (2026-03-25)
# - Added staged training curriculum support with three phases:
#     1) warmup      : identity reindex, Pass A only
#     2) stage1      : learned reindex enabled, soft tau_r anneal
#     3) stage2      : learned reindex enabled, sharper tau_r anneal
# - Moved token_encoder to the mapper-side optimizer.
# - Reindexer optimizer now updates reindexer only.
# - Switched optimizers from Adam to AdamW.
# - Added dynamic tau_r scheduling and stage-specific alpha_loc / beta_cons.
# - Added detailed per-step logging for stage, tau_r_active, and active aux weights.
# - Preserved the old two-pass behavior when use_staged_curriculum=False.
# =============================================================================

# src/kmw/training/trainer.py

"""
Trainer for the KMW project.

What this file does
-------------------
This file implements two training modes:

1) Legacy two-pass mode
   - enabled when TrainerConfig.use_staged_curriculum = False

2) Staged curriculum mode
   - enabled when TrainerConfig.use_staged_curriculum = True
   - phases:
       warmup: identity reindex, mapper/token-encoder only
       stage1: joint two-pass training with soft tau_r anneal
       stage2: joint two-pass training with sharper tau_r anneal

Pass A: mapper update
---------------------
Goal:
    update mapper + token_encoder parameters

Rule:
    - warmup:
        use identity reindex outputs
    - non-warmup:
        run reindexer
        DETACH reindex outputs before feeding mapper
    - compute mapper task loss
    - backprop only through mapper + token_encoder

Pass B: reindexer update
------------------------
Goal:
    update reindexer parameters only

Rule:
    - run reindexer again without detach
    - freeze mapper + token_encoder parameters
    - keep mapper/token_encoder forward differentiable (NO torch.no_grad())
    - compute reindex objective:
          L_reindex = L_task + alpha_loc * L_loc + beta_cons * L_cons
    - backprop only through reindexer

Important architectural note
----------------------------
The mapper freeze policy belongs here, not in model.py.
That was a deliberate implementation decision:
- architecture stays in model.py
- training policy stays in trainer.py
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
    reindexer_lr: float = 5e-5
    weight_decay: float = 1e-4

    # Epoch / checkpoint behavior
    num_epochs: int = 60
    grad_clip_norm: Optional[float] = 1.0
    log_every_steps: int = 10
    save_every_epochs: int = 1

    # Legacy fixed tau_r (used only when staged curriculum is disabled)
    tau_r: float = 1.0

    # Staged curriculum
    use_staged_curriculum: bool = False
    warmup_epochs: int = 10
    stage1_epochs: int = 15
    stage2_epochs: int = 35
    tau_r_start: float = 1.0
    tau_r_mid: float = 0.60
    tau_r_end: float = 0.15
    tau_r_schedule: str = "cosine"  # {"cosine", "linear"}
    stage1_alpha_loc: float = 0.02
    stage1_beta_cons: float = 0.0
    stage2_alpha_loc: float = 0.05
    stage2_beta_cons: float = 0.0
    warmup_reindex_mode: str = "identity"  # currently only "identity" supported

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
    """Fail loudly if a tensor contains NaN or Inf."""
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def _assert_required_batch_keys(batch: Dict, required_keys: Iterable[str]) -> None:
    """Check that the batch dictionary contains the required keys."""
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
    """Zero gradients for one or more optimizers."""
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def clip_gradients_if_needed_for_modules(
    modules: Iterable[nn.Module],
    max_norm: Optional[float],
) -> Optional[float]:
    """
    Clip gradients over a collection of modules as one parameter set.

    Returns:
        total_norm reported by torch, or None if clipping is disabled.
    """
    if max_norm is None:
        return None

    params = []
    seen = set()

    for module in modules:
        for param in module.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            params.append(param)

    if not params:
        return None

    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    return float(total_norm)


def assert_module_gradients_finite(module: nn.Module, module_name: str) -> None:
    """Fail loudly if any gradient inside a module is NaN/Inf."""
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise ValueError(f"Non-finite gradient detected in {module_name}.{name}")


def assert_modules_gradients_finite(modules: Dict[str, nn.Module]) -> None:
    """Fail loudly if any gradient inside any listed module is NaN/Inf."""
    for module_name, module in modules.items():
        assert_module_gradients_finite(module, module_name)


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
    """Count trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_identity_reindex_outputs(native: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Build identity reindex outputs with the same structure used by the learned reindexer.

    This keeps the downstream mapper/task-loss path unchanged during warmup.
    """
    A = native["A"]
    batch_size, n, _ = A.shape

    eye = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()

    return {
        "R_L": eye.clone(),
        "R_H": eye.clone(),
        "A_star": native["A"].clone(),
        "m_star": native["m"].clone(),
        "B_star": native["B"].clone(),
        "c1_star": native["c1"].clone(),
        "c2_star": native["c2"].clone(),
        "D_star": native["D"].clone(),
    }


# =============================================================================
# 3) Optimizer builders
# =============================================================================

def build_mapper_optimizer(model: KMWModel, config: TrainerConfig) -> torch.optim.Optimizer:
    """
    Build optimizer for mapper + token_encoder parameters.
    """
    params = list(model.mapper.parameters()) + list(model.token_encoder.parameters())

    return torch.optim.AdamW(
        params,
        lr=config.mapper_lr,
        weight_decay=config.weight_decay,
    )


def build_reindexer_optimizer(model: KMWModel, config: TrainerConfig) -> torch.optim.Optimizer:
    """
    Build optimizer for reindexer parameters only.
    """
    return torch.optim.AdamW(
        model.reindexer.parameters(),
        lr=config.reindexer_lr,
        weight_decay=config.weight_decay,
    )


# =============================================================================
# 4) Main trainer class
# =============================================================================

class KMWTrainer:
    """
    Main trainer implementing:
    - legacy two-pass training, or
    - staged curriculum training
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
        """Append one JSON object to the metrics JSONL log."""
        record = detach_scalar_dict(metrics)
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def state_dict(self) -> Dict:
        """Trainer state for checkpoints."""
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
        """Save a checkpoint file and return the path as string."""
        path = self.checkpoint_dir / filename
        torch.save(self.state_dict(), path)
        return str(path)

    def save_epoch_checkpoint(self, epoch: int) -> Dict[str, str]:
        """Save standard epoch checkpoints."""
        saved = {}
        name = f"{self.trainer_config.checkpoint_prefix}_epoch_{epoch:04d}.pt"
        saved["epoch"] = self.save_checkpoint(name)

        if self.trainer_config.save_latest:
            saved["latest"] = self.save_checkpoint("latest.pt")

        return saved

    # -------------------------------------------------------------------------
    # 4.2) Curriculum / scheduling helpers
    # -------------------------------------------------------------------------

    def validate_curriculum_config(self, num_epochs: int) -> None:
        """Sanity-check curriculum settings before training starts."""
        cfg = self.trainer_config

        if not cfg.use_staged_curriculum:
            return

        if cfg.warmup_reindex_mode != "identity":
            raise ValueError(
                f"Unsupported warmup_reindex_mode={cfg.warmup_reindex_mode!r}. "
                "Current implementation supports only 'identity'."
            )

        for name in ("warmup_epochs", "stage1_epochs", "stage2_epochs"):
            value = getattr(cfg, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if cfg.tau_r_schedule not in {"cosine", "linear"}:
            raise ValueError(
                f"Unsupported tau_r_schedule={cfg.tau_r_schedule!r}. "
                "Expected 'cosine' or 'linear'."
            )

        total_planned = cfg.warmup_epochs + cfg.stage1_epochs + cfg.stage2_epochs
        if total_planned > num_epochs:
            raise ValueError(
                "Curriculum phase lengths exceed total epochs: "
                f"warmup({cfg.warmup_epochs}) + stage1({cfg.stage1_epochs}) + "
                f"stage2({cfg.stage2_epochs}) = {total_planned} > num_epochs({num_epochs})."
            )

    def get_active_stage(self, epoch: Optional[int] = None) -> str:
        """
        Resolve the current training stage.

        Returns:
            "joint"   : legacy two-pass mode without staged curriculum
            "warmup"  : identity reindex, Pass A only
            "stage1"  : learned reindex, softer tau_r
            "stage2"  : learned reindex, sharper tau_r
        """
        cfg = self.trainer_config

        if epoch is None:
            epoch = self.current_epoch

        if not cfg.use_staged_curriculum:
            return "joint"

        if epoch <= cfg.warmup_epochs:
            return "warmup"

        stage1_end = cfg.warmup_epochs + cfg.stage1_epochs
        if epoch <= stage1_end:
            return "stage1"

        return "stage2"

    def _interpolate(self, start: float, end: float, progress: float) -> float:
        """
        Interpolate between two values using the configured schedule.
        """
        progress = max(0.0, min(1.0, float(progress)))

        if self.trainer_config.tau_r_schedule == "linear":
            return start + (end - start) * progress

        # cosine by default
        cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        return start + (end - start) * cosine_factor

    def _stage_progress(
        self,
        stage: str,
        step_idx: int,
        num_steps_in_epoch: Optional[int],
    ) -> float:
        """
        Compute normalized progress in the active stage.

        The schedule is step-based when num_steps_in_epoch is available.
        """
        cfg = self.trainer_config

        if num_steps_in_epoch is None or num_steps_in_epoch <= 0:
            return 0.0

        if stage == "stage1":
            stage_start_epoch = cfg.warmup_epochs + 1
            stage_epochs = max(cfg.stage1_epochs, 1)
        elif stage == "stage2":
            stage_start_epoch = cfg.warmup_epochs + cfg.stage1_epochs + 1
            stage_epochs = max(cfg.stage2_epochs, 1)
        else:
            return 0.0

        completed_before_current = (
            max(self.current_epoch - stage_start_epoch, 0) * num_steps_in_epoch
            + max(step_idx - 1, 0)
        )
        total_steps = max(stage_epochs * num_steps_in_epoch, 1)

        if total_steps == 1:
            return 1.0

        return min(max(completed_before_current / float(total_steps - 1), 0.0), 1.0)

    def get_active_tau_r(
        self,
        stage: str,
        step_idx: int,
        num_steps_in_epoch: Optional[int],
    ) -> float:
        """
        Resolve the active tau_r for the current batch.
        """
        cfg = self.trainer_config

        if not cfg.use_staged_curriculum:
            return float(cfg.tau_r)

        if stage == "warmup":
            return float(cfg.tau_r_start)

        if stage == "stage1":
            progress = self._stage_progress(stage, step_idx, num_steps_in_epoch)
            return float(self._interpolate(cfg.tau_r_start, cfg.tau_r_mid, progress))

        if stage == "stage2":
            progress = self._stage_progress(stage, step_idx, num_steps_in_epoch)
            return float(self._interpolate(cfg.tau_r_mid, cfg.tau_r_end, progress))

        return float(cfg.tau_r)

    def build_active_loss_config(self, stage: str) -> LossConfig:
        """
        Build the loss config active for the current stage.

        We keep the base LossConfig intact and only override the reindex auxiliaries
        when staged curriculum is enabled.
        """
        cfg = self.trainer_config

        if not cfg.use_staged_curriculum:
            return self.loss_config

        if stage == "warmup":
            return replace(self.loss_config, alpha_loc=0.0, beta_cons=0.0)

        if stage == "stage1":
            return replace(
                self.loss_config,
                alpha_loc=cfg.stage1_alpha_loc,
                beta_cons=cfg.stage1_beta_cons,
            )

        if stage == "stage2":
            return replace(
                self.loss_config,
                alpha_loc=cfg.stage2_alpha_loc,
                beta_cons=cfg.stage2_beta_cons,
            )

        return self.loss_config

    # -------------------------------------------------------------------------
    # 4.3) Core forward helpers
    # -------------------------------------------------------------------------

    def run_pass_a_mapper_forward(
        self,
        native: Dict[str, torch.Tensor],
        tau_r: float,
        use_identity_reindex: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Pass A forward path.

        - warmup uses identity reindex outputs
        - other stages use detached learned reindex outputs
        """
        if use_identity_reindex:
            reidx_for_mapper = build_identity_reindex_outputs(native)
        else:
            reidx = self.model.reindexer(
                A=native["A"],
                m=native["m"],
                Bmat=native["B"],
                c1=native["c1"],
                c2=native["c2"],
                D=native["D"],
                tau_r=tau_r,
            )
            reidx_for_mapper = maybe_detach_reindex_outputs(reidx, detach=True)

        T_hw_star = self.model.token_encoder(
            B_star=reidx_for_mapper["B_star"],
            c2_star=reidx_for_mapper["c2_star"],
            c1_star=reidx_for_mapper["c1_star"],
        )

        A_star_spatial = reidx_for_mapper["A_star"].unsqueeze(1)
        S_star = self.model.mapper(A_star_spatial, T_hw_star)

        return {
            **reidx_for_mapper,
            "T_hw_star": T_hw_star,
            "S_star": S_star,
        }

    def run_pass_b_reindex_forward(
        self,
        native: Dict[str, torch.Tensor],
        tau_r: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Pass B forward path.

        - reindex outputs are NOT detached
        - mapper/token_encoder parameters are frozen
        - forward remains differentiable
        """
        reidx = self.model.reindexer(
            A=native["A"],
            m=native["m"],
            Bmat=native["B"],
            c1=native["c1"],
            c2=native["c2"],
            D=native["D"],
            tau_r=tau_r,
        )

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
    # 4.4) Pass A / Pass B step functions
    # -------------------------------------------------------------------------

    def pass_a_mapper_step(
        self,
        native: Dict[str, torch.Tensor],
        stage: str,
        tau_r_active: float,
        active_loss_config: LossConfig,
    ) -> Dict:
        """
        One Pass-A update:
            mapper + token_encoder only
        """
        self.model.train()

        # Mapper-side path should receive gradients
        set_module_requires_grad(self.model.mapper, True)
        set_module_requires_grad(self.model.token_encoder, True)

        # Reindexer is frozen in Pass A
        set_module_requires_grad(self.model.reindexer, False)

        # Keep the frozen reindexer deterministic when used in Pass A
        self.model.reindexer.eval()
        self.model.mapper.train()
        self.model.token_encoder.train()

        zero_optimizer_grads(self.mapper_optimizer, self.reindexer_optimizer)

        use_identity_reindex = self.trainer_config.use_staged_curriculum and stage == "warmup"

        outputs = self.run_pass_a_mapper_forward(
            native=native,
            tau_r=tau_r_active,
            use_identity_reindex=use_identity_reindex,
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
            config=active_loss_config,
        )

        loss = task_losses["L_task"]
        _assert_finite_tensor(loss, "Pass-A L_task")

        if self.trainer_config.fail_on_nonfinite_loss and not torch.isfinite(loss):
            raise ValueError("Pass-A task loss is non-finite.")

        loss.backward()

        if self.trainer_config.fail_on_nonfinite_grad:
            assert_modules_gradients_finite(
                {
                    "mapper": self.model.mapper,
                    "token_encoder": self.model.token_encoder,
                }
            )

        grad_norm = clip_gradients_if_needed_for_modules(
            [self.model.mapper, self.model.token_encoder],
            self.trainer_config.grad_clip_norm,
        )

        self.mapper_optimizer.step()

        metrics = {
            "pass": "A",
            "stage": stage,
            "tau_r_active": tau_r_active,
            "alpha_loc_active": active_loss_config.alpha_loc,
            "beta_cons_active": active_loss_config.beta_cons,
            "L_task": task_losses["L_task"],
            "L_pst_proxy_1q": task_losses["L_pst_proxy_1q"],
            "L_pst_proxy_2q": task_losses["L_pst_proxy_2q"],
            "L_pst_proxy_total": task_losses["L_pst_proxy_total"],
            "L_swap": task_losses["L_swap"],
            "L_depth": task_losses["L_depth"],
        }

        if grad_norm is not None:
            metrics["mapper_side_grad_norm"] = grad_norm

        return metrics

    def pass_b_reindex_step(
        self,
        native: Dict[str, torch.Tensor],
        stage: str,
        tau_r_active: float,
        active_loss_config: LossConfig,
    ) -> Dict:
        """
        One Pass-B update:
            reindexer only

        In warmup, Pass B is skipped entirely.
        """
        if self.trainer_config.use_staged_curriculum and stage == "warmup":
            return {
                "pass": "B",
                "stage": stage,
                "tau_r_active": tau_r_active,
                "alpha_loc_active": active_loss_config.alpha_loc,
                "beta_cons_active": active_loss_config.beta_cons,
                "skipped": True,
                "L_task": None,
                "L_loc": None,
                "L_cons": None,
                "L_reindex": None,
                "L_pst_proxy_1q": None,
                "L_pst_proxy_2q": None,
                "L_pst_proxy_total": None,
                "L_swap": None,
                "L_depth": None,
            }

        self.model.train()

        # Reindexer receives gradients
        set_module_requires_grad(self.model.reindexer, True)

        # Mapper and token_encoder are frozen operators in Pass B
        set_module_requires_grad(self.model.mapper, False)
        set_module_requires_grad(self.model.token_encoder, False)

        self.model.reindexer.train()
        self.model.mapper.eval()
        self.model.token_encoder.eval()

        zero_optimizer_grads(self.mapper_optimizer, self.reindexer_optimizer)

        outputs = self.run_pass_b_reindex_forward(native=native, tau_r=tau_r_active)

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
            tau_r=tau_r_active,
            config=active_loss_config,
            generator=None,
        )

        loss = reindex_losses["L_reindex"]
        _assert_finite_tensor(loss, "Pass-B L_reindex")

        if self.trainer_config.fail_on_nonfinite_loss and not torch.isfinite(loss):
            raise ValueError("Pass-B reindex loss is non-finite.")

        loss.backward()

        if self.trainer_config.fail_on_nonfinite_grad:
            assert_module_gradients_finite(self.model.reindexer, "reindexer")

        grad_norm = clip_gradients_if_needed_for_modules(
            [self.model.reindexer],
            self.trainer_config.grad_clip_norm,
        )

        self.reindexer_optimizer.step()

        metrics = {
            "pass": "B",
            "stage": stage,
            "tau_r_active": tau_r_active,
            "alpha_loc_active": active_loss_config.alpha_loc,
            "beta_cons_active": active_loss_config.beta_cons,
            "skipped": False,
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

        if grad_norm is not None:
            metrics["reindexer_grad_norm"] = grad_norm

        return metrics

    # -------------------------------------------------------------------------
    # 4.5) Public step / epoch methods
    # -------------------------------------------------------------------------

    def train_one_batch(
        self,
        batch: Dict,
        step_idx: int,
        num_steps_in_epoch: Optional[int],
    ) -> Dict:
        """
        Run the active training logic for a single batch.
        """
        batch = move_batch_to_device(batch, self.device)
        native = extract_native_batch(batch)

        stage = self.get_active_stage()
        tau_r_active = self.get_active_tau_r(
            stage=stage,
            step_idx=step_idx,
            num_steps_in_epoch=num_steps_in_epoch,
        )
        active_loss_config = self.build_active_loss_config(stage)

        start_time = time.time()

        pass_a_metrics = self.pass_a_mapper_step(
            native=native,
            stage=stage,
            tau_r_active=tau_r_active,
            active_loss_config=active_loss_config,
        )

        pass_b_metrics = self.pass_b_reindex_step(
            native=native,
            stage=stage,
            tau_r_active=tau_r_active,
            active_loss_config=active_loss_config,
        )

        self.global_step += 1
        elapsed = time.time() - start_time

        metrics = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "stage": stage,
            "step_time_sec": elapsed,
            "tau_r_active": tau_r_active,
            "alpha_loc_active": active_loss_config.alpha_loc,
            "beta_cons_active": active_loss_config.beta_cons,

            # Pass A
            "passA_L_task": pass_a_metrics["L_task"],
            "passA_L_pst_proxy_1q": pass_a_metrics["L_pst_proxy_1q"],
            "passA_L_pst_proxy_2q": pass_a_metrics["L_pst_proxy_2q"],
            "passA_L_pst_proxy_total": pass_a_metrics["L_pst_proxy_total"],
            "passA_L_swap": pass_a_metrics["L_swap"],
            "passA_L_depth": pass_a_metrics["L_depth"],

            # Pass B
            "passB_skipped": pass_b_metrics["skipped"],
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

        if "mapper_side_grad_norm" in pass_a_metrics:
            metrics["mapper_side_grad_norm"] = pass_a_metrics["mapper_side_grad_norm"]
        if "reindexer_grad_norm" in pass_b_metrics:
            metrics["reindexer_grad_norm"] = pass_b_metrics["reindexer_grad_norm"]

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
        num_steps_in_epoch = len(loader) if hasattr(loader, "__len__") else None
        stage_name = self.get_active_stage()

        for step_idx, batch in enumerate(loader, start=1):
            metrics = self.train_one_batch(
                batch=batch,
                step_idx=step_idx,
                num_steps_in_epoch=num_steps_in_epoch,
            )
            running.append(detach_scalar_dict(metrics))

            if step_idx % self.trainer_config.log_every_steps == 0:
                if metrics["passB_skipped"]:
                    print(
                        f"[Epoch {self.current_epoch:03d} | Step {step_idx:04d} | {stage_name}] "
                        f"tau_r={metrics['tau_r_active']:.4f} | "
                        f"PassA L_task={metrics['passA_L_task']:.6f}"
                    )
                else:
                    print(
                        f"[Epoch {self.current_epoch:03d} | Step {step_idx:04d} | {stage_name}] "
                        f"tau_r={metrics['tau_r_active']:.4f} | "
                        f"PassA L_task={metrics['passA_L_task']:.6f} | "
                        f"PassB L_reindex={metrics['passB_L_reindex']:.6f}"
                    )

        epoch_summary = average_metric_dicts(running)
        epoch_summary["epoch"] = self.current_epoch
        epoch_summary["stage_name"] = stage_name

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

        self.validate_curriculum_config(num_epochs)

        history = []

        print("Starting KMW training")
        print(f"Device: {self.device}")
        print(f"Staged curriculum enabled: {self.trainer_config.use_staged_curriculum}")
        print(f"Mapper+token_encoder trainable params: {count_parameters(self.model.mapper) + count_parameters(self.model.token_encoder):,}")
        print(f"Reindexer trainable params: {count_parameters(self.model.reindexer):,}")

        if self.trainer_config.use_staged_curriculum:
            total_planned = (
                self.trainer_config.warmup_epochs
                + self.trainer_config.stage1_epochs
                + self.trainer_config.stage2_epochs
            )
            print(
                "Curriculum phases: "
                f"warmup={self.trainer_config.warmup_epochs}, "
                f"stage1={self.trainer_config.stage1_epochs}, "
                f"stage2={self.trainer_config.stage2_epochs}, "
                f"planned_total={total_planned}, "
                f"requested_total={num_epochs}"
            )
            if total_planned < num_epochs:
                print(
                    "Note: requested_total exceeds planned curriculum length. "
                    "Extra epochs will continue in stage2."
                )

        for _ in range(num_epochs):
            summary = self.train_one_epoch(train_loader)
            history.append(summary)

            stage_name = summary.get("stage_name", "unknown")
            tau_r_mean = summary.get("tau_r_active", float("nan"))

            if summary.get("passB_skipped", 0.0) >= 0.5:
                print(
                    f"[Epoch {summary['epoch']:03d} done | {stage_name}] "
                    f"tau_r≈{tau_r_mean:.4f} | "
                    f"PassA L_task={summary.get('passA_L_task', float('nan')):.6f}"
                )
            else:
                print(
                    f"[Epoch {summary['epoch']:03d} done | {stage_name}] "
                    f"tau_r≈{tau_r_mean:.4f} | "
                    f"PassA L_task={summary.get('passA_L_task', float('nan')):.6f} | "
                    f"PassB L_reindex={summary.get('passB_L_reindex', float('nan')):.6f}"
                )

        return history

    # -------------------------------------------------------------------------
    # 4.6) Lightweight validation helper
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_mapper_task_only(self, loader: Iterable[Dict]) -> Dict:
        """
        Lightweight evaluation helper.

        Purpose
        -------
        This is NOT the final full evaluation pipeline.
        It is only a quick validation helper for training-time monitoring.
        """
        self.model.eval()

        rows = []

        stage = self.get_active_stage(epoch=max(self.current_epoch, 1))
        active_loss_config = self.build_active_loss_config(stage)
        tau_r_active = (
            self.trainer_config.tau_r_start
            if self.trainer_config.use_staged_curriculum and stage == "warmup"
            else (
                self.trainer_config.tau_r_end
                if self.trainer_config.use_staged_curriculum
                else self.trainer_config.tau_r
            )
        )

        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            native = extract_native_batch(batch)

            if self.trainer_config.use_staged_curriculum and stage == "warmup":
                outputs = self.run_pass_a_mapper_forward(
                    native=native,
                    tau_r=tau_r_active,
                    use_identity_reindex=True,
                )
            else:
                outputs = self.model(
                    A=native["A"],
                    m=native["m"],
                    Bmat=native["B"],
                    c1=native["c1"],
                    c2=native["c2"],
                    D=native["D"],
                    tau_r=tau_r_active,
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
                config=active_loss_config,
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
    """Small convenience constructor."""
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

# token_encoder is intentionally trained with the mapper-side optimizer.
# During warmup, this lets the hardware-conditioning frontend stabilize together
# with the mapper under a fixed identity reindex frame.

# Pass B freezes mapper + token_encoder as differentiable operators.
# We do NOT wrap their forward pass in torch.no_grad(), because gradients still
# need to flow into the reindexer outputs.

# When use_staged_curriculum=False, this file preserves the old high-level
# behavior: two-pass training every batch with fixed tau_r.