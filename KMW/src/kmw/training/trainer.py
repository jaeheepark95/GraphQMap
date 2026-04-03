# =============================================================================
# UPDATE LOG (2026-03-30, v1.4.1)
# - Updated trainer-side task-loss calls and logging to use the v1.4.1
#   execution-surrogate metrics:
#     * L_1q, L_ro, L_2q, L_native, L_route, L_task
#     * S_proxy_exec
# - Native batches now carry raw backend tensors and logical count tensors for
#   the loss path: D_raw, e1q, ero, e2q, n1q, nmeas.
# - The mapper/reindexer forward path still consumes only A, m, B, c1, c2, D.
# =============================================================================
# =============================================================================
# UPDATE LOG (2026-03-30)
# - Added C: reindexer diagnostics logging to train_metrics.jsonl and epoch means:
#     RL_entropy_mean, RH_entropy_mean,
#     RL/RH row-max sharpness fractions at thresholds 0.5 / 0.7 / 0.9,
#     plus persistent logging of passB_L_loc and passB_L_cons.
# - Added D: freeze_hardware_reindex ablation mode.
#     In this mode R_H = I for the full run, only logical reindexing is learnable,
#     and evaluation can use the same identity-R_H behavior.
# - Added E: optional deterministic logical canonical-teacher pretraining.
#     This pretrains R_L only, keeps R_H = I, and then hands off to the staged
#     curriculum without restoring the old canonical pipeline as the main path.
# =============================================================================
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

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kmw.losses.loss import (
    LossConfig,
    compute_reindex_objective,
    compute_task_loss_from_logits,
)
from kmw.models.model import (
    KMWModel,
    build_logical_canonical_teacher_permutation,
    maybe_detach_reindex_outputs,
)


@dataclass
class TrainerConfig:
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
    tau_r_schedule: str = "cosine"
    stage1_alpha_loc: float = 0.02
    stage1_beta_cons: float = 0.0
    stage2_alpha_loc: float = 0.05
    stage2_beta_cons: float = 0.0
    warmup_reindex_mode: str = "identity"

    # C/D/E extensions
    freeze_hardware_reindex: bool = False
    enable_canonical_pretrain: bool = False
    pretrain_epochs: int = 0
    pretrain_tau_r: float = 1.0
    pretrain_teacher_loss_weight: float = 1.0

    # Fail-fast behavior
    fail_on_nonfinite_loss: bool = True
    fail_on_nonfinite_grad: bool = True

    # Output
    checkpoint_dir: str = "checkpoints"
    checkpoint_prefix: str = "kmw"
    save_latest: bool = True
    metrics_jsonl_name: str = "train_metrics.jsonl"


def _assert_finite_tensor(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def _assert_required_batch_keys(batch: Dict, required_keys: Iterable[str]) -> None:
    missing = [k for k in required_keys if k not in batch]
    if missing:
        raise KeyError(f"Batch is missing required keys: {missing}")


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def extract_native_batch(batch: Dict) -> Dict[str, torch.Tensor]:
    required = ("A", "m", "B", "c1", "c2", "D", "D_raw", "e1q", "ero", "e2q", "n1q", "nmeas")
    _assert_required_batch_keys(batch, required)
    native = {k: batch[k] for k in required}
    for key, value in native.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Batch key '{key}' must contain a tensor, got {type(value)}")
        _assert_finite_tensor(value, key)
    return native


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def zero_optimizer_grads(*optimizers: torch.optim.Optimizer) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float((g * g).sum().item())
    return float(total ** 0.5)


def clip_gradients_if_needed_for_modules(
    modules: Iterable[nn.Module],
    max_norm: Optional[float],
) -> Optional[float]:
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
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise ValueError(f"Non-finite gradient detected in {module_name}.{name}")


def assert_modules_gradients_finite(modules: Dict[str, nn.Module]) -> None:
    for module_name, module in modules.items():
        assert_module_gradients_finite(module, module_name)


def detach_scalar_dict(metrics: Dict) -> Dict:
    out = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_identity_reindex_outputs(native: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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


def _mean_row_entropy(P: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = P.clamp(min=eps)
    H = -(probs * probs.log()).sum(dim=-1)
    denom = math.log(max(P.shape[-1], 2))
    if denom > 0:
        H = H / denom
    return H.mean()


def _rowmax_fraction_ge(P: torch.Tensor, threshold: float) -> torch.Tensor:
    row_max = P.max(dim=-1).values
    return (row_max >= threshold).to(P.dtype).mean()


def build_reindex_diagnostics(R_L: torch.Tensor, R_H: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "RL_entropy_mean": _mean_row_entropy(R_L),
        "RH_entropy_mean": _mean_row_entropy(R_H),
        "RL_rowmax_frac_ge_0_5": _rowmax_fraction_ge(R_L, 0.5),
        "RL_rowmax_frac_ge_0_7": _rowmax_fraction_ge(R_L, 0.7),
        "RL_rowmax_frac_ge_0_9": _rowmax_fraction_ge(R_L, 0.9),
        "RH_rowmax_frac_ge_0_5": _rowmax_fraction_ge(R_H, 0.5),
        "RH_rowmax_frac_ge_0_7": _rowmax_fraction_ge(R_H, 0.7),
        "RH_rowmax_frac_ge_0_9": _rowmax_fraction_ge(R_H, 0.9),
    }


def build_mapper_optimizer(model: KMWModel, config: TrainerConfig) -> torch.optim.Optimizer:
    params = list(model.mapper.parameters()) + list(model.token_encoder.parameters())
    return torch.optim.AdamW(params, lr=config.mapper_lr, weight_decay=config.weight_decay)


def build_reindexer_optimizer(model: KMWModel, config: TrainerConfig) -> torch.optim.Optimizer:
    if config.freeze_hardware_reindex:
        params = list(model.reindexer.logical_branch.parameters())
    else:
        params = list(model.reindexer.parameters())
    return torch.optim.AdamW(params, lr=config.reindexer_lr, weight_decay=config.weight_decay)


class KMWTrainer:
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
        self.model.reindexer.force_identity_hardware = self.trainer_config.freeze_hardware_reindex

        self.mapper_optimizer = build_mapper_optimizer(self.model, self.trainer_config)
        self.reindexer_optimizer = build_reindexer_optimizer(self.model, self.trainer_config)
        self.global_step = 0
        self.current_epoch = 0
        self.checkpoint_dir = Path(self.trainer_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl_path = self.checkpoint_dir / self.trainer_config.metrics_jsonl_name

    def log_metrics(self, metrics: Dict) -> None:
        record = detach_scalar_dict(metrics)
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def state_dict(self) -> Dict:
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
        path = self.checkpoint_dir / filename
        torch.save(self.state_dict(), path)
        return str(path)

    def save_epoch_checkpoint(self, epoch: int) -> Dict[str, str]:
        saved = {}
        name = f"{self.trainer_config.checkpoint_prefix}_epoch_{epoch:04d}.pt"
        saved["epoch"] = self.save_checkpoint(name)
        if self.trainer_config.save_latest:
            saved["latest"] = self.save_checkpoint("latest.pt")
        return saved

    def validate_curriculum_config(self, num_epochs: int) -> None:
        cfg = self.trainer_config
        if not cfg.use_staged_curriculum and not cfg.enable_canonical_pretrain:
            return
        if cfg.enable_canonical_pretrain and cfg.pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be >= 0")
        if cfg.enable_canonical_pretrain and not cfg.use_staged_curriculum:
            raise ValueError("enable_canonical_pretrain requires use_staged_curriculum=True")
        if cfg.warmup_reindex_mode != "identity":
            raise ValueError(f"Unsupported warmup_reindex_mode={cfg.warmup_reindex_mode!r}. Current implementation supports only 'identity'.")
        for name in ("warmup_epochs", "stage1_epochs", "stage2_epochs"):
            value = getattr(cfg, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if cfg.tau_r_schedule not in {"cosine", "linear"}:
            raise ValueError(f"Unsupported tau_r_schedule={cfg.tau_r_schedule!r}.")
        total_planned = cfg.pretrain_epochs + cfg.warmup_epochs + cfg.stage1_epochs + cfg.stage2_epochs
        if total_planned > num_epochs:
            raise ValueError(
                "Curriculum phase lengths exceed total epochs: "
                f"pretrain({cfg.pretrain_epochs}) + warmup({cfg.warmup_epochs}) + "
                f"stage1({cfg.stage1_epochs}) + stage2({cfg.stage2_epochs}) = {total_planned} > num_epochs({num_epochs})."
            )

    def get_active_stage(self, epoch: Optional[int] = None) -> str:
        cfg = self.trainer_config
        if epoch is None:
            epoch = self.current_epoch
        if not cfg.use_staged_curriculum:
            return "joint"
        cursor = 0
        if cfg.enable_canonical_pretrain:
            cursor += cfg.pretrain_epochs
            if epoch <= cursor:
                return "pretrain"
        cursor += cfg.warmup_epochs
        if epoch <= cursor:
            return "warmup"
        cursor += cfg.stage1_epochs
        if epoch <= cursor:
            return "stage1"
        return "stage2"

    def _interpolate(self, start: float, end: float, progress: float) -> float:
        progress = max(0.0, min(1.0, float(progress)))
        if self.trainer_config.tau_r_schedule == "linear":
            return start + (end - start) * progress
        cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        return start + (end - start) * cosine_factor

    def _stage_progress(self, stage: str, step_idx: int, num_steps_in_epoch: Optional[int]) -> float:
        cfg = self.trainer_config
        if num_steps_in_epoch is None or num_steps_in_epoch <= 0:
            return 0.0
        offset = cfg.pretrain_epochs if cfg.enable_canonical_pretrain else 0
        if stage == "stage1":
            stage_start_epoch = offset + cfg.warmup_epochs + 1
            stage_epochs = max(cfg.stage1_epochs, 1)
        elif stage == "stage2":
            stage_start_epoch = offset + cfg.warmup_epochs + cfg.stage1_epochs + 1
            stage_epochs = max(cfg.stage2_epochs, 1)
        else:
            return 0.0
        completed_before_current = max(self.current_epoch - stage_start_epoch, 0) * num_steps_in_epoch + max(step_idx - 1, 0)
        total_steps = max(stage_epochs * num_steps_in_epoch, 1)
        if total_steps == 1:
            return 1.0
        return min(max(completed_before_current / float(total_steps - 1), 0.0), 1.0)

    def get_active_tau_r(self, stage: str, step_idx: int, num_steps_in_epoch: Optional[int]) -> float:
        cfg = self.trainer_config
        if not cfg.use_staged_curriculum:
            return float(cfg.tau_r)
        if stage == "pretrain":
            return float(cfg.pretrain_tau_r)
        if stage == "warmup":
            return float(cfg.tau_r_start)
        if stage == "stage1":
            return float(self._interpolate(cfg.tau_r_start, cfg.tau_r_mid, self._stage_progress(stage, step_idx, num_steps_in_epoch)))
        if stage == "stage2":
            return float(self._interpolate(cfg.tau_r_mid, cfg.tau_r_end, self._stage_progress(stage, step_idx, num_steps_in_epoch)))
        return float(cfg.tau_r)

    def build_active_loss_config(self, stage: str) -> LossConfig:
        cfg = self.trainer_config
        if not cfg.use_staged_curriculum:
            return self.loss_config
        if stage in {"pretrain", "warmup"}:
            return replace(self.loss_config, alpha_loc=0.0, beta_cons=0.0)
        if stage == "stage1":
            return replace(self.loss_config, alpha_loc=cfg.stage1_alpha_loc, beta_cons=cfg.stage1_beta_cons)
        if stage == "stage2":
            beta_cons = cfg.stage2_beta_cons
            return replace(self.loss_config, alpha_loc=cfg.stage2_alpha_loc, beta_cons=beta_cons)
        return self.loss_config

    def run_pretrain_reindex_forward(self, native: Dict[str, torch.Tensor], tau_r: float) -> Dict[str, torch.Tensor]:
        return self.model.reindexer(
            A=native["A"],
            m=native["m"],
            Bmat=native["B"],
            c1=native["c1"],
            c2=native["c2"],
            D=native["D"],
            tau_r=tau_r,
            freeze_hardware_reindex=True,
        )

    def run_pass_a_mapper_forward(self, native: Dict[str, torch.Tensor], tau_r: float, use_identity_reindex: bool) -> Dict[str, torch.Tensor]:
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
                freeze_hardware_reindex=self.trainer_config.freeze_hardware_reindex,
            )
            reidx_for_mapper = maybe_detach_reindex_outputs(reidx, detach=True)
        T_hw_star = self.model.token_encoder(
            B_star=reidx_for_mapper["B_star"],
            c2_star=reidx_for_mapper["c2_star"],
            c1_star=reidx_for_mapper["c1_star"],
        )
        A_star_spatial = reidx_for_mapper["A_star"].unsqueeze(1)
        S_star = self.model.mapper(A_star_spatial, T_hw_star)
        return {**reidx_for_mapper, "T_hw_star": T_hw_star, "S_star": S_star}

    def run_pass_b_reindex_forward(self, native: Dict[str, torch.Tensor], tau_r: float) -> Dict[str, torch.Tensor]:
        reidx = self.model.reindexer(
            A=native["A"],
            m=native["m"],
            Bmat=native["B"],
            c1=native["c1"],
            c2=native["c2"],
            D=native["D"],
            tau_r=tau_r,
            freeze_hardware_reindex=self.trainer_config.freeze_hardware_reindex,
        )
        T_hw_star = self.model.token_encoder(
            B_star=reidx["B_star"],
            c2_star=reidx["c2_star"],
            c1_star=reidx["c1_star"],
        )
        A_star_spatial = reidx["A_star"].unsqueeze(1)
        S_star = self.model.mapper(A_star_spatial, T_hw_star)
        return {**reidx, "T_hw_star": T_hw_star, "S_star": S_star}

    def canonical_pretrain_step(self, native: Dict[str, torch.Tensor], stage: str, tau_r_active: float) -> Dict:
        self.model.train()
        set_module_requires_grad(self.model.mapper, False)
        set_module_requires_grad(self.model.token_encoder, False)
        set_module_requires_grad(self.model.reindexer.logical_branch, True)
        set_module_requires_grad(self.model.reindexer.hardware_branch, False)
        self.model.reindexer.force_identity_hardware = True
        self.model.reindexer.train()
        zero_optimizer_grads(self.mapper_optimizer, self.reindexer_optimizer)
        outputs = self.run_pretrain_reindex_forward(native=native, tau_r=tau_r_active)
        teacher = build_logical_canonical_teacher_permutation(native["A"], native["m"])
        loss = self.trainer_config.pretrain_teacher_loss_weight * F.mse_loss(outputs["R_L"], teacher)
        _assert_finite_tensor(loss, "pretrain_L_teacher")
        loss.backward()
        if self.trainer_config.fail_on_nonfinite_grad:
            assert_module_gradients_finite(self.model.reindexer.logical_branch, "reindexer.logical_branch")
        clip_gradients_if_needed_for_modules([self.model.reindexer.logical_branch], self.trainer_config.grad_clip_norm)
        self.reindexer_optimizer.step()
        metrics = {
            "stage": stage,
            "tau_r_active": tau_r_active,
            "pretrain_L_teacher": loss,
            "passB_L_loc": None,
            "passB_L_cons": None,
            **build_reindex_diagnostics(outputs["R_L"], outputs["R_H"]),
            "reindexer_grad_norm": _module_grad_norm(self.model.reindexer.logical_branch),
        }
        return metrics

    def pass_a_mapper_step(self, native: Dict[str, torch.Tensor], stage: str, tau_r_active: float, active_loss_config: LossConfig) -> Dict:
        self.model.train()
        set_module_requires_grad(self.model.mapper, True)
        set_module_requires_grad(self.model.token_encoder, True)
        set_module_requires_grad(self.model.reindexer, False)
        self.model.reindexer.eval()
        self.model.mapper.train()
        self.model.token_encoder.train()
        zero_optimizer_grads(self.mapper_optimizer, self.reindexer_optimizer)
        use_identity_reindex = self.trainer_config.use_staged_curriculum and stage == "warmup"
        outputs = self.run_pass_a_mapper_forward(native=native, tau_r=tau_r_active, use_identity_reindex=use_identity_reindex)
        task_losses = compute_task_loss_from_logits(
            S_star=outputs["S_star"],
            R_L=outputs["R_L"],
            R_H=outputs["R_H"],
            A=native["A"], m=native["m"], Bmat=native["B"], D_raw=native["D_raw"], n1q=native["n1q"], nmeas=native["nmeas"], e1q=native["e1q"], ero=native["ero"], e2q=native["e2q"],
            config=active_loss_config,
        )
        loss = task_losses["L_task"]
        _assert_finite_tensor(loss, "Pass-A L_task")
        loss.backward()
        if self.trainer_config.fail_on_nonfinite_grad:
            assert_modules_gradients_finite({"mapper": self.model.mapper, "token_encoder": self.model.token_encoder})
        clip_gradients_if_needed_for_modules([self.model.mapper, self.model.token_encoder], self.trainer_config.grad_clip_norm)
        mapper_grad_norm = _module_grad_norm(self.model.mapper)
        token_encoder_grad_norm = _module_grad_norm(self.model.token_encoder)
        self.mapper_optimizer.step()
        metrics = {
            "pass": "A",
            "stage": stage,
            "tau_r_active": tau_r_active,
            "alpha_loc_active": active_loss_config.alpha_loc,
            "beta_cons_active": active_loss_config.beta_cons,
            "L_task": task_losses["L_task"],
            "L_1q": task_losses["L_1q"],
            "L_ro": task_losses["L_ro"],
            "L_2q": task_losses["L_2q"],
            "L_native": task_losses["L_native"],
            "L_route": task_losses["L_route"],
            "S_proxy_exec": task_losses["S_proxy_exec"],
            "mapper_grad_norm": mapper_grad_norm,
            "token_encoder_grad_norm": token_encoder_grad_norm,
            **build_reindex_diagnostics(outputs["R_L"], outputs["R_H"]),
        }
        return metrics

    def pass_b_reindex_step(self, native: Dict[str, torch.Tensor], stage: str, tau_r_active: float, active_loss_config: LossConfig) -> Dict:
        if self.trainer_config.use_staged_curriculum and stage in {"warmup", "pretrain"}:
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
                "L_1q": None,
                "L_2q": None,
                "L_native": None,
                "L_route": None,
                "L_ro": None,
            }
        self.model.train()
        set_module_requires_grad(self.model.reindexer, True)
        if self.trainer_config.freeze_hardware_reindex:
            set_module_requires_grad(self.model.reindexer.hardware_branch, False)
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
            A=native["A"], m=native["m"], Bmat=native["B"], c1=native["c1"], c2=native["c2"], D=native["D"], D_raw=native["D_raw"], n1q=native["n1q"], nmeas=native["nmeas"], e1q=native["e1q"], ero=native["ero"], e2q=native["e2q"],
            tau_r=tau_r_active,
            config=active_loss_config,
            generator=None,
        )
        loss = reindex_losses["L_reindex"]
        _assert_finite_tensor(loss, "Pass-B L_reindex")
        loss.backward()
        if self.trainer_config.fail_on_nonfinite_grad:
            assert_module_gradients_finite(self.model.reindexer, "reindexer")
        clip_gradients_if_needed_for_modules([self.model.reindexer], self.trainer_config.grad_clip_norm)
        reindexer_grad_norm = _module_grad_norm(self.model.reindexer)
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
            "L_1q": reindex_losses["L_1q"],
            "L_ro": reindex_losses["L_ro"],
            "L_2q": reindex_losses["L_2q"],
            "L_native": reindex_losses["L_native"],
            "L_route": reindex_losses["L_route"],
            "S_proxy_exec": reindex_losses["S_proxy_exec"],
            "reindexer_grad_norm": reindexer_grad_norm,
            **build_reindex_diagnostics(outputs["R_L"], outputs["R_H"]),
        }
        return metrics

    def train_one_batch(self, batch: Dict, step_idx: int, num_steps_in_epoch: Optional[int]) -> Dict:
        batch = move_batch_to_device(batch, self.device)
        native = extract_native_batch(batch)
        stage = self.get_active_stage()
        tau_r_active = self.get_active_tau_r(stage=stage, step_idx=step_idx, num_steps_in_epoch=num_steps_in_epoch)
        active_loss_config = self.build_active_loss_config(stage)
        start_time = time.time()

        if stage == "pretrain":
            pre_metrics = self.canonical_pretrain_step(native=native, stage=stage, tau_r_active=tau_r_active)
            self.global_step += 1
            metrics = {
                "global_step": self.global_step,
                "epoch": self.current_epoch,
                "stage": stage,
                "step_time_sec": time.time() - start_time,
                "tau_r_active": tau_r_active,
                "alpha_loc_active": 0.0,
                "beta_cons_active": 0.0,
                "pretrain_L_teacher": pre_metrics["pretrain_L_teacher"],
                "passA_L_task": None,
                "passA_L_1q": None,
                "passA_L_ro": None,
                "passA_L_2q": None,
                "passA_L_native": None,
                "passA_L_route": None,
                "passA_S_proxy_exec": None,
                "passB_skipped": True,
                "passB_L_task": None,
                "passB_L_loc": None,
                "passB_L_cons": None,
                "passB_L_reindex": None,
                "passB_L_1q": None,
                "passB_L_ro": None,
                "passB_L_2q": None,
                "passB_L_native": None,
                "passB_L_route": None,
                "passB_S_proxy_exec": None,
                "mapper_grad_norm": None,
                "token_encoder_grad_norm": None,
                "reindexer_grad_norm": pre_metrics["reindexer_grad_norm"],
                "RL_entropy_mean": pre_metrics["RL_entropy_mean"],
                "RH_entropy_mean": pre_metrics["RH_entropy_mean"],
                "RL_rowmax_frac_ge_0_5": pre_metrics["RL_rowmax_frac_ge_0_5"],
                "RL_rowmax_frac_ge_0_7": pre_metrics["RL_rowmax_frac_ge_0_7"],
                "RL_rowmax_frac_ge_0_9": pre_metrics["RL_rowmax_frac_ge_0_9"],
                "RH_rowmax_frac_ge_0_5": pre_metrics["RH_rowmax_frac_ge_0_5"],
                "RH_rowmax_frac_ge_0_7": pre_metrics["RH_rowmax_frac_ge_0_7"],
                "RH_rowmax_frac_ge_0_9": pre_metrics["RH_rowmax_frac_ge_0_9"],
            }
            self.log_metrics(metrics)
            return metrics

        pass_a_metrics = self.pass_a_mapper_step(native=native, stage=stage, tau_r_active=tau_r_active, active_loss_config=active_loss_config)
        pass_b_metrics = self.pass_b_reindex_step(native=native, stage=stage, tau_r_active=tau_r_active, active_loss_config=active_loss_config)
        diag_source = pass_b_metrics if not pass_b_metrics.get("skipped", False) else pass_a_metrics
        self.global_step += 1
        metrics = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "stage": stage,
            "step_time_sec": time.time() - start_time,
            "tau_r_active": tau_r_active,
            "alpha_loc_active": active_loss_config.alpha_loc,
            "beta_cons_active": active_loss_config.beta_cons,
            "pretrain_L_teacher": None,
            "passA_L_task": pass_a_metrics["L_task"],
            "passA_L_1q": pass_a_metrics["L_1q"],
            "passA_L_2q": pass_a_metrics["L_2q"],
            "passA_L_native": pass_a_metrics["L_native"],
            "passA_L_route": pass_a_metrics["L_route"],
            "passA_L_ro": pass_a_metrics["L_ro"],
            "passA_S_proxy_exec": pass_a_metrics["S_proxy_exec"],
            "passB_skipped": pass_b_metrics["skipped"],
            "passB_L_task": pass_b_metrics["L_task"],
            "passB_L_loc": pass_b_metrics["L_loc"],
            "passB_L_cons": pass_b_metrics["L_cons"],
            "passB_L_reindex": pass_b_metrics["L_reindex"],
            "passB_L_1q": pass_b_metrics["L_1q"],
            "passB_L_2q": pass_b_metrics["L_2q"],
            "passB_L_native": pass_b_metrics["L_native"],
            "passB_L_route": pass_b_metrics["L_route"],
            "passB_L_ro": pass_b_metrics["L_ro"],
            "passB_S_proxy_exec": pass_b_metrics.get("S_proxy_exec"),
            "mapper_grad_norm": pass_a_metrics.get("mapper_grad_norm"),
            "token_encoder_grad_norm": pass_a_metrics.get("token_encoder_grad_norm"),
            "reindexer_grad_norm": pass_b_metrics.get("reindexer_grad_norm"),
            "RL_entropy_mean": diag_source["RL_entropy_mean"],
            "RH_entropy_mean": diag_source["RH_entropy_mean"],
            "RL_rowmax_frac_ge_0_5": diag_source["RL_rowmax_frac_ge_0_5"],
            "RL_rowmax_frac_ge_0_7": diag_source["RL_rowmax_frac_ge_0_7"],
            "RL_rowmax_frac_ge_0_9": diag_source["RL_rowmax_frac_ge_0_9"],
            "RH_rowmax_frac_ge_0_5": diag_source["RH_rowmax_frac_ge_0_5"],
            "RH_rowmax_frac_ge_0_7": diag_source["RH_rowmax_frac_ge_0_7"],
            "RH_rowmax_frac_ge_0_9": diag_source["RH_rowmax_frac_ge_0_9"],
        }
        self.log_metrics(metrics)
        return metrics

    def train_one_epoch(self, loader: Iterable[Dict]) -> Dict:
        self.current_epoch += 1
        self.model.train()
        running: List[Dict] = []
        num_steps_in_epoch = len(loader) if hasattr(loader, "__len__") else None
        stage_name = self.get_active_stage()
        for step_idx, batch in enumerate(loader, start=1):
            metrics = self.train_one_batch(batch=batch, step_idx=step_idx, num_steps_in_epoch=num_steps_in_epoch)
            running.append(detach_scalar_dict(metrics))
            if step_idx % self.trainer_config.log_every_steps == 0:
                if stage_name == "pretrain":
                    print(f"[Epoch {self.current_epoch:03d} | Step {step_idx:04d} | {stage_name}] tau_r={metrics['tau_r_active']:.4f} | L_teacher={metrics['pretrain_L_teacher']:.6f} | RL_H={metrics['RL_entropy_mean']:.4f}")
                elif metrics["passB_skipped"]:
                    print(f"[Epoch {self.current_epoch:03d} | Step {step_idx:04d} | {stage_name}] tau_r={metrics['tau_r_active']:.4f} | PassA L_task={metrics['passA_L_task']:.6f} | RL_H={metrics['RL_entropy_mean']:.4f}")
                else:
                    print(f"[Epoch {self.current_epoch:03d} | Step {step_idx:04d} | {stage_name}] tau_r={metrics['tau_r_active']:.4f} | PassA L_task={metrics['passA_L_task']:.6f} | PassB L_reindex={metrics['passB_L_reindex']:.6f} | RL_H={metrics['RL_entropy_mean']:.4f}")
        epoch_summary = average_metric_dicts(running)
        epoch_summary["epoch"] = self.current_epoch
        epoch_summary["stage_name"] = stage_name
        if self.current_epoch % self.trainer_config.save_every_epochs == 0:
            saved = self.save_epoch_checkpoint(self.current_epoch)
            epoch_summary["checkpoint_epoch"] = saved.get("epoch")
            if "latest" in saved:
                epoch_summary["checkpoint_latest"] = saved["latest"]
        return epoch_summary

    def fit(self, train_loader: Iterable[Dict], num_epochs: Optional[int] = None) -> List[Dict]:
        if num_epochs is None:
            num_epochs = self.trainer_config.num_epochs
        self.validate_curriculum_config(num_epochs)
        history = []
        print("Starting KMW training")
        print(f"Device: {self.device}")
        print(f"Staged curriculum enabled: {self.trainer_config.use_staged_curriculum}")
        print(f"Canonical pretrain enabled: {self.trainer_config.enable_canonical_pretrain}")
        print(f"Freeze hardware reindex: {self.trainer_config.freeze_hardware_reindex}")
        print(f"Mapper+token_encoder trainable params: {count_parameters(self.model.mapper) + count_parameters(self.model.token_encoder):,}")
        print(f"Reindexer trainable params: {count_parameters(self.model.reindexer):,}")
        if self.trainer_config.use_staged_curriculum:
            total_planned = self.trainer_config.pretrain_epochs + self.trainer_config.warmup_epochs + self.trainer_config.stage1_epochs + self.trainer_config.stage2_epochs
            print("Curriculum phases: "
                  f"pretrain={self.trainer_config.pretrain_epochs}, warmup={self.trainer_config.warmup_epochs}, "
                  f"stage1={self.trainer_config.stage1_epochs}, stage2={self.trainer_config.stage2_epochs}, "
                  f"planned_total={total_planned}, requested_total={num_epochs}")
        for _ in range(num_epochs):
            summary = self.train_one_epoch(train_loader)
            history.append(summary)
            stage_name = summary.get("stage_name", "unknown")
            tau_r_mean = summary.get("tau_r_active", float("nan"))
            if stage_name == "pretrain":
                print(f"[Epoch {summary['epoch']:03d} done | {stage_name}] tau_r≈{tau_r_mean:.4f} | L_teacher={summary.get('pretrain_L_teacher', float('nan')):.6f} | RL_H={summary.get('RL_entropy_mean', float('nan')):.4f}")
            elif summary.get("passB_skipped", 0.0) >= 0.5:
                print(f"[Epoch {summary['epoch']:03d} done | {stage_name}] tau_r≈{tau_r_mean:.4f} | PassA L_task={summary.get('passA_L_task', float('nan')):.6f} | RL_H={summary.get('RL_entropy_mean', float('nan')):.4f}")
            else:
                print(f"[Epoch {summary['epoch']:03d} done | {stage_name}] tau_r≈{tau_r_mean:.4f} | PassA L_task={summary.get('passA_L_task', float('nan')):.6f} | PassB L_reindex={summary.get('passB_L_reindex', float('nan')):.6f} | RL_H={summary.get('RL_entropy_mean', float('nan')):.4f}")
        return history

    @torch.no_grad()
    def evaluate_mapper_task_only(self, loader: Iterable[Dict]) -> Dict:
        self.model.eval()
        rows = []
        stage = self.get_active_stage(epoch=max(self.current_epoch, 1))
        active_loss_config = self.build_active_loss_config(stage)
        tau_r_active = self.get_active_tau_r(stage=stage, step_idx=1, num_steps_in_epoch=1)
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            native = extract_native_batch(batch)
            if stage in {"pretrain", "warmup"}:
                outputs = self.run_pass_a_mapper_forward(native=native, tau_r=tau_r_active, use_identity_reindex=True)
            else:
                outputs = self.model(
                    A=native["A"], m=native["m"], Bmat=native["B"], c1=native["c1"], c2=native["c2"], D=native["D"],
                    tau_r=tau_r_active,
                    freeze_hardware_reindex=self.trainer_config.freeze_hardware_reindex,
                )
            task_losses = compute_task_loss_from_logits(
                S_star=outputs["S_star"], R_L=outputs["R_L"], R_H=outputs["R_H"],
                A=native["A"], m=native["m"], Bmat=native["B"], D_raw=native["D_raw"], n1q=native["n1q"], nmeas=native["nmeas"], e1q=native["e1q"], ero=native["ero"], e2q=native["e2q"],
                config=active_loss_config,
            )
            rows.append(detach_scalar_dict({
                "L_task": task_losses["L_task"],
                "L_1q": task_losses["L_1q"],
                "L_2q": task_losses["L_2q"],
                "L_native": task_losses["L_native"],
                "L_route": task_losses["L_route"],
                "L_ro": task_losses["L_ro"],
                **build_reindex_diagnostics(outputs["R_L"], outputs["R_H"]),
            }))
        return average_metric_dicts(rows)


def average_metric_dicts(rows: List[Dict]) -> Dict:
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


def build_trainer(
    model: KMWModel,
    loss_config: Optional[LossConfig] = None,
    trainer_config: Optional[TrainerConfig] = None,
    device: Optional[torch.device] = None,
) -> KMWTrainer:
    return KMWTrainer(model=model, loss_config=loss_config, trainer_config=trainer_config, device=device)


__all__ = [
    "TrainerConfig",
    "KMWTrainer",
    "move_batch_to_device",
    "extract_native_batch",
    "set_module_requires_grad",
    "average_metric_dicts",
    "build_trainer",
]
