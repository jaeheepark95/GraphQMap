from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kmw1.evaluation.evaluate import EvalConfig, evaluate_model
from kmw1.losses.loss import LossConfig, compute_task_loss_from_logits
from kmw1.models.model import KMWCanonicalModel
from kmw1.utils import append_jsonl, count_parameters, detach_scalar_dict, ensure_dir, move_to_device, tensor_stats, write_json


@dataclass
class TrainerConfig:
    run_dir: str = "runs/kmw1"
    epochs: int = 60
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 1
    num_workers: int = 0
    seed: int = 42
    log_every_steps: int = 10
    checkpoint_every_epochs: int = 1
    device: str = "auto"
    fail_on_nonfinite_grad: bool = True
    max_steps_per_epoch: int | None = None
    save_best_on: str = "val_L_task"
    backend_name: str = "fake_toronto_v2"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _assert_finite_tensor(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def assert_module_gradients_finite(module: nn.Module, module_name: str) -> None:
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise ValueError(f"Non-finite gradient detected in {module_name}.{name}")


class CanonicalTrainer:
    def __init__(
        self,
        *,
        model: KMWCanonicalModel,
        trainer_config: TrainerConfig,
        loss_config: LossConfig | None = None,
        eval_config: EvalConfig | None = None,
    ) -> None:
        self.model = model
        self.trainer_config = trainer_config
        self.loss_config = loss_config or LossConfig()
        self.eval_config = eval_config or EvalConfig()
        self.device = resolve_device(trainer_config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=trainer_config.lr,
            weight_decay=trainer_config.weight_decay,
        )
        self.run_dir = ensure_dir(trainer_config.run_dir)
        self.checkpoints_dir = ensure_dir(Path(self.run_dir) / "checkpoints")
        self.logs_dir = ensure_dir(Path(self.run_dir) / "logs")
        self.failure_dir = ensure_dir(Path(self.run_dir) / "failures")

        write_json(
            Path(self.run_dir) / "run_config.json",
            {
                "trainer_config": asdict(self.trainer_config),
                "loss_config": asdict(self.loss_config),
                "eval_config": asdict(self.eval_config),
                "trainable_parameters": count_parameters(self.model),
            },
        )

    def _forward_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        outputs = self.model(
            A=batch["A"],
            B_can=batch["B_can"],
            c1_can=batch["c1_can"],
            c2_can=batch["c2_can"],
        )
        _assert_finite_tensor(outputs["S_can"], "S_can")
        losses = compute_task_loss_from_logits(
            S_can=outputs["S_can"],
            p=batch["p"],
            A=batch["A"],
            m=batch["m"],
            B_nat=batch["B_nat"],
            D_raw_nat=batch["D_raw_nat"],
            n1q=batch["n1q"],
            nmeas=batch["nmeas"],
            e1q_nat=batch["e1q_nat"],
            ero_nat=batch["ero_nat"],
            e2q_nat=batch["e2q_nat"],
            config=self.loss_config,
        )
        losses["S_can_tensor"] = outputs["S_can"]
        return losses

    def _record_failure(
        self,
        *,
        exc: Exception,
        epoch: int,
        step: int,
        batch: dict[str, Any],
        failing_component: str,
    ) -> None:
        metadata = batch.get("metadata", [{}])[0] if isinstance(batch.get("metadata"), list) else {}
        failure_report = {
            "run_dir": str(self.run_dir),
            "epoch": epoch,
            "step": step,
            "circuit_id": metadata.get("id"),
            "backend_id": metadata.get("backend", {}).get("backend_name", self.trainer_config.backend_name),
            "canonical_permutation_p": batch["p"][0].detach().cpu().tolist() if "p" in batch else None,
            "failing_component": failing_component,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "A_stats": tensor_stats(batch["A"][0]) if "A" in batch else None,
            "B_can_stats": tensor_stats(batch["B_can"][0]) if "B_can" in batch else None,
        }
        write_json(Path(self.failure_dir) / f"failure_e{epoch:03d}_s{step:06d}.json", failure_report)

    def train(
        self,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        best_metric = None
        best_checkpoint_path: str | None = None

        for epoch in range(1, self.trainer_config.epochs + 1):
            self.model.train()
            epoch_scalar_logs: list[dict[str, Any]] = []

            for step, batch in enumerate(train_loader, start=1):
                if self.trainer_config.max_steps_per_epoch is not None and step > self.trainer_config.max_steps_per_epoch:
                    break

                batch = move_to_device(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)

                try:
                    losses = self._forward_loss(batch)
                    loss = losses["L_task"]
                    _assert_finite_tensor(loss, "L_task")
                    loss.backward()

                    if self.trainer_config.fail_on_nonfinite_grad:
                        assert_module_gradients_finite(self.model, "model")

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.trainer_config.grad_clip_norm,
                    )
                    self.optimizer.step()
                except Exception as exc:
                    self._record_failure(
                        exc=exc,
                        epoch=epoch,
                        step=step,
                        batch=batch,
                        failing_component="train_step",
                    )
                    raise

                row = {
                    "epoch": epoch,
                    "step": step,
                    "split": "train",
                    "grad_norm": float(grad_norm),
                    **detach_scalar_dict(losses),
                }
                append_jsonl(Path(self.logs_dir) / "train_metrics.jsonl", row)
                epoch_scalar_logs.append(row)

            epoch_summary = self._summarize_epoch(epoch_scalar_logs, epoch=epoch, split="train")

            if epoch % self.trainer_config.checkpoint_every_epochs == 0:
                ckpt_path = self.save_checkpoint(epoch=epoch, tag=f"epoch_{epoch:03d}")
                epoch_summary["checkpoint_path"] = ckpt_path

            if val_loader is not None:
                eval_cfg = EvalConfig(**asdict(self.eval_config))
                eval_cfg.eval_split_name = "val"
                eval_cfg.print_console_summary = False
                eval_cfg.per_circuit_csv_path = str(Path(self.run_dir) / "val_per_circuit.csv")
                eval_cfg.summary_json_path = str(Path(self.run_dir) / "val_summary.json")
                val_summary = evaluate_model(
                    model=self.model,
                    loader=val_loader,
                    device=self.device,
                    eval_config=eval_cfg,
                    loss_config=self.loss_config,
                )
                epoch_summary.update({f"val_{k}": v for k, v in val_summary.items() if isinstance(v, (int, float, str, bool))})

                metric = epoch_summary.get(self.trainer_config.save_best_on)
                if metric is not None and (best_metric is None or float(metric) < float(best_metric)):
                    best_metric = float(metric)
                    best_checkpoint_path = self.save_checkpoint(epoch=epoch, tag="best")

            append_jsonl(Path(self.logs_dir) / "epoch_metrics.jsonl", epoch_summary)

        return {
            "run_dir": str(self.run_dir),
            "best_metric": best_metric,
            "best_checkpoint_path": best_checkpoint_path,
        }

    def _summarize_epoch(self, rows: list[dict[str, Any]], *, epoch: int, split: str) -> dict[str, Any]:
        summary: dict[str, Any] = {"epoch": epoch, "split": split}
        if not rows:
            return summary
        numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
        for key in numeric_keys:
            vals = [float(r[key]) for r in rows if key in r]
            if vals:
                summary[key] = float(sum(vals) / len(vals))
        return summary

    def save_checkpoint(self, *, epoch: int, tag: str) -> str:
        ckpt_path = Path(self.checkpoints_dir) / f"{tag}.pt"
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_config": asdict(self.trainer_config),
            "loss_config": asdict(self.loss_config),
            "eval_config": asdict(self.eval_config),
        }
        torch.save(payload, ckpt_path)
        return str(ckpt_path)


def build_trainer(
    *,
    model: KMWCanonicalModel,
    trainer_config: TrainerConfig,
    loss_config: LossConfig | None = None,
    eval_config: EvalConfig | None = None,
) -> CanonicalTrainer:
    return CanonicalTrainer(
        model=model,
        trainer_config=trainer_config,
        loss_config=loss_config,
        eval_config=eval_config,
    )
