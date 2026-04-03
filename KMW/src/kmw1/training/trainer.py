from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from kmw1.data.dataset import KMW1Dataset, build_source_datasets, kmw1_collate_fn
from kmw1.evaluation.evaluate import EvalConfig, evaluate_model
from kmw1.losses.loss import LossConfig, compute_task_loss_from_logits
from kmw1.models.model import KMWCanonicalModel
from kmw1.utils import append_jsonl, count_parameters, detach_scalar_dict, ensure_dir, move_to_device, tensor_stats, write_json


@dataclass
class TrainerConfig:
    run_dir: str = "runs/kmw1"
    device: str = "auto"
    seed: int = 42
    log_every_steps: int = 10
    checkpoint_every_epochs: int = 1
    fail_on_nonfinite_grad: bool = True
    backend_name: str = "fake_toronto_v2"
    lr_stage0: float = 1e-4
    lr_stage1: float = 1e-4
    lr_stage2: float = 1e-4
    lr_stage3: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    epoch_draws: int = 6000
    batch_size: int = 1
    num_workers: int = 0
    stage0_epochs: int = 1
    stage1_epochs: int = 10
    stage2_epochs: int = 25
    stage3_epochs: int = 10


@dataclass
class StageScheduleConfig:
    source_manifest_root: str = "data/manifests/full/source_manifests"
    smoke_manifest: str | None = None
    synth_sources: tuple[str, ...] = ("queko", "mlqd", "mqt_bench")
    real_sources: tuple[str, ...] = ("qasmbench", "revlib")
    stage2_synth_prob: float = 0.70
    stage2_real_prob: float = 0.30
    stage3_synth_prob: float = 0.50
    stage3_real_prob: float = 0.50
    guardrail_factor: float = 1.10


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


class SourceBalancedSampler(Sampler[int]):
    def __init__(self, datasets_by_source: dict[str, KMW1Dataset], stage: str, epoch_draws: int, seed: int,
                 synth_sources: tuple[str, ...], real_sources: tuple[str, ...], stage2_synth_prob: float,
                 stage2_real_prob: float, stage3_synth_prob: float, stage3_real_prob: float) -> None:
        self.datasets_by_source = datasets_by_source
        self.stage = stage
        self.epoch_draws = epoch_draws
        self.seed = seed
        self.synth_sources = tuple(s for s in synth_sources if s in datasets_by_source)
        self.real_sources = tuple(r for r in real_sources if r in datasets_by_source)
        self.stage2_synth_prob = stage2_synth_prob
        self.stage2_real_prob = stage2_real_prob
        self.stage3_synth_prob = stage3_synth_prob
        self.stage3_real_prob = stage3_real_prob
        self.global_indices: dict[str, list[int]] = {}
        offset = 0
        for source, ds in datasets_by_source.items():
            self.global_indices[source] = list(range(offset, offset + len(ds)))
            offset += len(ds)
        self.total_len = offset
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.epoch_draws

    def _pick_source(self, rng: random.Random) -> str:
        if self.stage == "stage1":
            return rng.choice(self.synth_sources)
        if self.stage == "stage2":
            group = "synth" if rng.random() < self.stage2_synth_prob else "real"
            if group == "synth":
                return rng.choice(self.synth_sources)
            weights = [len(self.global_indices[s]) for s in self.real_sources]
            return rng.choices(list(self.real_sources), weights=weights, k=1)[0]
        if self.stage == "stage3":
            group = "synth" if rng.random() < self.stage3_synth_prob else "real"
            if group == "synth":
                return rng.choice(self.synth_sources)
            weights = [len(self.global_indices[s]) for s in self.real_sources]
            return rng.choices(list(self.real_sources), weights=weights, k=1)[0]
        # fallback for smoke/other single-source loaders
        return next(iter(self.datasets_by_source.keys()))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.epoch_draws):
            source = self._pick_source(rng)
            yield rng.choice(self.global_indices[source])


class ConcatSourceDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_by_source: dict[str, KMW1Dataset]) -> None:
        self.datasets_by_source = datasets_by_source
        self.sources = list(datasets_by_source.keys())
        self.offsets: list[tuple[str, int, int]] = []
        offset = 0
        for source, ds in datasets_by_source.items():
            self.offsets.append((source, offset, offset + len(ds)))
            offset += len(ds)
        self.total_len = offset

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, index: int):
        for source, start, end in self.offsets:
            if start <= index < end:
                return self.datasets_by_source[source][index - start]
        raise IndexError(index)


class CanonicalTrainer:
    def __init__(self, *, model: KMWCanonicalModel, trainer_config: TrainerConfig, schedule_config: StageScheduleConfig,
                 loss_config: LossConfig | None = None, eval_config: EvalConfig | None = None) -> None:
        self.model = model
        self.trainer_config = trainer_config
        self.schedule_config = schedule_config
        self.loss_config = loss_config or LossConfig()
        self.eval_config = eval_config or EvalConfig()
        self.device = resolve_device(trainer_config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=trainer_config.lr_stage1, weight_decay=trainer_config.weight_decay)
        self.run_dir = ensure_dir(trainer_config.run_dir)
        self.checkpoints_dir = ensure_dir(Path(self.run_dir) / "checkpoints")
        self.logs_dir = ensure_dir(Path(self.run_dir) / "logs")
        self.failure_dir = ensure_dir(Path(self.run_dir) / "failures")
        write_json(Path(self.run_dir) / "run_config.json", {
            "trainer_config": asdict(self.trainer_config),
            "schedule_config": asdict(self.schedule_config),
            "loss_config": asdict(self.loss_config),
            "eval_config": asdict(self.eval_config),
            "trainable_parameters": count_parameters(self.model),
        })
        self.real_stage_baseline: float | None = None
        self.best_real_metric: float | None = None
        self.best_checkpoint_path: str | None = None

    def _set_stage_lr(self, stage: str) -> None:
        lr_map = {
            "stage0": self.trainer_config.lr_stage0,
            "stage1": self.trainer_config.lr_stage1,
            "stage2": self.trainer_config.lr_stage2,
            "stage3": self.trainer_config.lr_stage3,
        }
        lr = lr_map[stage]
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _forward_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        outputs = self.model(A=batch["A"], B_can=batch["B_can"], c1_can=batch["c1_can"], c2_can=batch["c2_can"])
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
        return losses

    def _record_failure(self, *, exc: Exception, stage: str, epoch: int, step: int, batch: dict[str, Any], failing_component: str) -> None:
        metadata = batch.get("metadata", [{}])[0] if isinstance(batch.get("metadata"), list) else {}
        write_json(Path(self.failure_dir) / f"failure_{stage}_e{epoch:03d}_s{step:06d}.json", {
            "run_dir": str(self.run_dir),
            "stage": stage,
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
        })

    def _build_loader_for_stage(self, project_root: str | Path, stage: str, force_recompute: bool = False):
        if stage == "stage0":
            smoke_manifest = self.schedule_config.smoke_manifest
            if smoke_manifest is None:
                raise ValueError("stage0 requires schedule_config.smoke_manifest")
            ds = KMW1Dataset(project_root=project_root, manifest_path=smoke_manifest, backend_name=self.trainer_config.backend_name, force_recompute=force_recompute)
            return DataLoader(ds, batch_size=self.trainer_config.batch_size, shuffle=True, num_workers=self.trainer_config.num_workers, collate_fn=kmw1_collate_fn)
        if stage == "stage1":
            required = list(self.schedule_config.synth_sources)
        elif stage in {"stage2", "stage3"}:
            required = list(self.schedule_config.synth_sources + self.schedule_config.real_sources)
        else:
            raise ValueError(stage)
        ds_by_source = build_source_datasets(project_root=project_root, source_manifest_root=self.schedule_config.source_manifest_root, split="train", backend_name=self.trainer_config.backend_name, force_recompute=force_recompute, required_sources=required)
        concat = ConcatSourceDataset(ds_by_source)
        sampler = SourceBalancedSampler(ds_by_source, stage=stage, epoch_draws=self.trainer_config.epoch_draws, seed=self.trainer_config.seed, synth_sources=self.schedule_config.synth_sources, real_sources=self.schedule_config.real_sources, stage2_synth_prob=self.schedule_config.stage2_synth_prob, stage2_real_prob=self.schedule_config.stage2_real_prob, stage3_synth_prob=self.schedule_config.stage3_synth_prob, stage3_real_prob=self.schedule_config.stage3_real_prob)
        return concat, sampler, DataLoader(concat, batch_size=self.trainer_config.batch_size, sampler=sampler, num_workers=self.trainer_config.num_workers, collate_fn=kmw1_collate_fn)

    def _evaluate_macro(self, datasets_by_source: dict[str, KMW1Dataset], split_name: str, stage: str, epoch: int) -> tuple[float, dict[str, float]]:
        per_source = {}
        for source, ds in datasets_by_source.items():
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=self.trainer_config.num_workers, collate_fn=kmw1_collate_fn)
            ev = EvalConfig(per_circuit_csv_path=str(Path(self.run_dir)/"logs"/f"{stage}_{split_name}_{source}_per_circuit.csv"), summary_json_path=str(Path(self.run_dir)/"logs"/f"{stage}_{split_name}_{source}_summary.json"), print_console_summary=False, fail_fast=False, eval_split_name=f"{split_name}_{source}", project_root=".", backend_name=self.trainer_config.backend_name, route_final_eval=False)
            summary = evaluate_model(model=self.model, loader=loader, device=self.device, eval_config=ev, loss_config=self.loss_config)
            per_source[source] = float(summary.get("mean_L_task_hard", float("inf")))
        macro = sum(per_source.values()) / max(len(per_source), 1)
        row = {"stage": stage, "epoch": epoch, f"{split_name}_macro": macro, **{f"{split_name}_{k}": v for k, v in per_source.items()}}
        append_jsonl(Path(self.run_dir)/"logs"/"val_metrics.jsonl", row)
        return macro, per_source

    def _maybe_update_best(self, stage: str, epoch: int, val_synth_macro: float, val_real_macro: float) -> None:
        if stage == "stage2" and self.real_stage_baseline is None:
            self.real_stage_baseline = val_synth_macro
        if stage in {"stage2", "stage3"} and self.real_stage_baseline is not None:
            if val_synth_macro > self.schedule_config.guardrail_factor * self.real_stage_baseline:
                return
        if self.best_real_metric is None or val_real_macro < self.best_real_metric:
            self.best_real_metric = val_real_macro
            self.best_checkpoint_path = self.save_checkpoint(epoch=epoch, stage=stage, tag="best")

    def save_checkpoint(self, *, epoch: int, stage: str, tag: str) -> str:
        ckpt_path = Path(self.checkpoints_dir) / f"{tag}.pt"
        torch.save({
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_config": asdict(self.trainer_config),
            "schedule_config": asdict(self.schedule_config),
            "loss_config": asdict(self.loss_config),
            "eval_config": asdict(self.eval_config),
        }, ckpt_path)
        return str(ckpt_path)

    def run(self, project_root: str | Path, force_recompute: bool = False) -> dict[str, Any]:
        stages = [("stage0", self.trainer_config.stage0_epochs), ("stage1", self.trainer_config.stage1_epochs), ("stage2", self.trainer_config.stage2_epochs), ("stage3", self.trainer_config.stage3_epochs)]
        global_epoch = 0
        for stage, epochs in stages:
            if epochs <= 0:
                continue
            self._set_stage_lr(stage)
            if stage == "stage0":
                loader = self._build_loader_for_stage(project_root, stage, force_recompute=force_recompute)
                concat = None
                sampler = None
            else:
                concat, sampler, loader = self._build_loader_for_stage(project_root, stage, force_recompute=force_recompute)
            for _ in range(epochs):
                global_epoch += 1
                self.model.train()
                if sampler is not None:
                    sampler.set_epoch(global_epoch)
                epoch_rows = []
                for step, batch in enumerate(loader, start=1):
                    batch = move_to_device(batch, self.device)
                    self.optimizer.zero_grad(set_to_none=True)
                    try:
                        losses = self._forward_loss(batch)
                        loss = losses["L_task"]
                        _assert_finite_tensor(loss, "L_task")
                        loss.backward()
                        if self.trainer_config.fail_on_nonfinite_grad:
                            assert_module_gradients_finite(self.model, "model")
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.trainer_config.grad_clip_norm)
                        self.optimizer.step()
                    except Exception as exc:
                        self._record_failure(exc=exc, stage=stage, epoch=global_epoch, step=step, batch=batch, failing_component="train_step")
                        raise
                    row = {"stage": stage, "epoch": global_epoch, "step": step, "split": "train", "grad_norm": float(grad_norm), **detach_scalar_dict(losses)}
                    append_jsonl(Path(self.logs_dir)/"train_metrics.jsonl", row)
                    epoch_rows.append(row)
                # save epoch checkpoint
                if global_epoch % self.trainer_config.checkpoint_every_epochs == 0:
                    self.save_checkpoint(epoch=global_epoch, stage=stage, tag=f"{stage}_epoch_{global_epoch:03d}")
                # validation
                source_root = self.schedule_config.source_manifest_root
                synth_val = build_source_datasets(project_root=project_root, source_manifest_root=source_root, split="val", backend_name=self.trainer_config.backend_name, required_sources=list(self.schedule_config.synth_sources))
                val_synth_macro, synth_terms = self._evaluate_macro(synth_val, "val_synth", stage, global_epoch)
                if stage in {"stage2", "stage3"}:
                    real_val = build_source_datasets(project_root=project_root, source_manifest_root=source_root, split="val", backend_name=self.trainer_config.backend_name, required_sources=list(self.schedule_config.real_sources))
                    val_real_macro, real_terms = self._evaluate_macro(real_val, "val_real", stage, global_epoch)
                    self._maybe_update_best(stage, global_epoch, val_synth_macro, val_real_macro)
                else:
                    val_real_macro, real_terms = float("inf"), {}
                append_jsonl(Path(self.logs_dir)/"epoch_metrics.jsonl", {"stage": stage, "epoch": global_epoch, "mean_train_L_task": sum(r["L_task"] for r in epoch_rows)/max(len(epoch_rows),1), "val_synth_macro": val_synth_macro, "val_real_macro": val_real_macro, **{f"val_synth_{k}": v for k,v in synth_terms.items()}, **{f"val_real_{k}": v for k,v in real_terms.items()}})
        return {"run_dir": str(self.run_dir), "best_metric": self.best_real_metric, "best_checkpoint_path": self.best_checkpoint_path}


def build_source_manifest_trainer(*, model: KMWCanonicalModel, trainer_config: TrainerConfig, schedule_config: StageScheduleConfig, loss_config: LossConfig | None = None, eval_config: EvalConfig | None = None) -> CanonicalTrainer:
    return CanonicalTrainer(model=model, trainer_config=trainer_config, schedule_config=schedule_config, loss_config=loss_config, eval_config=eval_config)
