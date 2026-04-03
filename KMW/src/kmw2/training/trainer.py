from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ..data.dataset import QubitMappingDataset
from ..losses.loss import MappingProxyLoss
from ..models.model import AssignmentHead, UNetMapping
from ..preprocessing.pipeline import build_model_inputs
from ..utils import backend_from_name, ensure_dir, load_merged_configs, save_json, save_jsonl
from .samplers import BalancedBucketSampler, build_source_index


@dataclass
class TrainerArtifacts:
    checkpoint_path: Path
    metrics_path: Path


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        forced_device = config.get('runtime', {}).get('device')
        if forced_device:
            self.device = torch.device(forced_device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        backend_cfg = config.get('backend', 'FakeTorontoV2')
        self.backend = backend_cfg if not isinstance(backend_cfg, str) else backend_from_name(backend_cfg)

        model_cfg = config.get('model', {})
        self.model = UNetMapping(
            in_channels=int(model_cfg.get('in_channels', 5)),
            token_dim=int(model_cfg.get('token_dim', 128)),
        ).to(self.device)

        loss_cfg = config.get('loss', {})
        self.loss_fn = MappingProxyLoss(
            lambda_p=float(loss_cfg.get('lambda_p', 1.0)),
            lambda_s=float(loss_cfg.get('lambda_s', 0.1)),
            lambda_d=float(loss_cfg.get('lambda_d', 0.1)),
            kappa=float(loss_cfg.get('kappa', 1.0)),
        )

        train_cfg = config.get('training', {})
        self.default_epochs = int(train_cfg.get('epochs', 50))
        self.default_batch_size = int(train_cfg.get('batch_size', 16))
        self.default_num_workers = int(train_cfg.get('num_workers', 0))
        self.grad_clip_norm = float(train_cfg.get('grad_clip_norm', 1.0))
        self.sinkhorn_tau = float(train_cfg.get('sinkhorn_tau', 0.5))
        self.sinkhorn_iters = int(train_cfg.get('sinkhorn_iters', 30))
        self.scheduler_patience = int(train_cfg.get('scheduler_patience', 5))
        self.scheduler_factor = float(train_cfg.get('scheduler_factor', 0.5))
        self.base_lr = float(train_cfg.get('lr', 1e-4))
        self.weight_decay = float(train_cfg.get('weight_decay', 0.0))
        self.seed = int(train_cfg.get('seed', 42))
        self.save_epoch_checkpoints = bool(train_cfg.get('save_epoch_checkpoints', True))

        self.optimizer = self._build_optimizer(self.base_lr, self.weight_decay)
        self.scheduler = self._build_scheduler()

        paths_cfg = config.get('paths', {})
        self.run_dir = ensure_dir(paths_cfg.get('run_dir', 'runs/kmw2/train_main'))
        self.checkpoints_dir = ensure_dir(self.run_dir / 'checkpoints')
        self.stages_dir = ensure_dir(self.run_dir / 'stages')
        self.logs_dir = ensure_dir(self.run_dir / 'logs')
        self.metrics_path = self.logs_dir / 'train_metrics.json'
        self.train_metrics_jsonl = self.logs_dir / 'train_metrics.jsonl'
        self.epoch_metrics_jsonl = self.logs_dir / 'epoch_metrics.jsonl'
        self.run_config_path = self.run_dir / 'run_config.json'
        self.checkpoint_path = Path(paths_cfg.get('checkpoint_path') or (self.run_dir / paths_cfg.get('checkpoint_name', 'model_final.pth')))

    def _build_optimizer(self, lr: float, weight_decay: float):
        return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _build_scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
        )

    def _build_dataset(self, split: str = 'train', dataset_cfg: Optional[Dict[str, Any]] = None) -> QubitMappingDataset:
        data_cfg = dataset_cfg or self.config.get('dataset', {})
        recipe_path = data_cfg.get('recipe_path')
        if recipe_path:
            dataset = QubitMappingDataset.from_recipe(recipe_path, split=split, backend=self.backend)
        else:
            manifest_key = f'{split}_manifest_path'
            manifest_path = data_cfg.get(manifest_key) or data_cfg.get('manifest_path')
            manifest_paths = data_cfg.get(f'{split}_manifest_paths') or data_cfg.get('manifest_paths')
            if manifest_paths:
                combined_records: List[Dict[str, Any]] = []
                for path in manifest_paths:
                    ds = QubitMappingDataset(backend=self.backend, manifest_path=path)
                    combined_records.extend([
                        {
                            'circuit_id': rec.circuit_id,
                            'source': rec.source,
                            'qasm_path': rec.qasm_path,
                            'logical_qubits': rec.logical_qubits,
                        }
                        for rec in ds.records
                    ])
                dataset = QubitMappingDataset(backend=self.backend, records=combined_records)
            elif manifest_path:
                dataset = QubitMappingDataset(backend=self.backend, manifest_path=manifest_path)
            else:
                qasm_paths = data_cfg.get('qasm_paths')
                if qasm_paths:
                    dataset = QubitMappingDataset(backend=self.backend, qasm_paths=qasm_paths)
                else:
                    raise ValueError('No dataset recipe, manifest, manifest_paths, or direct qasm_paths provided.')

        allowed_sources = data_cfg.get('allowed_sources')
        if allowed_sources:
            dataset = dataset.subset_by_sources(allowed_sources)
        return dataset

    def _build_dataloader(self, dataset: QubitMappingDataset, batch_size: int, sampler_cfg: Optional[Dict[str, Any]], epoch: int, num_workers: int):
        sampler_cfg = sampler_cfg or {}
        strategy = sampler_cfg.get('kind', 'shuffle')
        common = {'batch_size': batch_size, 'num_workers': num_workers}
        if strategy == 'shuffle':
            return DataLoader(dataset, shuffle=True, **common)

        source_index = build_source_index(dataset)
        sampler = BalancedBucketSampler(
            source_to_indices=source_index,
            num_samples=int(sampler_cfg.get('epoch_samples', len(dataset))),
            strategy=strategy,
            source_weights=sampler_cfg.get('source_weights'),
            groups=sampler_cfg.get('groups'),
            group_weights=sampler_cfg.get('group_weights'),
            seed=int(sampler_cfg.get('seed', self.seed)),
        )
        sampler.set_epoch(epoch)
        return DataLoader(dataset, sampler=sampler, **common)

    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        device_batch = {}
        for key, value in batch.items():
            device_batch[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
        X, Tlog_raw, Tphy_raw = build_model_inputs(device_batch['W'], device_batch['m'], device_batch['A'], device_batch['c1'], device_batch['c2'])
        logits = self.model(X, Tlog_raw, Tphy_raw)
        P = AssignmentHead.sinkhorn(logits, tau=self.sinkhorn_tau, iterations=self.sinkhorn_iters)
        return self.loss_fn(P, device_batch['W'], device_batch['c1'], device_batch['c2'], device_batch['D'], device_batch['m'])

    def _evaluate_loss_dataset(self, dataset: Optional[QubitMappingDataset], batch_size: int, num_workers: int) -> Optional[float]:
        if dataset is None:
            return None
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.model.eval()
        total = 0.0
        seen = 0
        with torch.no_grad():
            for batch in dataloader:
                loss = self._compute_loss(batch)
                if torch.isnan(loss):
                    continue
                total += float(loss.item())
                seen += 1
        self.model.train()
        return (total / seen) if seen else None

    def _save_model_state(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def _load_model_state(self, path: str | Path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    @staticmethod
    def _copy_file(src: str | Path, dst: str | Path):
        src = Path(src)
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    def _write_run_config(self):
        save_json(self.run_config_path, self.config)

    def _append_jsonl(self, path: Path, row: Dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, sort_keys=False))
            f.write('\n')

    def _train_single_stage(
        self,
        *,
        stage_name: str,
        dataset_cfg: Dict[str, Any],
        val_cfg: Optional[Dict[str, Any]],
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        num_workers: int,
        sampler_cfg: Optional[Dict[str, Any]],
        load_from: Optional[str | Path] = None,
    ):
        stage_dir = ensure_dir(self.stages_dir / stage_name)
        self.optimizer = self._build_optimizer(lr, weight_decay)
        self.scheduler = self._build_scheduler()
        if load_from:
            self._load_model_state(load_from)

        train_dataset = self._build_dataset(split='train', dataset_cfg=dataset_cfg)
        val_dataset = self._build_dataset(split='val', dataset_cfg=val_cfg) if val_cfg else None

        history = {
            'stage_name': stage_name,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset) if val_dataset is not None else 0,
            'lr': lr,
            'weight_decay': weight_decay,
            'epochs': [],
        }

        best_metric = None
        best_path = stage_dir / 'best.pt'
        last_path = stage_dir / 'last.pt'
        self.model.train()

        for epoch in range(epochs):
            dataloader = self._build_dataloader(train_dataset, batch_size=batch_size, sampler_cfg=sampler_cfg, epoch=epoch, num_workers=num_workers)
            epoch_loss = 0.0
            batches_seen = 0
            started = time.perf_counter()
            for batch in dataloader:
                self.optimizer.zero_grad()
                loss = self._compute_loss(batch)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()
                epoch_loss += float(loss.item())
                batches_seen += 1

            elapsed = time.perf_counter() - started
            nan_tag = ''
            if batches_seen == 0:
                avg_loss = float('nan')
                nan_tag = '  !! ALL BATCHES NaN !!'
            else:
                avg_loss = epoch_loss / batches_seen
            val_loss = self._evaluate_loss_dataset(val_dataset, batch_size=batch_size, num_workers=num_workers) if val_dataset is not None else None
            metric_for_scheduler = val_loss if val_loss is not None else (avg_loss if not nan_tag else 1e6)
            self.scheduler.step(metric_for_scheduler)
            cur_lr = self.optimizer.param_groups[0]['lr']
            val_str = f'  val={val_loss:.4f}' if val_loss is not None else ''
            best_tag = ''
            metric_chk = val_loss if val_loss is not None else avg_loss
            if not nan_tag and (best_metric is None or metric_chk < best_metric):
                best_tag = '  *best*'
            print(f'[{stage_name}] epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}{val_str}  lr={cur_lr:.2e}  ({elapsed:.1f}s){best_tag}{nan_tag}', flush=True)

            epoch_row = {
                'stage_name': stage_name,
                'epoch': epoch + 1,
                'avg_loss': avg_loss,
                'val_loss': val_loss,
                'lr': cur_lr,
                'batches_seen': batches_seen,
                'seconds': elapsed,
            }
            history['epochs'].append(epoch_row)
            self._append_jsonl(self.epoch_metrics_jsonl, epoch_row)
            self._save_model_state(last_path)
            if self.save_epoch_checkpoints:
                self._save_model_state(self.checkpoints_dir / f'{stage_name}_epoch_{epoch+1:03d}.pt')
            metric_for_best = val_loss if val_loss is not None else avg_loss
            if best_metric is None or metric_for_best < best_metric:
                best_metric = metric_for_best
                self._save_model_state(best_path)

        save_json(stage_dir / 'metrics.json', history)
        return {
            'stage_name': stage_name,
            'best_path': best_path,
            'last_path': last_path,
            'history': history,
            'best_metric': best_metric,
        }

    def train(self) -> TrainerArtifacts:
        self._write_run_config()
        dataset = self._build_dataset(split='train')
        dataloader = DataLoader(dataset, batch_size=self.default_batch_size, shuffle=True, num_workers=self.default_num_workers)
        self.model.train()
        history = {'device': str(self.device), 'epochs': [], 'config': self.config}

        for epoch in range(self.default_epochs):
            epoch_loss = 0.0
            batches_seen = 0
            started = time.perf_counter()
            for batch in dataloader:
                self.optimizer.zero_grad()
                loss = self._compute_loss(batch)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()
                epoch_loss += float(loss.item())
                batches_seen += 1

            elapsed = time.perf_counter() - started
            nan_tag = ''
            if batches_seen == 0:
                avg_loss = float('nan')
                nan_tag = '  !! ALL BATCHES NaN !!'
            else:
                avg_loss = epoch_loss / batches_seen
            self.scheduler.step(avg_loss if not nan_tag else 1e6)
            cur_lr = self.optimizer.param_groups[0]['lr']
            print(f'epoch {epoch+1:3d}/{self.default_epochs}  loss={avg_loss:.4f}  lr={cur_lr:.2e}  ({elapsed:.1f}s){nan_tag}', flush=True)

            epoch_row = {
                'epoch': epoch + 1,
                'avg_loss': avg_loss,
                'lr': cur_lr,
                'batches_seen': batches_seen,
                'seconds': elapsed,
            }
            history['epochs'].append(epoch_row)
            self._append_jsonl(self.train_metrics_jsonl, epoch_row)
            self._append_jsonl(self.epoch_metrics_jsonl, epoch_row)
            if self.save_epoch_checkpoints:
                self._save_model_state(self.checkpoints_dir / f'epoch_{epoch+1:03d}.pt')

        self._save_model_state(self.checkpoint_path)
        self._copy_file(self.checkpoint_path, self.checkpoints_dir / 'last.pt')
        save_json(self.metrics_path, history)
        return TrainerArtifacts(checkpoint_path=self.checkpoint_path, metrics_path=self.metrics_path)

    def train_staged(self) -> TrainerArtifacts:
        self._write_run_config()
        staged_cfg = self.config.get('staged_training', {})
        stages = staged_cfg.get('stages', [])
        if not stages:
            raise ValueError('staged_training.stages is required for train-staged.')

        history = {'device': str(self.device), 'config': self.config, 'stages': []}
        previous_best = None
        previous_last = None
        global_val_cfg = self.config.get('validation')

        for idx, stage in enumerate(stages, start=1):
            stage_name = stage.get('name') or f'stage{idx}'
            dataset_cfg = dict(self.config.get('dataset', {}))
            dataset_cfg.update(stage.get('dataset', {}))

            val_cfg = dict(global_val_cfg or {}) if global_val_cfg else None
            if stage.get('validation'):
                val_cfg = dict(val_cfg or {})
                val_cfg.update(stage.get('validation', {}))

            load_from = stage.get('load_from')
            if load_from == 'previous_best':
                load_from = previous_best
            elif load_from and isinstance(load_from, str) and ':' in load_from:
                ref_stage, ref_kind = load_from.split(':', 1)
                load_from = self.stages_dir / ref_stage / ('best.pt' if ref_kind == 'best' else 'last.pt')
            elif load_from is None and idx > 1:
                load_from = previous_best or previous_last

            result = self._train_single_stage(
                stage_name=stage_name,
                dataset_cfg=dataset_cfg,
                val_cfg=val_cfg,
                epochs=int(stage.get('epochs', self.default_epochs)),
                batch_size=int(stage.get('batch_size', self.default_batch_size)),
                lr=float(stage.get('lr', self.base_lr)),
                weight_decay=float(stage.get('weight_decay', self.weight_decay)),
                num_workers=int(stage.get('num_workers', self.default_num_workers)),
                sampler_cfg=stage.get('sampler'),
                load_from=load_from,
            )
            history['stages'].append(result['history'])
            previous_best = result['best_path']
            previous_last = result['last_path']
            self._copy_file(previous_best, self.checkpoints_dir / f'{stage_name}_best.pt')
            self._copy_file(previous_last, self.checkpoints_dir / f'{stage_name}_last.pt')

        final_choice = staged_cfg.get('final_checkpoint', 'last')
        final_src = previous_best if final_choice == 'best' else previous_last
        if final_src is None:
            raise RuntimeError('No stage checkpoint produced by staged training.')
        self._copy_file(final_src, self.checkpoint_path)
        if previous_best is not None:
            self._copy_file(previous_best, self.checkpoints_dir / 'best.pt')
        if previous_last is not None:
            self._copy_file(previous_last, self.checkpoints_dir / 'last.pt')
        save_json(self.metrics_path, history)
        save_jsonl(self.train_metrics_jsonl, [{'stage': s['stage_name'], 'epochs': s['epochs']} for s in history['stages']])
        return TrainerArtifacts(checkpoint_path=self.checkpoint_path, metrics_path=self.metrics_path)


def train_from_config(config_paths, staged: bool = False) -> TrainerArtifacts:
    if isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]
    cfg = load_merged_configs(config_paths)
    trainer = Trainer(cfg)
    return trainer.train_staged() if staged else trainer.train()
