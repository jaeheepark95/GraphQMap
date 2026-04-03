from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ..data.dataset import QubitMappingDataset
from ..losses.loss import MappingProxyLoss
from ..models.model import AssignmentHead, UNetMapping
from ..preprocessing.pipeline import build_model_inputs
from ..utils import backend_from_name, ensure_dir, move_tensor_batch, resolve_config, save_json


@dataclass
class TrainerArtifacts:
    checkpoint_path: Path
    metrics_path: Path


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
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
        self.epochs = int(train_cfg.get('epochs', 50))
        self.batch_size = int(train_cfg.get('batch_size', 16))
        self.grad_clip_norm = float(train_cfg.get('grad_clip_norm', 1.0))
        self.sinkhorn_tau = float(train_cfg.get('sinkhorn_tau', 0.5))
        self.sinkhorn_iters = int(train_cfg.get('sinkhorn_iters', 30))

        self.optimizer = optim.Adam(self.model.parameters(), lr=float(train_cfg.get('lr', 1e-4)))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=int(train_cfg.get('scheduler_patience', 5)),
            factor=float(train_cfg.get('scheduler_factor', 0.5)),
        )

        paths_cfg = config.get('paths', {})
        run_dir = ensure_dir(paths_cfg.get('run_dir', 'runs/kmw2/train_main'))
        self.metrics_path = run_dir / 'train_metrics.json'
        self.checkpoint_path = run_dir / paths_cfg.get('checkpoint_name', 'model_final.pth')

    def _build_dataset(self, split: str = 'train') -> QubitMappingDataset:
        data_cfg = self.config.get('dataset', {})
        recipe_path = data_cfg.get('recipe_path')
        if recipe_path:
            return QubitMappingDataset.from_recipe(recipe_path, split=split, backend=self.backend)
        manifest_key = f'{split}_manifest_path'
        manifest_path = data_cfg.get(manifest_key) or data_cfg.get('manifest_path')
        if manifest_path:
            return QubitMappingDataset(backend=self.backend, manifest_path=manifest_path)
        qasm_paths = data_cfg.get('qasm_paths')
        if qasm_paths:
            return QubitMappingDataset(backend=self.backend, qasm_paths=qasm_paths)
        raise ValueError('No dataset recipe, manifest, or direct qasm_paths provided.')

    def train(self) -> TrainerArtifacts:
        dataset = self._build_dataset(split='train')
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

        history = {
            'device': str(self.device),
            'epochs': [],
            'config': self.config,
        }

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches_seen = 0
            started = time.perf_counter()
            for batch in dataloader:
                batch = move_tensor_batch(batch, self.device)
                X, Tlog_raw, Tphy_raw = build_model_inputs(batch['W'], batch['m'], batch['A'], batch['c1'], batch['c2'])
                self.optimizer.zero_grad()
                logits = self.model(X, Tlog_raw, Tphy_raw)
                P = AssignmentHead.sinkhorn(logits, tau=self.sinkhorn_tau, iterations=self.sinkhorn_iters)
                loss = self.loss_fn(P, batch['W'], batch['c1'], batch['c2'], batch['D'], batch['m'])
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()
                epoch_loss += float(loss.item())
                batches_seen += 1

            avg_loss = epoch_loss / max(batches_seen, 1)
            self.scheduler.step(avg_loss)
            history['epochs'].append(
                {
                    'epoch': epoch + 1,
                    'avg_loss': avg_loss,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'batches_seen': batches_seen,
                    'seconds': time.perf_counter() - started,
                }
            )

        torch.save(self.model.state_dict(), self.checkpoint_path)
        save_json(self.metrics_path, history)
        return TrainerArtifacts(checkpoint_path=self.checkpoint_path, metrics_path=self.metrics_path)


def train_from_config(config_path: str | Path) -> TrainerArtifacts:
    cfg = resolve_config(config_path)
    trainer = Trainer(cfg)
    return trainer.train()
