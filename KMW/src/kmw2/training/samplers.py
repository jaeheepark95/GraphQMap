from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Optional

from torch.utils.data import Sampler

from ..utils import normalize_source_name


def _weighted_choice(rng: random.Random, labels: List[str], weights: List[float]) -> str:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError('Sampling weights must sum to a positive value.')
    pick = rng.random() * total
    acc = 0.0
    for label, weight in zip(labels, weights):
        acc += float(weight)
        if pick <= acc:
            return label
    return labels[-1]


class BalancedBucketSampler(Sampler[int]):
    """Replacement sampler that chooses source/group first, then samples uniformly within the chosen bucket."""

    def __init__(
        self,
        *,
        source_to_indices: Mapping[str, List[int]],
        num_samples: int,
        strategy: str = 'shuffle',
        source_weights: Optional[Mapping[str, float]] = None,
        groups: Optional[Mapping[str, Iterable[str]]] = None,
        group_weights: Optional[Mapping[str, float]] = None,
        seed: int = 42,
    ):
        self.source_to_indices = {
            normalize_source_name(source): list(indices)
            for source, indices in source_to_indices.items()
            if indices
        }
        if not self.source_to_indices:
            raise ValueError('Sampler requires at least one non-empty source bucket.')
        self.num_samples = int(num_samples)
        self.strategy = strategy
        self.seed = int(seed)
        self._epoch = 0
        self.source_weights = {
            normalize_source_name(k): float(v)
            for k, v in (source_weights or {}).items()
        }
        self.groups = None
        self.group_weights = None
        if groups:
            normalized_groups = {}
            for group_name, sources in groups.items():
                present = [
                    normalize_source_name(src)
                    for src in sources
                    if normalize_source_name(src) in self.source_to_indices
                ]
                if present:
                    normalized_groups[str(group_name)] = present
            if normalized_groups:
                self.groups = normalized_groups
                self.group_weights = {
                    str(k): float(v)
                    for k, v in (group_weights or {}).items()
                    if str(k) in normalized_groups
                }

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def _draw_source(self, rng: random.Random) -> str:
        sources = list(self.source_to_indices.keys())
        if self.strategy == 'group_balanced' and self.groups:
            group_names = list(self.groups.keys())
            gweights = [self.group_weights.get(g, 1.0) for g in group_names]
            chosen_group = _weighted_choice(rng, group_names, gweights)
            group_sources = self.groups[chosen_group]
            return rng.choice(group_sources)

        if self.strategy == 'source_balanced':
            weights = [self.source_weights.get(src, 1.0) for src in sources]
            return _weighted_choice(rng, sources, weights)

        return rng.choice(sources)

    def __iter__(self):
        rng = random.Random(self.seed + 1009 * self._epoch)
        if self.strategy == 'shuffle':
            flat = [idx for indices in self.source_to_indices.values() for idx in indices]
            rng.shuffle(flat)
            if not flat:
                return iter(())
            if self.num_samples <= len(flat):
                return iter(flat[:self.num_samples])
            expanded = list(flat)
            while len(expanded) < self.num_samples:
                expanded.append(rng.choice(flat))
            return iter(expanded[:self.num_samples])

        draws = []
        for _ in range(self.num_samples):
            source = self._draw_source(rng)
            draws.append(rng.choice(self.source_to_indices[source]))
        return iter(draws)


def build_source_index(dataset) -> Dict[str, List[int]]:
    buckets: Dict[str, List[int]] = {}
    for idx, rec in enumerate(dataset.records):
        buckets.setdefault(normalize_source_name(rec.source), []).append(idx)
    return buckets
