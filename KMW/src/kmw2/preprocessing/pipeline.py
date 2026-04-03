from __future__ import annotations

from typing import Dict, Iterable, Mapping

import networkx as nx
import numpy as np
import torch

from .canonical_indexer import CanonicalIndexer
from .extractor import BackendV2Extractor
from .featurizer import CircuitFeaturizer


def validate_hardware_bundle(bundle: Mapping[str, np.ndarray]) -> None:
    p = np.asarray(bundle['p'])
    p_inv = np.asarray(bundle['p_inv'])
    n = len(p)
    if sorted(p.tolist()) != list(range(n)):
        raise ValueError('Invalid canonical permutation p.')
    if not np.array_equal(p_inv[p], np.arange(n)):
        raise ValueError('Invalid inverse permutation p_inv.')
    for name in ['Anat', 'c1nat', 'c2nat', 'A', 'c1', 'c2', 'D']:
        arr = np.asarray(bundle[name])
        if not np.all(np.isfinite(arr)):
            raise ValueError(f'Non-finite values detected in {name}.')


def build_hardware_cache(backend, n: int = 27) -> Dict[str, np.ndarray]:
    extractor = BackendV2Extractor(n=n)
    indexer = CanonicalIndexer(n=n)
    Anat, c1nat, c2nat = extractor.extract(backend)
    p, p_inv = indexer.get_permutation(Anat, c1nat, c2nat)
    A, c1, c2 = indexer.canonicalize(Anat, c1nat, c2nat, p)

    G = nx.from_numpy_array(A)
    D = np.asarray(nx.floyd_warshall_numpy(G), dtype=np.float32)
    D_max = float(D.max())
    if D_max > 0:
        D = D / D_max

    bundle = {
        'Anat': Anat,
        'c1nat': c1nat,
        'c2nat': c2nat,
        'p': p,
        'p_inv': p_inv,
        'A': A,
        'c1': c1,
        'c2': c2,
        'D': D,
    }
    validate_hardware_bundle(bundle)
    return bundle


def build_model_inputs(W, m, A, c1, c2):
    if not torch.is_tensor(W):
        W = torch.as_tensor(W, dtype=torch.float32)
    if not torch.is_tensor(m):
        m = torch.as_tensor(m, dtype=torch.float32)
    if not torch.is_tensor(A):
        A = torch.as_tensor(A, dtype=torch.float32)
    if not torch.is_tensor(c1):
        c1 = torch.as_tensor(c1, dtype=torch.float32)
    if not torch.is_tensor(c2):
        c2 = torch.as_tensor(c2, dtype=torch.float32)

    if W.ndim == 2:
        W = W.unsqueeze(0)
    if m.ndim == 1:
        m = m.unsqueeze(0)
    if A.ndim == 2:
        A = A.unsqueeze(0)
    if c1.ndim == 1:
        c1 = c1.unsqueeze(0)
    if c2.ndim == 2:
        c2 = c2.unsqueeze(0)

    X3 = c1.unsqueeze(1).repeat(1, 27, 1)
    X4 = m.unsqueeze(2).repeat(1, 1, 27)
    X = torch.stack([W, A, c2, X3, X4], dim=1)

    Tlog_raw = torch.stack([W.sum(dim=-1), torch.zeros_like(m), m], dim=-1)
    Tphy_raw = torch.stack([c1, A.sum(dim=-1), c2.mean(dim=-1), c2.min(dim=-1).values], dim=-1)
    validate_model_inputs(X, Tlog_raw, Tphy_raw)
    return X, Tlog_raw, Tphy_raw


def validate_model_inputs(X: torch.Tensor, Tlog_raw: torch.Tensor, Tphy_raw: torch.Tensor) -> None:
    if X.shape[1:] != (5, 27, 27):
        raise ValueError(f'Unexpected X shape: {tuple(X.shape)}')
    if Tlog_raw.shape[-1] != 3:
        raise ValueError('Logical token dimension must be 3.')
    if Tphy_raw.shape[-1] != 4:
        raise ValueError('Physical token dimension must be 4.')
    for name, tensor in [('X', X), ('Tlog_raw', Tlog_raw), ('Tphy_raw', Tphy_raw)]:
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Non-finite values detected in {name}.')


def canonical_indices_to_native_ids(canonical_indices, p):
    p = np.asarray(p.cpu() if isinstance(p, torch.Tensor) else p)
    canonical_indices = np.asarray(canonical_indices.cpu() if isinstance(canonical_indices, torch.Tensor) else canonical_indices, dtype=np.int64)
    return p[canonical_indices]


def canonical_mapping_to_native(hard_assignment: torch.Tensor, logical_qubits: int, p) -> Dict[int, int]:
    if hard_assignment.ndim == 3:
        hard_assignment = hard_assignment[0]
    canonical_indices = torch.argmax(hard_assignment[:logical_qubits], dim=1).detach().cpu().numpy()
    native_ids = canonical_indices_to_native_ids(canonical_indices, p)
    return {int(i): int(native_ids[i]) for i in range(logical_qubits)}


def featurize_circuit(circuit, n: int = 27):
    featurizer = CircuitFeaturizer(n=n)
    return featurizer.featurize(circuit)
