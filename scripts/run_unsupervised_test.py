"""End-to-end unsupervised (Stage 2) sanity check for GraphQMap.

Loads circuits from QUEKO/QASMBench, builds hardware graphs for selected
backends, creates Stage 2 training samples (no labels needed), and runs
the surrogate loss training loop to verify loss decreases.

Usage:
    python scripts/run_unsupervised_test.py
    python scripts/run_unsupervised_test.py --config configs/unsupervised_test.yaml
    python scripts/run_unsupervised_test.py --max-circuits 20 --epochs 10
"""

from __future__ import annotations

import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import torch
from qiskit import QuantumCircuit
from scipy.sparse.csgraph import floyd_warshall

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph, extract_circuit_features, load_circuit
from data.dataset import (
    MappingDataset,
    MappingSample,
    create_dataloader,
)
from data.hardware_graph import (
    build_hardware_graph,
    get_backend,
    get_hw_node_features,
    precompute_error_distance,
)
from models.graphqmap import GraphQMap
from training.losses import SurrogateLoss
from training.quality_score import QualityScore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy Qiskit loggers
logging.getLogger("backend_converter").setLevel(logging.WARNING)
logging.getLogger("qiskit").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def prepare_hw_node_features_for_qscore(backend) -> np.ndarray:
    """Prepare (h, 5) node features for QualityScore.

    Features: [readout_error, single_qubit_error, degree, t1_cx_ratio, t2_cx_ratio]
    Z-score normalized within backend.
    """
    return get_hw_node_features(backend)


def precompute_hop_distance(backend) -> np.ndarray:
    """Precompute hop-count shortest path distances for L_hop loss."""
    h = backend.target.num_qubits
    adj = np.full((h, h), np.inf)
    np.fill_diagonal(adj, 0)
    for edge in backend.coupling_map.get_edges():
        p, q = edge
        adj[p][q] = 1.0
        adj[q][p] = 1.0
    d_hw = floyd_warshall(adj)
    return d_hw.astype(np.float32)


def load_circuits_from_dir(
    circuit_dir: Path,
    max_circuits: int | None = None,
    max_qubits: int | None = None,
) -> list[tuple[str, QuantumCircuit]]:
    """Load .qasm circuits from a directory, filtering by qubit count."""
    circuits = []
    qasm_files = sorted(circuit_dir.glob("*.qasm"))
    for f in qasm_files:
        # Skip transpiled circuits from QASMBench
        if "transpiled" in f.name:
            continue
        try:
            qc = load_circuit(f)
            if max_qubits is not None and qc.num_qubits > max_qubits:
                continue
            # Skip circuits with no 2-qubit gates
            has_2q = any(
                len(inst.qubits) == 2
                for inst in qc.data
            )
            if not has_2q:
                continue
            circuits.append((f.name, qc))
        except Exception as e:
            logger.warning(f"Failed to load {f.name}: {e}")
    if max_circuits is not None:
        random.shuffle(circuits)
        circuits = circuits[:max_circuits]
    return circuits


def build_stage2_sample(
    circuit: QuantumCircuit,
    backend,
    backend_name: str,
    hw_graph,
    d_hw: np.ndarray,
    d_error: np.ndarray,
    hw_node_features: np.ndarray,
) -> MappingSample | None:
    """Build a single Stage 2 training sample."""
    num_logical = circuit.num_qubits
    num_physical = backend.target.num_qubits

    if num_logical >= num_physical:
        return None

    # Build circuit graph (no global summary for single-circuit test)
    circuit_graph = build_circuit_graph(circuit)

    # Extract circuit edge pairs, weights, and qubit importance
    feats = extract_circuit_features(circuit)
    edge_pairs = list(feats["edge_list"])
    edge_weights = feats["edge_features"][:, 0].tolist()
    # Qubit importance = number of 2-qubit gates per qubit
    importance = np.array(feats["node_features_dict"]["two_qubit_gate_count"])

    return MappingSample(
        circuit_graph=circuit_graph,
        hardware_graph=hw_graph,
        backend_name=backend_name,
        num_logical=num_logical,
        num_physical=num_physical,
        d_error=d_error,
        d_hw=d_hw,
        hw_node_features=hw_node_features,
        circuit_edge_pairs=edge_pairs,
        circuit_edge_weights=edge_weights,
        qubit_importance=importance,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_unsupervised(
    model: GraphQMap,
    quality_score: QualityScore,
    criterion: SurrogateLoss,
    dataloader,
    device: torch.device,
    num_epochs: int = 30,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    tau: float = 0.05,
) -> list[dict[str, float]]:
    """Run Stage 2 training and return per-epoch loss history."""
    all_params = list(model.parameters()) + list(quality_score.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6,
    )

    history = []

    for epoch in range(num_epochs):
        model.train()

        # Warmup
        if epoch < warmup_epochs and warmup_epochs > 0:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr * warmup_factor

        accum: dict[str, float] = {"total": 0.0}
        for name in criterion.component_names:
            accum[name] = 0.0
        num_batches = 0

        for batch in dataloader:
            circuit_batch = batch["circuit_batch"].to(device)
            hardware_batch = batch["hardware_batch"].to(device)
            num_logical = batch["num_logical"][0]
            num_physical = batch["num_physical"]
            batch_size = batch["batch_size"]

            optimizer.zero_grad()

            P = model(
                circuit_batch, hardware_batch,
                batch_size=batch_size,
                num_logical=num_logical,
                num_physical=num_physical,
                tau=tau,
            )

            loss_kwargs: dict = {
                "num_logical": num_logical,
                "circuit_edge_pairs": batch["circuit_edge_pairs"],
                "circuit_edge_weights": batch.get("circuit_edge_weights", []),
                "qubit_importance": batch["qubit_importance"].to(device),
                "hw_node_features": batch["hw_node_features"].to(device),
                "cross_circuit_pairs": batch.get("cross_circuit_pairs", []),
            }
            if "d_hw" in batch:
                loss_kwargs["d_hw"] = batch["d_hw"].to(device)
            if "d_error" in batch:
                loss_kwargs["d_error"] = batch["d_error"].to(device)

            losses = criterion(P, **loss_kwargs)

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            for k in accum:
                if k in losses:
                    accum[k] += losses[k].item()
            num_batches += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        avg = {k: v / max(num_batches, 1) for k, v in accum.items()}
        current_lr = optimizer.param_groups[0]["lr"]

        parts = [f"Epoch {epoch:3d}/{num_epochs} | LR={current_lr:.6f} | L_total={avg['total']:+.6f}"]
        for name in criterion.component_names:
            parts.append(f"{name}={avg[name]:+.6f}")
        logger.info(" | ".join(parts))
        history.append(avg)

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GraphQMap Unsupervised Test")
    parser.add_argument("--config", type=str, default="configs/unsupervised_test.yaml")
    parser.add_argument("--max-circuits", type=int, default=None,
                        help="Max circuits per source directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("GraphQMap — Unsupervised (Stage 2) Sanity Check")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")

    # ---- Step 1: Prepare backends ----
    backend_names = [b.replace("Fake", "").lower() for b in cfg.backends.training]
    backends = {}
    hw_graphs = {}
    d_hws = {}
    d_errors = {}
    hw_features = {}

    logger.info(f"Loading {len(backend_names)} backends...")
    for name in backend_names:
        try:
            b = get_backend(name)
            backends[name] = b
            hw_graphs[name] = build_hardware_graph(b)
            d_hws[name] = precompute_hop_distance(b)
            d_errors[name] = precompute_error_distance(b)
            hw_features[name] = prepare_hw_node_features_for_qscore(b)
            nq = b.target.num_qubits
            logger.info(f"  {name}: {nq}Q, {len(b.coupling_map.get_edges())//2} edges")
        except Exception as e:
            logger.error(f"  Failed to load {name}: {e}")

    if not backends:
        logger.error("No backends loaded. Exiting.")
        return

    # ---- Step 2: Load circuits ----
    circuit_dirs = [Path(d) for d in cfg.data.circuit_dirs]
    max_per_dir = args.max_circuits or 50

    all_circuits: list[tuple[str, QuantumCircuit]] = []
    for cdir in circuit_dirs:
        if not cdir.exists():
            logger.warning(f"Circuit dir not found: {cdir}")
            continue
        max_q = max(b.target.num_qubits for b in backends.values()) - 1
        loaded = load_circuits_from_dir(cdir, max_circuits=max_per_dir, max_qubits=max_q)
        logger.info(f"  {cdir}: loaded {len(loaded)} circuits")
        all_circuits.extend(loaded)

    if not all_circuits:
        logger.error("No circuits loaded. Exiting.")
        return

    logger.info(f"Total circuits: {len(all_circuits)}")

    # ---- Step 3: Build dataset ----
    dataset = MappingDataset()
    pair_count = 0
    skip_count = 0

    total_pairs = len(all_circuits) * len(backends)
    for i, (name, qc) in enumerate(all_circuits):
        for bname, backend in backends.items():
            sample = build_stage2_sample(
                qc, backend, bname,
                hw_graphs[bname],
                d_hws[bname],
                d_errors[bname],
                hw_features[bname],
            )
            if sample is not None:
                dataset.add_sample(sample)
                pair_count += 1
            else:
                skip_count += 1
        if (i + 1) % 10 == 0 or i == len(all_circuits) - 1:
            logger.info(f"  Building dataset: {i+1}/{len(all_circuits)} circuits processed")

    logger.info(f"Dataset: {pair_count} samples ({skip_count} skipped, circuit >= backend)")

    if len(dataset) == 0:
        logger.error("Empty dataset. Exiting.")
        return

    # Log group distribution
    groups: dict[tuple[str, int], int] = defaultdict(int)
    for s in dataset.samples:
        groups[(s.backend_name, s.num_logical)] += 1
    logger.info(f"Batching groups (backend, num_logical): {len(groups)}")
    for (bname, nl), count in sorted(groups.items()):
        logger.info(f"  ({bname}, {nl}Q): {count} samples")

    dataloader = create_dataloader(
        dataset,
        max_total_nodes=cfg.batching.max_total_nodes,
        shuffle=True,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    logger.info(f"DataLoader: {len(dataloader)} batches")

    # ---- Step 4: Build model ----
    model = GraphQMap.from_config(cfg).to(device)
    quality_score = QualityScore().to(device)
    components = [
        {"name": c.name, "weight": c.weight}
        for c in cfg.loss.components
    ]
    criterion = SurrogateLoss(
        components=components,
        quality_score=quality_score,
    )

    total_params = sum(p.numel() for p in model.parameters())
    qs_params = sum(p.numel() for p in quality_score.parameters())
    logger.info(f"Model params: {total_params:,} | QualityScore params: {qs_params}")

    # ---- Step 5: Train ----
    num_epochs = args.epochs or cfg.training.max_epochs
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training: {num_epochs} epochs")
    logger.info(f"{'='*60}")

    t0 = time.time()
    history = train_unsupervised(
        model=model,
        quality_score=quality_score,
        criterion=criterion,
        dataloader=dataloader,
        device=device,
        num_epochs=num_epochs,
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        tau=getattr(cfg.sinkhorn, "tau", getattr(cfg.sinkhorn, "tau_min", 0.05)),
    )
    elapsed = time.time() - t0

    # ---- Step 6: Summary ----
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete in {elapsed:.1f}s")
    logger.info(f"{'='*60}")

    first = history[0]
    last = history[-1]

    logger.info(f"L_total: {first['total']:+.6f} → {last['total']:+.6f}")
    for name in criterion.component_names:
        logger.info(f"  {name}: {first[name]:+.6f} → {last[name]:+.6f}")

    # Check if losses decreased
    decreased = last["total"] < first["total"]
    logger.info(f"\nSanity check {'PASSED' if decreased else 'FAILED'}: "
                f"total loss {'decreased' if decreased else 'did NOT decrease'}")

    # Log quality score parameters
    qs_params_count = sum(p.numel() for p in quality_score.parameters())
    logger.info(f"\nQualityScore MLP: {qs_params_count} parameters")


if __name__ == "__main__":
    main()
