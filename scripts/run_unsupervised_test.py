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
    extract_qubit_properties,
    get_backend,
    precompute_error_distance,
)
from models.graphqmap import GraphQMap
from training.losses import Stage2Loss
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

    Features: [T1_norm, T2_norm, (1-readout_err_norm), (1-sq_err_norm), freq_norm]
    Z-score normalized within backend, errors inverted.
    """
    props = extract_qubit_properties(backend)
    h = backend.target.num_qubits
    eps = 1e-8

    def znorm(arr: np.ndarray) -> np.ndarray:
        m, s = arr.mean(), arr.std()
        return (arr - m) / (s + eps)

    t1_n = znorm(props["t1"])
    t2_n = znorm(props["t2"])
    freq_n = znorm(props["frequency"])
    # Invert errors: (1 - error) then normalize
    readout_inv = znorm(1.0 - props["readout_error"])
    sq_inv = znorm(1.0 - props["single_qubit_error"])

    features = np.stack([t1_n, t2_n, readout_inv, sq_inv, freq_n], axis=1)
    return features.astype(np.float32)


def precompute_hop_distance(backend) -> np.ndarray:
    """Precompute hop-count shortest path distances for separation loss."""
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
    d_error: np.ndarray,
    d_hw: np.ndarray,
    hw_node_features: np.ndarray,
) -> MappingSample | None:
    """Build a single Stage 2 training sample."""
    num_logical = circuit.num_qubits
    num_physical = backend.target.num_qubits

    if num_logical >= num_physical:
        return None

    # Build circuit graph (no global summary for single-circuit test)
    circuit_graph = build_circuit_graph(circuit)

    # Extract circuit edge pairs and qubit importance
    feats = extract_circuit_features(circuit)
    edge_pairs = list(feats["edge_list"])
    # Qubit importance = number of 2-qubit gates per qubit
    importance = feats["node_features"][:, 1].numpy()  # two_qubit_gate_count column

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
        cross_circuit_pairs=[],  # single circuit, no cross pairs
        qubit_importance=importance,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_unsupervised(
    model: GraphQMap,
    quality_score: QualityScore,
    criterion: Stage2Loss,
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

        accum = {"total": 0.0, "l_surr": 0.0, "l_node": 0.0, "l_sep": 0.0}
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

            losses = criterion(
                P=P,
                d_error=batch["d_error"].to(device),
                d_hw=batch["d_hw"].to(device),
                hw_node_features=batch["hw_node_features"].to(device),
                circuit_edge_pairs=batch["circuit_edge_pairs"],
                cross_circuit_pairs=batch["cross_circuit_pairs"],
                qubit_importance=batch["qubit_importance"].to(device),
                num_logical=num_logical,
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            for k in accum:
                accum[k] += losses[k].item()
            num_batches += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        avg = {k: v / max(num_batches, 1) for k, v in accum.items()}
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{num_epochs} | LR={current_lr:.6f} | "
            f"L_total={avg['total']:+.6f} | L_surr={avg['l_surr']:+.6f} | "
            f"L_node={avg['l_node']:+.6f} | L_sep={avg['l_sep']:+.6f}"
        )
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
    d_errors = {}
    d_hws = {}
    hw_features = {}

    logger.info(f"Loading {len(backend_names)} backends...")
    for name in backend_names:
        try:
            b = get_backend(name)
            backends[name] = b
            hw_graphs[name] = build_hardware_graph(b)
            d_errors[name] = precompute_error_distance(b)
            d_hws[name] = precompute_hop_distance(b)
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
                d_errors[bname],
                d_hws[bname],
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
    criterion = Stage2Loss(
        quality_score=quality_score,
        alpha=cfg.loss.weights.alpha,
        lambda_sep=cfg.loss.weights.lambda_sep,
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
        tau=cfg.sinkhorn.tau,
    )
    elapsed = time.time() - t0

    # ---- Step 6: Summary ----
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete in {elapsed:.1f}s")
    logger.info(f"{'='*60}")

    first = history[0]
    last = history[-1]

    logger.info(f"L_total: {first['total']:+.6f} → {last['total']:+.6f}")
    logger.info(f"L_surr:  {first['l_surr']:+.6f} → {last['l_surr']:+.6f}")
    logger.info(f"L_node:  {first['l_node']:+.6f} → {last['l_node']:+.6f}")
    logger.info(f"L_sep:   {first['l_sep']:+.6f} → {last['l_sep']:+.6f}")

    # Check if losses decreased
    decreased = last["total"] < first["total"]
    logger.info(f"\nSanity check {'PASSED' if decreased else 'FAILED'}: "
                f"total loss {'decreased' if decreased else 'did NOT decrease'}")

    # Log quality score weights
    w = quality_score.weights.data.cpu().numpy()
    b = quality_score.bias.data.cpu().item()
    logger.info(f"\nLearned QualityScore weights:")
    logger.info(f"  T1={w[0]:.4f}, T2={w[1]:.4f}, readout={w[2]:.4f}, "
                f"sq_err={w[3]:.4f}, freq={w[4]:.4f}, bias={b:.4f}")


if __name__ == "__main__":
    main()
