"""GraphQMap training entry point.

Usage:
    python train.py --config configs/stage1.yaml
    python train.py --config configs/stage2.yaml
    python train.py --config configs/stage1.yaml --override training.optimizer.lr=0.0005
"""

from __future__ import annotations

import logging

import torch

import numpy as np

from configs.config_loader import parse_args_with_config
from data.dataset import create_dataloader, load_split
from data.hardware_graph import configure_hw_features
from models.graphqmap import GraphQMap
from training.trainer import Stage1Trainer, Stage2Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy Qiskit logging
logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("stevedore").setLevel(logging.WARNING)


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_training_backends(cfg) -> list[str]:
    """Extract training backend names from config."""
    backends = []
    for name in cfg.backends.training:
        # Normalize: FakeAthens -> athens, queko_aspen4 stays as-is
        if name.startswith("Fake"):
            backends.append(name[4:].lower())
        else:
            backends.append(name)
    return backends


def _build_val_pst_fn(cfg, device: torch.device):
    """Build a PST validation function for Stage 2.

    Uses a subset of benchmark circuits on the first test backend to
    measure actual PST during training. Simulators are created once
    and reused across validation calls.
    """
    from torch_geometric.data import Batch

    from data.circuit_graph import build_circuit_graph
    from data.hardware_graph import build_hardware_graph, get_backend
    from evaluation.benchmark import BENCHMARK_CIRCUITS, load_benchmark_circuit
    from evaluation.pst import compute_pst, create_ideal_simulator, create_noisy_simulator
    from evaluation.transpiler import transpile_with_timing

    # Use first test backend for validation
    val_backend_name = cfg.backends.test[0]
    if val_backend_name.startswith("Fake"):
        val_backend_name = val_backend_name[4:].lower()
    backend = get_backend(val_backend_name)
    num_physical = backend.target.num_qubits
    hw_graph = build_hardware_graph(backend)
    tau = getattr(cfg.sinkhorn, "tau_min", getattr(cfg.sinkhorn, "tau", 0.05))

    ideal_sim = create_ideal_simulator(backend)
    noisy_sim = create_noisy_simulator(backend)

    # Preload benchmark circuits that fit this backend
    val_circuits = []
    for cname in BENCHMARK_CIRCUITS:
        try:
            circ = load_benchmark_circuit(cname, measure=True)
            if 2 <= circ.num_qubits <= num_physical:
                val_circuits.append((cname, circ))
        except Exception:
            continue

    logger.info("PST validation: %d circuits on %s (%dQ)",
                len(val_circuits), val_backend_name, num_physical)

    def val_pst_fn(model, epoch: int) -> float:
        model.eval()
        pst_values = []

        node_fnames = getattr(cfg.model.circuit_gnn, "node_features", None)
        rk = getattr(cfg.model.circuit_gnn, "rwpe_k", 0)

        for cname, circuit in val_circuits:
            num_logical = circuit.num_qubits
            circuit_graph = build_circuit_graph(
                circuit, node_feature_names=node_fnames, rwpe_k=rk,
            )
            circuit_batch = Batch.from_data_list([circuit_graph]).to(device)
            hw_batch = Batch.from_data_list([hw_graph]).to(device)

            with torch.no_grad():
                layouts = model.predict(
                    circuit_batch, hw_batch,
                    batch_size=1,
                    num_logical=num_logical,
                    num_physical=num_physical,
                    tau=tau,
                )
            layout = list(layouts[0].values())

            try:
                tc, _ = transpile_with_timing(
                    circuit, backend,
                    initial_layout=layout,
                    layout_method="given",
                    routing_method="sabre",
                    seed=cfg.seed,
                )
                ideal_counts = ideal_sim.run(tc, shots=4096).result().get_counts()

                # Average over multiple noisy reps to reduce measurement noise
                rep_psts = []
                for _ in range(3):
                    noisy_counts = noisy_sim.run(tc, shots=4096).result().get_counts()
                    p = compute_pst(noisy_counts, ideal_counts)
                    if isinstance(p, list):
                        p = sum(p) / len(p)
                    rep_psts.append(p)
                pst_values.append(float(np.mean(rep_psts)))
            except Exception as e:
                logger.warning("  PST val skip %s: %s", cname, e)

        model.train()
        avg_pst = float(np.mean(pst_values)) if pst_values else 0.0
        return avg_pst

    return val_pst_fn


def _generate_plots(cfg) -> None:
    """Generate training visualizations after training completes."""
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")

    from scripts.visualize import detect_stage, plot_stage1, plot_stage2

    run_dir = Path(cfg.checkpoint_dir).parent
    save_dir = run_dir / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    stage = detect_stage(run_dir)

    if stage == 1:
        plot_stage1([run_dir], save_dir=save_dir)
    elif stage == 2:
        plot_stage2([run_dir], save_dir=save_dir)
    else:
        logger.warning("Could not detect stage for plotting")
        return

    logger.info("Plots saved to %s", save_dir)


def main() -> None:
    cfg = parse_args_with_config()
    stage = cfg.training.stage
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    _set_seed(cfg.seed)

    # Configure HW feature dimensionality (7dim includes t1/t2)
    hw_input_dim = getattr(cfg.model.hardware_gnn, "node_input_dim", 5)
    configure_hw_features(include_t1_t2=(hw_input_dim == 7))

    logger.info("=== GraphQMap Training — Stage %d ===", stage)
    logger.info("Device: %s", device)
    logger.info("Run directory: %s", cfg.checkpoint_dir.rsplit("/checkpoints", 1)[0])

    data_root = cfg.data.data_root
    max_nodes = cfg.batching.max_total_nodes
    num_workers = cfg.num_workers
    training_backends = _get_training_backends(cfg)

    # Circuit feature config (from YAML, passed to dataset loaders)
    node_feature_names = getattr(cfg.model.circuit_gnn, "node_features", None)
    rwpe_k = getattr(cfg.model.circuit_gnn, "rwpe_k", 0)
    feature_kwargs = {"node_feature_names": node_feature_names, "rwpe_k": rwpe_k}

    if stage == 1:
        model = GraphQMap.from_config(cfg)
        trainer = Stage1Trainer(model, cfg, device)

        # Load main supervised train/val datasets
        logger.info("Loading Stage 1 supervised training data...")
        train_ds = load_split(cfg.data.splits.train, data_root=data_root, **feature_kwargs)
        val_ds = load_split(cfg.data.splits.val, data_root=data_root, **feature_kwargs)
        logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

        train_loader = create_dataloader(
            train_ds, max_total_nodes=max_nodes, shuffle=True,
            num_workers=num_workers, seed=cfg.seed,
        )
        val_loader = create_dataloader(
            val_ds, max_total_nodes=max_nodes, shuffle=False,
            num_workers=num_workers, seed=cfg.seed,
        )

        # Load QUEKO-only data for fine-tuning phase
        queko_train_loader = None
        queko_val_loader = None
        queko_train_path = getattr(cfg.data.splits, "queko_train", None)
        queko_val_path = getattr(cfg.data.splits, "queko_val", None)
        if queko_train_path and queko_val_path:
            logger.info("Loading QUEKO fine-tuning data...")
            queko_train_ds = load_split(queko_train_path, data_root=data_root, **feature_kwargs)
            queko_val_ds = load_split(queko_val_path, data_root=data_root, **feature_kwargs)
            logger.info("QUEKO Train: %d, Val: %d", len(queko_train_ds), len(queko_val_ds))

            queko_train_loader = create_dataloader(
                queko_train_ds, max_total_nodes=max_nodes, shuffle=True,
                num_workers=num_workers, seed=cfg.seed,
            )
            queko_val_loader = create_dataloader(
                queko_val_ds, max_total_nodes=max_nodes, shuffle=False,
                num_workers=num_workers, seed=cfg.seed,
            )

        trainer.run(train_loader, val_loader, queko_train_loader, queko_val_loader)
        _generate_plots(cfg)

    elif stage == 2:
        model = GraphQMap.from_config(cfg)

        # Load Stage 1 checkpoint
        pretrained = getattr(cfg, "pretrained_checkpoint", None)
        if pretrained:
            logger.info("Loading pretrained checkpoint: %s", pretrained)
            ckpt = torch.load(pretrained, map_location=device, weights_only=True)
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if missing:
                logger.info("New keys (randomly initialized): %s", missing)
            if unexpected:
                logger.warning("Unexpected keys in checkpoint: %s", unexpected)

        trainer = Stage2Trainer(model, cfg, device)

        # Load all circuits for Stage 2 surrogate training
        logger.info("Loading Stage 2 training data (all circuits)...")
        train_ds = load_split(
            cfg.data.splits.train,
            data_root=data_root,
            training_backends=training_backends,
            include_stage2_fields=True,
            **feature_kwargs,
        )
        logger.info("Train: %d samples", len(train_ds))

        train_loader = create_dataloader(
            train_ds, max_total_nodes=max_nodes, shuffle=True,
            num_workers=num_workers, seed=cfg.seed,
        )

        # Validation data for surrogate loss early stopping
        val_loader = None
        val_split = getattr(cfg.data.splits, "val", None)
        if val_split:
            logger.info("Loading Stage 2 validation data...")
            val_ds = load_split(
                val_split,
                data_root=data_root,
                training_backends=training_backends,
                include_stage2_fields=True,
                **feature_kwargs,
            )
            logger.info("Val: %d samples", len(val_ds))
            val_loader = create_dataloader(
                val_ds, max_total_nodes=max_nodes, shuffle=False,
                num_workers=num_workers, seed=cfg.seed,
            )

        # PST validation function using benchmark circuits on a test backend (monitoring only)
        val_pst_fn = _build_val_pst_fn(cfg, device)
        trainer.run(train_loader, val_loader=val_loader, val_pst_fn=val_pst_fn)
        _generate_plots(cfg)

    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
