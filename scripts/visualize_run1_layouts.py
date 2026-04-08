"""One-off layout visualization for Run 1 (20260329_235420_scratch_error_dist_adj).

Run 1 uses a deprecated configuration that does not match current code defaults:
- Circuit features: old set [gate_count, two_qubit_gate_count, degree, depth_participation]
- HW node features: 7dim [t1, t2, readout, sq_err, degree, t1_cx, t2_cx]
- HW edge features: 1dim [2q_gate_error]
- score_head.noise_bias_dim = 7 (loaded but inert; predict() does not pass HW features)

This script handles the dim mismatches by:
1. Calling configure_hw_features(include_t1_t2=True) so the HW builder emits
   8dim node features. The GNNEncoder slicing then trims 8→7 (Run 1 order).
2. Injecting node_features into the loaded config so build_circuit_graph uses
   the old feature set rather than DEFAULT_NODE_FEATURES.
3. Reusing visualize_layouts.visualize_circuit_layouts for the actual plotting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# IMPORTANT: configure HW features BEFORE importing build_hardware_graph users
from data.hardware_graph import configure_hw_features  # noqa: E402
configure_hw_features(include_t1_t2=True)

from configs.config_loader import load_config  # noqa: E402
from data.hardware_graph import build_hardware_graph, get_backend  # noqa: E402
from evaluation.benchmark import BENCHMARK_CIRCUITS, load_benchmark_circuit  # noqa: E402
from models.graphqmap import GraphQMap  # noqa: E402
from scripts.visualize_layouts import (  # noqa: E402
    BACKENDS,
    visualize_circuit_layouts,
)


RUN_DIR = Path("runs/stage2/20260329_235420_scratch_error_dist_adj")
CONFIG_PATH = RUN_DIR / "config.yaml"
CHECKPOINT_PATH = RUN_DIR / "checkpoints" / "best.pt"

OLD_NODE_FEATURES = [
    "gate_count",
    "two_qubit_gate_count",
    "degree",
    "depth_participation",
]


def main() -> None:
    print(f"Loading Run 1 config: {CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)

    # Inject old feature set so build_circuit_graph uses the right inputs.
    # The Config class supports attribute assignment.
    cfg.model.circuit_gnn.node_features = OLD_NODE_FEATURES
    if not hasattr(cfg.model.circuit_gnn, "rwpe_k"):
        cfg.model.circuit_gnn.rwpe_k = 0

    print(f"  HW node_input_dim: {cfg.model.hardware_gnn.node_input_dim} "
          f"(builder emits 8, encoder slices to 7)")
    print(f"  HW edge_input_dim: {cfg.model.hardware_gnn.edge_input_dim} "
          f"(builder emits 2, encoder slices to 1)")
    print(f"  Circuit node_features: {OLD_NODE_FEATURES}")
    print(f"  Circuit edge_input_dim: {cfg.model.circuit_gnn.edge_input_dim}")
    print(f"  noise_bias_dim: {cfg.model.score_head.noise_bias_dim} "
          f"(loaded but unused at inference)")

    # Build model and load checkpoint
    model = GraphQMap.from_config(cfg)
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model.eval()

    # Output dir: runs/eval/<RUN>/layout_plots
    run_name = RUN_DIR.name
    output_dir = Path("runs/eval") / run_name / "layout_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput dir: {output_dir}")

    # Load all 3 test backends
    backends_dict = {}
    hw_graphs = {}
    for name in BACKENDS.keys():
        print(f"Loading backend: {name}")
        backend = get_backend(name)
        backends_dict[name] = backend
        hw_graphs[name] = build_hardware_graph(backend)
        x_dim = hw_graphs[name].x.shape[-1]
        e_dim = hw_graphs[name].edge_attr.shape[-1] if hw_graphs[name].edge_attr.numel() > 0 else 0
        print(f"  HW graph: node {x_dim}dim, edge {e_dim}dim")

    # Visualize all benchmark circuits
    print(f"\nVisualizing {len(BENCHMARK_CIRCUITS)} circuits × "
          f"{len(backends_dict)} backends × 3 methods\n")

    for cname in BENCHMARK_CIRCUITS:
        print(f"[{cname}]")
        try:
            circuit = load_benchmark_circuit(cname, measure=True)
        except Exception as e:
            print(f"  Skip (load failed): {e}")
            continue

        try:
            visualize_circuit_layouts(
                cname, circuit, backends_dict, model, hw_graphs, cfg, output_dir,
            )
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
