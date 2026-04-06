"""Visualize initial layouts on backend topology.

Compares 3 layout methods (SABRE, OURS+NASSC, QAP+NASSC) across test backends.
Highlighted (colored) nodes = mapped physical qubits, with virtual qubit labels.

Usage:
    python scripts/visualize_layouts.py \
        --config configs/stage2.yaml \
        --checkpoint runs/stage2/<RUN>/checkpoints/best.pt \
        --circuit toffoli_3 fredkin_3 alu-v0_27

    # All benchmark circuits on all test backends
    python scripts/visualize_layouts.py \
        --config configs/stage2.yaml \
        --checkpoint runs/stage2/<RUN>/checkpoints/best.pt

    # Specific backend only
    python scripts/visualize_layouts.py \
        --config configs/stage2.yaml \
        --checkpoint runs/stage2/<RUN>/checkpoints/best.pt \
        --backend toronto --circuit toffoli_3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch
from qiskit.visualization.gate_map import plot_gate_map
from qiskit_ibm_runtime.fake_provider import FakeTorontoV2, FakeBrooklynV2, FakeTorino

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph
from data.hardware_graph import build_hardware_graph, get_backend
from evaluation.benchmark import load_benchmark_circuit, BENCHMARK_CIRCUITS
from evaluation.transpiler import build_transpiler
from models.graphqmap import GraphQMap


BACKENDS = {
    "toronto": FakeTorontoV2,
    "brooklyn": FakeBrooklynV2,
    "torino": FakeTorino,
}

METHODS = [
    # (label, layout_method, routing_method, uses_model)
    ("SABRE", "sabre", "sabre", False),
    ("OURS+NASSC", None, "nassc", True),
    ("QAP+NASSC", "qap", "nassc", False),
]

# Colors for mapped qubits per method
METHOD_COLORS = {
    "SABRE": "#4CAF50",       # green
    "OURS+NASSC": "#2196F3",  # blue
    "QAP+NASSC": "#FF9800",   # orange
}


def get_layout_from_transpiled(circuit, backend, layout_method, routing_method, seed=43):
    """Transpile circuit and extract the initial layout mapping."""
    pm = build_transpiler(
        backend,
        layout_method=layout_method,
        routing_method=routing_method,
        seed=seed,
    )
    tc = pm.run(circuit)
    # Extract layout: virtual qubit index -> physical qubit index
    layout = {}
    virt_bits = tc._layout.initial_layout.get_virtual_bits()
    for qubit, phys_idx in virt_bits.items():
        reg = None
        for register in tc._layout.initial_layout.get_registers():
            if qubit in register:
                reg = register
                break
        if reg is not None and reg.name == "ancilla":
            continue
        # Find the virtual index
        for r in tc._layout.initial_layout.get_registers():
            if qubit in r and r.name != "ancilla":
                virt_idx = list(r).index(qubit)
                layout[virt_idx] = phys_idx
                break
    return layout


def get_model_layout(model, circuit, backend, hw_graph, cfg):
    """Get layout from GraphQMap model prediction."""
    node_features = getattr(cfg.model.circuit_gnn, "node_features", None)
    rwpe_k = getattr(cfg.model.circuit_gnn, "rwpe_k", 0)
    circuit_graph = build_circuit_graph(
        circuit, node_feature_names=node_features, rwpe_k=rwpe_k,
    )
    circuit_batch = Batch.from_data_list([circuit_graph])
    hw_batch = Batch.from_data_list([hw_graph])

    num_logical = circuit.num_qubits
    num_physical = backend.target.num_qubits
    tau = getattr(cfg.sinkhorn, "tau_min", getattr(cfg.sinkhorn, "tau", 0.05))

    with torch.no_grad():
        layouts = model.predict(
            circuit_batch, hw_batch,
            batch_size=1,
            num_logical=num_logical,
            num_physical=num_physical,
            tau=tau,
        )
    return layouts[0]  # dict {logical: physical}


def plot_layout_on_backend(backend, layout, ax, title, color="#2196F3"):
    """Plot backend topology with layout highlighted.

    Args:
        backend: FakeBackendV2 instance.
        layout: dict {virtual_qubit_idx: physical_qubit_idx}.
        ax: matplotlib axes.
        title: subplot title.
        color: color for mapped qubits.
    """
    num_qubits = backend.num_qubits
    cmap = backend.coupling_map
    mapped_physical = set(layout.values())

    # Node colors and labels
    q_colors = ["#dddddd"] * num_qubits
    q_labels = [str(i) for i in range(num_qubits)]

    for virt_idx, phys_idx in layout.items():
        q_colors[phys_idx] = color
        q_labels[phys_idx] = f"q{virt_idx}"

    # Edge colors: highlight edges between mapped qubits
    line_colors = []
    for edge in cmap.get_edges():
        if edge[0] in mapped_physical and edge[1] in mapped_physical:
            line_colors.append(color)
        else:
            line_colors.append("#cccccc")

    qubit_size = 28 if num_qubits > 5 else 20
    if num_qubits > 100:
        qubit_size = 18

    plot_gate_map(
        backend,
        qubit_color=q_colors,
        qubit_labels=q_labels,
        line_color=line_colors,
        qubit_size=qubit_size,
        line_width=3,
        plot_directed=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def visualize_circuit_layouts(
    circuit_name, circuit, backends_dict, model, hw_graphs, cfg, output_dir
):
    """Create a grid of layout visualizations: methods × backends."""
    n_methods = len(METHODS)
    n_backends = len(backends_dict)

    fig, axes = plt.subplots(
        n_methods, n_backends,
        figsize=(7 * n_backends, 6 * n_methods),
    )
    if n_backends == 1:
        axes = axes.reshape(-1, 1)
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    for col, (backend_name, backend) in enumerate(backends_dict.items()):
        for row, (label, layout_method, routing_method, uses_model) in enumerate(METHODS):
            ax = axes[row, col]
            try:
                if uses_model:
                    layout = get_model_layout(
                        model, circuit, backend, hw_graphs[backend_name], cfg
                    )
                else:
                    layout = get_layout_from_transpiled(
                        circuit, backend, layout_method, routing_method
                    )

                title = f"{label} on {backend_name} ({backend.num_qubits}Q)"
                mapping_str = ", ".join(f"q{v}→{p}" for v, p in sorted(layout.items()))
                title += f"\n[{mapping_str}]"

                plot_layout_on_backend(
                    backend, layout, ax, title,
                    color=METHOD_COLORS[label],
                )
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="red")
                ax.set_title(f"{label} on {backend_name}", fontsize=12)

    fig.suptitle(
        f"Initial Layout Comparison — {circuit_name} ({circuit.num_qubits}Q)",
        fontsize=16, fontweight="bold", y=1.0,
    )
    fig.tight_layout()

    out_path = output_dir / f"layout_{circuit_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize initial layouts on backend topology")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument(
        "--backend", nargs="+", default=list(BACKENDS.keys()),
        choices=list(BACKENDS.keys()),
        help="Test backends (default: all)",
    )
    parser.add_argument(
        "--circuit", nargs="+", default=None,
        help="Circuit names (default: all benchmark circuits)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ckpt_path = Path(args.checkpoint)
        run_dir = ckpt_path.parent.parent
        run_name = run_dir.name
        eval_dir = run_dir.parent.parent / "eval" / run_name
        output_dir = eval_dir / "layout_plots"
    # Load config and model (before creating output dir to avoid empty dirs on failure)
    cfg = load_config(args.config)
    model = GraphQMap.from_config(cfg)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load backends
    backends_dict = {}
    hw_graphs = {}
    for name in args.backend:
        print(f"Loading backend: {name}")
        backend = get_backend(name)
        backends_dict[name] = backend
        hw_graphs[name] = build_hardware_graph(backend)

    # Load circuits
    circuit_names = args.circuit or BENCHMARK_CIRCUITS
    print(f"\nVisualizing {len(circuit_names)} circuits × {len(backends_dict)} backends × {len(METHODS)} methods\n")

    for cname in circuit_names:
        print(f"[{cname}]")
        try:
            circuit = load_benchmark_circuit(cname, measure=True)
        except Exception as e:
            print(f"  Skip (load failed): {e}")
            continue

        visualize_circuit_layouts(
            cname, circuit, backends_dict, model, hw_graphs, cfg, output_dir
        )

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
