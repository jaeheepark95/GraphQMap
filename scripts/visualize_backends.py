"""Visualize test backend topologies with error maps.

Plots gate map with readout error (node color) and 2Q gate error (edge color)
for the 3 unseen test backends: FakeToronto (27Q), FakeBrooklyn (65Q), FakeTorino (133Q).

Usage:
    python scripts/visualize_backends.py
    python scripts/visualize_backends.py --backend toronto brooklyn
    python scripts/visualize_backends.py --no-show          # save only
    python scripts/visualize_backends.py --output-dir /tmp   # custom output dir
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colorbar
from matplotlib import gridspec, ticker
import numpy as np
import seaborn as sns
from qiskit.visualization.gate_map import plot_gate_map

from qiskit_ibm_runtime.fake_provider import FakeTorontoV2, FakeBrooklynV2, FakeTorino


BACKENDS = {
    "toronto": FakeTorontoV2,
    "brooklyn": FakeBrooklynV2,
    "torino": FakeTorino,
}


def plot_error_map(backend, figsize=(15, 12), qubit_coordinates=None):
    """Plot error map for a BackendV2 backend.

    Shows topology with readout error rate (node color) and
    2-qubit gate error rate (edge color).

    Based on references/colleague/mqm/_utils/_visualization.py
    """
    color_map = sns.cubehelix_palette(reverse=True, as_cmap=True)

    backend_name = backend.name
    num_qubits = backend.num_qubits
    cmap = backend.coupling_map

    two_q_error_map = {}
    single_gate_errors = [0.0] * num_qubits
    read_err = [0.0] * num_qubits

    for gate, prop_dict in backend.target.items():
        if prop_dict is None or None in prop_dict:
            continue
        for qargs, inst_props in prop_dict.items():
            if inst_props is None:
                continue
            if gate == "measure":
                if inst_props.error is not None:
                    read_err[qargs[0]] = inst_props.error
            elif len(qargs) == 1:
                if inst_props.error is not None:
                    single_gate_errors[qargs[0]] = max(
                        single_gate_errors[qargs[0]], inst_props.error
                    )
            elif len(qargs) == 2:
                if inst_props.error is not None:
                    two_q_error_map[qargs] = max(
                        two_q_error_map.get(qargs, 0), inst_props.error
                    )

    # CX/ECR errors per coupling edge
    cx_errors = []
    for line in cmap.get_edges():
        err = two_q_error_map.get(tuple(line), 0)
        cx_errors.append(err)

    # Convert to percent
    read_err_pct = 100 * np.asarray(read_err)
    cx_errors_pct = 100 * np.asarray(cx_errors)

    avg_read_err = np.mean(read_err_pct)
    avg_cx_err = np.mean(cx_errors_pct)

    # Node colors from readout error
    read_norm = matplotlib.colors.Normalize(
        vmin=min(read_err_pct), vmax=max(read_err_pct)
    )
    q_colors = [matplotlib.colors.to_hex(color_map(read_norm(e))) for e in read_err_pct]

    # Edge colors from 2Q gate error
    cx_norm = matplotlib.colors.Normalize(
        vmin=min(cx_errors_pct), vmax=max(cx_errors_pct)
    )
    line_colors = [matplotlib.colors.to_hex(color_map(cx_norm(e))) for e in cx_errors_pct]

    # Layout
    fig = plt.figure(figsize=figsize)
    grid_spec = gridspec.GridSpec(
        12, 12,
        height_ratios=[1] * 11 + [0.5],
        width_ratios=[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    )

    main_ax = plt.subplot(grid_spec[:11, 1:11])
    bleft_ax = plt.subplot(grid_spec[-1, :5])
    bright_ax = plt.subplot(grid_spec[-1, 7:])

    qubit_size = 28 if num_qubits > 5 else 20
    plot_gate_map(
        backend,
        qubit_color=q_colors,
        line_color=line_colors,
        qubit_size=qubit_size,
        line_width=5,
        plot_directed=False,
        ax=main_ax,
        qubit_coordinates=qubit_coordinates,
    )
    main_ax.axis("off")
    main_ax.set_aspect(1)

    # Readout error colorbar
    single_cb = matplotlib.colorbar.ColorbarBase(
        bleft_ax, cmap=color_map, norm=read_norm, orientation="horizontal"
    )
    single_cb.locator = ticker.MaxNLocator(nbins=5)
    single_cb.update_ticks()
    bleft_ax.set_title(f"Readout error rate (%) [Avg. = {avg_read_err:.3f}]")

    # 2Q gate error colorbar
    cx_cb = matplotlib.colorbar.ColorbarBase(
        bright_ax, cmap=color_map, norm=cx_norm, orientation="horizontal"
    )
    cx_cb.locator = ticker.MaxNLocator(nbins=5)
    cx_cb.update_ticks()
    bright_ax.set_title(f"2Q gate error rate (%) [Avg. = {avg_cx_err:.3f}]")

    fig.suptitle(f"{backend_name} ({num_qubits}Q) Error Map", fontsize=20, y=0.95)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize test backend error maps")
    parser.add_argument(
        "--backend", nargs="+", default=list(BACKENDS.keys()),
        choices=list(BACKENDS.keys()),
        help="Backends to visualize (default: all test backends)",
    )
    parser.add_argument("--no-show", action="store_true", help="Save only, don't display")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for PNGs")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).resolve().parent.parent / "runs" / "backend_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.backend:
        print(f"Plotting {name}...")
        backend = BACKENDS[name]()
        fig = plot_error_map(backend)
        out_path = out_dir / f"error_map_{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to {out_path}")
        if not args.no_show:
            plt.show()
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
