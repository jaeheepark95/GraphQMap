"""Visualize GraphQMap training metrics and evaluation results.

Usage:
    # Training metrics (single run)
    python scripts/visualize.py runs/train/20260323_025733_baseline_v1

    # Compare multiple training runs
    python scripts/visualize.py runs/train/RUN_A runs/train/RUN_B

    # Evaluation results
    python scripts/visualize.py --eval results/eval_toronto.csv

    # Both training + evaluation
    python scripts/visualize.py runs/train/RUN --eval results/eval_toronto.csv

    # Save plots without showing (for headless servers)
    python scripts/visualize.py runs/RUN --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training(run_dirs: list[Path], save_dir: Path | None = None) -> list[plt.Figure]:
    """Plot training metrics: loss components and Val PST."""
    figures = []

    # Detect if any run has PST data
    has_pst = False
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.csv"
        if metrics_path.exists():
            df_check = pd.read_csv(metrics_path)
            if "val_pst" in df_check.columns and df_check["val_pst"].notna().any():
                has_pst = True
                break

    ncols = 3 if has_pst else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    fig.suptitle("Training", fontsize=14)

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            print(f"  Skip (no metrics.csv): {run_dir}")
            continue

        df = pd.read_csv(metrics_path)
        label = run_dir.name

        # Loss components: L_total + all active components
        ax = axes[0]
        ax.plot(df["epoch"], df["l_total"], label=f"{label} L_total", linewidth=2)
        # Plot all loss component columns (everything except metadata columns)
        skip_cols = {"epoch", "tau", "lr", "l_total", "val_pst"}
        component_cols = [c for c in df.columns if c not in skip_cols]
        for i, col in enumerate(component_cols):
            ax.plot(df["epoch"], df[col], label=f"{label} {col}", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Components")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Second panel: individual component detail (use first component if available)
        ax = axes[1]
        if component_cols:
            for i, col in enumerate(component_cols):
                ax.plot(df["epoch"], df[col], label=f"{label} {col}", color=f"C{i+2}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Individual Components")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Val PST
        if has_pst and "val_pst" in df.columns:
            pst_df = df.dropna(subset=["val_pst"])
            if not pst_df.empty:
                ax = axes[2]
                ax.plot(pst_df["epoch"], pst_df["val_pst"], marker="o",
                        markersize=4, label=label)
                best_idx = pst_df["val_pst"].idxmax()
                ax.axhline(y=pst_df.loc[best_idx, "val_pst"], color="gray",
                           linestyle="--", alpha=0.5)
                ax.annotate(
                    f"best={pst_df.loc[best_idx, 'val_pst']:.4f}",
                    xy=(pst_df.loc[best_idx, "epoch"], pst_df.loc[best_idx, "val_pst"]),
                    fontsize=8, ha="left",
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("PST")
                ax.set_title("Validation PST")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

    fig.tight_layout()
    figures.append(fig)

    if save_dir:
        fig.savefig(save_dir / "training.png", dpi=150, bbox_inches="tight")

    return figures


def plot_eval(
    csv_path: Path,
    save_dir: Path | None = None,
    backend_label: str | None = None,
) -> list[plt.Figure]:
    """Plot evaluation results: PST bar chart + per-circuit heatmap.

    Args:
        csv_path: Path to evaluation CSV.
        save_dir: Directory to save PNG files.
        backend_label: Backend name for file naming (e.g. 'toronto').
            If None, uses the backend column from CSV.
    """
    df = pd.read_csv(csv_path)
    figures = []

    title_backend = backend_label or df["backend"].iloc[0]
    file_suffix = f"_{backend_label}" if backend_label else ""

    # Filter by backend if specified and column exists
    if backend_label and "backend" in df.columns:
        df = df[df["backend"] == backend_label]
        if df.empty:
            return figures

    # --- PST Bar Chart (mean per method) ---
    agg = df.groupby("method")["pst"].agg(["mean", "std"]).reset_index()
    # Sort: baselines first, then model variants
    method_order = ["SABRE", "NASSC", "QAP+SABRE", "QAP+NASSC", "OURS+SABRE", "OURS+NASSC"]
    present = [m for m in method_order if m in agg["method"].values]
    extra = [m for m in agg["method"].values if m not in method_order]
    order = present + extra
    agg = agg.set_index("method").loc[order].reset_index()

    colors = ["#aaaaaa"] * len(agg)
    for i, m in enumerate(agg["method"]):
        if m.startswith("OURS"):
            colors[i] = "#4c72b0"

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agg["method"], agg["mean"], yerr=agg["std"], capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("PST (mean)")
    ax.set_title(f"PST Comparison — {title_backend}")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, agg["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    figures.append(fig)

    if save_dir:
        fig.savefig(save_dir / f"pst_comparison{file_suffix}.png", dpi=150, bbox_inches="tight")

    return figures


def plot_pst_table(
    csv_path: Path,
    save_dir: Path | None = None,
) -> plt.Figure | None:
    """Render a per-circuit PST summary table as a PNG image.

    Produces a single table with backends as column groups, methods as
    sub-columns, and circuits as rows.  Per-row and per-backend-avg best
    values are rendered in bold.

    Args:
        csv_path: Path to eval_results.csv.
        save_dir: Directory to save the PNG.

    Returns:
        The matplotlib Figure, or None if data is insufficient.
    """
    raw_df = pd.read_csv(csv_path)
    if raw_df.empty:
        return None

    # Method display order
    method_order = ["SABRE", "NASSC", "QAP+SABRE", "QAP+NASSC", "OURS+SABRE", "OURS+NASSC"]
    present_methods = [m for m in method_order if m in raw_df["method"].unique()]
    if not present_methods:
        return None

    # Average across reps, pivot
    agg = raw_df.groupby(["backend", "circuit", "method"])["pst"].mean().reset_index()
    backends = sorted(agg["backend"].unique())
    circuits = sorted(agg["circuit"].unique())

    # Backend display names with qubit count
    backend_sizes = {"toronto": 27, "rochester": 53, "washington": 127,
                     "brooklyn": 65, "torino": 133, "mumbai": 27, "manhattan": 65}
    backend_labels = {}
    for b in backends:
        size = backend_sizes.get(b, "")
        backend_labels[b] = f"{b.capitalize()} ({size}q)" if size else b.capitalize()

    # Short method labels for compactness
    short_method = {"SABRE": "SABRE", "NASSC": "NASSC",
                    "QAP+SABRE": "QAP+SABRE", "QAP+NASSC": "QAP+NASSC",
                    "OURS+SABRE": "Ours+SABRE", "OURS+NASSC": "Ours+NASSC"}

    # Build data matrix: rows = circuits + AVG, cols = backend × method
    n_rows = len(circuits) + 1  # +1 for AVG
    n_cols = len(backends) * len(present_methods)
    cell_text = []
    cell_colors = []
    raw_vals = np.full((n_rows, n_cols), np.nan)

    for r, circ in enumerate(circuits):
        row_text = []
        row_colors = []
        for bi, backend in enumerate(backends):
            for mi, method in enumerate(present_methods):
                col_idx = bi * len(present_methods) + mi
                subset = agg[(agg["backend"] == backend) & (agg["circuit"] == circ) & (agg["method"] == method)]
                if not subset.empty and not np.isnan(subset["pst"].values[0]):
                    val = subset["pst"].values[0]
                    raw_vals[r, col_idx] = val
                    row_text.append(f"{val * 100:.1f}")
                else:
                    row_text.append("-")
                row_colors.append("white")
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    # AVG row
    avg_text = []
    avg_colors = []
    for ci in range(n_cols):
        col_vals = raw_vals[:-1, ci]
        valid = col_vals[~np.isnan(col_vals)]
        if len(valid) > 0:
            avg_val = np.mean(valid)
            raw_vals[-1, ci] = avg_val
            avg_text.append(f"{avg_val * 100:.1f}")
        else:
            avg_text.append("-")
        avg_colors.append("#e8e8e8")
    cell_text.append(avg_text)
    cell_colors.append(avg_colors)

    # Determine best per row (bold) — find max across all methods for each row
    is_best = np.zeros_like(raw_vals, dtype=bool)
    for r in range(n_rows):
        row = raw_vals[r]
        if np.all(np.isnan(row)):
            continue
        max_val = np.nanmax(row)
        # Bold all cells equal to max (ties)
        is_best[r] = (row == max_val) & ~np.isnan(row)

    # Also mark best per backend-group for AVG row
    avg_r = n_rows - 1
    for bi, backend in enumerate(backends):
        start = bi * len(present_methods)
        end = start + len(present_methods)
        group = raw_vals[avg_r, start:end]
        if np.all(np.isnan(group)):
            continue
        max_val = np.nanmax(group)
        for ci in range(start, end):
            if raw_vals[avg_r, ci] == max_val and not np.isnan(raw_vals[avg_r, ci]):
                is_best[avg_r, ci] = True

    # Apply bold formatting via fontweight on cell objects (done after table creation)
    bold_cells: list[tuple[int, int]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if is_best[r, c] and cell_text[r][c] != "-":
                bold_cells.append((r, c))

    # Build column labels
    col_labels = []
    for backend in backends:
        for method in present_methods:
            col_labels.append(short_method.get(method, method))

    row_labels = circuits + ["Average"]

    # --- Render table ---
    col_width = 1.1
    row_height = 0.38
    fig_width = 1.5 + n_cols * col_width  # left margin for circuit names
    fig_height = 1.2 + n_rows * row_height

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#f0f0f0"] * n_rows,
        colColours=["#d9e2f3"] * n_cols,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Apply bold to best-in-row cells
    for r, c in bold_cells:
        table[r + 1, c].set_text_props(fontweight="bold")

    # Style header row
    for ci in range(n_cols):
        cell = table[0, ci]
        cell.set_text_props(fontweight="bold", fontsize=8)

    # Style row labels
    for r in range(n_rows):
        cell = table[r + 1, -1]
        cell.set_text_props(fontsize=8)
    # Bold the AVG row label
    table[n_rows, -1].set_text_props(fontweight="bold", fontsize=8)

    # Add backend group headers above method columns
    # Render first so cell bboxes are available in display coords
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for bi, backend in enumerate(backends):
        start_col = bi * len(present_methods)
        end_col = start_col + len(present_methods) - 1
        bbox_s = table[0, start_col].get_window_extent(renderer)
        bbox_e = table[0, end_col].get_window_extent(renderer)
        # Convert display coords → axes coords
        inv = ax.transAxes.inverted()
        x0_ax = inv.transform((bbox_s.x0, bbox_s.y1))[0]
        x1_ax = inv.transform((bbox_e.x1, bbox_e.y1))[0]
        y_ax = inv.transform((bbox_s.x0, bbox_s.y1))[1]
        ax.text(
            (x0_ax + x1_ax) / 2, y_ax + 0.02,
            backend_labels[backend],
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Separator lines between backend groups
    for bi in range(1, len(backends)):
        col_idx = bi * len(present_methods)
        for r in range(n_rows + 1):  # +1 for header
            cell = table[r, col_idx] if col_idx < n_cols else None
            if cell is not None:
                cell.set_edgecolor("black")
                cell.visible_edges = "BRTL" if r == 0 else "BRL"

    fig.suptitle("PST Evaluation Results (×100)", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        fig.savefig(save_dir / "pst_table.png", dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GraphQMap experiments")
    parser.add_argument(
        "run_dirs", nargs="*", type=Path,
        help="Run directories containing metrics.csv",
    )
    parser.add_argument(
        "--eval", type=Path, default=None, dest="eval_csv",
        help="Evaluation results CSV",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save plots without displaying (headless mode)",
    )
    parser.add_argument(
        "--save-dir", type=Path, default=None,
        help="Directory to save plots (default: first run_dir/plots/)",
    )
    args = parser.parse_args()

    if not args.run_dirs and not args.eval_csv:
        parser.error("Provide at least one run directory or --eval CSV")

    # Explicit --save-dir overrides per-stage defaults
    save_dir = args.save_dir
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    all_figures = []

    # Training plots
    if args.run_dirs:
        valid_dirs = [d for d in args.run_dirs if (d / "metrics.csv").exists()]
        missing = [d for d in args.run_dirs if not (d / "metrics.csv").exists()]
        if valid_dirs:
            s_save = save_dir if save_dir else valid_dirs[0] / "plots"
            s_save.mkdir(parents=True, exist_ok=True)
            all_figures.extend(plot_training(valid_dirs, s_save))
            print(f"Plots saved to {s_save}/")
        for d in missing:
            print(f"  Warning: no metrics.csv found in {d}")

    # Evaluation plots
    if args.eval_csv:
        if not args.eval_csv.exists():
            print(f"  Error: {args.eval_csv} not found")
        else:
            all_figures.extend(plot_eval(args.eval_csv, save_dir))

    if save_dir and args.eval_csv and all_figures:
        print(f"Eval plots saved to {save_dir}/")

    if not args.no_show and all_figures:
        plt.show()


if __name__ == "__main__":
    main()
