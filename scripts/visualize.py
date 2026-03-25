"""Visualize GraphQMap training metrics and evaluation results.

Usage:
    # Training metrics (single run)
    python scripts/visualize.py runs/stage1/20260323_025733_baseline_v1

    # Compare multiple training runs
    python scripts/visualize.py runs/stage1/RUN_A runs/stage1/RUN_B

    # Evaluation results
    python scripts/visualize.py --eval results/eval_toronto.csv

    # Both training + evaluation
    python scripts/visualize.py runs/stage2/RUN --eval results/eval_toronto.csv

    # Save plots without showing (for headless servers)
    python scripts/visualize.py runs/stage1/RUN --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_stage1(run_dirs: list[Path], save_dir: Path | None = None) -> list[plt.Figure]:
    """Plot Stage 1 training metrics: train/val loss."""
    figures = []

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fig.suptitle("Stage 1 Training", fontsize=14)

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            print(f"  Skip (no metrics.csv): {run_dir}")
            continue

        df = pd.read_csv(metrics_path)
        label = run_dir.name

        # Phase boundaries
        phases = df["phase"].unique()
        phase_colors = {"mlqd_queko_best": "C0", "queko_best": "C1"}

        for phase in phases:
            mask = df["phase"] == phase
            color = phase_colors.get(phase, "C2")
            offset = 0 if phase == phases[0] else df[mask].index[0]
            epochs = df.loc[mask, "epoch"] + offset
            ax.plot(epochs, df.loc[mask, "train_loss"], color=color, alpha=0.5,
                    linestyle="--", label=f"{label} train ({phase.split('_')[0]})")
            ax.plot(epochs, df.loc[mask, "val_loss"], color=color,
                    label=f"{label} val ({phase.split('_')[0]})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE Loss")
    ax.set_title("Train / Val Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    figures.append(fig)

    if save_dir:
        fig.savefig(save_dir / "stage1_training.png", dpi=150, bbox_inches="tight")

    return figures


def plot_stage2(run_dirs: list[Path], save_dir: Path | None = None) -> list[plt.Figure]:
    """Plot Stage 2 training metrics: loss components and Val PST."""
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
    fig.suptitle("Stage 2 Training", fontsize=14)

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
        fig.savefig(save_dir / "stage2_training.png", dpi=150, bbox_inches="tight")

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

    # --- Per-circuit Heatmap ---
    circuits = df["circuit"].unique()
    methods = order

    if len(circuits) >= 2:
        pivot = df.groupby(["circuit", "method"])["pst"].mean().unstack(fill_value=0)
        pivot = pivot.reindex(columns=[m for m in methods if m in pivot.columns])

        fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), max(4, len(circuits) * 0.5)))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val < 0.4 else "black")

        plt.colorbar(im, ax=ax, label="PST")
        ax.set_title(f"Per-Circuit PST — {title_backend}")
        fig.tight_layout()
        figures.append(fig)

        if save_dir:
            fig.savefig(save_dir / f"pst_heatmap{file_suffix}.png", dpi=150, bbox_inches="tight")

    return figures


def detect_stage(run_dir: Path) -> int | None:
    """Detect stage from metrics CSV columns."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path, nrows=0)
    if "phase" in df.columns:
        return 1
    if "l_total" in df.columns:
        return 2
    return None


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
        stage1_dirs = [d for d in args.run_dirs if detect_stage(d) == 1]
        stage2_dirs = [d for d in args.run_dirs if detect_stage(d) == 2]

        if stage1_dirs:
            s1_save = save_dir if save_dir else stage1_dirs[0] / "plots"
            s1_save.mkdir(parents=True, exist_ok=True)
            all_figures.extend(plot_stage1(stage1_dirs, s1_save))
            print(f"Plots saved to {s1_save}/")
        if stage2_dirs:
            s2_save = save_dir if save_dir else stage2_dirs[0] / "plots"
            s2_save.mkdir(parents=True, exist_ok=True)
            all_figures.extend(plot_stage2(stage2_dirs, s2_save))
            print(f"Plots saved to {s2_save}/")

        undetected = [d for d in args.run_dirs if detect_stage(d) is None]
        for d in undetected:
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
