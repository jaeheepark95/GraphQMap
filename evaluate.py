"""GraphQMap evaluation entry point.

Usage:
    # Model evaluation with baselines (auto-saves to runs/eval/<run_name>/)
    python evaluate.py --config configs/stage2.yaml --checkpoint runs/stage2/<RUN>/checkpoints/best.pt --backend toronto brooklyn torino --reps 3

    # Custom output path
    python evaluate.py --config configs/stage2.yaml --checkpoint runs/stage2/<RUN>/checkpoints/best.pt --backend toronto --reps 3 --output runs/eval/custom/eval_results.csv

    # Benchmark mode (baselines only, no model)
    python evaluate.py --benchmark --backend toronto brooklyn torino
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Batch

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph, extract_circuit_features
from data.hardware_graph import (
    build_hardware_graph,
    configure_hw_features,
    get_backend,
    precompute_c_eff,
)
from evaluation.benchmark import (
    BENCHMARK_CIRCUIT_DIR,
    BENCHMARK_CIRCUITS,
    execute_on_simulators,
    load_benchmark_circuit,
    run_benchmark_single,
)
from evaluation.metrics import (
    EvalResult,
    aggregate_results,
)
from evaluation.pst import (
    create_ideal_simulator,
    create_noisy_simulator,
)
from evaluation.transpiler import transpile_with_timing
from models.graphqmap import GraphQMap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Baseline methods: (routing_method, layout_method, label)
BASELINE_COMBOS = [
    ("sabre", "sabre", "SABRE"),
    ("nassc", "sabre", "NASSC"),
    ("sabre", "qap", "QAP+SABRE"),
    ("nassc", "qap", "QAP+NASSC"),
]

# Model routing variants: (routing_method, label)
MODEL_ROUTING_VARIANTS = [
    ("sabre", "OURS+SABRE"),
    ("nassc", "OURS+NASSC"),
]


def _save_pst_tables(raw_df: pd.DataFrame, eval_dir: Path) -> None:
    """Save PST summary tables as CSV and Markdown."""
    method_order = [label for _, _, label in BASELINE_COMBOS] + [label for _, label in MODEL_ROUTING_VARIANTS]

    # Average across reps
    pivot = raw_df.groupby(["backend", "circuit", "method"])["pst"].mean().reset_index()
    table = pivot.pivot_table(index=["backend", "circuit"], columns="method", values="pst")
    table = table[[m for m in method_order if m in table.columns]]

    # Build table with per-backend averages
    backends = sorted(table.index.get_level_values(0).unique())
    parts = []
    for backend in backends:
        sub = table.loc[backend]
        avg = pd.DataFrame([sub.mean()], index=pd.MultiIndex.from_tuples([(backend, "AVG")]))
        avg.columns = table.columns
        parts.append(pd.concat([table.loc[[backend]], avg]))
    full_table = pd.concat(parts)

    # Save CSV
    full_table.to_csv(eval_dir / "pst_summary.csv", float_format="%.4f")

    # Save Markdown (per-backend tables + per-circuit detail)
    with open(eval_dir / "pst_summary.md", "w") as f:
        f.write("# PST Evaluation Results\n\n")
        for backend in backends:
            f.write(f"## {backend.capitalize()}\n\n")
            sub = full_table.loc[backend]
            f.write(sub.to_markdown(floatfmt=".4f"))
            f.write("\n\n")

        # Per-circuit comparison across all backends
        f.write("## Per-Circuit Comparison (All Backends)\n\n")
        for backend in backends:
            f.write(f"### {backend.capitalize()}\n\n")
            sub = table.loc[backend]
            f.write("| Circuit |")
            for m in sub.columns:
                f.write(f" {m} |")
            f.write("\n|---------|")
            for _ in sub.columns:
                f.write(":---:|")
            f.write("\n")
            for circuit in sub.index:
                f.write(f"| {circuit} |")
                for m in sub.columns:
                    f.write(f" {sub.loc[circuit, m]:.4f} |")
                f.write("\n")
            avg = sub.mean()
            f.write(f"| **AVG** |")
            for m in sub.columns:
                f.write(f" **{avg[m]:.4f}** |")
            f.write("\n\n")

    logger.info("PST tables saved to %s", eval_dir)


def _safe_simulate(transpiled_circuit, ideal_sim, noisy_sim, shots):
    """Execute simulation with error handling.

    Returns:
        Tuple of (avg_pst, depth, psts) on success,
        or (None, None, None) on failure.
    """
    try:
        return execute_on_simulators(
            transpiled_circuit, ideal_sim, noisy_sim, shots=shots,
        )
    except Exception as e:
        logger.warning("  Simulation failed: %s", e)
        return None, None, None


def evaluate_model(args, cfg) -> None:
    """Evaluate GraphQMap model against baselines on benchmark circuits.

    Evaluation order: baselines first, then model. This prevents GPU state
    corruption from model-generated deep circuits affecting baseline results.
    On simulation failure, simulators are recreated and the run is marked NaN.
    """
    device = torch.device("cpu")

    logger.info("=== GraphQMap Evaluation ===")
    logger.info(
        "Backend: %s | Routing: %s | Reps: %d",
        args.backend, args.routing_method, args.reps,
    )

    # Load backend
    backend = get_backend(args.backend)
    num_physical = backend.target.num_qubits
    hw_graph = build_hardware_graph(backend)
    logger.info("Physical qubits: %d", num_physical)

    # Load model
    model = GraphQMap.from_config(cfg)
    ckpt_path = args.checkpoint or getattr(cfg, "pretrained_checkpoint", None)
    if ckpt_path:
        logger.info("Loading checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.warning("No checkpoint — using randomly initialized model")
    model.to(device)
    model.eval()

    # Create simulators once per backend (reuse for all circuits)
    ideal_sim = create_ideal_simulator(backend)
    noisy_sim = create_noisy_simulator(backend)

    def recreate_simulators():
        nonlocal ideal_sim, noisy_sim
        logger.info("  Recreating simulators after failure...")
        ideal_sim = create_ideal_simulator(backend)
        noisy_sim = create_noisy_simulator(backend)

    # Determine test circuits and model params
    circuit_names = args.circuits or BENCHMARK_CIRCUITS
    tau = getattr(cfg.sinkhorn, "tau_min", getattr(cfg.sinkhorn, "tau", 0.05))

    # Results storage (baselines first, then model — matches evaluation order)
    method_labels = (
        [label for _, _, label in BASELINE_COMBOS]
        + [label for _, label in MODEL_ROUTING_VARIANTS]
    )
    all_pst = {m: [] for m in method_labels}
    all_results: list[EvalResult] = []

    for i, cname in enumerate(circuit_names):
        logger.info("[%d/%d] %s", i + 1, len(circuit_names), cname)

        # Load circuit
        try:
            circuit = load_benchmark_circuit(
                cname, args.circuit_dir, measure=True
            )
        except Exception as e:
            logger.warning("  Skip (load failed): %s", e)
            continue

        num_logical = circuit.num_qubits
        if num_logical > num_physical or num_logical < 2:
            logger.warning("  Skip (qubits: %d > %d)", num_logical, num_physical)
            continue

        # --- Baseline evaluations (run first to avoid GPU state corruption) ---
        for routing, layout, label in BASELINE_COMBOS:
            baseline_result = EvalResult(
                circuit_name=cname,
                backend_name=args.backend,
                method=label,
            )

            for rep in range(args.reps):
                try:
                    tc, metadata = transpile_with_timing(
                        circuit, backend,
                        layout_method=layout,
                        routing_method=routing,
                        seed=args.seed + rep,
                    )
                except Exception as e:
                    logger.warning("  Transpile failed (%s): %s", label, e)
                    baseline_result.pst_values.append(float("nan"))
                    baseline_result.depths.append(0)
                    baseline_result.swap_counts.append(0)
                    continue

                avg_pst, depth, _ = _safe_simulate(
                    tc, ideal_sim, noisy_sim, args.shots,
                )
                if avg_pst is None:
                    recreate_simulators()
                    avg_pst, depth = float("nan"), tc.depth()
                baseline_result.pst_values.append(avg_pst)
                baseline_result.depths.append(depth)
                baseline_result.swap_counts.append(metadata["map_cx"])

            all_results.append(baseline_result)
            all_pst[label].append(baseline_result.pst_mean)
            logger.info("  %s PST: %.4f", label, baseline_result.pst_mean)

        # --- Model evaluation (both routing variants) ---
        node_fnames = getattr(cfg.model.circuit_gnn, "node_features", None)
        rk = getattr(cfg.model.circuit_gnn, "rwpe_k", 0)
        edge_dim = getattr(cfg.model.circuit_gnn, "edge_input_dim", None)
        circuit_graph = build_circuit_graph(
            circuit, node_feature_names=node_fnames, rwpe_k=rk,
            edge_dim=edge_dim,
        )
        circuit_batch = Batch.from_data_list([circuit_graph])
        hw_batch = Batch.from_data_list([hw_graph])

        # Build C_eff and circuit_adj for iterative refinement
        eval_c_eff = None
        eval_circuit_adj = None
        if model.refine_iterations > 0:
            eval_c_eff = torch.tensor(
                precompute_c_eff(backend), dtype=torch.float32,
            ).to(device)
            feats = extract_circuit_features(circuit)
            eval_circuit_adj = torch.zeros(
                num_logical, num_logical, dtype=torch.float32,
            )
            for (ci, cj), w in zip(
                feats["edge_list"],
                feats["edge_features"][:, 0].tolist(),
            ):
                eval_circuit_adj[ci, cj] = w
                eval_circuit_adj[cj, ci] = w
            eval_circuit_adj = eval_circuit_adj.to(device)

        t0 = time.perf_counter()
        layouts = model.predict(
            circuit_batch, hw_batch,
            batch_size=1,
            num_logical=num_logical,
            num_physical=num_physical,
            tau=tau,
            c_eff=eval_c_eff,
            circuit_adj=eval_circuit_adj,
        )
        model_inference_time = time.perf_counter() - t0
        model_layout = list(layouts[0].values())

        for model_routing, model_label in MODEL_ROUTING_VARIANTS:
            model_result = EvalResult(
                circuit_name=cname,
                backend_name=args.backend,
                method=model_label,
            )

            for rep in range(args.reps):
                model_result.inference_times.append(model_inference_time)

                try:
                    tc, metadata = transpile_with_timing(
                        circuit, backend,
                        initial_layout=model_layout,
                        layout_method="given",
                        routing_method=model_routing,
                        seed=args.seed + rep,
                    )
                except Exception as e:
                    logger.warning("  Transpile failed (%s): %s", model_label, e)
                    model_result.pst_values.append(float("nan"))
                    model_result.depths.append(0)
                    model_result.swap_counts.append(0)
                    continue

                avg_pst, depth, _ = _safe_simulate(
                    tc, ideal_sim, noisy_sim, args.shots,
                )
                if avg_pst is None:
                    recreate_simulators()
                    avg_pst, depth = float("nan"), tc.depth()
                model_result.pst_values.append(avg_pst)
                model_result.depths.append(depth)
                model_result.swap_counts.append(metadata["map_cx"])

            all_results.append(model_result)
            all_pst[model_label].append(model_result.pst_mean)
            logger.info("  %s PST: %.4f", model_label, model_result.pst_mean)

    # --- DataFrame Report ---
    evaluated_circuits = [
        r.circuit_name for r in all_results
        if r.method == MODEL_ROUTING_VARIANTS[-1][1]
    ]
    data = {("PST", m): all_pst[m] for m in method_labels}
    df = pd.DataFrame(data, index=evaluated_circuits)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.loc["Avg"] = df.mean(numeric_only=True, axis=0)

    pd.options.display.float_format = "{:.4f}".format
    print(f"\n=== Results for {args.backend} ===")
    print(df)

    # Aggregate summary
    print("\n=== Aggregated ===")
    for method in method_labels:
        method_results = [r for r in all_results if r.method == method]
        agg = aggregate_results(method_results)
        print(
            f"{method}: PST={agg['pst_mean']:.4f}±{agg['pst_std']:.4f} | "
            f"Depth={agg['depth_mean']:.1f} | n={agg['num_circuits']}"
        )

    # Return rows for combined CSV saving
    rows = []
    for r in all_results:
        for i in range(len(r.pst_values)):
            rows.append({
                "circuit": r.circuit_name,
                "backend": r.backend_name,
                "method": r.method,
                "rep": i,
                "pst": r.pst_values[i],
                "depth": r.depths[i] if i < len(r.depths) else "",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphQMap Evaluation")

    # Mode selection
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark mode (baselines only, no model)",
    )

    # Common args
    parser.add_argument(
        "--backend", nargs="+", default=["toronto"],
        help="Backend name(s)",
    )
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--circuits", nargs="*", default=None,
        help="Circuit names (default: standard 8)",
    )
    parser.add_argument(
        "--circuit-dir", default=BENCHMARK_CIRCUIT_DIR,
        help="Directory containing benchmark .qasm files",
    )

    # Model evaluation args
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--routing-method", default="sabre",
        choices=["sabre", "nassc"],
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save evaluation results to CSV (e.g. results/eval_toronto.csv)",
    )

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark_single(
            circuit_names=args.circuits,
            backend_names=args.backend,
            reps=args.reps,
            shots=args.shots,
            circuit_dir=args.circuit_dir,
        )
    else:
        if not args.config:
            parser.error("--config is required for model evaluation")

        cfg = load_config(args.config)

        # Configure HW feature dimensionality (default 6dim; 8dim includes raw t1/t2)
        hw_input_dim = getattr(cfg.model.hardware_gnn, "node_input_dim", 6)
        exclude_degree = getattr(cfg.model.hardware_gnn, "exclude_degree", False)
        configure_hw_features(
            include_t1_t2=(hw_input_dim in (7, 8)),
            exclude_degree=exclude_degree,
        )

        # Collect all results across backends
        all_rows = []
        for backend_name in args.backend:
            args_copy = argparse.Namespace(**vars(args))
            args_copy.backend = backend_name
            # Disable per-backend CSV save; collect results instead
            args_copy.output = None
            rows = evaluate_model(args_copy, cfg)
            if rows:
                all_rows.extend(rows)

        # Determine eval output directory
        if args.output:
            output_path = Path(args.output)
            eval_dir = output_path.parent
        elif args.checkpoint:
            # Auto-derive from checkpoint: runs/stage2/<RUN>/checkpoints/best.pt → runs/eval/<RUN>/
            ckpt_path = Path(args.checkpoint)
            run_name = ckpt_path.parent.parent.name  # e.g. 20260324_091153_rerun_consistency
            eval_dir = Path("runs/eval") / run_name
            output_path = eval_dir / "eval_results.csv"
        else:
            eval_dir = Path("runs/eval/default")
            output_path = eval_dir / "eval_results.csv"

        # Save combined CSV + eval plots + summary tables
        if all_rows:
            eval_dir.mkdir(parents=True, exist_ok=True)
            raw_df = pd.DataFrame(all_rows)
            raw_df.to_csv(output_path, index=False)
            logger.info("Results saved to %s", output_path)

            # Generate PST summary tables (per-backend + combined)
            _save_pst_tables(raw_df, eval_dir)

            # Generate plots
            import matplotlib
            matplotlib.use("Agg")
            from scripts.visualize import plot_eval
            for bname in args.backend:
                plot_eval(output_path, save_dir=eval_dir, backend_label=bname)
            logger.info("Eval results saved to %s", eval_dir)


if __name__ == "__main__":
    main()
