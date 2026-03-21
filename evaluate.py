"""GraphQMap evaluation entry point.

Usage:
    # Model evaluation with baselines
    python evaluate.py --config configs/stage2.yaml --backend toronto --reps 3

    # Specific layout/routing combinations
    python evaluate.py --config configs/stage2.yaml --backend toronto --routing-method nassc

    # Benchmark mode (no model, just baselines comparison)
    python evaluate.py --benchmark --backend toronto brooklyn torino
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph, load_circuit
from data.hardware_graph import build_hardware_graph, get_backend
from evaluation.benchmark import (
    BENCHMARK_CIRCUIT_DIR,
    BENCHMARK_CIRCUITS,
    DEFAULT_BACKENDS,
    execute_on_simulators,
    load_benchmark_circuit,
    run_benchmark_single,
)
from evaluation.metrics import (
    EvalResult,
    aggregate_results,
    format_results_table,
)
from evaluation.pst import (
    create_ideal_simulator,
    create_noisy_simulator,
    measure_pst,
)
from evaluation.transpiler import build_transpiler, transpile_with_timing
from models.graphqmap import GraphQMap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_model(args, cfg) -> None:
    """Evaluate GraphQMap model against baselines on benchmark circuits."""
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

    # Determine test circuits
    circuit_names = args.circuits or BENCHMARK_CIRCUITS
    tau = getattr(cfg.sinkhorn, "tau", 0.05)

    # Baseline methods to compare
    baseline_combos = [
        ("sabre", "sabre", "SABRE"),
        ("nassc", "sabre", "NASSC"),
    ]
    if args.routing_method == "nassc":
        model_label = "OURS+NASSC"
    else:
        model_label = "OURS+SABRE"

    # Results storage
    method_labels = [model_label] + [label for _, _, label in baseline_combos]
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

        # --- Model evaluation ---
        circuit_graph = build_circuit_graph(circuit)
        circuit_batch = Batch.from_data_list([circuit_graph])
        hw_batch = Batch.from_data_list([hw_graph])

        model_result = EvalResult(
            circuit_name=cname,
            backend_name=args.backend,
            method=model_label,
        )

        for rep in range(args.reps):
            import time as time_module

            t0 = time_module.perf_counter()
            layouts = model.predict(
                circuit_batch, hw_batch,
                batch_size=1,
                num_logical=num_logical,
                num_physical=num_physical,
                tau=tau,
            )
            t1 = time_module.perf_counter()

            layout = list(layouts[0].values())
            model_result.inference_times.append(t1 - t0)

            # Transpile with custom transpiler + measure PST
            tc, metadata = transpile_with_timing(
                circuit, backend,
                initial_layout=layout,
                layout_method="given",
                routing_method=args.routing_method,
                seed=args.seed + rep,
            )
            avg_pst, depth, _ = execute_on_simulators(
                tc, ideal_sim, noisy_sim, shots=args.shots,
            )
            model_result.pst_values.append(avg_pst)
            model_result.depths.append(depth)
            model_result.swap_counts.append(metadata["map_cx"])

        all_results.append(model_result)
        all_pst[model_label].append(model_result.pst_mean)
        logger.info("  %s PST: %.4f", model_label, model_result.pst_mean)

        # --- Baseline evaluations ---
        for routing, layout, label in baseline_combos:
            baseline_result = EvalResult(
                circuit_name=cname,
                backend_name=args.backend,
                method=label,
            )

            for rep in range(args.reps):
                pm = build_transpiler(
                    backend,
                    layout_method=layout,
                    routing_method=routing,
                    seed=args.seed + rep,
                )
                tc = pm.run(circuit)
                avg_pst, depth, _ = execute_on_simulators(
                    tc, ideal_sim, noisy_sim, shots=args.shots,
                )
                baseline_result.pst_values.append(avg_pst)
                baseline_result.depths.append(depth)

            all_results.append(baseline_result)
            all_pst[label].append(baseline_result.pst_mean)
            logger.info("  %s PST: %.4f", label, baseline_result.pst_mean)

    # --- DataFrame Report ---
    data = {("PST", m): all_pst[m] for m in method_labels}
    evaluated_circuits = [
        r.circuit_name for r in all_results if r.method == model_label
    ]
    df = pd.DataFrame(data, index=evaluated_circuits)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.loc["Avg"] = df.mean(numeric_only=True, axis=0)

    pd.options.display.float_format = "{:.4f}".format
    print(f"\n=== Results for {args.backend} ===")
    print(df)

    # Aggregate summary
    print("\n=== Aggregated ===")
    methods = set(r.method for r in all_results)
    for method in sorted(methods):
        method_results = [r for r in all_results if r.method == method]
        agg = aggregate_results(method_results)
        print(
            f"{method}: PST={agg['pst_mean']:.4f}±{agg['pst_std']:.4f} | "
            f"Depth={agg['depth_mean']:.1f} | n={agg['num_circuits']}"
        )


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

    args = parser.parse_args()

    if args.benchmark:
        # Benchmark mode: baselines comparison only
        run_benchmark_single(
            circuit_names=args.circuits,
            backend_names=args.backend,
            reps=args.reps,
            shots=args.shots,
            circuit_dir=args.circuit_dir,
        )
    else:
        # Model evaluation mode
        if not args.config:
            parser.error("--config is required for model evaluation")

        cfg = load_config(args.config)

        # Evaluate per backend
        for backend_name in args.backend:
            args_copy = argparse.Namespace(**vars(args))
            args_copy.backend = backend_name
            evaluate_model(args_copy, cfg)


if __name__ == "__main__":
    main()
