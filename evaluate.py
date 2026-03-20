"""GraphQMap evaluation entry point.

Usage:
    python evaluate.py --config configs/stage2.yaml --backend toronto --scenario single
    python evaluate.py --config configs/stage2.yaml --backend brooklyn --scenario dual --reps 5
    python evaluate.py --config configs/stage2.yaml --backend toronto --max-circuits 20 --reps 3
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import torch
from torch_geometric.data import Batch

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph, load_circuit
from data.hardware_graph import build_hardware_graph, get_backend
from evaluation.baselines import evaluate_baseline
from evaluation.metrics import (
    EvalResult,
    aggregate_results,
    evaluate_model_on_circuit,
    format_results_table,
)
from models.graphqmap import GraphQMap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SCENARIOS = {"single": 1, "dual": 2, "quad": 4}


def _collect_test_circuits(
    data_root: str,
    max_circuits: int | None = None,
    max_qubits: int | None = None,
) -> list[tuple[str, Path]]:
    """Collect test circuit paths from all datasets.

    Args:
        data_root: Root of data/circuits/ directory.
        max_circuits: Maximum number of circuits to use.
        max_qubits: Maximum qubit count filter.

    Returns:
        List of (circuit_name, path) tuples.
    """
    import re

    qasm_root = Path(data_root) / "qasm"
    circuits = []

    for dataset_dir in sorted(qasm_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for qasm_file in sorted(dataset_dir.glob("*.qasm")):
            # Quick qubit count check from file
            with open(qasm_file) as f:
                content = f.read(500)
            qregs = re.findall(r"qreg\s+\w+\[(\d+)\]", content)
            if not qregs:
                continue
            num_qubits = sum(int(q) for q in qregs)
            if num_qubits < 2:
                continue
            if max_qubits and num_qubits > max_qubits:
                continue
            name = f"{dataset_dir.name}/{qasm_file.stem}"
            circuits.append((name, qasm_file))

    if max_circuits and len(circuits) > max_circuits:
        # Sample evenly across datasets
        import random
        random.seed(42)
        random.shuffle(circuits)
        circuits = circuits[:max_circuits]

    return circuits


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphQMap Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backend", type=str, required=True,
                        help="Backend name (e.g. toronto, brooklyn, torino)")
    parser.add_argument("--scenario", type=str, default="single",
                        choices=["single", "dual", "quad"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--reps", type=int, default=1,
                        help="Repetitions per circuit for statistical reliability")
    parser.add_argument("--max-circuits", type=int, default=50,
                        help="Max circuits to evaluate")
    parser.add_argument("--opt-level", type=int, default=3,
                        help="Qiskit transpiler optimization level")
    parser.add_argument("--baselines", nargs="*", default=["sabre", "dense", "trivial"])
    parser.add_argument("--data-root", type=str, default="data/circuits")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")

    logger.info("=== GraphQMap Evaluation ===")
    logger.info("Backend: %s | Scenario: %s | Reps: %d", args.backend, args.scenario, args.reps)

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

    # Collect test circuits
    test_circuits = _collect_test_circuits(
        args.data_root,
        max_circuits=args.max_circuits,
        max_qubits=num_physical,
    )
    logger.info("Test circuits: %d", len(test_circuits))

    if not test_circuits:
        logger.error("No test circuits found!")
        return

    # Evaluate
    all_results: list[EvalResult] = []
    tau = getattr(cfg.sinkhorn, "tau", 0.05)

    for i, (circuit_name, qasm_path) in enumerate(test_circuits):
        logger.info("[%d/%d] %s", i + 1, len(test_circuits), circuit_name)

        try:
            circuit = load_circuit(qasm_path)
        except Exception as e:
            logger.warning("  Skip (load failed): %s", e)
            continue

        num_logical = circuit.num_qubits
        if num_logical > num_physical or num_logical < 2:
            continue

        circuit_graph = build_circuit_graph(circuit)

        # Batch for single inference
        circuit_batch = Batch.from_data_list([circuit_graph])
        hw_batch = Batch.from_data_list([hw_graph])

        # Model evaluation
        model_result = evaluate_model_on_circuit(
            model=model,
            circuit=circuit,
            circuit_graph=circuit_batch,
            hardware_graph=hw_batch,
            backend=backend,
            circuit_name=circuit_name,
            backend_name=args.backend,
            num_logical=num_logical,
            num_physical=num_physical,
            tau=tau,
            shots=args.shots,
            num_repetitions=args.reps,
        )
        all_results.append(model_result)
        logger.info("  GraphQMap PST: %.4f", model_result.pst_mean)

        # Baselines
        for method in args.baselines:
            baseline_results = EvalResult(
                circuit_name=circuit_name,
                backend_name=args.backend,
                method=method,
            )
            for rep in range(args.reps):
                try:
                    br = evaluate_baseline(
                        circuit, backend, method=method,
                        shots=args.shots, seed=rep,
                        optimization_level=args.opt_level,
                    )
                    baseline_results.pst_values.append(br["pst"])
                    baseline_results.swap_counts.append(br["swap_count"])
                    baseline_results.depths.append(br["depth"])
                except Exception as e:
                    logger.warning("  Baseline %s rep %d failed: %s", method, rep, e)

            if baseline_results.pst_values:
                all_results.append(baseline_results)
                logger.info("  %s PST: %.4f", method, baseline_results.pst_mean)

    # Report
    logger.info("\n" + format_results_table(all_results))

    # Aggregate by method
    methods = set(r.method for r in all_results)
    logger.info("\n=== Aggregated Results ===")
    for method in sorted(methods):
        method_results = [r for r in all_results if r.method == method]
        agg = aggregate_results(method_results)
        logger.info(
            "%s: PST=%.4f±%.4f | SWAP=%.1f±%.1f | Depth=%.1f±%.1f | n=%d",
            method,
            agg["pst_mean"], agg["pst_std"],
            agg["swap_mean"], agg["swap_std"],
            agg["depth_mean"], agg["depth_std"],
            agg["num_circuits"],
        )


if __name__ == "__main__":
    main()
