"""Compare GraphQMap model vs SABRE baseline on PST.

Loads a trained model checkpoint (or uses random init), runs inference
on test circuits, measures PST via noisy simulation, and compares
against Qiskit SABRE layout baseline.

Usage:
    # With trained checkpoint
    python scripts/compare_pst.py --checkpoint checkpoints/unsupervised_test/best.pt

    # Random init (sanity check)
    python scripts/compare_pst.py --no-checkpoint

    # Custom settings
    python scripts/compare_pst.py --backend montreal --max-circuits 10 --shots 8192 --opt-level 3
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from qiskit import QuantumCircuit

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph, extract_circuit_features, load_circuit
from data.hardware_graph import build_hardware_graph, get_backend
from evaluation.baselines import evaluate_baseline
from evaluation.pst import measure_pst
from models.graphqmap import GraphQMap
from torch_geometric.data import Batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
# Suppress Qiskit noise
for name in ["backend_converter", "qiskit"]:
    logging.getLogger(name).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def load_test_circuits(
    circuit_dirs: list[Path],
    max_circuits: int,
    max_qubits: int,
) -> list[tuple[str, QuantumCircuit]]:
    """Load circuits suitable for testing."""
    circuits = []
    for cdir in circuit_dirs:
        if not cdir.exists():
            continue
        for f in sorted(cdir.glob("*.qasm")):
            if "transpiled" in f.name:
                continue
            try:
                qc = load_circuit(f)
                if qc.num_qubits > max_qubits:
                    continue
                if qc.num_qubits < 2:
                    continue
                has_2q = any(len(inst.qubits) == 2 for inst in qc.data)
                if not has_2q:
                    continue
                # Skip circuits with mid-circuit measurements (incompatible with statevector)
                has_mid_meas = any(
                    inst.operation.name == "measure"
                    for inst in qc.data[:-qc.num_qubits]  # ignore final measurements
                    if inst.operation.name == "measure"
                )
                if has_mid_meas and qc.num_clbits > 0:
                    continue
                circuits.append((f.name, qc))
            except Exception:
                pass
    random.shuffle(circuits)
    return circuits[:max_circuits]


def model_predict_layout(
    model: GraphQMap,
    circuit: QuantumCircuit,
    backend,
    hw_graph,
    device: torch.device,
    tau: float = 0.05,
) -> list[int]:
    """Get layout from GraphQMap model for a single circuit."""
    circuit_graph = build_circuit_graph(circuit)

    # Batch of 1
    circuit_batch = Batch.from_data_list([circuit_graph])
    hardware_batch = Batch.from_data_list([hw_graph])

    circuit_batch = circuit_batch.to(device)
    hardware_batch = hardware_batch.to(device)

    layouts = model.predict(
        circuit_batch, hardware_batch,
        batch_size=1,
        num_logical=circuit.num_qubits,
        num_physical=backend.target.num_qubits,
        tau=tau,
    )
    # layouts[0] is a dict {logical: physical}
    layout_dict = layouts[0]
    return [layout_dict[i] for i in range(circuit.num_qubits)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GraphQMap vs SABRE on PST")
    parser.add_argument("--config", type=str, default="configs/unsupervised_test.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Use randomly initialized model (sanity check)")
    parser.add_argument("--backend", type=str, default="montreal",
                        help="Backend name for evaluation")
    parser.add_argument("--max-circuits", type=int, default=10,
                        help="Max circuits to evaluate")
    parser.add_argument("--shots", type=int, default=4096,
                        help="Simulation shots per circuit")
    parser.add_argument("--opt-level", type=int, default=3,
                        help="Qiskit transpiler optimization level (0-3)")
    parser.add_argument("--max-qubits", type=int, default=10,
                        help="Max logical qubits per circuit (statevector scales as 2^n)")
    parser.add_argument("--baselines", nargs="*", default=["sabre"],
                        help="Baseline methods to compare")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("GraphQMap vs Baseline PST Comparison")
    logger.info("=" * 70)
    logger.info(f"Backend: {args.backend} | Shots: {args.shots} | Opt Level: {args.opt_level}")
    logger.info(f"Baselines: {args.baselines}")

    # Load backend
    backend = get_backend(args.backend)
    num_physical = backend.target.num_qubits
    hw_graph = build_hardware_graph(backend)
    logger.info(f"Backend {args.backend}: {num_physical}Q")

    # Load model
    model = GraphQMap.from_config(cfg).to(device)
    if args.checkpoint and not args.no_checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.info("Using randomly initialized model")
    model.eval()

    # Load circuits
    max_q = min(args.max_qubits, num_physical - 1)
    circuit_dirs = [Path("data/circuits/queko"), Path("data/circuits/qasmbench")]
    circuits = load_test_circuits(circuit_dirs, args.max_circuits, max_q)
    logger.info(f"Max qubits per circuit: {max_q} (statevector: 2^{max_q} = {2**max_q:,} dims)")
    logger.info(f"Test circuits: {len(circuits)}")

    if not circuits:
        logger.error("No circuits found. Exiting.")
        return

    # Evaluate
    results: list[dict] = []
    logger.info("")
    header = f"{'Circuit':40s} | {'Qubits':>6s} | {'Method':>10s} | {'PST':>8s} | {'SWAPs':>5s} | {'Depth':>5s} | {'Time':>7s}"
    logger.info(header)
    logger.info("-" * len(header))

    for name, qc in circuits:
        row_results = {"circuit": name, "num_qubits": qc.num_qubits}

        # --- GraphQMap model ---
        t0 = time.perf_counter()
        try:
            layout = model_predict_layout(model, qc, backend, hw_graph, device)
            inference_time = time.perf_counter() - t0

            pst_result = measure_pst(
                qc, backend, layout,
                shots=args.shots,
                optimization_level=args.opt_level,
            )
            row_results["model_pst"] = pst_result["pst"]
            row_results["model_swaps"] = pst_result["swap_count"]
            row_results["model_depth"] = pst_result["depth"]
            row_results["model_time"] = inference_time

            logger.info(
                f"{name:40s} | {qc.num_qubits:6d} | {'GraphQMap':>10s} | "
                f"{pst_result['pst']:8.4f} | {pst_result['swap_count']:5d} | "
                f"{pst_result['depth']:5d} | {inference_time:6.3f}s"
            )
        except Exception as e:
            logger.warning(f"{name:40s} | GraphQMap FAILED: {e}")
            row_results["model_pst"] = None

        # --- Baselines ---
        for method in args.baselines:
            t0 = time.perf_counter()
            try:
                bl_result = evaluate_baseline(
                    qc, backend, method=method,
                    shots=args.shots, seed=args.seed,
                    optimization_level=args.opt_level,
                )
                bl_time = time.perf_counter() - t0

                row_results[f"{method}_pst"] = bl_result["pst"]
                row_results[f"{method}_swaps"] = bl_result["swap_count"]
                row_results[f"{method}_depth"] = bl_result["depth"]

                logger.info(
                    f"{' ':40s} | {' ':6s} | {method:>10s} | "
                    f"{bl_result['pst']:8.4f} | {bl_result['swap_count']:5d} | "
                    f"{bl_result['depth']:5d} | {bl_time:6.3f}s"
                )
            except Exception as e:
                logger.warning(f"{' ':40s} | {method} FAILED: {e}")

        results.append(row_results)
        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    model_psts = [r["model_pst"] for r in results if r.get("model_pst") is not None]

    if model_psts:
        logger.info(f"GraphQMap  | PST mean: {np.mean(model_psts):.4f} ± {np.std(model_psts):.4f}")

    for method in args.baselines:
        bl_psts = [r[f"{method}_pst"] for r in results if r.get(f"{method}_pst") is not None]
        if bl_psts:
            logger.info(f"{method:10s} | PST mean: {np.mean(bl_psts):.4f} ± {np.std(bl_psts):.4f}")

    if model_psts:
        for method in args.baselines:
            bl_psts = [r[f"{method}_pst"] for r in results if r.get(f"{method}_pst") is not None]
            if bl_psts and len(bl_psts) == len(model_psts):
                wins = sum(m > b for m, b in zip(model_psts, bl_psts))
                logger.info(f"GraphQMap vs {method}: {wins}/{len(model_psts)} wins")


if __name__ == "__main__":
    main()
