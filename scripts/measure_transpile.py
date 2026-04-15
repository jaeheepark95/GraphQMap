"""Measure transpilation depth and additional CNOT count (no simulation).

Metrics:
  - Depth: final circuit depth after full transpilation
  - Additional CNOT: (final 2Q gates) - (original 2Q gates), i.e. routing overhead

Usage:
    python scripts/measure_transpile.py --config configs/base.yaml \
      --checkpoint runs/stage2/<RUN>/checkpoints/best.pt \
      --backend torino --reps 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from qiskit import QuantumCircuit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph
from data.hardware_graph import get_backend, build_hardware_graph, get_hw_node_features
from evaluation.transpiler import transpile_with_timing
from models.graphqmap import GraphQMap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path("data/circuits/qasm/benchmarks")

BASELINE_COMBOS = [
    ("sabre", "sabre", "SABRE"),
    ("sabre", "nassc", "NASSC"),
    ("qap", "sabre", "QAP+SABRE"),
    ("qap", "nassc", "QAP+NASSC"),
]

MODEL_ROUTING = [
    ("sabre", "OURS+SABRE"),
    ("nassc", "OURS+NASSC"),
]

TWO_QUBIT_GATES = ("cx", "ecr", "cz")


def count_2q_gates(circuit: QuantumCircuit) -> int:
    """Count 2-qubit gates in a circuit."""
    return sum(1 for inst in circuit.data if inst.operation.name in TWO_QUBIT_GATES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", required=True, nargs="+")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = GraphQMap.from_config(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    tau = getattr(cfg.sinkhorn, "tau_min", getattr(cfg.sinkhorn, "tau", 0.05))

    circuit_files = sorted(BENCHMARK_DIR.glob("*.qasm"))
    all_rows = []

    for backend_name in args.backend:
        backend = get_backend(backend_name)
        hw_graph = build_hardware_graph(backend).to(device)
        hw_feats = torch.tensor(get_hw_node_features(backend), dtype=torch.float32).to(device)
        num_physical = backend.target.num_qubits
        logger.info("Backend: %s (%dQ)", backend_name, num_physical)

        for qasm_path in circuit_files:
            cname = qasm_path.stem
            circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
            num_logical = circuit.num_qubits
            orig_2q = count_2q_gates(circuit)

            if num_logical > num_physical:
                logger.warning("Skip %s: %d > %d qubits", cname, num_logical, num_physical)
                continue

            logger.info("  %s (%dQ, orig_cx=%d)", cname, num_logical, orig_2q)

            # --- Baselines ---
            for layout_method, routing_method, label in BASELINE_COMBOS:
                for rep in range(args.reps):
                    try:
                        tc, metadata = transpile_with_timing(
                            circuit, backend,
                            layout_method=layout_method,
                            routing_method=routing_method,
                            seed=args.seed + rep,
                        )
                        final_2q = count_2q_gates(tc)
                        all_rows.append({
                            "backend": backend_name, "circuit": cname,
                            "qubits": num_logical, "orig_cx": orig_2q,
                            "method": label, "rep": rep,
                            "depth": tc.depth(), "add_cx": final_2q - orig_2q,
                        })
                    except Exception as e:
                        logger.warning("    %s rep%d failed: %s", label, rep, e)

            # --- Model ---
            circuit_graph = build_circuit_graph(circuit).to(device)
            with torch.no_grad():
                layouts = model.predict(
                    circuit_graph, hw_graph,
                    batch_size=1, num_logical=num_logical, num_physical=num_physical,
                    tau=tau, hw_node_features=hw_feats,
                )
            model_layout = list(layouts[0].values())

            for routing_method, label in MODEL_ROUTING:
                for rep in range(args.reps):
                    try:
                        tc, metadata = transpile_with_timing(
                            circuit, backend,
                            initial_layout=model_layout,
                            layout_method="given",
                            routing_method=routing_method,
                            seed=args.seed + rep,
                        )
                        final_2q = count_2q_gates(tc)
                        all_rows.append({
                            "backend": backend_name, "circuit": cname,
                            "qubits": num_logical, "orig_cx": orig_2q,
                            "method": label, "rep": rep,
                            "depth": tc.depth(), "add_cx": final_2q - orig_2q,
                        })
                    except Exception as e:
                        logger.warning("    %s rep%d failed: %s", label, rep, e)

    df = pd.DataFrame(all_rows)

    # Summary: average across reps
    summary = df.groupby(["backend", "circuit", "qubits", "orig_cx", "method"]).agg(
        depth=("depth", "mean"),
        add_cx=("add_cx", "mean"),
    ).reset_index()

    method_order = ["SABRE", "NASSC", "QAP+SABRE", "QAP+NASSC", "OURS+SABRE", "OURS+NASSC"]

    # Output per backend
    output_dir = Path(args.checkpoint).parent.parent / "transpile_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "raw.csv", index=False)

    for bname in args.backend:
        bsub = summary[summary["backend"] == bname]
        idx = ["circuit", "qubits", "orig_cx"]

        pivot_depth = bsub.pivot_table(index=idx, columns="method", values="depth")
        pivot_depth = pivot_depth[[m for m in method_order if m in pivot_depth.columns]]

        pivot_add_cx = bsub.pivot_table(index=idx, columns="method", values="add_cx")
        pivot_add_cx = pivot_add_cx[[m for m in method_order if m in pivot_add_cx.columns]]

        # Add AVG row
        avg_depth = pivot_depth.mean().to_frame().T
        avg_depth.index = pd.MultiIndex.from_tuples([("AVG", "", "")], names=idx)
        pivot_depth = pd.concat([pivot_depth, avg_depth])

        avg_add_cx = pivot_add_cx.mean().to_frame().T
        avg_add_cx.index = pd.MultiIndex.from_tuples([("AVG", "", "")], names=idx)
        pivot_add_cx = pd.concat([pivot_add_cx, avg_add_cx])

        print(f"\n=== {bname.upper()} — Depth ===")
        print(pivot_depth.to_string(float_format="%.1f"))

        print(f"\n=== {bname.upper()} — Additional CNOT ===")
        print(pivot_add_cx.to_string(float_format="%.1f"))

        pivot_depth.to_csv(output_dir / f"depth_{bname}.csv", float_format="%.1f")
        pivot_add_cx.to_csv(output_dir / f"add_cx_{bname}.csv", float_format="%.1f")

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
