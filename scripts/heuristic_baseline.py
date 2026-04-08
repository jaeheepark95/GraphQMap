"""Pure-heuristic initial layout baseline.

Question: How much of the GraphQMap problem is just "place busy logical qubits
on good HW spots, and keep hot interaction pairs adjacent"? This script
implements that as 50 lines of code, runs the same eval pipeline, and reports
PST vs SABRE / QAP+SABRE on the standard benchmark circuits.

Heuristic recipe:
  1. logical busyness   = sum of 2Q gates touching each logical qubit
  2. hardware quality   = -(readout_err + sq_err + 2 * mean_neighbor_2q_err)
  3. seed               = busiest logical → best HW qubit
  4. expand (BFS by interaction weight): repeatedly take the unplaced logical
     with the highest cumulative weight to the placed set, and put it on the
     free HW qubit that maximises (adjacency_benefit, hw_quality).

No learning, no GNN, no QAP solver. Pure greedy.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import pandas as pd
from qiskit import QuantumCircuit

from data.hardware_graph import get_backend
from evaluation.benchmark import (
    BENCHMARK_CIRCUITS,
    execute_on_simulators,
    load_benchmark_circuit,
)
from evaluation.pst import create_ideal_simulator, create_noisy_simulator
from evaluation.transpiler import transpile_with_timing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
logging.getLogger("qiskit").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Heuristic layout
# ---------------------------------------------------------------------------- #

def _logical_interactions(circuit: QuantumCircuit) -> tuple[Counter, Counter]:
    pair_w: Counter = Counter()
    busy: Counter = Counter()
    for instr in circuit.data:
        qargs = instr.qubits
        if len(qargs) != 2:
            continue
        a, b = sorted((circuit.find_bit(qargs[0]).index, circuit.find_bit(qargs[1]).index))
        pair_w[(a, b)] += 1
        busy[a] += 1
        busy[b] += 1
    return pair_w, busy


def _hw_quality(backend: Any, hw_graph: nx.Graph) -> dict[int, float]:
    props = backend.properties()
    num_phys = backend.target.num_qubits
    two_q_names = ("cx", "ecr", "cz")

    quality: dict[int, float] = {}
    for p in range(num_phys):
        # readout
        try:
            ro = float(props.readout_error(p))
        except Exception:
            ro = 0.0
        # 1q gate error (sx preferred)
        sq = 0.0
        for gname in ("sx", "x"):
            try:
                sq = float(props.gate_error(gname, p))
                if sq > 0:
                    break
            except Exception:
                pass
        # mean 2q error over incident edges
        edge_errs = []
        for nbr in hw_graph.neighbors(p):
            for gname in two_q_names:
                try:
                    e = float(props.gate_error(gname, [p, nbr]))
                    if e > 0:
                        edge_errs.append(e)
                        break
                except Exception:
                    continue
        mean_2q = sum(edge_errs) / len(edge_errs) if edge_errs else 0.05
        quality[p] = -(ro + sq + 2.0 * mean_2q)
    return quality


def heuristic_layout(circuit: QuantumCircuit, backend: Any) -> list[int]:
    """Return a logical→physical layout as a list (length = num_logical)."""
    n = circuit.num_qubits
    coupling = backend.configuration().coupling_map
    G_hw = nx.Graph()
    G_hw.add_edges_from(coupling)
    num_phys = backend.target.num_qubits
    G_hw.add_nodes_from(range(num_phys))

    pair_w, busy = _logical_interactions(circuit)
    quality = _hw_quality(backend, G_hw)

    # logical placement order: BFS-by-weight from busiest
    if not pair_w:
        order = sorted(range(n), key=lambda q: busy.get(q, 0), reverse=True)
    else:
        seed = busy.most_common(1)[0][0]
        order = [seed]
        remaining = set(range(n)) - {seed}
        while remaining:
            best_q, best_w = None, -1
            for q in remaining:
                w = sum(pair_w.get(tuple(sorted((q, p))), 0) for p in order)
                if w > best_w or (w == best_w and busy.get(q, 0) > busy.get(best_q or -1, -1)):
                    best_w, best_q = w, q
            order.append(best_q)
            remaining.remove(best_q)

    placed: dict[int, int] = {}
    used: set[int] = set()
    for idx, lq in enumerate(order):
        if idx == 0:
            phys = max(range(num_phys), key=lambda p: quality[p])
        else:
            best_p, best_score = None, None
            for p in range(num_phys):
                if p in used:
                    continue
                adj_benefit = 0.0
                for plq, pp in placed.items():
                    if pp in G_hw[p]:
                        adj_benefit += pair_w.get(tuple(sorted((lq, plq))), 0)
                score = (adj_benefit, quality[p])
                if best_score is None or score > best_score:
                    best_score, best_p = score, p
            phys = best_p
        placed[lq] = phys
        used.add(phys)

    return [placed[lq] for lq in range(n)]


# ---------------------------------------------------------------------------- #
# Eval loop
# ---------------------------------------------------------------------------- #

def run(args) -> None:
    rows = []
    for backend_name in args.backend:
        logger.info("=== Backend: %s ===", backend_name)
        backend = get_backend(backend_name)
        num_phys = backend.target.num_qubits
        ideal = create_ideal_simulator(backend)
        noisy = create_noisy_simulator(backend)

        circuits = args.circuits or BENCHMARK_CIRCUITS
        for cname in circuits:
            try:
                circuit = load_benchmark_circuit(cname, args.circuit_dir, measure=True)
            except Exception as e:
                logger.warning("  Skip %s: %s", cname, e)
                continue
            if circuit.num_qubits > num_phys or circuit.num_qubits < 2:
                continue

            # heuristic layout once (deterministic; reps only re-run sim)
            layout = heuristic_layout(circuit, backend)

            for method, kwargs in [
                ("HEURISTIC+SABRE", dict(initial_layout=layout, layout_method="given", routing_method="sabre")),
                ("SABRE", dict(layout_method="sabre", routing_method="sabre")),
                ("QAP+SABRE", dict(layout_method="qap", routing_method="sabre")),
            ]:
                psts = []
                for rep in range(args.reps):
                    try:
                        tc, _ = transpile_with_timing(
                            circuit, backend, seed=args.seed + rep, **kwargs
                        )
                        avg_pst, _, _ = execute_on_simulators(tc, ideal, noisy, shots=args.shots)
                        psts.append(avg_pst)
                    except Exception as e:
                        logger.warning("  %s/%s rep%d failed: %s", cname, method, rep, e)
                        psts.append(float("nan"))
                pst_mean = sum(p for p in psts if p == p) / max(1, sum(1 for p in psts if p == p))
                rows.append(dict(backend=backend_name, circuit=cname, method=method, pst=pst_mean))
                logger.info("  %-20s %-18s PST=%.4f", cname, method, pst_mean)

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    logger.info("Saved raw → %s", out)

    # summary
    print("\n=== PER-CIRCUIT × BACKEND ===")
    pivot = df.pivot_table(index=["backend", "circuit"], columns="method", values="pst")
    cols = [c for c in ["HEURISTIC+SABRE", "SABRE", "QAP+SABRE"] if c in pivot.columns]
    pivot = pivot[cols]
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== PER-BACKEND AVG ===")
    avg = df.groupby(["backend", "method"])["pst"].mean().unstack()[cols]
    print(avg.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== OVERALL AVG ===")
    print(df.groupby("method")["pst"].mean()[cols].to_string(float_format=lambda x: f"{x:.4f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", nargs="+", default=["toronto", "brooklyn", "torino"])
    ap.add_argument("--circuits", nargs="*", default=None)
    ap.add_argument("--circuit-dir", type=Path,
                    default=Path("data/circuits/qasm/benchmarks"))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--output", type=Path, default=Path("runs/eval/heuristic/heuristic_results.csv"))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
