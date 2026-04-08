"""Why does the 50-line heuristic lose to QAP on certain circuits?

For each (circuit, backend) failure case from `scripts/heuristic_baseline.py`,
this dumps a side-by-side report of:

  - top-K logical hot edges (sorted by 2Q-gate count)
  - HW hop-distance of each hot edge under HEURISTIC vs QAP layouts
  - which edges the heuristic placed worse than QAP, and the routing-cost gap
    (3·max(hop-1, 0) — i.e. SWAPs needed)

Also writes a per-case PNG: hardware topology with mapped physical qubits
labelled `q{logical}`, hot edges drawn between them (width ∝ weight,
green = adjacent, red = needs SWAPs).

Run:
  python scripts/analyze_heuristic_failures.py
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag

from data.hardware_graph import get_backend
from evaluation.benchmark import load_benchmark_circuit
from evaluation.prev_methods.qap import QAPLayout
from scripts.heuristic_baseline import _logical_interactions, heuristic_layout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("qiskit").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# (circuit, backend) cases where heuristic underperformed QAP+SABRE in the
# previous run.
FAILURE_CASES = [
    ("4mod5-v1_22", "torino"),
    ("mod5mils_65", "torino"),
    ("4gt13_92", "torino"),
    ("4mod5-v1_22", "brooklyn"),
    ("toffoli_3", "brooklyn"),  # tiny but heuristic lost here too
]

OUT_DIR = Path("runs/eval/heuristic/failure_analysis")


# ---------------------------------------------------------------------------- #
# Layout extractors
# ---------------------------------------------------------------------------- #

def qap_layout(circuit: QuantumCircuit, backend: Any) -> list[int]:
    """Run QAPLayout pass and extract logical→physical list."""
    pass_ = QAPLayout(backend)
    dag: DAGCircuit = circuit_to_dag(circuit)
    pass_.run(dag)
    layout = pass_.property_set["layout"]
    # Layout is phys→logical via Qubit objects; build logical_idx → phys_idx
    n = circuit.num_qubits
    out = [0] * n
    v2p = layout.get_virtual_bits()  # {Qubit: phys_idx}
    for q, p in v2p.items():
        try:
            li = circuit.find_bit(q).index
        except Exception:
            continue
        if li < n:
            out[li] = p
    return out


# ---------------------------------------------------------------------------- #
# Analysis
# ---------------------------------------------------------------------------- #

def hw_graph(backend) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(backend.target.num_qubits))
    G.add_edges_from(backend.configuration().coupling_map)
    return G


def edge_hop(layout: list[int], a: int, b: int, sp: dict) -> int:
    pa, pb = layout[a], layout[b]
    return sp[pa].get(pb, 99)


def analyze_case(circuit_name: str, backend_name: str, top_k: int = 10):
    backend = get_backend(backend_name)
    circuit = load_benchmark_circuit(circuit_name, Path("data/circuits/qasm/benchmarks"), measure=True)
    n = circuit.num_qubits
    Ghw = hw_graph(backend)
    # Pre-compute single-source shortest paths for placed physicals only later
    # Use full-graph BFS (acceptable for ≤133Q)
    sp = dict(nx.all_pairs_shortest_path_length(Ghw))

    pair_w, busy = _logical_interactions(circuit)
    h_layout = heuristic_layout(circuit, backend)
    q_layout = qap_layout(circuit, backend)

    top_pairs = pair_w.most_common(top_k)

    rows = []
    swap_h = swap_q = 0
    for (a, b), w in top_pairs:
        ha = edge_hop(h_layout, a, b, sp)
        qa = edge_hop(q_layout, a, b, sp)
        sh = 3 * max(ha - 1, 0)
        sq = 3 * max(qa - 1, 0)
        swap_h += sh * w
        swap_q += sq * w
        rows.append((a, b, w, ha, qa, sh - sq))

    # Total weighted SWAP cost over ALL pairs (not just top_k)
    total_h = sum(w * 3 * max(edge_hop(h_layout, a, b, sp) - 1, 0)
                  for (a, b), w in pair_w.items())
    total_q = sum(w * 3 * max(edge_hop(q_layout, a, b, sp) - 1, 0)
                  for (a, b), w in pair_w.items())

    print(f"\n┌── {circuit_name} on {backend_name} (n={n}, |E|={len(pair_w)}, "
          f"total 2Q={sum(pair_w.values())}) ──")
    print(f"│  HEURISTIC  layout (logical→phys): {h_layout}")
    print(f"│  QAP        layout (logical→phys): {q_layout}")
    print(f"│")
    print(f"│  Hot edges (top {top_k}):")
    print(f"│    {'L_pair':>8s}  {'×':>4s}  {'h.hop':>6s}  {'q.hop':>6s}  {'h-q swaps':>10s}")
    for a, b, w, ha, qa, dswap in rows:
        marker = "  ⚠" if dswap > 0 else ("  ✓" if dswap < 0 else "")
        print(f"│    ({a:2d},{b:2d})  {w:4d}  {ha:6d}  {qa:6d}  {dswap*w:+10d}{marker}")
    print(f"│")
    print(f"│  Weighted SWAP cost (all edges):  HEURISTIC={total_h:>5d}  QAP={total_q:>5d}  "
          f"Δ={total_h - total_q:+d}")
    print(f"└────────────────────────────────────────────────────────────────")

    # PNG
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot(circuit_name, backend_name, Ghw, h_layout, q_layout, pair_w, busy)

    return dict(circuit=circuit_name, backend=backend_name,
                heuristic_swap=total_h, qap_swap=total_q, delta=total_h - total_q)


def _plot(cname, bname, Ghw, h_layout, q_layout, pair_w, busy):
    pos = nx.kamada_kawai_layout(Ghw)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    n_log = len(h_layout)
    max_w = max(pair_w.values()) if pair_w else 1
    busiest = max(busy.values()) if busy else 1

    for ax, layout, title in [
        (axes[0], h_layout, f"HEURISTIC — {cname} / {bname}"),
        (axes[1], q_layout, f"QAP — {cname} / {bname}"),
    ]:
        # HW background
        nx.draw_networkx_edges(Ghw, pos, ax=ax, edge_color="lightgray", width=1.0)
        nx.draw_networkx_nodes(Ghw, pos, ax=ax, node_color="whitesmoke",
                               node_size=120, edgecolors="lightgray")

        # Mapped physical qubits, sized by busyness
        used = [layout[i] for i in range(n_log)]
        sizes = [220 + 600 * busy.get(i, 0) / busiest for i in range(n_log)]
        labels = {layout[i]: f"q{i}" for i in range(n_log)}
        nx.draw_networkx_nodes(Ghw, pos, nodelist=used, ax=ax,
                               node_color="#ffd166", edgecolors="black",
                               node_size=sizes, linewidths=1.5)
        nx.draw_networkx_labels(Ghw, pos, labels=labels, ax=ax, font_size=8)

        # Hot edges (top-K)
        for (a, b), w in pair_w.most_common(12):
            pa, pb = layout[a], layout[b]
            adj = Ghw.has_edge(pa, pb)
            color = "#06a35a" if adj else "#d63031"
            style = "solid" if adj else "dashed"
            width = 0.8 + 3.5 * (w / max_w)
            nx.draw_networkx_edges(Ghw, pos, edgelist=[(pa, pb)], ax=ax,
                                   edge_color=color, style=style, width=width,
                                   alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()

    fig.suptitle(
        f"{cname} on {bname} — yellow nodes = mapped, size ∝ logical busyness; "
        f"green solid = adjacent hot edge, red dashed = non-adjacent (needs SWAPs)",
        fontsize=10, y=0.98,
    )
    fig.tight_layout()
    out = OUT_DIR / f"{cname}__{bname}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("  saved %s", out)


def main():
    summary = []
    for cname, bname in FAILURE_CASES:
        try:
            summary.append(analyze_case(cname, bname))
        except Exception as e:
            logger.error("Failed %s/%s: %s", cname, bname, e)

    print("\n=== SUMMARY (weighted SWAP cost over all interaction edges) ===")
    print(f"{'circuit':<18s}{'backend':<12s}{'heuristic':>12s}{'qap':>10s}{'delta':>10s}")
    for r in summary:
        print(f"{r['circuit']:<18s}{r['backend']:<12s}{r['heuristic_swap']:>12d}"
              f"{r['qap_swap']:>10d}{r['delta']:>+10d}")


if __name__ == "__main__":
    main()
