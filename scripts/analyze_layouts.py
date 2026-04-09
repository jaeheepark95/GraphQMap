"""Quantitative layout analysis for GraphQMap diagnosis (Phase D-2).

For each (circuit, backend, method), extracts the initial layout and computes:
  1. Centrality of mapped physical qubits (closeness, betweenness, degree)
  2. Pairwise hop distance distribution for circuit edges
  3. Noise quality (avg readout error, avg 2Q gate error)
  4. Layout overlap across circuits (Jaccard, per backend × method)
  5. Per-qubit weighted hop-cost variance (fairness / tail risk)

Produces a markdown report comparing OURS / QAP+NASSC / SABRE.

Usage:
    python scripts/analyze_layouts.py \\
        --config runs/stage2/<RUN>/config.yaml \\
        --checkpoint runs/stage2/<RUN>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph
from data.hardware_graph import (
    build_hardware_graph,
    extract_qubit_properties,
    extract_edge_properties,
    get_backend,
    precompute_hop_distance,
    _get_two_qubit_gate_name,
)
from evaluation.benchmark import BENCHMARK_CIRCUITS, load_benchmark_circuit
from evaluation.transpiler import build_transpiler
from models.graphqmap import GraphQMap


BACKENDS = ["toronto", "brooklyn", "torino"]

METHODS = [
    # (label, layout_method, routing_method, uses_model)
    ("OURS", None, "nassc", True),
    ("QAP+NASSC", "qap", "nassc", False),
    ("SABRE", "sabre", "sabre", False),
]


# ---------------------------------------------------------------------------
# Layout extraction
# ---------------------------------------------------------------------------

def get_transpiled_layout(circuit, backend, layout_method, routing_method, seed=43):
    pm = build_transpiler(
        backend, layout_method=layout_method,
        routing_method=routing_method, seed=seed,
    )
    tc = pm.run(circuit)
    layout = {}
    virt_bits = tc._layout.initial_layout.get_virtual_bits()
    for qubit, phys_idx in virt_bits.items():
        for r in tc._layout.initial_layout.get_registers():
            if qubit in r and r.name != "ancilla":
                virt_idx = list(r).index(qubit)
                layout[virt_idx] = phys_idx
                break
    return layout


def get_model_layout(model, circuit, backend, hw_graph, cfg):
    node_features = getattr(cfg.model.circuit_gnn, "node_features", None)
    rwpe_k = getattr(cfg.model.circuit_gnn, "rwpe_k", 0)
    edge_dim = getattr(cfg.model.circuit_gnn, "edge_input_dim", None)
    cg = build_circuit_graph(
        circuit, node_feature_names=node_features,
        rwpe_k=rwpe_k, edge_dim=edge_dim,
    )
    cb = Batch.from_data_list([cg])
    hb = Batch.from_data_list([hw_graph])
    num_logical = circuit.num_qubits
    num_physical = backend.target.num_qubits
    tau = getattr(cfg.sinkhorn, "tau_min", 0.05)
    with torch.no_grad():
        layouts = model.predict(
            cb, hb, batch_size=1,
            num_logical=num_logical,
            num_physical=num_physical, tau=tau,
        )
    return layouts[0]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def get_circuit_edges(circuit) -> list[tuple[int, int]]:
    """Extract logical 2Q gate pairs from a circuit (multi-set as list)."""
    edges = []
    for instr in circuit.data:
        if instr.operation.num_qubits == 2:
            q0 = circuit.find_bit(instr.qubits[0]).index
            q1 = circuit.find_bit(instr.qubits[1]).index
            if q0 == q1:
                continue
            edges.append((min(q0, q1), max(q0, q1)))
    return edges


def backend_to_nx(backend) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(backend.target.num_qubits))
    for u, v in backend.coupling_map.get_edges():
        g.add_edge(u, v)
    return g


def get_qubit_noise(backend) -> dict[str, np.ndarray]:
    qp = extract_qubit_properties(backend)
    return {
        "readout_error": qp["readout_error"],
        "single_qubit_error": qp["single_qubit_error"],
    }


def get_edge_noise(backend) -> dict[tuple[int, int], float]:
    edge_list, edge_error, _ = extract_edge_properties(backend)
    out = {}
    for (u, v), e in zip(edge_list, edge_error):
        out[(min(u, v), max(u, v))] = float(e)
    return out


def compute_metrics(
    layout: dict[int, int],
    circuit_edges: list[tuple[int, int]],
    bk_nx: nx.Graph,
    centralities: dict[str, dict],
    hop_dist: np.ndarray,
    qubit_noise: dict[str, np.ndarray],
    edge_noise: dict[tuple[int, int], float],
) -> dict:
    mapped_phys = list(layout.values())

    # 1. Centrality of mapped qubits
    cent = {}
    for name, cmap in centralities.items():
        vals = [cmap[p] for p in mapped_phys]
        cent[f"{name}_mean"] = float(np.mean(vals))
        cent[f"{name}_std"] = float(np.std(vals))

    # 2. Hop distance for circuit edges
    hops = []
    for (i, j) in circuit_edges:
        if i in layout and j in layout:
            hops.append(int(hop_dist[layout[i]][layout[j]]))
    if hops:
        hop_metrics = {
            "hop_mean": float(np.mean(hops)),
            "hop_max": int(np.max(hops)),
            "hop_p90": float(np.percentile(hops, 90)),
            "hop_adj_frac": float(np.mean([h <= 1 for h in hops])),
        }
    else:
        hop_metrics = {"hop_mean": 0.0, "hop_max": 0, "hop_p90": 0.0, "hop_adj_frac": 1.0}

    # 3. Noise quality of used qubits/edges
    used_readout = float(np.mean([qubit_noise["readout_error"][p] for p in mapped_phys]))
    used_sq = float(np.mean([qubit_noise["single_qubit_error"][p] for p in mapped_phys]))

    used_edge_errors = []
    for (i, j) in set(circuit_edges):
        if i in layout and j in layout:
            p, q = layout[i], layout[j]
            key = (min(p, q), max(p, q))
            if key in edge_noise:
                used_edge_errors.append(edge_noise[key])
    used_2q = float(np.mean(used_edge_errors)) if used_edge_errors else 0.0

    # 5. Per-qubit weighted cost variance
    # For each logical i: sum of hop_dist(layout[i], layout[j]) * weight over j
    edge_weight: dict[tuple[int, int], float] = defaultdict(float)
    for (i, j) in circuit_edges:
        edge_weight[(i, j)] += 1.0
    per_q_cost = defaultdict(float)
    for (i, j), w in edge_weight.items():
        if i in layout and j in layout:
            d = float(hop_dist[layout[i]][layout[j]])
            per_q_cost[i] += w * d
            per_q_cost[j] += w * d
    if per_q_cost:
        cost_arr = np.array(list(per_q_cost.values()))
        cost_var = float(np.var(cost_arr))
        cost_max = float(np.max(cost_arr))
    else:
        cost_var = 0.0
        cost_max = 0.0

    return {
        **cent,
        **hop_metrics,
        "used_readout_err": used_readout,
        "used_sq_err": used_sq,
        "used_2q_err": used_2q,
        "qubit_cost_var": cost_var,
        "qubit_cost_max": cost_max,
        "_mapped_set": frozenset(mapped_phys),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(headers: list[str], rows: list[list]) -> str:
    """Render a markdown table."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def aggregate_per_method(
    results: dict, backend_name: str, metric: str
) -> dict[str, float]:
    """Average a metric across circuits for each method on a backend."""
    out = {}
    for method in [m[0] for m in METHODS]:
        vals = []
        for circuit_name, by_backend in results.items():
            r = by_backend.get(backend_name, {}).get(method)
            if r is not None and metric in r:
                vals.append(r[metric])
        out[method] = float(np.mean(vals)) if vals else float("nan")
    return out


def compute_jaccard_overlap(
    results: dict, backend_name: str
) -> dict[str, float]:
    """Mean pairwise Jaccard of mapped qubit sets across circuits, per method.
    Higher = layouts reuse the same physical region (more circuit-agnostic).
    Only meaningful for circuits with comparable num_logical sizes — we
    aggregate over all pairs."""
    out = {}
    for method in [m[0] for m in METHODS]:
        sets = []
        for circuit_name, by_backend in results.items():
            r = by_backend.get(backend_name, {}).get(method)
            if r is not None:
                sets.append(r["_mapped_set"])
        jaccards = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                a, b = sets[i], sets[j]
                u = len(a | b)
                if u > 0:
                    jaccards.append(len(a & b) / u)
        out[method] = float(np.mean(jaccards)) if jaccards else 0.0
    return out


def write_report(results: dict, output_path: Path):
    lines = ["# Layout Diagnosis Report (Phase D-2)\n"]
    lines.append(
        "Comparison of OURS (model+NASSC), QAP+NASSC, and SABRE layouts across "
        f"{len(BENCHMARK_CIRCUITS)} benchmark circuits × {len(BACKENDS)} test backends.\n"
    )

    method_names = [m[0] for m in METHODS]

    # ---------- Section 1: Per-backend aggregate metrics ----------
    lines.append("## 1. Per-backend aggregate metrics (mean across 8 circuits)\n")
    metrics_to_show = [
        ("hop_mean", "Mean hop dist (lower=better)"),
        ("hop_max", "Max hop dist"),
        ("hop_adj_frac", "Frac adjacent edges"),
        ("used_readout_err", "Avg readout err of used qubits"),
        ("used_2q_err", "Avg 2Q err of used edges"),
        ("qubit_cost_var", "Per-qubit cost variance (fairness)"),
        ("qubit_cost_max", "Max per-qubit cost (tail)"),
        ("closeness_mean", "Closeness centrality (used qubits)"),
        ("betweenness_mean", "Betweenness centrality"),
        ("degree_mean", "Degree (used qubits)"),
    ]
    for backend_name in BACKENDS:
        lines.append(f"### Backend: {backend_name}\n")
        rows = []
        for metric_key, metric_label in metrics_to_show:
            agg = aggregate_per_method(results, backend_name, metric_key)
            rows.append([metric_label] + [agg[m] for m in method_names])
        lines.append(format_table(["Metric"] + method_names, rows))
        lines.append("")

    # ---------- Section 2: Layout overlap (circuit-agnostic check) ----------
    lines.append("## 2. Layout overlap across circuits (Jaccard, per backend)\n")
    lines.append(
        "Mean pairwise Jaccard of mapped qubit sets across the 8 circuits. "
        "**Higher = more circuit-agnostic** (uses similar physical region regardless "
        "of circuit). OURS being notably higher than baselines would indicate the "
        "core-periphery collapse hypothesis.\n"
    )
    rows = []
    for backend_name in BACKENDS:
        ov = compute_jaccard_overlap(results, backend_name)
        rows.append([backend_name] + [ov[m] for m in method_names])
    lines.append(format_table(["Backend"] + method_names, rows))
    lines.append("")

    # ---------- Section 3: Per-circuit detail (worst-case spotting) ----------
    lines.append("## 3. Per-circuit hop_mean and qubit_cost_max\n")
    lines.append(
        "Drill-down for each circuit. Look for circuits where OURS is sharply worse "
        "than baselines.\n"
    )
    for backend_name in BACKENDS:
        lines.append(f"### Backend: {backend_name}\n")
        rows = []
        for cname in BENCHMARK_CIRCUITS:
            row = [cname]
            for method in method_names:
                r = results.get(cname, {}).get(backend_name, {}).get(method)
                if r is None:
                    row.append("—")
                else:
                    row.append(f"{r['hop_mean']:.2f}/{r['qubit_cost_max']:.1f}")
            rows.append(row)
        lines.append(format_table(["Circuit (hop_mean / cost_max)"] + method_names, rows))
        lines.append("")

    # ---------- Section 4: Centrality detail ----------
    lines.append("## 4. Centrality of mapped qubits\n")
    lines.append(
        "If OURS shows much higher closeness/betweenness than baselines, the model "
        "is preferring central hub qubits (core-periphery collapse).\n"
    )

    output_path.write_text("\n".join(lines))
    print(f"\nReport written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Output path: default = sibling of layout_plots
    if args.output:
        output_path = Path(args.output)
    else:
        ckpt_path = Path(args.checkpoint)
        run_name = ckpt_path.parent.parent.name
        eval_dir = ckpt_path.parent.parent.parent.parent / "eval" / run_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = eval_dir / "diagnosis_report.md"

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = GraphQMap.from_config(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Pre-compute backend artifacts
    print("Pre-computing backend artifacts...")
    backends = {}
    bk_nx = {}
    centralities = {}
    hop_dists = {}
    qubit_noises = {}
    edge_noises = {}
    hw_graphs = {}
    for name in BACKENDS:
        bk = get_backend(name)
        backends[name] = bk
        g = backend_to_nx(bk)
        bk_nx[name] = g
        print(f"  [{name}] computing centralities ({g.number_of_nodes()}Q)...")
        centralities[name] = {
            "closeness": nx.closeness_centrality(g),
            "betweenness": nx.betweenness_centrality(g),
            "degree": dict(g.degree()),
        }
        hop_dists[name] = precompute_hop_distance(bk)
        qubit_noises[name] = get_qubit_noise(bk)
        edge_noises[name] = get_edge_noise(bk)
        hw_graphs[name] = build_hardware_graph(bk)

    # Run all (circuit, backend, method) layouts
    print("\nExtracting layouts and computing metrics...")
    results: dict = {}  # results[circuit][backend][method] = metrics
    for cname in BENCHMARK_CIRCUITS:
        print(f"[{cname}]")
        try:
            circuit = load_benchmark_circuit(cname, measure=True)
        except Exception as e:
            print(f"  Skip ({e})")
            continue
        circuit_edges = get_circuit_edges(circuit)
        results[cname] = {}
        for backend_name in BACKENDS:
            results[cname][backend_name] = {}
            for label, layout_method, routing_method, uses_model in METHODS:
                try:
                    if uses_model:
                        layout = get_model_layout(
                            model, circuit, backends[backend_name],
                            hw_graphs[backend_name], cfg,
                        )
                    else:
                        layout = get_transpiled_layout(
                            circuit, backends[backend_name],
                            layout_method, routing_method,
                        )
                    metrics = compute_metrics(
                        layout, circuit_edges, bk_nx[backend_name],
                        centralities[backend_name], hop_dists[backend_name],
                        qubit_noises[backend_name], edge_noises[backend_name],
                    )
                    results[cname][backend_name][label] = metrics
                except Exception as e:
                    print(f"  {backend_name}/{label}: ERROR {e}")

    write_report(results, output_path)


if __name__ == "__main__":
    main()
