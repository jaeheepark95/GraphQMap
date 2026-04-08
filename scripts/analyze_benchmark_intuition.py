"""Per-benchmark intuitive analysis: who interacts with whom, how busy each qubit is.

Question: do these circuits even need a complex GNN, or is "place interacting
qubits adjacent + put busy qubits on best hardware spots" enough?

For each benchmark circuit prints:
  - shape  : #qubits, 2Q count, depth
  - busy   : per-logical-qubit 2Q-gate count
  - edges  : (u,v): count, sorted desc
  - graph  : detected interaction-graph topology (line / ring / star / complete / tree / dense)
  - hint   : one-line intuitive mapping recipe
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from qiskit import QuantumCircuit

BENCH_DIR = Path(__file__).resolve().parents[1] / "data/circuits/qasm/benchmarks"


def load(qasm: Path) -> QuantumCircuit:
    return QuantumCircuit.from_qasm_file(str(qasm))


def interaction_stats(qc: QuantumCircuit):
    n = qc.num_qubits
    pair_count: Counter = Counter()
    qubit_2q: Counter = Counter()
    for instr, qargs, _ in qc.data:
        if len(qargs) != 2:
            continue
        a, b = sorted((qc.find_bit(qargs[0]).index, qc.find_bit(qargs[1]).index))
        pair_count[(a, b)] += 1
        qubit_2q[a] += 1
        qubit_2q[b] += 1
    return n, pair_count, qubit_2q


def classify(n: int, edges: list[tuple[int, int]]) -> str:
    if not edges:
        return "no-2Q"
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    m = G.number_of_edges()
    degs = sorted((d for _, d in G.degree()), reverse=True)
    max_deg, min_deg = degs[0], degs[-1]
    complete_m = n * (n - 1) // 2

    if m == complete_m:
        return f"complete K{n}"
    if max_deg == n - 1 and sum(1 for d in degs if d == 1) == n - 1:
        return f"star (hub=q?)"
    if max_deg == 2 and min_deg == 2 and m == n:
        return "ring"
    if max_deg == 2 and m == n - 1:
        return "line / path"
    if nx.is_tree(G):
        return "tree"
    density = m / complete_m if complete_m else 0
    if density >= 0.7:
        return f"dense ({m}/{complete_m} edges)"
    return f"sparse ({m}/{complete_m} edges)"


def hint(n, pair_count, qubit_2q, topo) -> str:
    if not pair_count:
        return "no 2Q gates — any layout works"
    busy_q, busy_n = qubit_2q.most_common(1)[0]
    top_pair, top_w = pair_count.most_common(1)[0]
    parts = []
    if "complete" in topo:
        parts.append(f"K{n}: needs all-to-all → pick a tightly connected hardware clique")
    elif "star" in topo:
        # find hub
        deg = Counter()
        for a, b in pair_count:
            deg[a] += 1
            deg[b] += 1
        hub = deg.most_common(1)[0][0]
        parts.append(f"star hub=q{hub} → place hub on a high-degree HW node, leaves on its neighbors")
    elif "line" in topo:
        parts.append("line → embed onto any path of length n on HW")
    elif "ring" in topo:
        parts.append("ring → embed onto an HW cycle (or path; closing edge needs 1 SWAP)")
    elif "tree" in topo:
        parts.append("tree → embed root on hub, BFS-place children on neighbors")
    else:
        parts.append(f"{topo}: hot edge q{top_pair[0]}-q{top_pair[1]} (×{top_w}) MUST be HW-adjacent")
    parts.append(f"busiest q{busy_q} ({busy_n} 2Q) → put on a low-error HW qubit")
    return "; ".join(parts)


def analyze_one(qasm: Path) -> dict:
    qc = load(qasm)
    qc.remove_final_measurements(inplace=True)
    n, pair_count, qubit_2q = interaction_stats(qc)
    edges = list(pair_count.keys())
    topo = classify(n, edges)
    return {
        "name": qasm.stem,
        "n": n,
        "depth": qc.depth(),
        "n_2q": sum(pair_count.values()),
        "n_pairs": len(pair_count),
        "topology": topo,
        "busy": qubit_2q,
        "pairs": pair_count,
        "hint": hint(n, pair_count, qubit_2q, topo),
    }


def fmt_busy(busy: Counter, n: int) -> str:
    return " ".join(f"q{i}={busy.get(i,0)}" for i in range(n))


def fmt_pairs(pairs: Counter, k: int = 6) -> str:
    items = pairs.most_common(k)
    return ", ".join(f"({a},{b}):{w}" for (a, b), w in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=BENCH_DIR)
    args = ap.parse_args()

    files = sorted(args.dir.glob("*.qasm"))
    rows = [analyze_one(f) for f in files]
    rows.sort(key=lambda r: (r["n"], r["n_2q"]))

    print(f"\n{'='*100}")
    print(f"BENCHMARK INTUITION  ({len(rows)} circuits)")
    print(f"{'='*100}\n")

    for r in rows:
        print(f"── {r['name']}  (n={r['n']}, depth={r['depth']}, 2Q={r['n_2q']}, "
              f"unique pairs={r['n_pairs']}/{r['n']*(r['n']-1)//2})")
        print(f"   topology : {r['topology']}")
        print(f"   busy     : {fmt_busy(r['busy'], r['n'])}")
        print(f"   hot pairs: {fmt_pairs(r['pairs'])}")
        print(f"   hint     : {r['hint']}")
        print()

    # summary
    print(f"{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    topo_count = Counter()
    for r in rows:
        key = r["topology"].split(" (")[0]
        topo_count[key] += 1
    for t, c in topo_count.most_common():
        print(f"  {t:30s} : {c}")
    n_complete = sum(1 for r in rows if r["n_pairs"] == r["n"] * (r["n"] - 1) // 2)
    print(f"\n  fully connected (K_n) : {n_complete}/{len(rows)}")
    avg_density = sum(r["n_pairs"] / max(1, r["n"] * (r["n"] - 1) // 2) for r in rows) / len(rows)
    print(f"  avg interaction-graph density : {avg_density:.2f}")


if __name__ == "__main__":
    main()
