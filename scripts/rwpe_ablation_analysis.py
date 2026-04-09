"""RWPE ablation analysis: measure RWPE contribution to qubit distinguishability."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.circuit_graph import build_circuit_graph, load_circuit

FEATURES = ["gate_count", "two_qubit_gate_count", "single_qubit_gate_ratio", "critical_path_fraction"]
SIZE_BUCKETS = [
    ("tiny(2-3Q)", 2, 3),
    ("small(4-5Q)", 4, 5),
    ("medium(6-10Q)", 6, 10),
    ("large(11-20Q)", 11, 20),
    ("xlarge(21Q+)", 21, 9999),
]


def get_bucket(nq):
    for label, lo, hi in SIZE_BUCKETS:
        if lo <= nq <= hi:
            return label
    return "unknown"


def indist_rate(x, n):
    x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    cos = x_norm @ x_norm.T
    mask = ~torch.eye(n, dtype=torch.bool)
    return (cos[mask] > 0.95).float().mean().item()


def analyze_set(qasm_list, label):
    bucket_no = {b[0]: [] for b in SIZE_BUCKETS}
    bucket_yes = {b[0]: [] for b in SIZE_BUCKETS}
    bucket_rwpe_dead = {b[0]: [] for b in SIZE_BUCKETS}
    bucket_rwpe_var = {b[0]: [] for b in SIZE_BUCKETS}

    for qf in qasm_list:
        try:
            circ = load_circuit(qf)
            g_no = build_circuit_graph(circ, node_feature_names=FEATURES, rwpe_k=0)
            g_yes = build_circuit_graph(circ, node_feature_names=FEATURES, rwpe_k=2)
        except Exception:
            continue
        n = g_no.x.shape[0]
        if n < 2:
            continue
        bucket = get_bucket(n)

        bucket_no[bucket].append(indist_rate(g_no.x, n))
        bucket_yes[bucket].append(indist_rate(g_yes.x, n))

        rwpe_part = g_yes.x[:, len(FEATURES):]
        dead = (rwpe_part.std(dim=0) < 1e-6).sum().item()
        bucket_rwpe_dead[bucket].append(dead)

        total_var = g_yes.x.var(dim=0).sum().item()
        rwpe_var = rwpe_part.var(dim=0).sum().item()
        bucket_rwpe_var[bucket].append(rwpe_var / total_var if total_var > 0 else 0)

    print(f"=== {label} ===")
    header = f"{'Bucket':<16} {'N':>4} {'Indist(no)':>10} {'Indist(+RWPE)':>13} {'Delta':>8} {'RWPE dead':>10} {'RWPE var%':>10}"
    print(header)
    for bname, _, _ in SIZE_BUCKETS:
        cnt = len(bucket_no[bname])
        if cnt == 0:
            continue
        avg_no = np.mean(bucket_no[bname]) * 100
        avg_yes = np.mean(bucket_yes[bname]) * 100
        delta = avg_no - avg_yes
        dead = np.mean(bucket_rwpe_dead[bname])
        rvar = np.mean(bucket_rwpe_var[bname]) * 100
        print(f"{bname:<16} {cnt:>4} {avg_no:>9.2f}% {avg_yes:>12.2f}% {delta:>+7.2f}pp {dead:>9.2f} {rvar:>9.2f}%")


def main():
    qasm_root = Path("data/circuits/qasm")
    all_qasm = []
    for src in sorted(qasm_root.iterdir()):
        if src.is_dir() and src.name != "benchmarks":
            all_qasm.extend(sorted(src.glob("*.qasm")))

    rng = np.random.RandomState(42)
    sampled = list(rng.choice(all_qasm, size=500, replace=False))
    bench_qasm = sorted((qasm_root / "benchmarks").glob("*.qasm"))

    analyze_set(sampled, "Training Circuits (500 sampled)")
    print()
    analyze_set(bench_qasm, "Benchmark Circuits (23)")

    # Per-circuit RWPE analysis on benchmarks
    print("\n=== Per-Benchmark Circuit RWPE Detail ===")
    print(f"{'Circuit':<30} {'NQ':>3} {'Indist(no)':>10} {'Indist(+RWPE)':>13} {'RWPE dead':>10} {'RWPE var%':>10}")
    for qf in bench_qasm:
        try:
            circ = load_circuit(qf)
            g_no = build_circuit_graph(circ, node_feature_names=FEATURES, rwpe_k=0)
            g_yes = build_circuit_graph(circ, node_feature_names=FEATURES, rwpe_k=2)
        except Exception:
            continue
        n = g_no.x.shape[0]
        if n < 2:
            continue
        r_no = indist_rate(g_no.x, n) * 100
        r_yes = indist_rate(g_yes.x, n) * 100
        rwpe_part = g_yes.x[:, len(FEATURES):]
        dead = int((rwpe_part.std(dim=0) < 1e-6).sum().item())
        total_var = g_yes.x.var(dim=0).sum().item()
        rwpe_var = rwpe_part.var(dim=0).sum().item()
        rvar = rwpe_var / total_var * 100 if total_var > 0 else 0
        name = qf.stem
        print(f"{name:<30} {n:>3} {r_no:>9.2f}% {r_yes:>12.2f}% {dead:>10} {rvar:>9.2f}%")


if __name__ == "__main__":
    main()
