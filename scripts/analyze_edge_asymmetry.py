"""Analyze hardware graph features across all training backends.

Phase 1: Edge asymmetry analysis (error_asymmetry → confirmed negligible)
Phase 2: Data availability, per-backend distributions, feature correlations
  - Node features: existing 5 + candidates (T2/T1 ratio, measurement_duration, T1/T2 raw)
  - Edge features: existing 1 + candidates (2q_gate_duration, edge_coherence_ratio)
"""

import sys
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

sys.path.insert(0, ".")
from data.hardware_graph import _get_two_qubit_gate_name


# ─────────────────────────────────────────────────────────────
# Phase 1: Edge asymmetry (kept for reference)
# ─────────────────────────────────────────────────────────────

def analyze_asymmetry(backend) -> dict | None:
    """Analyze edge asymmetry for a single backend."""
    try:
        target = backend.target
        gate_name = _get_two_qubit_gate_name(backend)
    except (ValueError, AttributeError):
        return None

    cx_props = target[gate_name]

    directed: dict[tuple[int, int], dict] = {}
    for qargs, props in cx_props.items():
        if props is None:
            continue
        a, b = qargs
        entry = {}
        if props.error is not None:
            entry["error"] = props.error
        if props.duration is not None:
            entry["duration"] = props.duration
        if entry:
            directed[(a, b)] = entry

    error_ratios = []
    duration_ratios = []
    visited = set()
    for (a, b), props_ab in directed.items():
        key = (min(a, b), max(a, b))
        if key in visited:
            continue
        visited.add(key)
        props_ba = directed.get((b, a))
        if props_ba is None:
            continue
        if "error" in props_ab and "error" in props_ba:
            e_ab, e_ba = props_ab["error"], props_ba["error"]
            if e_ab > 0 and e_ba > 0:
                error_ratios.append(max(e_ab, e_ba) / min(e_ab, e_ba))
        if "duration" in props_ab and "duration" in props_ba:
            d_ab, d_ba = props_ab["duration"], props_ba["duration"]
            if d_ab > 0 and d_ba > 0:
                duration_ratios.append(max(d_ab, d_ba) / min(d_ab, d_ba))

    n_oneway = sum(1 for (a, b) in directed if (b, a) not in directed)
    return {
        "name": backend.name,
        "num_qubits": backend.num_qubits,
        "gate": gate_name,
        "n_pairs": len(visited),
        "n_oneway": n_oneway,
        "error_ratio_mean": float(np.mean(error_ratios)) if error_ratios else None,
        "error_pct_sym": float(np.mean(np.array(error_ratios) < 1.01) * 100) if error_ratios else None,
        "dur_ratio_mean": float(np.mean(duration_ratios)) if duration_ratios else None,
        "dur_pct_sym": float(np.mean(np.array(duration_ratios) < 1.01) * 100) if duration_ratios else None,
    }


# ─────────────────────────────────────────────────────────────
# Phase 2: Feature availability & distribution analysis
# ─────────────────────────────────────────────────────────────

def extract_all_features(backend) -> dict | None:
    """Extract all current + candidate features from a backend.

    Returns dict with:
      - node_features: {name: np.ndarray(num_qubits,)}
      - edge_features: {name: np.ndarray(num_edges,)}
      - meta: backend info
    """
    try:
        target = backend.target
        gate_name = _get_two_qubit_gate_name(backend)
    except (ValueError, AttributeError):
        return None

    num_qubits = target.num_qubits

    # ── Node features ──
    t1 = np.zeros(num_qubits)
    t2 = np.zeros(num_qubits)
    readout_error = np.zeros(num_qubits)
    single_qubit_error = np.zeros(num_qubits)
    measurement_duration = np.zeros(num_qubits)

    for q in range(num_qubits):
        qp = target.qubit_properties[q]
        t1[q] = qp.t1 if qp.t1 is not None else 0.0
        t2[q] = qp.t2 if qp.t2 is not None else 0.0

        # Readout error & measurement duration
        if "measure" in target.operation_names:
            meas_props = target["measure"]
            if (q,) in meas_props and meas_props[(q,)] is not None:
                readout_error[q] = meas_props[(q,)].error or 0.0
                measurement_duration[q] = meas_props[(q,)].duration or 0.0

        # Single-qubit error
        sq_errors = []
        for gn in ["sx", "x"]:
            if gn in target.operation_names:
                gate_props = target[gn]
                if (q,) in gate_props and gate_props[(q,)] is not None:
                    err = gate_props[(q,)].error
                    if err is not None:
                        sq_errors.append(err)
        single_qubit_error[q] = np.mean(sq_errors) if sq_errors else 0.0

    # Degree
    coupling_map = backend.coupling_map
    degree = np.zeros(num_qubits)
    neighbor_map: list[set[int]] = [set() for _ in range(num_qubits)]
    for edge in coupling_map.get_edges():
        neighbor_map[edge[0]].add(edge[1])
        neighbor_map[edge[1]].add(edge[0])
    for q in range(num_qubits):
        degree[q] = len(neighbor_map[q])

    # Per-qubit mean cx_duration
    cx_props = target[gate_name]
    cx_dur_sum = np.zeros(num_qubits)
    cx_dur_count = np.zeros(num_qubits)
    for qargs, props in cx_props.items():
        if props is None or props.duration is None:
            continue
        p, q = qargs
        cx_dur_sum[p] += props.duration
        cx_dur_sum[q] += props.duration
        cx_dur_count[p] += 1
        cx_dur_count[q] += 1
    mean_cx_dur = np.where(cx_dur_count > 0, cx_dur_sum / cx_dur_count, 1.0)

    t1_cx_ratio = np.where(mean_cx_dur > 0, t1 / mean_cx_dur, 0.0)
    t2_cx_ratio = np.where(mean_cx_dur > 0, t2 / mean_cx_dur, 0.0)

    # Candidate: T2/T1 ratio
    t2_t1_ratio = np.where(t1 > 0, t2 / t1, 0.0)

    node_features = {
        "readout_error": readout_error,
        "single_qubit_error": single_qubit_error,
        "degree": degree,
        "t1_cx_ratio": t1_cx_ratio,
        "t2_cx_ratio": t2_cx_ratio,
        "T1_raw": t1,
        "T2_raw": t2,
        "T2/T1_ratio": t2_t1_ratio,
        "meas_duration": measurement_duration,
    }

    # ── Edge features ──
    edge_dict: dict[tuple[int, int], dict] = {}
    for qargs, props in cx_props.items():
        if props is None:
            continue
        a, b = qargs
        key = (min(a, b), max(a, b))
        error = props.error if props.error is not None else 0.0
        duration = props.duration if props.duration is not None else 0.0
        if key not in edge_dict:
            edge_dict[key] = {"errors": [error], "durations": [duration],
                              "qubits": (min(a, b), max(a, b))}
        else:
            edge_dict[key]["errors"].append(error)
            edge_dict[key]["durations"].append(duration)

    edge_list = sorted(edge_dict.keys())
    gate_error = np.array([np.mean(edge_dict[e]["errors"]) for e in edge_list])
    gate_duration = np.array([np.mean(edge_dict[e]["durations"]) for e in edge_list])

    # Candidate: edge_coherence_ratio = cx_duration / min(T1_u, T2_v, T2_u, T1_v)
    # More precisely: cx_duration / min(T1_u, T1_v, T2_u, T2_v) — how much coherence budget this gate consumes
    edge_coherence = np.zeros(len(edge_list))
    for i, (u, v) in enumerate(edge_list):
        min_coherence = min(t1[u], t1[v], t2[u], t2[v])
        if min_coherence > 0:
            edge_coherence[i] = gate_duration[i] / min_coherence
        else:
            edge_coherence[i] = 0.0

    edge_features = {
        "2q_gate_error": gate_error,
        "2q_gate_duration": gate_duration,
        "edge_coherence_ratio": edge_coherence,
    }

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "meta": {
            "name": backend.name,
            "num_qubits": num_qubits,
            "gate": gate_name,
            "num_edges": len(edge_list),
        },
    }


def print_availability(all_data: list[dict]):
    """Step 1: Check data availability for all features across backends."""
    print("\n" + "=" * 90)
    print("STEP 1: DATA AVAILABILITY")
    print("=" * 90)

    # Node features
    node_names = list(all_data[0]["node_features"].keys())
    print(f"\n{'Backend':<22} {'Q':>4}", end="")
    for name in node_names:
        print(f" {name[:10]:>11}", end="")
    print()
    print("-" * (28 + 12 * len(node_names)))

    for d in all_data:
        m = d["meta"]
        print(f"{m['name']:<22} {m['num_qubits']:>4}", end="")
        for name in node_names:
            arr = d["node_features"][name]
            nonzero = np.count_nonzero(arr)
            total = len(arr)
            if nonzero == 0:
                print(f"   {'---':>8}", end="")
            else:
                print(f" {nonzero:>4}/{total:<4}", end="")
        print()

    # Edge features
    edge_names = list(all_data[0]["edge_features"].keys())
    print(f"\n{'Backend':<22} {'E':>4}", end="")
    for name in edge_names:
        print(f" {name[:14]:>15}", end="")
    print()
    print("-" * (28 + 16 * len(edge_names)))

    for d in all_data:
        m = d["meta"]
        print(f"{m['name']:<22} {m['num_edges']:>4}", end="")
        for name in edge_names:
            arr = d["edge_features"][name]
            nonzero = np.count_nonzero(arr)
            total = len(arr)
            if nonzero == 0:
                print(f"      {'---':>9}", end="")
            else:
                print(f"  {nonzero:>5}/{total:<5}", end="")
        print()


def print_distributions(all_data: list[dict]):
    """Step 2: Per-backend feature distributions (CoV = std/mean)."""
    print("\n" + "=" * 90)
    print("STEP 2: PER-BACKEND FEATURE DISTRIBUTIONS (Coefficient of Variation = std/|mean|)")
    print("=" * 90)

    # Node features
    node_names = list(all_data[0]["node_features"].keys())
    print("\n── Node Features ──")
    print(f"{'Backend':<22} {'Q':>4}", end="")
    for name in node_names:
        print(f" {name[:10]:>11}", end="")
    print()
    print("-" * (28 + 12 * len(node_names)))

    # Collect CoV across backends for summary
    cov_summary = {name: [] for name in node_names}

    for d in all_data:
        m = d["meta"]
        print(f"{m['name']:<22} {m['num_qubits']:>4}", end="")
        for name in node_names:
            arr = d["node_features"][name]
            if np.all(arr == 0):
                print(f"      {'n/a':>5}", end="")
            else:
                mean = np.mean(arr)
                std = np.std(arr)
                cov = std / abs(mean) if abs(mean) > 1e-15 else 0.0
                cov_summary[name].append(cov)
                print(f" {cov:>10.3f}", end="")
        print()

    # Summary row
    print("-" * (28 + 12 * len(node_names)))
    print(f"{'MEAN CoV':<22} {'':>4}", end="")
    for name in node_names:
        vals = cov_summary[name]
        if vals:
            print(f" {np.mean(vals):>10.3f}", end="")
        else:
            print(f"      {'n/a':>5}", end="")
    print()

    # Edge features
    edge_names = list(all_data[0]["edge_features"].keys())
    print("\n── Edge Features ──")
    print(f"{'Backend':<22} {'E':>4}", end="")
    for name in edge_names:
        print(f" {name[:14]:>15}", end="")
    print()
    print("-" * (28 + 16 * len(edge_names)))

    edge_cov_summary = {name: [] for name in edge_names}

    for d in all_data:
        m = d["meta"]
        print(f"{m['name']:<22} {m['num_edges']:>4}", end="")
        for name in edge_names:
            arr = d["edge_features"][name]
            if np.all(arr == 0):
                print(f"       {'n/a':>8}", end="")
            else:
                mean = np.mean(arr)
                std = np.std(arr)
                cov = std / abs(mean) if abs(mean) > 1e-15 else 0.0
                edge_cov_summary[name].append(cov)
                print(f" {cov:>14.3f}", end="")
        print()

    print("-" * (28 + 16 * len(edge_names)))
    print(f"{'MEAN CoV':<22} {'':>4}", end="")
    for name in edge_names:
        vals = edge_cov_summary[name]
        if vals:
            print(f" {np.mean(vals):>14.3f}", end="")
        else:
            print(f"       {'n/a':>8}", end="")
    print()


def print_correlations(all_data: list[dict]):
    """Step 3: Feature correlations (pooled across backends after z-score)."""
    print("\n" + "=" * 90)
    print("STEP 3: FEATURE CORRELATIONS (pooled across backends, per-backend z-score)")
    print("=" * 90)

    # ── Node features: pool z-scored features across all backends ──
    node_names = list(all_data[0]["node_features"].keys())
    pooled_node = {name: [] for name in node_names}

    for d in all_data:
        for name in node_names:
            arr = d["node_features"][name].copy()
            if np.std(arr) > 1e-15:
                arr = (arr - np.mean(arr)) / np.std(arr)
            else:
                arr = np.zeros_like(arr)
            pooled_node[name].append(arr)

    # Stack
    pooled_node = {name: np.concatenate(arrs) for name, arrs in pooled_node.items()}
    n_total = len(pooled_node[node_names[0]])

    print(f"\n── Node Feature Correlation Matrix (n={n_total} qubits) ──\n")

    # Print header
    short_names = [n[:10] for n in node_names]
    print(f"{'':>18}", end="")
    for sn in short_names:
        print(f" {sn:>11}", end="")
    print()

    # Correlation matrix
    node_matrix = np.column_stack([pooled_node[n] for n in node_names])
    corr = np.corrcoef(node_matrix.T)

    for i, name in enumerate(node_names):
        print(f"{name[:18]:<18}", end="")
        for j in range(len(node_names)):
            val = corr[i, j]
            marker = " **" if abs(val) > 0.9 and i != j else "   " if abs(val) > 0.7 and i != j else "   "
            if i == j:
                print(f"      {'1.000':>5}", end="")
            else:
                print(f" {val:>10.3f}{'' if abs(val) <= 0.9 else '*'}", end="")
        print()

    # Flag high correlations
    print("\n  ** High correlations (|r| > 0.9) → redundancy candidates:")
    found = False
    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):
            if abs(corr[i, j]) > 0.9:
                print(f"     {node_names[i]} ↔ {node_names[j]}: r = {corr[i, j]:.4f}")
                found = True
    if not found:
        print("     (none)")

    print("\n  Notable correlations (0.7 < |r| < 0.9):")
    found = False
    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):
            if 0.7 < abs(corr[i, j]) <= 0.9:
                print(f"     {node_names[i]} ↔ {node_names[j]}: r = {corr[i, j]:.4f}")
                found = True
    if not found:
        print("     (none)")

    # ── Edge features ──
    edge_names = list(all_data[0]["edge_features"].keys())
    pooled_edge = {name: [] for name in edge_names}

    for d in all_data:
        for name in edge_names:
            arr = d["edge_features"][name].copy()
            if np.std(arr) > 1e-15:
                arr = (arr - np.mean(arr)) / np.std(arr)
            else:
                arr = np.zeros_like(arr)
            pooled_edge[name].append(arr)

    pooled_edge = {name: np.concatenate(arrs) for name, arrs in pooled_edge.items()}
    n_edges_total = len(pooled_edge[edge_names[0]])

    print(f"\n── Edge Feature Correlation Matrix (n={n_edges_total} edges) ──\n")

    print(f"{'':>22}", end="")
    for en in edge_names:
        print(f" {en[:18]:>19}", end="")
    print()

    edge_matrix = np.column_stack([pooled_edge[n] for n in edge_names])
    ecorr = np.corrcoef(edge_matrix.T)

    for i, name in enumerate(edge_names):
        print(f"{name[:22]:<22}", end="")
        for j in range(len(edge_names)):
            if i == j:
                print(f"              {'1.000':>5}", end="")
            else:
                print(f" {ecorr[i, j]:>18.3f}", end="")
        print()

    print("\n  ** High correlations (|r| > 0.9):")
    found = False
    for i in range(len(edge_names)):
        for j in range(i + 1, len(edge_names)):
            if abs(ecorr[i, j]) > 0.9:
                print(f"     {edge_names[i]} ↔ {edge_names[j]}: r = {ecorr[i, j]:.4f}")
                found = True
    if not found:
        print("     (none)")

    print("\n  Notable correlations (0.7 < |r| < 0.9):")
    found = False
    for i in range(len(edge_names)):
        for j in range(i + 1, len(edge_names)):
            if 0.7 < abs(ecorr[i, j]) <= 0.9:
                print(f"     {edge_names[i]} ↔ {edge_names[j]}: r = {ecorr[i, j]:.4f}")
                found = True
    if not found:
        print("     (none)")


def main():
    provider = FakeProviderForBackendV2()
    backends = provider.backends()
    backends = sorted(backends, key=lambda b: b.num_qubits)

    print(f"Analyzing {len(backends)} backends...\n")

    # ── Phase 1: Edge asymmetry (compact summary) ──
    print("=" * 70)
    print("PHASE 1: EDGE ASYMMETRY SUMMARY (detailed run done previously)")
    print("=" * 70)
    asym_results = [analyze_asymmetry(b) for b in backends]
    asym_results = [r for r in asym_results if r is not None]

    err_sym = [r["error_pct_sym"] for r in asym_results if r["error_pct_sym"] is not None]
    dur_sym = [r["dur_pct_sym"] for r in asym_results if r["dur_pct_sym"] is not None]
    oneway = [r for r in asym_results if r["n_oneway"] > 0]

    print(f"  Error: {np.mean(err_sym):.1f}% of edges are symmetric (<1% diff) → NEGLIGIBLE")
    print(f"  Duration: {np.mean(dur_sym):.1f}% of edges are symmetric (<1% diff) → ~10% asymmetry exists")
    print(f"  One-way-only backends: {len(oneway)} ({', '.join(r['name'] for r in oneway)})")
    print("  → error_asymmetry feature: EXCLUDED")
    print("  → duration averaged for undirected edges")

    # ── Phase 2: Feature analysis ──
    all_data = []
    skipped = []
    for b in backends:
        d = extract_all_features(b)
        if d is not None:
            all_data.append(d)
        else:
            skipped.append(b.name)

    if skipped:
        print(f"\n  Skipped backends (no supported 2Q gate): {skipped}")

    print_availability(all_data)
    print_distributions(all_data)
    print_correlations(all_data)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("SUMMARY: FEATURE CANDIDATE ASSESSMENT")
    print("=" * 70)


if __name__ == "__main__":
    main()
