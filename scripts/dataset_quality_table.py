"""Per-category dataset quality summary table.

Reads cached preprocessed circuits and computes a single comparison table
across the 5 training datasets (queko, mlqd, mqt_bench, qasmbench, revlib).
The goal is to identify categories that warrant additional filtering.

Metrics:
  Volume:    total cached, in active train_all split
  Size:      num_qubits (mean / median / max)
  Workload:  total_gates (mean / median), 2q_gates (mean), gates_per_qubit
  Edge:      num_edges (mean), edge_density (edges / max_possible)
  Quality:   indist_rate (mean), high_indist (% with rate>0.3),
             eff_dim (mean), sqr_const %, cpf_const %
  Hygiene:   num_qubits<2, empty_2q (no 2-qubit gates), 2q-singleton qubits

Output:
  - stdout markdown table
  - data/circuits/splits/dataset_quality.md  (saved table)
  - data/circuits/splits/dataset_quality.csv (machine-readable)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.circuit_graph import build_circuit_graph_from_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_ROOT = Path("data/circuits/cache")
SPLITS_DIR = Path("data/circuits/splits")
DATASETS = ["queko", "mlqd", "mqt_bench", "qasmbench", "revlib"]
INDIST_THRESHOLD = 0.3  # consistent with current filter


def analyze_one(cache_path: Path) -> dict | None:
    try:
        d = torch.load(cache_path, weights_only=False, map_location="cpu")
    except Exception:
        return None
    if not isinstance(d, dict) or "node_features_dict" not in d:
        return None

    nfd = d["node_features_dict"]
    nq = int(d.get("num_logical", len(next(iter(nfd.values())))))
    if nq < 2:
        return {"num_qubits": nq, "_skip_quality": True}

    gc = np.asarray(nfd["gate_count"], dtype=np.float64)
    twoqc = np.asarray(nfd["two_qubit_gate_count"], dtype=np.float64)
    sqr = np.asarray(nfd["single_qubit_gate_ratio"], dtype=np.float64)
    cpf = np.asarray(nfd["critical_path_fraction"], dtype=np.float64)

    edge_list = d.get("edge_list", [])
    num_edges = len(edge_list)
    max_edges = nq * (nq - 1) // 2
    edge_density = num_edges / max_edges if max_edges > 0 else 0.0

    total_gates = float(gc.sum())
    total_2q = float(twoqc.sum() / 2)  # each 2q gate counted on both endpoints
    gates_per_qubit = total_gates / nq

    # Build z-scored feature matrix (current default features + RWPE) for indist
    edge_features = d.get("edge_features", torch.zeros((0, 3), dtype=torch.float32))
    if isinstance(edge_features, np.ndarray):
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
    graph = build_circuit_graph_from_raw(
        node_features_dict=nfd,
        edge_list=edge_list,
        edge_features=edge_features,
        num_qubits=nq,
        node_feature_names=["gate_count", "two_qubit_gate_count",
                            "single_qubit_gate_ratio", "critical_path_fraction"],
        rwpe_k=2,
    )
    x = graph.x  # (nq, 6)

    # Effective dim
    _, S, _ = torch.svd(x)
    eff_dim = int((S > S[0] * 0.01).sum().item()) if S[0] > 0 else 0

    # Indist rate (cosine > 0.95)
    xn = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    cos = (xn @ xn.T).triu(diagonal=1)
    total_pairs = nq * (nq - 1) // 2
    indist_pairs = int((cos > 0.95).sum().item())
    indist_rate = indist_pairs / total_pairs if total_pairs > 0 else 0.0

    sqr_constant = len(set(f"{v:.6f}" for v in sqr)) <= 1
    cpf_constant = len(set(f"{v:.6f}" for v in cpf)) <= 1

    # Hygiene: qubits with no 2q interactions at all
    isolated_2q = int((twoqc == 0).sum())

    return {
        "num_qubits": nq,
        "total_gates": total_gates,
        "total_2q": total_2q,
        "gates_per_qubit": gates_per_qubit,
        "num_edges": num_edges,
        "edge_density": edge_density,
        "eff_dim": eff_dim,
        "indist_rate": indist_rate,
        "sqr_constant": sqr_constant,
        "cpf_constant": cpf_constant,
        "isolated_2q": isolated_2q,
        "isolated_2q_frac": isolated_2q / nq,
        "no_2q_circuit": total_2q == 0,
    }


def aggregate(records: list[dict]) -> dict:
    valid = [r for r in records if not r.get("_skip_quality")]
    if not valid:
        return {"n": 0}
    nq = np.array([r["num_qubits"] for r in valid])
    tg = np.array([r["total_gates"] for r in valid])
    t2 = np.array([r["total_2q"] for r in valid])
    gpq = np.array([r["gates_per_qubit"] for r in valid])
    ne = np.array([r["num_edges"] for r in valid])
    ed = np.array([r["edge_density"] for r in valid])
    ef = np.array([r["eff_dim"] for r in valid])
    ir = np.array([r["indist_rate"] for r in valid])
    return {
        "n": len(valid),
        "nq_mean": nq.mean(),
        "nq_median": np.median(nq),
        "nq_max": nq.max(),
        "tg_mean": tg.mean(),
        "tg_median": np.median(tg),
        "t2_mean": t2.mean(),
        "gpq_mean": gpq.mean(),
        "ne_mean": ne.mean(),
        "ed_mean": ed.mean(),
        "ef_mean": ef.mean(),
        "ir_mean": ir.mean(),
        "high_indist_pct": float(np.mean(ir > INDIST_THRESHOLD) * 100),
        "sqr_const_pct": float(np.mean([r["sqr_constant"] for r in valid]) * 100),
        "cpf_const_pct": float(np.mean([r["cpf_constant"] for r in valid]) * 100),
        "no_2q_pct": float(np.mean([r["no_2q_circuit"] for r in valid]) * 100),
        "isolated_2q_frac_mean": float(np.mean([r["isolated_2q_frac"] for r in valid]) * 100),
    }


def main() -> None:
    # Active set membership (post-filtering)
    stage2 = json.load(open(SPLITS_DIR / "train_all.json"))
    active = {(e["source"], Path(e["file"]).stem) for e in stage2}

    cat_stats: dict[str, dict] = {}

    for src in DATASETS:
        cdir = CACHE_ROOT / src
        if not cdir.exists():
            logger.warning("missing cache dir: %s", cdir)
            continue
        files = sorted(cdir.glob("*.pt"))
        logger.info("Analyzing %s (%d files)...", src, len(files))

        all_recs: list[dict] = []
        active_recs: list[dict] = []
        for f in files:
            r = analyze_one(f)
            if r is None:
                continue
            all_recs.append(r)
            if (src, f.stem) in active:
                active_recs.append(r)

        agg_all = aggregate(all_recs)
        agg_active = aggregate(active_recs)
        cat_stats[src] = {
            "cached": len(all_recs),
            "active": agg_active["n"],
            "removed": len(all_recs) - agg_active["n"],
            "removed_pct": (len(all_recs) - agg_active["n"]) / max(len(all_recs), 1) * 100,
            **{f"all_{k}": v for k, v in agg_all.items()},
            **{f"act_{k}": v for k, v in agg_active.items()},
        }

    # ---- Build markdown table ----
    lines = []
    lines.append("# Dataset Quality Summary (per category)\n")
    lines.append(f"Computed on cached preprocessed circuits. Quality metrics use active "
                 f"(post-filter) circuits only. indist threshold = {INDIST_THRESHOLD}.\n")

    headers = [
        "dataset", "cached", "active", "removed%",
        "nq_med", "nq_max",
        "gates_med", "2q_avg", "g/q",
        "edges_avg", "edge_dens",
        "eff_dim", "indist_avg", "high_indist%",
        "sqr_C%", "cpf_C%", "no_2q%", "iso_q%",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    csv_rows = [headers]
    for src in DATASETS:
        s = cat_stats.get(src)
        if not s:
            continue
        row = [
            src,
            s["cached"],
            s["active"],
            f"{s['removed_pct']:.1f}",
            f"{s['act_nq_median']:.0f}",
            f"{s['act_nq_max']:.0f}",
            f"{s['act_tg_median']:.0f}",
            f"{s['act_t2_mean']:.1f}",
            f"{s['act_gpq_mean']:.1f}",
            f"{s['act_ne_mean']:.1f}",
            f"{s['act_ed_mean']:.2f}",
            f"{s['act_ef_mean']:.2f}",
            f"{s['act_ir_mean']:.3f}",
            f"{s['act_high_indist_pct']:.1f}",
            f"{s['act_sqr_const_pct']:.1f}",
            f"{s['act_cpf_const_pct']:.1f}",
            f"{s['act_no_2q_pct']:.1f}",
            f"{s['act_isolated_2q_frac_mean']:.1f}",
        ]
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
        csv_rows.append(row)

    table = "\n".join(lines)
    print()
    print(table)
    print()

    md_path = SPLITS_DIR / "dataset_quality.md"
    md_path.write_text(table + "\n")
    csv_path = SPLITS_DIR / "dataset_quality.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    logger.info("Saved: %s, %s", md_path, csv_path)

    # ---- Filtering recommendations ----
    print("\n## Filtering signals\n")
    for src in DATASETS:
        s = cat_stats.get(src)
        if not s:
            continue
        flags = []
        if s["act_high_indist_pct"] > 10:
            flags.append(f"high_indist={s['act_high_indist_pct']:.0f}%")
        if s["act_sqr_const_pct"] > 30:
            flags.append(f"sqr_const={s['act_sqr_const_pct']:.0f}%")
        if s["act_cpf_const_pct"] > 30:
            flags.append(f"cpf_const={s['act_cpf_const_pct']:.0f}%")
        if s["act_no_2q_pct"] > 1:
            flags.append(f"no_2q={s['act_no_2q_pct']:.1f}%")
        if s["act_isolated_2q_frac_mean"] > 20:
            flags.append(f"isolated_q={s['act_isolated_2q_frac_mean']:.0f}%")
        if s["act_ef_mean"] < 3.0:
            flags.append(f"low_eff_dim={s['act_ef_mean']:.1f}")
        if flags:
            print(f"  {src}: " + ", ".join(flags))
        else:
            print(f"  {src}: clean")


if __name__ == "__main__":
    main()
