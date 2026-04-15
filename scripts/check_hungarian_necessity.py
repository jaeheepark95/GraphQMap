"""Check if Hungarian algorithm is necessary at inference.

Analyzes the P matrix after Sinkhorn to determine:
1. How close P is to a permutation matrix (max values per row)
2. Whether row-wise argmax produces conflicts (same physical qubit for multiple logical)
3. Whether argmax layout matches Hungarian layout

Usage:
    python scripts/check_hungarian_necessity.py \
        --config configs/base.yaml \
        --checkpoint runs/stage2/<RUN>/checkpoints/best.pt \
        --backend toronto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch_geometric.data import Batch

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph
from data.hardware_graph import build_hardware_graph, get_backend
from evaluation.benchmark import (
    BENCHMARK_CIRCUIT_DIR,
    BENCHMARK_CIRCUITS,
    load_benchmark_circuit,
)
from models.graphqmap import GraphQMap
from models.hungarian import hungarian_decode_batch


def analyze_p_matrix(
    P: torch.Tensor, num_logical: int,
) -> dict:
    """Analyze a single P matrix (h, h)."""
    P_np = P.numpy()
    l, h = num_logical, P_np.shape[0]

    # Only look at the logical rows (first l rows)
    P_logical = P_np[:l]  # (l, h)

    # Per-row max values — how "sharp" is the assignment?
    row_maxes = P_logical.max(axis=1)  # (l,)
    row_argmaxes = P_logical.argmax(axis=1)  # (l,)

    # Check for argmax conflicts
    unique_assignments = len(set(row_argmaxes))
    has_conflict = unique_assignments < l

    # Hungarian result
    hungarian_layout = hungarian_decode_batch(P.unsqueeze(0), num_logical)[0]
    hungarian_assignments = np.array([hungarian_layout[i] for i in range(l)])

    # Do argmax and Hungarian agree?
    argmax_matches_hungarian = np.array_equal(row_argmaxes, hungarian_assignments)

    # Row entropy (lower = sharper)
    row_entropies = -np.sum(P_logical * np.log(P_logical + 1e-12), axis=1)

    # Dummy rows analysis
    P_dummy = P_np[l:]  # (h-l, h)
    dummy_row_maxes = P_dummy.max(axis=1) if l < h else np.array([])

    return {
        "num_logical": l,
        "num_physical": h,
        "row_max_min": float(row_maxes.min()),
        "row_max_mean": float(row_maxes.mean()),
        "row_max_max": float(row_maxes.max()),
        "row_entropy_mean": float(row_entropies.mean()),
        "argmax_assignments": row_argmaxes.tolist(),
        "hungarian_assignments": hungarian_assignments.tolist(),
        "has_argmax_conflict": has_conflict,
        "argmax_matches_hungarian": argmax_matches_hungarian,
        "dummy_row_max_mean": float(dummy_row_maxes.mean()) if len(dummy_row_maxes) > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", default="toronto")
    parser.add_argument("--tau", type=float, default=None, help="Override tau (default: use config tau_min)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")

    # Load model
    model = GraphQMap.from_config(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load backend
    backend = get_backend(args.backend)
    num_physical = backend.target.num_qubits
    hw_graph = build_hardware_graph(backend)

    tau = args.tau or getattr(cfg.sinkhorn, "tau_min", getattr(cfg.sinkhorn, "tau", 0.05))
    print(f"Backend: {args.backend} ({num_physical}Q), tau={tau}")
    print(f"Circuits: {len(BENCHMARK_CIRCUITS)}")
    print("=" * 80)

    total_conflicts = 0
    total_mismatches = 0
    total_circuits = 0

    for cname in BENCHMARK_CIRCUITS:
        circuit = load_benchmark_circuit(cname, BENCHMARK_CIRCUIT_DIR)
        if circuit is None:
            print(f"  SKIP {cname}: not found")
            continue

        num_logical = circuit.num_qubits
        if num_logical > num_physical:
            print(f"  SKIP {cname}: {num_logical}Q > {num_physical}Q")
            continue

        circuit_graph = build_circuit_graph(circuit)
        circuit_batch = Batch.from_data_list([circuit_graph])
        hw_batch = Batch.from_data_list([hw_graph])

        with torch.no_grad():
            P = model.forward(
                circuit_batch, hw_batch,
                batch_size=1,
                num_logical=num_logical,
                num_physical=num_physical,
                tau=tau,
            )  # (1, h, h)

        result = analyze_p_matrix(P[0], num_logical)
        total_circuits += 1
        if result["has_argmax_conflict"]:
            total_conflicts += 1
        if not result["argmax_matches_hungarian"]:
            total_mismatches += 1

        conflict_str = "CONFLICT" if result["has_argmax_conflict"] else "ok"
        match_str = "MISMATCH" if not result["argmax_matches_hungarian"] else "match"

        print(f"\n{cname} ({num_logical}Q → {num_physical}Q)")
        print(f"  Row max:  min={result['row_max_min']:.6f}  mean={result['row_max_mean']:.6f}  max={result['row_max_max']:.6f}")
        print(f"  Entropy:  mean={result['row_entropy_mean']:.6f}")
        if result["dummy_row_max_mean"] is not None:
            print(f"  Dummy row max mean: {result['dummy_row_max_mean']:.6f}")
        print(f"  Argmax:    {result['argmax_assignments']}  [{conflict_str}]")
        print(f"  Hungarian: {result['hungarian_assignments']}  [{match_str}]")

    print("\n" + "=" * 80)
    print(f"Summary: {total_circuits} circuits")
    print(f"  Argmax conflicts:              {total_conflicts}/{total_circuits}")
    print(f"  Argmax != Hungarian:           {total_mismatches}/{total_circuits}")
    print(f"  Hungarian strictly necessary:  {'YES' if total_conflicts > 0 else 'NO (argmax sufficient)'}")


if __name__ == "__main__":
    main()
