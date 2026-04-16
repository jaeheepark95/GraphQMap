"""Evaluate our trained model using reference paper's PST measurement pipeline.

Uses:
- add_measurements (measures only used qubits) from attn_map paper
- build_transpiler (custom PassManager) from butils
- PSTv2 from butils

Compares: SABRE baseline vs OURS (our model layout).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "references/Attention_Qubit_Mapping"))
sys.path.insert(0, str(ROOT / "references/Attention_Qubit_Mapping/MQM/tests2"))

from butils import PSTv2, build_transpiler  # type: ignore

from configs.config_loader import load_config
from data.circuit_graph import build_circuit_graph
from data.hardware_graph import build_hardware_graph, get_backend
from evaluation.benchmark import POZZI_TEST2_CIRCUITS
from models.graphqmap import GraphQMap


SHOTS = 8192
OPT_LEVEL = int(os.environ.get("OPT_LEVEL", "3"))
USE_NORM = os.environ.get("USE_NORM", "1") == "1"
BENCH_DIR = ROOT / "data/circuits/qasm/pozzi_benchmarks"  # raw Clifford+T
NORMALIZED_DIR = ROOT / "data/circuits/qasm/benchmarks"


def load_circuit_no_measure(name: str, use_normalized: bool = True) -> QuantumCircuit:
    path = (NORMALIZED_DIR if use_normalized else BENCH_DIR) / f"{name}.qasm"
    qc = QuantumCircuit.from_qasm_file(str(path))
    return qc.decompose() if use_normalized else qc


def add_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Add measurements to USED qubits only (matching ref paper)."""
    if qc.count_ops().get("measure", 0) > 0:
        return qc
    used = set()
    for inst in qc.data:
        for q in inst.qubits:
            used.add(qc.find_bit(q).index)
    new_qc = qc.copy()
    cr = ClassicalRegister(len(used), "meas")
    new_qc.add_register(cr)
    for idx, q in enumerate(sorted(used)):
        new_qc.measure(q, idx)
    return new_qc


def measure_pst(tc: QuantumCircuit, backend) -> float:
    noisy_sim = AerSimulator.from_backend(backend)
    ideal_sim = AerSimulator.from_backend(backend, noise_model=None)
    ic = ideal_sim.run(tc, shots=SHOTS).result().get_counts()
    nc = noisy_sim.run(tc, shots=SHOTS).result().get_counts()
    pst = PSTv2(nc, ic)
    return (sum(pst) / len(pst)) if isinstance(pst, list) else pst


def predict_layout(model, cfg, circ: QuantumCircuit, backend, device) -> dict:
    """Run our model to predict layout (logical -> physical)."""
    node_fnames = getattr(cfg.model.circuit_gnn, "node_features", None)
    rk = getattr(cfg.model.circuit_gnn, "rwpe_k", 0)
    edim = getattr(cfg.model.circuit_gnn, "edge_input_dim", None)
    tau = getattr(cfg.sinkhorn, "tau_min", 0.5)

    # Used qubits only (for consistency with add_measurements)
    used = sorted({circ.find_bit(q).index for inst in circ.data for q in inst.qubits})
    # Build a reduced circuit that only has the used qubits for graph construction
    l = len(used)
    num_physical = backend.target.num_qubits

    c_graph = build_circuit_graph(
        circ, node_feature_names=node_fnames, rwpe_k=rk, edge_dim=edim,
    )
    hw_graph = build_hardware_graph(backend)
    c_batch = Batch.from_data_list([c_graph]).to(device)
    hw_batch = Batch.from_data_list([hw_graph]).to(device)

    with torch.no_grad():
        # Our model forward: full logical count = qreg size typically 16
        # Score head returns (1, l_total, h). We need only used qubits' rows.
        C, H = model.encode(c_batch, hw_batch, 1,
                            c_graph.num_nodes, num_physical)
        if model.cross_attention is not None:
            C_prime, H_prime = model.cross_attention(C, H)
        else:
            C_prime, H_prime = C, H
        S = model.score_head(C_prime, H_prime, None)
        P = model.sinkhorn(S, c_graph.num_nodes, num_physical, tau)
        if P.shape[1] != c_graph.num_nodes:
            P = P[:, :c_graph.num_nodes, :]

    P_np = P[0].cpu().numpy()  # (l_total, h)

    # Greedy assignment with conflict avoidance, in confidence order
    layout = {}  # logical idx -> physical idx
    used_phys = set()
    order = np.argsort(-P_np.max(axis=1))  # most confident first
    for i in order:
        ranked = np.argsort(-P_np[i])
        for p in ranked:
            if int(p) not in used_phys:
                layout[int(i)] = int(p)
                used_phys.add(int(p))
                break
    return layout


def run(config_path: str, checkpoint_path: str, backends: list[str],
        circuits: list[str]) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphQMap.from_config(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    backend_map = {
        "toronto": fake_provider.FakeTorontoV2(),
        "rochester": fake_provider.FakeRochesterV2(),
        "washington": fake_provider.FakeWashingtonV2(),
    }

    print(f"{'Circuit':<20} {'Backend':<12} {'SABRE':>8} {'OURS':>8}")
    print("-" * 50)

    for bname in backends:
        backend = backend_map[bname]
        for cname in circuits:
            try:
                qc = load_circuit_no_measure(cname, use_normalized=USE_NORM)
                qc_m = add_measurements(qc)

                # SABRE baseline via ref build_transpiler
                pm_sabre = build_transpiler(backend, layout_method="sabre",
                                             routing_method="sabre",
                                             optimization_level=OPT_LEVEL)
                tc_sabre = pm_sabre.run(qc_m)
                pst_sabre = measure_pst(tc_sabre, backend)

                # OURS: predict layout, then ref transpiler with given layout
                layout = predict_layout(model, cfg, qc, backend, device)
                # Convert dict to list: list[logical_idx] = physical_idx
                l_count = qc.num_qubits
                layout_list = [layout.get(i, 0) for i in range(l_count)]
                pm_ours = build_transpiler(backend, initial_layout=layout_list,
                                            layout_method="sabre",
                                            routing_method="sabre",
                                            optimization_level=OPT_LEVEL)
                tc_ours = pm_ours.run(qc_m)
                pst_ours = measure_pst(tc_ours, backend)

                print(f"{cname:<20} {bname:<12} {pst_sabre:>8.2f} {pst_ours:>8.2f}")
            except Exception as e:
                print(f"{cname:<20} {bname:<12} ERROR: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backends", nargs="+", default=["toronto", "rochester"])
    ap.add_argument("--circuits", nargs="+", default=None)
    args = ap.parse_args()

    circuits = args.circuits if args.circuits else POZZI_TEST2_CIRCUITS
    run(args.config, args.checkpoint, args.backends, circuits)
