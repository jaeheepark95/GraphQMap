"""QAP (Quadratic Assignment Problem) based qubit layout pass.

Ported from MQM project (colleague) for use as a baseline comparison.
Solves initial layout via gradient-based QAP formulation with
adjacency-aware greedy rounding.

Reference: MQM project — mathematical matrix-based qubit mapping optimization.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import rustworkx as rx

from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.dagcircuit import DAGCircuit

BIG = np.inf


class QAPLayout(AnalysisPass):
    """Quadratic Assignment Problem (QAP) based qubit layout pass.

    Formulates qubit mapping as a QAP optimization:
        minimize  2·W·(X·Ac) + γ·(E_read + E_gate)

    Where:
        - W: Floyd-Warshall distance matrix weighted by log gate errors
        - Ac: circuit adjacency matrix (2Q gate connectivity)
        - E_read: readout error cost matrix
        - E_gate: single-qubit gate error cost matrix
        - X: assignment matrix (physical → logical)

    The relaxed solution is converted to a discrete mapping via
    adjacency-aware greedy rounding (largest circuit first, then by
    degree and gradient cost).

    Args:
        backend: Target FakeBackendV2 for transpilation.
    """

    def __init__(self, backend):
        super().__init__()
        self.Ah, self.W, self.Rr, self.Rs = self._parse_topology_to_matrices(backend)
        self.backend = backend
        self.num_qubits = backend.num_qubits

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run QAP layout pass on the given DAG circuit.

        Sets property_set["layout"] with the computed layout.
        """
        Ac, I, nqc = self._parse_dag_to_matrix(dag)
        E_read = np.outer(self.Rr, np.ones(Ac.shape[0], dtype=float))
        E_gate = self.Rs @ I.T

        # Relaxed assignment matrix X (uniform initialization)
        X = np.ones((self.Ah.shape[0], Ac.shape[0]), dtype=float)
        X /= X.sum(axis=0, keepdims=True) + 1e-12

        # Compute gradient: 2·W·X·A + γ·(E_read + E_gate)
        gTwo = 2 * (self.W @ (X @ Ac))
        gOne = E_read + E_gate
        # Normalize to prevent dominance of distance term
        gOne *= gTwo.mean() / (gOne.mean() + 1e-12)
        grad = gTwo + gOne

        maps = self._continuous_to_binary(grad, Ac, nqc)
        self.property_set["layout"] = Layout({
            phys: logi for logi, phys in zip(dag.qubits, maps)
        })
        return dag

    def _parse_topology_to_matrices(self, backend) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ]:
        """Extract hardware matrices from backend.

        Returns:
            Ah: Coupling map adjacency matrix (directed).
            W: Floyd-Warshall distance matrix weighted by -log(1-error).
            Rr: Readout error vector per qubit (-log(1-p_readout)).
            Rs: Single-qubit gate error vector per qubit (-log(1-p_gate)).
        """
        n_qubits = backend.num_qubits
        coupling_map = backend.coupling_map.graph.copy()
        props = backend.properties()

        def get_edge_error(two_qubit_gate):
            v = next(iter(two_qubit_gate.values()))
            return -np.log(np.maximum(1.0 - v.error, 1e-12))

        Ah = rx.digraph_adjacency_matrix(coupling_map)
        W = rx.digraph_floyd_warshall_numpy(coupling_map, weight_fn=get_edge_error)

        # Collect single-qubit gate names
        gate_names = []
        for inst in backend.operations:
            if inst.num_qubits == 1 and inst.name not in ["measure", "reset", "delay"]:
                gate_names.append(inst.name)

        Rr = np.zeros(n_qubits, dtype=float)
        Rs = np.zeros(n_qubits, dtype=float)
        for i in range(n_qubits):
            read_error = props.readout_error(i)
            Rr[i] = -np.log(np.maximum(1.0 - read_error, 1e-12))

            errors = [props.gate_error(gname, [i]) for gname in gate_names]
            max_err = np.max(errors) if errors else 0.0
            Rs[i] = -np.log(np.maximum(1.0 - max_err, 1e-12))

        Rs = Rs[:, None]
        return Ah, W, Rr, Rs

    def _parse_dag_to_matrix(self, dag: DAGCircuit) -> Tuple[
        np.ndarray, np.ndarray, List[List[int]],
    ]:
        """Parse DAG circuit into adjacency matrix and instruction count.

        Returns:
            Ac: Circuit adjacency matrix (k×k, 2Q gate connectivity).
            I: Instruction count vector (k×1, 1Q gate counts).
            qubits_for_each_circuit: List of qubit index lists per sub-circuit.
        """
        k = dag.num_qubits()
        Ac = np.zeros((k, k), dtype=float)
        I = np.zeros((k,), dtype=float)
        qubit_index = {q: idx for idx, q in enumerate(dag.qubits)}

        qubits_for_each_circuit = []
        for n, q in enumerate(dag.qubits):
            if q._index == 0:
                qubits_for_each_circuit.append([])
            qubits_for_each_circuit[-1].append(n)

        for gate in dag.gate_nodes():
            if gate.op.num_qubits == 1:
                i = qubit_index[gate.qargs[0]]
                I[i] += 1.0
            elif gate.op.num_qubits == 2:
                q1, q2 = gate.qargs[0], gate.qargs[1]
                i, j = qubit_index[q1], qubit_index[q2]
                Ac[i, j] = 1.0
                Ac[j, i] = 1.0

        I = I[:, None]
        return Ac, I, qubits_for_each_circuit

    def _continuous_to_binary(self, G, Ac, qubits_for_each_circuit):
        """Convert relaxed gradient matrix to discrete binary assignment.

        Uses adjacency-aware greedy rounding:
        1. Sort qubits by circuit size (largest first), then by degree and cost.
        2. For each logical qubit, select the best available physical qubit
           that neighbors already-assigned qubits.

        Args:
            G: Gradient cost matrix (n_physical × n_logical).
            Ac: Circuit adjacency matrix.
            qubits_for_each_circuit: List of qubit index lists per sub-circuit.

        Returns:
            Mapping array where mapping[logical] = physical.
        """
        sorted_qubit_order = []
        for nq in sorted(qubits_for_each_circuit, key=len, reverse=True):
            s, e = nq[0], nq[-1] + 1
            G_, Ac_ = G[:, s:e], Ac[s:e, s:e]

            col_maxvals = G_.max(axis=0)
            col_degrees = Ac_.sum(axis=0, dtype=int)
            sort_keys = np.lexsort((-col_maxvals, -col_degrees))

            sorted_qubit_order.extend([
                (col_maxvals[i], col_degrees[i], nq[i], idx == 0)
                for idx, i in enumerate(sort_keys)
            ])

        n = G.shape[0]
        X = np.zeros_like(G, dtype=int)
        temp_cost = np.zeros(n, dtype=float)
        nbr_mask = np.zeros(n, dtype=bool)

        for _, deg, j, is_other_circ_qubit in sorted_qubit_order:
            np.copyto(temp_cost, G[:, j])
            if not is_other_circ_qubit and deg > 0:
                temp_cost[~nbr_mask] = BIG
            r_best = int(np.argmin(temp_cost))
            if G[r_best, j] < BIG:
                X[r_best, j] = 1
                G[r_best, :] = BIG
                if is_other_circ_qubit:
                    nbr_mask[:] = False
                nbr_mask |= (self.Ah[r_best, :] > 0)
            else:
                raise ValueError(
                    "Failed to find valid mapping."
                    " Please check the circuit and backend connectivity."
                )

        return np.where(X.any(axis=0), np.argmax(X, axis=0), -1)
