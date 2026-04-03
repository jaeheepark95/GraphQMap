from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


class CircuitFeaturizer:
    """Exact v1.1 circuit featurization."""

    def __init__(self, n: int = 27):
        self.n = n

    def featurize(self, circuit: QuantumCircuit):
        K = circuit.num_qubits
        if K > self.n:
            raise ValueError(f'Circuit size K={K} exceeds fixed hardware size n={self.n}')

        W = np.zeros((self.n, self.n), dtype=np.float32)
        m = np.zeros(self.n, dtype=np.float32)

        for instruction in circuit.data:
            qargs = instruction.qubits
            if len(qargs) == 2:
                u = circuit.find_bit(qargs[0]).index
                v = circuit.find_bit(qargs[1]).index
                W[u, v] += 1.0
                W[v, u] += 1.0

        m[:K] = 1.0
        return W, m
