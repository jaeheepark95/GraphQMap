from __future__ import annotations

import numpy as np
from qiskit.transpiler import Target


class BackendV2Extractor:
    """Exact v1.1 backend extraction behavior."""

    def __init__(self, n: int = 27):
        self.n = n

    def extract(self, backend):
        return self.extract_tensors(backend)

    def extract_tensors(self, backend):
        target: Target = backend.target
        Anat = np.zeros((self.n, self.n), dtype=np.float32)
        c1nat = np.zeros(self.n, dtype=np.float32)
        c2nat = np.zeros((self.n, self.n), dtype=np.float32)

        two_q_gate = None
        for gate in ['cx', 'ecr']:
            if gate in target.operation_names:
                two_q_gate = gate
                break

        if two_q_gate is not None:
            for qargs, props in target[two_q_gate].items():
                if props and props.error is not None:
                    i, j = qargs
                    Anat[i, j] = 1.0
                    c2nat[i, j] = props.error

        for i in range(self.n):
            q_error_sum = 0.0
            if 'measure' in target.operation_names:
                m_props = target['measure'].get((i,), None)
                if m_props and m_props.error is not None:
                    q_error_sum += m_props.error

            one_q_errors = []
            for gate in ['sx', 'x', 'id']:
                if gate in target.operation_names:
                    props = target[gate].get((i,), None)
                    if props and props.error is not None:
                        one_q_errors.append(props.error)
            if one_q_errors:
                q_error_sum += float(np.max(one_q_errors))
            c1nat[i] = q_error_sum

        return Anat, c1nat, c2nat
