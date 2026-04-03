from __future__ import annotations

from collections import deque
import numpy as np


class CanonicalIndexer:
    """Exact v1.1 canonical BFS ordering with inverse permutation support."""

    def __init__(self, n: int = 27):
        self.n = n

    @staticmethod
    def _z_score(x: np.ndarray) -> np.ndarray:
        std = np.std(x)
        return (x - np.mean(x)) / std if std > 0 else x - np.mean(x)

    def get_permutation(self, Anat, c1nat, c2nat):
        adj = (Anat + Anat.T > 0).astype(int)
        degrees = adj.sum(axis=1)

        mean_edge_costs = []
        for i in range(self.n):
            neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) > 0:
                costs = [min(c2nat[i, j], c2nat[j, i]) for j in neighbors]
                mean_edge_costs.append(float(np.mean(costs)))
            else:
                mean_edge_costs.append(1.0)
        mean_edge_costs = np.asarray(mean_edge_costs, dtype=np.float32)

        qscores = (
            self._z_score(c1nat)
            + self._z_score(mean_edge_costs)
            - 0.3 * self._z_score(degrees)
        )

        p = []
        visited = [False] * self.n
        while len(p) < self.n:
            remaining = [i for i in range(self.n) if not visited[i]]
            if not remaining:
                break
            root = remaining[int(np.argmin(qscores[remaining]))]
            queue = deque([root])
            visited[root] = True
            p.append(root)

            while queue:
                u = queue.popleft()
                neighbors = [v for v in np.where(adj[u] > 0)[0] if not visited[v]]
                neighbors.sort(
                    key=lambda v: (
                        min(c2nat[u, v], c2nat[v, u]),
                        c1nat[v],
                        -degrees[v],
                        v,
                    )
                )
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
                        p.append(v)

        p = np.asarray(p, dtype=np.int64)
        p_inv = np.empty_like(p)
        p_inv[p] = np.arange(self.n, dtype=np.int64)
        return p, p_inv

    def canonicalize(self, Anat, c1nat, c2nat, p):
        A = Anat[p][:, p]
        c1 = c1nat[p]
        c2 = c2nat[p][:, p]
        return A, c1, c2
