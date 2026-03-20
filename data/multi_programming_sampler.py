"""Multi-programming data sampler for GraphQMap.

Generates training samples by combining multiple circuits for
multi-programming scenarios (1, 2, or 4 circuits per sample).

Combination rules:
  - Total logical qubits < physical qubit count
  - Occupancy range: 30-75%
  - Training ratio: 50% single / 30% dual / 20% quad
"""

from __future__ import annotations

import random
from typing import Any

from qiskit import QuantumCircuit


def sample_multi_programming_groups(
    circuits: list[QuantumCircuit],
    num_physical: int,
    ratios: dict[str, float] | None = None,
    occupancy_min: float = 0.30,
    occupancy_max: float = 0.75,
    num_samples: int | None = None,
    rng_seed: int | None = None,
) -> list[list[int]]:
    """Sample circuit groups for multi-programming training.

    Args:
        circuits: Pool of available circuits.
        num_physical: Number of physical qubits on target backend.
        ratios: Dict with keys 'single', 'dual', 'quad' and float ratios
                summing to 1.0. Defaults to 50/30/20.
        occupancy_min: Minimum occupancy ratio (total_logical / num_physical).
        occupancy_max: Maximum occupancy ratio.
        num_samples: Total number of groups to generate.
                     Defaults to len(circuits).
        rng_seed: Random seed.

    Returns:
        List of groups, where each group is a list of circuit indices.
    """
    if ratios is None:
        ratios = {"single": 0.5, "dual": 0.3, "quad": 0.2}

    if num_samples is None:
        num_samples = len(circuits)

    rng = random.Random(rng_seed)

    # Build size index for efficient lookup
    qubit_counts = [c.num_qubits for c in circuits]
    max_logical = int(num_physical * occupancy_max)
    min_logical = max(1, int(num_physical * occupancy_min))

    # Determine how many of each type
    n_single = int(num_samples * ratios.get("single", 0.5))
    n_dual = int(num_samples * ratios.get("dual", 0.3))
    n_quad = num_samples - n_single - n_dual

    groups: list[list[int]] = []

    # Single circuit groups
    groups.extend(_sample_groups(
        qubit_counts, 1, n_single, min_logical, max_logical, num_physical, rng,
    ))

    # Dual circuit groups
    groups.extend(_sample_groups(
        qubit_counts, 2, n_dual, min_logical, max_logical, num_physical, rng,
    ))

    # Quad circuit groups
    groups.extend(_sample_groups(
        qubit_counts, 4, n_quad, min_logical, max_logical, num_physical, rng,
    ))

    rng.shuffle(groups)
    return groups


def _sample_groups(
    qubit_counts: list[int],
    group_size: int,
    num_groups: int,
    min_logical: int,
    max_logical: int,
    num_physical: int,
    rng: random.Random,
    max_attempts_per_group: int = 100,
) -> list[list[int]]:
    """Sample groups of a specific size meeting occupancy constraints.

    Args:
        qubit_counts: Per-circuit qubit counts.
        group_size: Number of circuits per group (1, 2, or 4).
        num_groups: Number of groups to generate.
        min_logical: Minimum total logical qubits.
        max_logical: Maximum total logical qubits.
        num_physical: Number of physical qubits (total must be < this).
        rng: Random number generator.
        max_attempts_per_group: Max random sampling attempts per group.

    Returns:
        List of valid groups (each a list of circuit indices).
    """
    n = len(qubit_counts)
    if n == 0:
        return []

    # Filter circuits that are small enough to participate
    eligible = [i for i, q in enumerate(qubit_counts) if q < num_physical]
    if not eligible:
        return []

    groups: list[list[int]] = []

    for _ in range(num_groups):
        for _attempt in range(max_attempts_per_group):
            selected = rng.choices(eligible, k=group_size)
            total = sum(qubit_counts[i] for i in selected)

            if total < num_physical and min_logical <= total <= max_logical:
                groups.append(selected)
                break

    return groups
