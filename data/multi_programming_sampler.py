"""Multi-programming data sampler for GraphQMap.

Generates training samples by combining multiple circuits for
multi-programming scenarios. Supports arbitrary circuit counts
per group, configured via scenarios/proportions.

Combination rules:
  - Total logical qubits <= physical qubit count
  - Occupancy range: 30-75%
  - Scenarios and proportions configured via YAML
"""

from __future__ import annotations

import random

from qiskit import QuantumCircuit


def sample_multi_programming_groups(
    circuits: list[QuantumCircuit],
    num_physical: int,
    scenarios: list[int] | None = None,
    proportions: list[float] | None = None,
    occupancy_min: float = 0.30,
    occupancy_max: float = 0.75,
    num_samples: int | None = None,
    rng_seed: int | None = None,
) -> list[list[int]]:
    """Sample circuit groups for multi-programming training.

    Args:
        circuits: Pool of available circuits.
        num_physical: Number of physical qubits on target backend.
        scenarios: List of group sizes (number of circuits per group).
                   E.g. [1, 2, 4] or [1, 2, 3, 5]. Defaults to [1, 2, 4].
        proportions: Sampling proportion for each scenario, must sum to 1.0.
                     Must have the same length as scenarios. Defaults to [0.5, 0.3, 0.2].
        occupancy_min: Minimum occupancy ratio (total_logical / num_physical).
        occupancy_max: Maximum occupancy ratio.
        num_samples: Total number of groups to generate.
                     Defaults to len(circuits).
        rng_seed: Random seed.

    Returns:
        List of groups, where each group is a list of circuit indices.
    """
    if scenarios is None:
        scenarios = [1, 2, 4]
    if proportions is None:
        proportions = [0.5, 0.3, 0.2]

    if len(scenarios) != len(proportions):
        raise ValueError(
            f"scenarios ({len(scenarios)}) and proportions ({len(proportions)}) "
            f"must have the same length"
        )

    if num_samples is None:
        num_samples = len(circuits)

    rng = random.Random(rng_seed)

    # Build size index for efficient lookup
    qubit_counts = [c.num_qubits for c in circuits]
    max_logical = int(num_physical * occupancy_max)
    min_logical = max(1, int(num_physical * occupancy_min))

    # Determine how many groups for each scenario
    counts = []
    remaining = num_samples
    for i, prop in enumerate(proportions):
        if i == len(proportions) - 1:
            counts.append(remaining)
        else:
            n = int(num_samples * prop)
            counts.append(n)
            remaining -= n

    groups: list[list[int]] = []

    for group_size, num_groups in zip(scenarios, counts):
        groups.extend(_sample_groups(
            qubit_counts, group_size, num_groups,
            min_logical, max_logical, num_physical, rng,
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
        group_size: Number of circuits per group.
        num_groups: Number of groups to generate.
        min_logical: Minimum total logical qubits.
        max_logical: Maximum total logical qubits.
        num_physical: Number of physical qubits (total must not exceed this).
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

            if total <= num_physical and min_logical <= total <= max_logical:
                groups.append(selected)
                break

    return groups
