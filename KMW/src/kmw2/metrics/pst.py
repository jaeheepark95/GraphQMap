from __future__ import annotations
from typing import Mapping, Dict


def _normalize_counts(counts: Mapping[str, int | float]) -> Dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("Counts must have positive total shots.")
    return {k: float(v) / total for k, v in counts.items()}


def pst_overlap_percent(
    noisy_counts: Mapping[str, int | float],
    ideal_counts: Mapping[str, int | float],
) -> float:
    """
    PST = 100 * sum_x min(p_noisy(x), p_ideal(x))

    Interpretation:
      100.0 -> noisy and ideal output distributions match exactly
        0.0 -> no overlap

    This is 100 * (1 - TVD), where TVD is total variation distance.
    """
    p_noisy = _normalize_counts(noisy_counts)
    p_ideal = _normalize_counts(ideal_counts)

    keys = set(p_noisy) | set(p_ideal)
    overlap = sum(min(p_noisy.get(k, 0.0), p_ideal.get(k, 0.0)) for k in keys)
    return 100.0 * overlap


def pst_v2(
    noisy_counts: Mapping[str, int | float],
    ideal_counts: Mapping[str, int | float],
) -> float:
    """
    PSTv2 = 100 * (noisy hits of ideal top bitstring / total noisy shots)

    Measures how often the noisy circuit produces the ideal circuit's
    most probable bitstring (the "correct answer").
    """
    ideal_result = max(ideal_counts, key=lambda k: ideal_counts[k])
    total_noisy = sum(noisy_counts.values())
    if total_noisy <= 0:
        raise ValueError("Counts must have positive total shots.")
    return (noisy_counts.get(ideal_result, 0) / total_noisy) * 100.0
