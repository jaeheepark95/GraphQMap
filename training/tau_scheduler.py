"""Sinkhorn temperature (τ) scheduler for GraphQMap.

Stage 1: Exponential decay from τ_max to τ_min.
Stage 2: Fixed at τ_min.
"""

from __future__ import annotations

import math


class TauScheduler:
    """Sinkhorn temperature scheduler.

    Args:
        tau_max: Initial temperature.
        tau_min: Final temperature.
        schedule: 'exponential' or 'fixed'.
        total_epochs: Total epochs for exponential schedule.
    """

    def __init__(
        self,
        tau_max: float = 1.0,
        tau_min: float = 0.05,
        schedule: str = "exponential",
        total_epochs: int = 100,
    ) -> None:
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.schedule = schedule
        self.total_epochs = max(total_epochs, 1)

    def get_tau(self, epoch: int) -> float:
        """Get temperature for the given epoch.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            Temperature value.
        """
        if self.schedule == "fixed":
            return self.tau_min

        if self.schedule == "exponential":
            # τ(epoch) = τ_max * (τ_min / τ_max) ^ (epoch / total_epochs)
            ratio = epoch / self.total_epochs
            ratio = min(ratio, 1.0)
            tau = self.tau_max * math.pow(self.tau_min / self.tau_max, ratio)
            return max(tau, self.tau_min)

        raise ValueError(f"Unknown schedule: {self.schedule}")
