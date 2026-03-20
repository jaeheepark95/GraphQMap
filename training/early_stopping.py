"""Early stopping utility for GraphQMap training."""

from __future__ import annotations


class EarlyStopping:
    """Early stopping based on a monitored metric.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' for loss metrics, 'max' for PST-like metrics.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: float | None = None
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Check whether training should stop.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self._is_improvement(value):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        else:
            return value > self.best_value + self.min_delta
