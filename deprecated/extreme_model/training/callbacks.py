from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0

    best: float = float("inf")
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        if value + self.min_delta < self.best:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

