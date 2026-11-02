"""Interday momentum of chapter 6."""

from __future__ import annotations

from collections import deque

import numpy as np

from .reversion import HISTORY, target


class TsMomentum:
    """Example 6.1: hold the sum of the signs of the last `holddays - 1` lookback returns, `scale` dollars a unit.

    Before that many returns exist, the sum covers those there are.
    """

    def __init__(self, instrument: str, lookback: int = 250, holddays: int = 25, scale: float = 10_000_000) -> None:
        self.instrument, self.lookback, self.holddays, self.scale = instrument, lookback, holddays, scale
        self.tag = "ts-momentum"
        self.close: deque[float] = deque(maxlen=HISTORY)

    def on_bar(self, ctx, bar) -> None:
        if bar.symbol != self.instrument:
            return
        self.close.append(bar.close)
        close = np.asarray(self.close)
        terms = min(self.holddays - 1, max(0, len(close) - self.lookback))  # the lookback returns that exist
        units = np.sign(
            close[len(close) - terms :] - close[len(close) - terms - self.lookback : len(close) - self.lookback]
        ).sum()
        target(ctx, self.instrument, units * self.scale / close[-1])
