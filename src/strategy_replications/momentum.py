"""Interday momentum of chapter 6."""

from __future__ import annotations

from collections import deque

import numpy as np

from .reversion import HISTORY, state_targets, target


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


class CrossSectionalMomentum:
    """Example 6.2: each close, a long unit for the `top` best `lookback` returns, a short one for the `top` worst.

    A stock holds the sum of its units of the last `holddays - 1` closes, as in the book's Python port.
    A unit is `scale` dollars, restated in whole shares at every close.
    Only the stocks with a close today and `lookback` days before are ranked, ties in the universe's order.
    """

    def __init__(
        self, universe: list[str], lookback: int = 252, holddays: int = 25, top: int = 50, scale: float = 100000
    ) -> None:
        self.universe, self.lookback, self.holddays, self.top, self.scale = universe, lookback, holddays, top, scale
        self.tag = "cross-sectional-momentum"
        self.index = {symbol: i for i, symbol in enumerate(universe)}
        self.closes: deque[np.ndarray] = deque(maxlen=lookback + 1)  # one row of closes a day, NaN where none
        self.units: deque[np.ndarray] = deque(maxlen=holddays - 1)  # one row of units a day
        self.held = np.zeros(len(universe), dtype=np.int64)

    def on_bars(self, ctx, bars) -> None:
        close = np.full(len(self.universe), np.nan)
        for bar in bars:
            close[self.index[bar.symbol]] = bar.close
        self.closes.append(close)
        units = np.zeros(len(self.universe))
        if len(self.closes) > self.lookback:
            ret = close / self.closes[0] - 1
            ranked = np.flatnonzero(np.isfinite(ret))
            ranked = ranked[np.argsort(ret[ranked], kind="stable")]  # ascending
            units[ranked[len(ranked) - min(self.top, len(ranked)) :]] += 1
            units[ranked[: self.top]] -= 1
        self.units.append(units)
        wanted = np.sum(self.units, axis=0) * self.scale / close  # NaN, so nothing stated, where there is no close
        state_targets(ctx, self.universe, wanted, self.held)
