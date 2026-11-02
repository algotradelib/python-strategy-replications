"""Intraday mean reversion of chapter 4, on days replayed as three bars (see `backtesting.replay_intraday`)."""

from __future__ import annotations

import math
import warnings
from collections import deque

import numpy as np

from .reversion import state_targets

NS_PER_DAY = 86_400 * 10**9
NS_PER_MINUTE = 60 * 10**9


def part_of(period_end: int) -> int:
    """0: the open, 1: the day, 2: the close."""
    return (period_end % NS_PER_DAY) // NS_PER_MINUTE - 1


class BuyOnGap:
    """Example 4.1: buy at the open the `top` stocks that gapped down the most below the last low; out at the close.

    A stock qualifies when it opens more than `entry` deviations of its daily returns below the last low, and still
    above its moving average.
    """

    def __init__(
        self,
        universe: list[str],
        top: int = 10,
        lookback: int = 20,
        window: int = 90,
        entry: float = 1.0,
        capital: float = 1_000_000_000,
    ) -> None:
        self.universe, self.top, self.lookback, self.window = universe, top, lookback, window
        self.entry, self.capital = entry, capital
        self.tag = "buy-on-gap"
        self.index = {symbol: i for i, symbol in enumerate(universe)}
        self.closes: deque[np.ndarray] = deque(maxlen=max(window + 1, lookback))  # one row of closes a day
        self.prev_low = np.full(len(universe), np.nan)
        self.held = np.zeros(len(universe), dtype=bool)

    def _field(self, bars, name: str) -> np.ndarray:
        values = np.full(len(self.universe), np.nan)
        for bar in bars:
            values[self.index[bar.symbol]] = getattr(bar, name)
        return values

    def decide(self, ctx, bars) -> None:
        curr_open = self._field(bars, "open")
        if len(self.closes) < self.window + 1 or len(self.closes) < self.lookback:
            return
        closes = np.array(self.closes)
        recent = closes[-(self.window + 1) :]
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a stock with fewer than two returns has no deviation
            ret_sd = np.nanstd(recent[1:] / recent[:-1] - 1, axis=0, ddof=1)
            ma = np.nanmean(closes[-self.lookback :], axis=0)
            gap = curr_open / self.prev_low - 1
            gapped = np.isfinite(gap) & (curr_open < self.prev_low * (1 - self.entry * ret_sd)) & (curr_open > ma)
        if gapped.sum() > self.top:
            # The `top` largest gaps down: the smallest gap returns among the stocks that gapped.
            chosen = np.flatnonzero(gapped)[np.argsort(gap[gapped], kind="stable")[: self.top]]
            gapped[:] = False
            gapped[chosen] = True
        for i in np.flatnonzero(gapped):
            ctx.target(self.universe[i], math.floor(self.capital / self.top / curr_open[i]))
            self.held[i] = True

    def exit_all(self, ctx, bars) -> None:
        for i in np.flatnonzero(self.held):
            ctx.target(self.universe[i], 0)
        self.held[:] = False
        self.prev_low = self._field(bars, "low")

    def push_close(self, bars) -> None:
        self.closes.append(self._field(bars, "close"))

    def on_bars(self, ctx, bars) -> None:
        part = part_of(bars[0].time)
        if part == 0:
            self.decide(ctx, bars)
        elif part == 1:
            self.exit_all(ctx, bars)
        elif part == 2:
            self.push_close(bars)


class IntradayLongShort:
    """Example 4.4: at the open, weight every stock against its overnight return; out at the close.

    The weight is minus the return from the last close to the open, less the mean, scaled to `capital` dollars gross.
    """

    def __init__(self, universe: list[str], capital: float = 1_000_000_000) -> None:
        self.universe, self.capital = universe, capital
        self.tag = "intraday-long-short"
        self.index = {symbol: i for i, symbol in enumerate(universe)}
        n = len(universe)
        self.open, self.close = np.zeros(n), np.zeros(n)
        self.held = np.zeros(n, dtype=np.int64)
        self.count = 0

    def decide(self, ctx, bars) -> None:
        for bar in bars:
            self.open[self.index[bar.symbol]] = bar.open  # a stock without a bar keeps its last open
        self.count += 1
        if self.count == 1:
            return
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_change = self.open / self.close - 1
            finite = np.isfinite(pct_change)
            ret_mean = np.where(finite, pct_change, 0.0).sum() / finite.sum()
            weights = -(pct_change - ret_mean)
            weights /= np.abs(np.where(np.isfinite(weights), weights, 0.0)).sum()
            priced = np.isfinite(self.open) & (self.open > 0)
            wanted = np.where(priced, weights / self.open, 0.0) * self.capital
        state_targets(ctx, self.universe, wanted, self.held)

    def exit_all(self, ctx) -> None:
        for i in np.flatnonzero(self.held):
            ctx.target(self.universe[i], 0)
        self.held[:] = 0

    def push_close(self, bars) -> None:
        for bar in bars:
            self.close[self.index[bar.symbol]] = bar.close

    def on_bars(self, ctx, bars) -> None:
        part = part_of(bars[0].time)
        if part == 0:
            self.decide(ctx, bars)
        elif part == 1:
            self.exit_all(ctx)
        elif part == 2:
            self.push_close(bars)
