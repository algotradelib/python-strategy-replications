"""Intraday momentum of chapter 7, on days replayed as three bars (see `backtesting.replay_intraday`)."""

from __future__ import annotations

from collections import deque

import numpy as np

from .reversion_intraday import NS_PER_DAY, part_of
from .statistics import round_half_away


class OpeningGap:
    """Example 7.1 (`gapFutures_FSTX.m`): trade against the opening gap of one future; out at the close.

    Sell when the open is at or above yesterday's high by `entry` deviations of the daily returns.
    Buy when it is at or below yesterday's low by as much; when both hold (a zero deviation), the purchase wins.
    The deviation is the population one (the MATLAB's `smartstd`) of the finite returns among the last `window`.
    The returns are close to close, up to yesterday.
    A position is `capital` at the open, in whole units.
    """

    def __init__(self, instrument: str, entry: float = 0.1, window: int = 90, capital: float = 1_000_000_000) -> None:
        self.instrument, self.entry, self.window, self.capital = instrument, entry, window, capital
        self.tag = "opening-gap"
        self.returns: deque[float] = deque(maxlen=window)  # NaN for the first day, which has no last close
        self.high = self.low = self.close = np.nan
        self.held = 0

    def decide(self, ctx, bar) -> None:
        returns = np.asarray(self.returns)
        returns = returns[np.isfinite(returns)]
        if len(self.returns) < self.window or len(returns) == 0:
            return
        sd = returns.std()
        units = int(round_half_away(self.capital / bar.open))
        if bar.open <= self.low * (1 - self.entry * sd):
            self.held = units
        elif bar.open >= self.high * (1 + self.entry * sd):
            self.held = -units
        if self.held != 0:
            ctx.target(self.instrument, self.held)

    def exit(self, ctx, bar) -> None:
        if self.held != 0:
            ctx.target(self.instrument, 0)
            self.held = 0
        self.high, self.low = bar.high, bar.low

    def push_close(self, bar) -> None:
        self.returns.append(bar.close / self.close - 1)
        self.close = bar.close

    def on_bars(self, ctx, bars) -> None:
        bar = next((bar for bar in bars if bar.symbol == self.instrument), None)
        if bar is None:
            return
        part = part_of(bar.time)
        if part == 0:
            self.decide(ctx, bar)
        elif part == 1:
            self.exit(ctx, bar)
        elif part == 2:
            self.push_close(bar)


class PostEarningsDrift:
    """Example 7.2 (`pead.py`): trade a stock's opening gap on its announcement day, with the gap; out at the close.

    Buy when the return from the last close to the open is at least `entry` deviations of such returns.
    Sell short when it is at most minus that; when both hold (a zero deviation), the short wins, as in the port.
    The deviation is the sample one of the last `lookback` returns, today's included, and needs all of them.
    The replay holds only the days the announcements and the prices share, as the port's join does.
    A position is `capital / slots` dollars at the open, in whole shares.
    """

    def __init__(
        self,
        universe: list[str],
        announced: dict[int, np.ndarray],
        lookback: int = 90,
        entry: float = 0.5,
        slots: int = 30,
        capital: float = 1_000_000_000,
    ) -> None:
        self.universe, self.announced = universe, announced  # a row of flags by midnight, in ns since the epoch
        self.lookback, self.entry, self.slots, self.capital = lookback, entry, slots, capital
        self.tag = "post-earnings-drift"
        self.index = {symbol: i for i, symbol in enumerate(universe)}
        self.returns: deque[np.ndarray] = deque(maxlen=lookback)  # one row of returns a day, NaN where none
        self.close = np.full(len(universe), np.nan)
        self.held = np.zeros(len(universe), dtype=bool)

    def _field(self, bars, name: str) -> np.ndarray:
        values = np.full(len(self.universe), np.nan)
        for bar in bars:
            values[self.index[bar.symbol]] = getattr(bar, name)
        return values

    def decide(self, ctx, bars) -> None:
        curr_open = self._field(bars, "open")
        ret = curr_open / self.close - 1
        self.returns.append(ret)
        announced = self.announced.get(bars[0].time - bars[0].time % NS_PER_DAY)
        if announced is None or len(self.returns) < self.lookback:
            return
        sd = np.std(np.array(self.returns), axis=0, ddof=1)  # NaN where a return is missing
        side = np.zeros(len(self.universe), dtype=np.int64)
        side[announced & (ret >= self.entry * sd)] = 1
        side[announced & (ret <= -self.entry * sd)] = -1
        for i in np.flatnonzero(side):
            ctx.target(self.universe[i], int(side[i] * round_half_away(self.capital / self.slots / curr_open[i])))
            self.held[i] = True

    def exit_all(self, ctx) -> None:
        for i in np.flatnonzero(self.held):
            ctx.target(self.universe[i], 0)
        self.held[:] = False

    def on_bars(self, ctx, bars) -> None:
        part = part_of(bars[0].time)
        if part == 0:
            self.decide(ctx, bars)
        elif part == 1:
            self.exit_all(ctx)
        elif part == 2:
            self.close = self._field(bars, "close")
