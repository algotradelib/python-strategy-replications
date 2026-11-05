"""Mean reversion strategies of chapters 3 and 4: pairs, and the daily linear long-short model."""

from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np

from .statistics import KalmanFilter, hedge_ratio, mean_sd, round_half_away, shares

# Bars kept per instrument, as the C++ replay's series capacity.
HISTORY = 512


def target(ctx, symbol: str, value: float) -> None:
    """States a position in whole shares; a value that is not finite states nothing."""
    quantity = shares(value)
    if quantity is not None:
        ctx.target(symbol, quantity)


def state_targets(ctx, universe: list[str], wanted: np.ndarray, held: np.ndarray) -> None:
    """States every finite position, in whole shares, that differs from the last one stated; updates `held`."""
    finite = np.isfinite(wanted)
    rounded = held.copy()
    rounded[finite] = round_half_away(wanted[finite])
    for i in np.flatnonzero(rounded != held):
        held[i] = rounded[i]
        ctx.target(universe[i], int(rounded[i]))


class LinearSpread(Enum):
    PRICE = "price"  # price spread
    LOG = "log"  # log price spread
    RATIO = "ratio"  # price ratio


class Closes:
    """The closes of two instruments, each with its bar times."""

    def __init__(self, y: str, x: str) -> None:
        self.y, self.x = y, x
        self.times = {y: deque(maxlen=HISTORY), x: deque(maxlen=HISTORY)}
        self.closes = {y: deque(maxlen=HISTORY), x: deque(maxlen=HISTORY)}

    def push(self, bars) -> bool:
        """Records the bars of y and x; whether both are in this slice."""
        seen = 0
        for bar in bars:
            if bar.symbol in self.times:
                self.times[bar.symbol].append(bar.time)
                self.closes[bar.symbol].append(bar.close)
                seen += 1
        return seen == 2

    def shared_window(self, size: int) -> tuple[np.ndarray, np.ndarray] | None:
        """The last `size` closes on the dates both share, oldest first; None until there are that many."""
        _, iy, ix = np.intersect1d(self.times[self.y], self.times[self.x], assume_unique=True, return_indices=True)
        if len(iy) < size:
            return None
        return np.asarray(self.closes[self.y])[iy[-size:]], np.asarray(self.closes[self.x])[ix[-size:]]


class LinearReversion:
    """Example 3.1: hold minus the z-score of the spread, in units of `scale`, hedged by a rolling regression."""

    def __init__(
        self, y: str, x: str, spread: LinearSpread = LinearSpread.PRICE, lookback: int = 20, scale: float = 1000
    ) -> None:
        self.y, self.x, self.spread_type = y, x, spread
        self.lookback, self.scale = lookback, scale  # shares a unit (PRICE), or dollars a unit (LOG, RATIO)
        self.tag = "linear-reversion"
        self.closes = Closes(y, x)
        self.spread: list[float] = []

    def on_bars(self, ctx, bars) -> None:
        if not self.closes.push(bars) or self.lookback < 2:
            return
        # The ratio needs only today's closes; the two spreads need a regression window first.
        needed = 1 if self.spread_type is LinearSpread.RATIO else self.lookback
        window = self.closes.shared_window(needed)
        if window is None:
            return
        window_y, window_x = window
        y, x = window_y[-1], window_x[-1]
        hedge = 1.0
        if self.spread_type is LinearSpread.RATIO:
            self.spread.append(y / x)
        else:
            if self.spread_type is LinearSpread.LOG:
                window_y, window_x = np.log(window_y), np.log(window_x)
            hedge = hedge_ratio(window_y, window_x)
            self.spread.append(window_y[-1] - hedge * window_x[-1])
        if len(self.spread) < self.lookback:
            return  # warm-up
        mean, sd = mean_sd(self.spread[-self.lookback :])
        if not sd > 0:
            return
        units = -(self.spread[-1] - mean) / sd
        if self.spread_type is LinearSpread.PRICE:
            shares_y = units * self.scale
            shares_x = -hedge * units * self.scale
        elif self.spread_type is LinearSpread.LOG:
            shares_y = units * self.scale / y
            shares_x = -hedge * units * self.scale / x
        else:
            shares_y = units * self.scale / y
            shares_x = -units * self.scale / x
        target(ctx, self.y, shares_y)
        target(ctx, self.x, shares_x)


class JohansenLinear:
    """Example 2.8: hold minus the z-score of the portfolio the Johansen eigenvector weights, `scale` shares a unit.

    Each instrument's position is the units times its weight times `scale`, not divided by its price.
    The weights and the lookback come from the whole history (`johansen_portfolio`), so the figure looks ahead.
    """

    def __init__(self, universe: list[str], weights, lookback: int, scale: float = 100000) -> None:
        self.universe, self.weights = universe, np.asarray(weights)
        self.lookback, self.scale = lookback, scale
        self.tag = "johansen-linear"
        self.portfolio: list[float] = []

    def on_bars(self, ctx, bars) -> None:
        close = {bar.symbol: bar.close for bar in bars}
        if any(symbol not in close for symbol in self.universe) or self.lookback < 2:
            return
        self.portfolio.append(float(self.weights @ np.array([close[symbol] for symbol in self.universe])))
        if len(self.portfolio) < self.lookback:
            return  # warm-up
        mean, sd = mean_sd(self.portfolio[-self.lookback :])
        if not sd > 0:
            return
        units = -(self.portfolio[-1] - mean) / sd
        for symbol, weight in zip(self.universe, self.weights, strict=True):
            target(ctx, symbol, units * weight * self.scale)


class BollingerBand:
    """Example 3.2: enter when the spread's z-score leaves `entry` deviations, leave when it crosses `exit`."""

    def __init__(
        self, y: str, x: str, lookback: int = 20, entry: float = 1.0, exit: float = 0.0, scale: int = 1000
    ) -> None:
        self.y, self.x = y, x
        self.lookback, self.entry, self.exit, self.scale = lookback, entry, exit, scale
        self.tag = "bollinger-reversion"
        self.closes = Closes(y, x)
        self.spread: list[float] = []
        self.units_long = self.units_short = False

    def on_bars(self, ctx, bars) -> None:
        if not self.closes.push(bars) or self.lookback < 2 or not self.exit < self.entry:
            return
        window = self.closes.shared_window(self.lookback)
        if window is None:
            return
        window_y, window_x = window
        hedge = hedge_ratio(window_y, window_x)
        self.spread.append(window_y[-1] - hedge * window_x[-1])
        if len(self.spread) <= self.lookback:
            return  # warm-up
        mean, sd = mean_sd(self.spread[-self.lookback :])
        if sd > 0:
            zscore = (self.spread[-1] - mean) / sd
            if zscore < -self.entry:
                self.units_long = True
            elif zscore > -self.exit:
                self.units_long = False
            if zscore > self.entry:
                self.units_short = True
            elif zscore < self.exit:
                self.units_short = False
        units = int(self.units_long) - int(self.units_short)
        ctx.target(self.y, units * self.scale)
        target(ctx, self.x, -hedge * units * self.scale)


class KalmanReversion:
    """Example 3.3: a Kalman filter for the hedge ratio; enter when the forecast error leaves one deviation."""

    def __init__(self, y: str, x: str, delta: float = 0.0001, v_e: float = 0.001, scale: int = 1000) -> None:
        self.y, self.x, self.scale = y, x, scale
        self.tag = "kalman-reversion"
        self.kf = KalmanFilter(delta, v_e)
        self.units_long = self.units_short = False

    def on_bars(self, ctx, bars) -> None:
        close = {bar.symbol: bar.close for bar in bars}
        if self.y not in close or self.x not in close:
            return
        self.kf.forecast(close[self.y], close[self.x])
        resid, sqrt_q, hedge = self.kf.measurement_resid(), self.kf.measurement_sd(), self.kf.hedge_ratio()
        if resid < -sqrt_q:
            self.units_long = True
        elif resid > -sqrt_q:
            self.units_long = False
        if resid > sqrt_q:
            self.units_short = True
        elif resid < sqrt_q:
            self.units_short = False
        units = int(self.units_long) - int(self.units_short)
        ctx.target(self.y, units * self.scale)
        target(ctx, self.x, -hedge * units * self.scale)


class LinearLongShort:
    """Example 4.3: each day, weight every stock by minus its return less the mean return; `scale` dollars gross."""

    def __init__(self, universe: list[str], scale: float = 1000) -> None:
        self.universe, self.scale = universe, scale
        self.tag = "linear-long-short"
        self.index = {symbol: i for i, symbol in enumerate(universe)}
        self.curr = np.zeros(len(universe))
        self.held = np.zeros(len(universe), dtype=np.int64)
        self.count = 0

    def on_bars(self, ctx, bars) -> None:
        prev = self.curr.copy()
        for bar in bars:
            self.curr[self.index[bar.symbol]] = bar.close  # a stock without a bar keeps its last close
        self.count += 1
        if self.count == 1:
            return
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_change = self.curr / prev - 1
            finite = np.isfinite(pct_change)
            ret_mean = np.where(finite, pct_change, 0.0).sum() / finite.sum()
            weights = -(pct_change - ret_mean)
            weights /= np.abs(np.where(np.isfinite(weights), weights, 0.0)).sum()
            priced = np.isfinite(self.curr) & (self.curr > 0)
            wanted = np.where(priced, weights / self.curr, 0.0) * self.scale
        state_targets(ctx, self.universe, wanted, self.held)
