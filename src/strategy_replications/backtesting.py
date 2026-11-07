"""Replaying the book's tables through pyatl, and the book's own return figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyatl


def replay(
    strategy, closes: pd.DataFrame, names: list[str], *, cash: str = "100000", shorts: bool = True
) -> pyatl.BacktestResult:
    """Replays daily closes, each bar's open, high and low set to its close.

    A market order decided at close t fills at close t (`fill="decision-close"`), the book's convention.
    """
    c = closes[names].to_numpy()
    day, at = np.nonzero(~np.isnan(c))  # in time order, then in the order of names
    close = c[day, at]
    engine = pyatl.Replay(
        strategy, names, interval="1d", stamp="close", cash=cash, shorts=shorts, fill="decision-close"
    )
    engine.feed_bars(
        at,
        closes.index.to_numpy()[day],
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.full(len(day), 100000.0),
    )
    return engine.finish()


def replay_intraday(
    strategy,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    cash: str = "2000000000",
    shorts: bool = True,
) -> pyatl.BacktestResult:
    """Replays each day as three one-minute bars, so an order can fill at the open and another at the close.

    Part 0, (o, o, o, o): the strategy sees the open and submits entries.
    Part 1, (o, h, l, c): entries fill at the open; the strategy submits exits.
    Part 2, (c, c, c, c): exits fill at the close.
    """
    op, hi, lo, cl = (frame.to_numpy() for frame in (open_, high, low, close))
    parts = [(op, op, op, op), (op, hi, lo, cl), (cl, cl, cl, cl)]
    fields = [np.stack([part[field] for part in parts], axis=1) for field in range(4)]  # (day, part, column)
    valid = np.broadcast_to((~np.isnan(op) & ~np.isnan(cl))[:, None, :], fields[0].shape)
    day, part, at = np.nonzero(valid)
    time = close.index.to_numpy()[day] + part.astype("timedelta64[m]")
    engine = pyatl.Replay(strategy, list(close.columns), interval="1m", stamp="open", cash=cash, shorts=shorts)
    engine.feed_bars(
        at,
        time,
        open=fields[0][day, part, at],
        high=fields[1][day, part, at],
        low=fields[2][day, part, at],
        close=fields[3][day, part, at],
        volume=np.full(len(day), 1e9),
    )
    return engine.finish()


def replay_log(tag: str, result: pyatl.BacktestResult) -> None:
    print(f"{tag} replayed: {result.bars} bars and filled {len(result.fills['time'])} orders.")


def _report(ret: np.ndarray, label: str = "") -> None:
    apr = np.prod(1 + ret) ** (252.0 / len(ret)) - 1
    sharpe = ret.mean() * np.sqrt(252.0 * (len(ret) - 1.0)) / np.linalg.norm(ret - ret.mean())
    print(f"{label}apr={apr:.6g}, sharpe={sharpe:.6g}")


def _positions(prices: pd.DataFrame, result: pyatl.BacktestResult) -> pd.DataFrame:
    """The dollars held in each instrument at each close, from the fills."""
    fills = pd.DataFrame(result.fills)
    fills["shares"] = fills["quantity"].where(fills["side"] == "buy", -fills["quantity"])
    # A fill is held from the close that decided it: its time is at most the next date.
    next_dates = np.append(prices.index[1:], prices.index[-1:]).astype("datetime64[ns]").view(np.int64)
    fills["day"] = np.searchsorted(next_dates, fills["time"], side="left")
    held = (
        fills[fills["day"] < len(prices)]
        .pivot_table(index="day", columns="symbol", values="shares", aggfunc="sum")
        .reindex(index=range(len(prices)), columns=prices.columns)
        .fillna(0.0)
        .cumsum()
        .set_axis(prices.index)
    )
    return (held * prices).fillna(0.0)


def _pnl(prices: pd.DataFrame, positions: pd.DataFrame) -> pd.Series:
    """Each day's dollar change of the positions held from the close before."""
    return (positions.shift() * prices.pct_change(fill_method=None).fillna(0.0)).sum(axis=1)


def summarize(prices: pd.DataFrame, result: pyatl.BacktestResult) -> None:
    """The book's figures: each day's return is the dollar change of the positions held over the gross position."""
    positions = _positions(prices, result)
    ret = (_pnl(prices, positions) / positions.shift().abs().sum(axis=1)).iloc[1:].fillna(0.0)
    _report(ret.to_numpy())


def summarize_fixed(
    prices: pd.DataFrame, result: pyatl.BacktestResult, denominator: float, start: str, end: str, label: str
) -> None:
    """The book's figures over the days from `start` to `end`: each day's dollar change over a fixed `denominator`.

    The replay runs over the whole history, so the positions on `start` are those the history built.
    """
    ret = (_pnl(prices, _positions(prices, result)) / denominator).loc[start:end]
    _report(ret.to_numpy(), label)


def summarize_intraday(close: pd.DataFrame, result: pyatl.BacktestResult, capital: float, t_start: int) -> None:
    """The book's figures for a strategy flat every night: each day's change in equity over the capital.

    The days from `t_start` on are scored; the first day's change is from the opening equity.
    """
    times = np.asarray(result.equity["time"])
    ends = (close.index + pd.Timedelta(days=1)).to_numpy().astype("datetime64[ns]").view(np.int64)
    last = np.searchsorted(times, ends, side="right") - 1  # the last equity point by each day's end
    opening = float(result.opening_equity)
    level = pd.Series(np.where(last >= 0, np.asarray(result.equity["equity"])[last], opening))
    ret = (level - level.shift(fill_value=opening)) / capital
    _report(ret.iloc[t_start:].to_numpy())
