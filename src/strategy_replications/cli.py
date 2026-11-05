"""`atl-strategy -s NAME -d DATA [--vars Y,X] [--no-shorts]`: one book example, replayed and summarized."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import backtesting
from .data import load_csv, load_ohlc
from .momentum import CrossSectionalMomentum, TsMomentum
from .momentum_intraday import OpeningGap, PostEarningsDrift
from .reversion import (
    BollingerBand,
    JohansenLinear,
    KalmanReversion,
    LinearLongShort,
    LinearReversion,
    LinearSpread,
)
from .reversion_intraday import BuyOnGap, IntradayLongShort
from .statistics import johansen_portfolio, round_half_away

FROM, TO = "2007-01-03", "2011-12-30"  # the period of examples 4.3 and 4.4
BOOK_WINDOW = "2007-05-15", "2007-12-31"  # the period of example 6.2's figure


def pair(strategy, path: str, names: list[str], shorts: bool, cash: str = "100000") -> None:
    """Examples 3.1-3.3, and 6.1 on one instrument."""
    data = load_csv(path)
    for name in names:
        if name not in data.columns:
            raise SystemExit(f"atl-strategy: {path} has no column {name!r} (it has {', '.join(data.columns)})")
    result = backtesting.replay(strategy, data, names, cash=cash, shorts=shorts)
    backtesting.replay_log(strategy.tag, result)
    backtesting.summarize(data, result)


def johansen_linear(path: str, shorts: bool) -> None:
    """Example 2.8, on every column of the file, in its order."""
    data = load_csv(path)
    names = list(data.columns)
    weights, half_life = johansen_portfolio(data)
    lookback = int(round_half_away(half_life))
    print(f"eigenvector={[float(weight) for weight in weights]!r} ({', '.join(names)})")
    print(f"half-life={half_life!r}, lookback={lookback} (both estimated on the whole file: the figure looks ahead)")
    strategy = JohansenLinear(names, weights, lookback, scale=100000)
    result = backtesting.replay(strategy, data, names, shorts=shorts)
    backtesting.replay_log(strategy.tag, result)
    backtesting.summarize(data, result)


def linear_longshort(directory: str, shorts: bool) -> None:
    """Example 4.3."""
    data = load_ohlc(directory, "cl").loc[FROM:TO]
    names = list(data.columns)
    strategy = LinearLongShort(names, scale=100000)
    result = backtesting.replay(strategy, data, names, shorts=shorts)
    backtesting.replay_log(strategy.tag, result)
    backtesting.summarize(data, result)


def xsmom(directory: str, shorts: bool) -> None:
    """Example 6.2 over the whole history, then over the book's window with the book's fixed denominator."""
    data = load_ohlc(directory, "cl")
    names = list(data.columns)
    strategy = CrossSectionalMomentum(names)
    result = backtesting.replay(strategy, data, names, cash="1000000000", shorts=shorts)
    backtesting.replay_log(strategy.tag, result)
    backtesting.summarize(data, result)
    denominator = 2 * strategy.top * strategy.holddays * strategy.scale
    start, end = BOOK_WINDOW
    label = f"from {start} to {end}, over 2*{strategy.top}*{strategy.holddays}*{strategy.scale:g}: "
    backtesting.summarize_fixed(data, result, denominator, start, end, label)


def intraday(make, directory: str, shorts: bool, sliced: bool) -> None:
    """Examples 4.1 and 4.4."""
    tables = [load_ohlc(directory, kind) for kind in ("op", "hi", "lo", "cl")]
    if sliced:
        tables = [table.loc[FROM:TO] for table in tables]
    close = tables[-1]
    strategy = make(list(close.columns))
    result = backtesting.replay_intraday(strategy, *tables, shorts=shorts)
    backtesting.replay_log(strategy.tag, result)
    t_start = strategy.window + 1 if isinstance(strategy, BuyOnGap) else 1
    backtesting.summarize_intraday(close, result, strategy.capital, t_start)


def gap(path: str, shorts: bool) -> None:
    """Example 7.1, on one instrument's `Date,Open,High,Low,Close` file, named after the file."""
    data = load_csv(path)
    symbol = Path(path).stem
    tables = [data[[field]].set_axis([symbol], axis=1) for field in ("Open", "High", "Low", "Close")]
    strategy = OpeningGap(symbol)
    result = backtesting.replay_intraday(strategy, *tables, shorts=shorts)
    backtesting.replay_log(strategy.tag, result)
    backtesting.summarize_intraday(tables[-1], result, strategy.capital, 0)


def pead(directory: str, shorts: bool) -> None:
    """Example 7.2, over the days `earnannFile.csv` and the stock tables share."""
    tables = [load_ohlc(directory, kind) for kind in ("op", "hi", "lo", "cl")]
    announced = load_csv(Path(directory) / "earnannFile.csv")
    days = tables[-1].index.intersection(announced.index)
    tables = [table.loc[days] for table in tables]
    close = tables[-1]
    flags = announced.loc[days, close.columns].to_numpy() != 0
    by_day = dict(zip(days.to_numpy().astype("datetime64[ns]").view(np.int64), flags, strict=True))
    strategy = PostEarningsDrift(list(close.columns), by_day)
    result = backtesting.replay_intraday(strategy, *tables, shorts=shorts)
    backtesting.replay_log(strategy.tag, result)
    backtesting.summarize_intraday(close, result, strategy.capital, 0)


# The columns each strategy takes from --vars; the stock examples take none.
VARS = {"linear": 2, "linear-log": 2, "ratio": 2, "bollinger": 2, "kalman": 2, "tsmom": 1}

STRATEGIES = (
    "linear",
    "linear-log",
    "ratio",
    "bollinger",
    "kalman",
    "johansen-linear",
    "bog",
    "linear-longshort",
    "longshort",
    "tsmom",
    "xsmom",
    "gap",
    "pead",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atl-strategy", description="Book examples")
    parser.add_argument("-s", "--strategy", required=True, type=str.lower, choices=STRATEGIES, help="The strategy name")
    parser.add_argument(
        "-d", "--data", required=True, help="Path for the csv file, or the directory of the stock files"
    )
    parser.add_argument("--vars", type=lambda text: text.split(","), default=[], help="Variable names: Y,X for a pair")
    parser.add_argument("--no-shorts", dest="shorts", action="store_false", help="Do not use shorts if specified")
    args = parser.parse_args(argv)
    names, shorts = args.vars, args.shorts
    needed = VARS.get(args.strategy, 0)
    if len(names) != needed:
        what = "one column" if needed == 1 else "the pair as Y,X" if needed == 2 else "no --vars"
        parser.error(f"{args.strategy} takes {what}")
    match args.strategy:
        case "linear" | "linear-log" | "ratio":
            spread = {"linear": LinearSpread.PRICE, "linear-log": LinearSpread.LOG, "ratio": LinearSpread.RATIO}
            kind = spread[args.strategy]
            scale = 100000 if kind is LinearSpread.PRICE else 10000000
            pair(LinearReversion(names[0], names[1], kind, scale=scale), args.data, names, shorts)
        case "bollinger":
            pair(BollingerBand(names[0], names[1], scale=1000), args.data, names, shorts)
        case "kalman":
            pair(KalmanReversion(names[0], names[1], scale=1000), args.data, names, shorts)
        case "johansen-linear":
            johansen_linear(args.data, shorts)
        case "tsmom":
            pair(TsMomentum(names[0]), args.data, names, shorts, cash="1000000000")
        case "xsmom":
            xsmom(args.data, shorts)
        case "linear-longshort":
            linear_longshort(args.data, shorts)
        case "longshort":
            intraday(IntradayLongShort, args.data, shorts, sliced=True)
        case "bog":
            intraday(BuyOnGap, args.data, shorts, sliced=False)
        case "gap":
            gap(args.data, shorts)
        case "pead":
            pead(args.data, shorts)
    return 0
