"""`atl-strategy -s NAME -d DATA [--vars Y,X] [--no-shorts]`: one book example, replayed and summarized."""

from __future__ import annotations

import argparse

from . import backtesting
from .data import load_csv, load_ohlc
from .momentum import TsMomentum
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
        case "linear-longshort":
            linear_longshort(args.data, shorts)
        case "longshort":
            intraday(IntradayLongShort, args.data, shorts, sliced=True)
        case "bog":
            intraday(BuyOnGap, args.data, shorts, sliced=False)
    return 0
