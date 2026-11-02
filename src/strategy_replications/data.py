"""Loading the book's csv files: a yyyymmdd date column, then one column per instrument."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """One field of several instruments, indexed by date, with NaN where a cell is missing."""
    table = pd.read_csv(path, skipinitialspace=True, index_col=0)
    table.index = pd.to_datetime(table.index.astype(str), format="%Y%m%d").rename("date")
    table.columns = table.columns.str.strip()
    return table.astype("float64")


def load_ohlc(directory: str | Path, kind: str) -> pd.DataFrame:
    """One of the stock tables (`op`, `hi`, `lo`, `cl`), with the names from the second line of `*_stocks.csv`."""
    directory = Path(directory)
    table = load_csv(directory / f"inputDataOHLCDaily_20120424_{kind}.csv")
    # The first line holds placeholders (stocks1, stocks2, ...); the names are on the second.
    names = pd.read_csv(directory / "inputDataOHLCDaily_20120424_stocks.csv", header=None, skiprows=1, nrows=1)
    table.columns = names.iloc[0].astype(str).str.strip()
    return table
