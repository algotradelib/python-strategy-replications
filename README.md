# python-strategy-replications

Replications of [Ernest P. Chan (2013), *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley](https://www.amazon.com/dp/1118460146?lv=shuf&channelId=500&plpRedirect=mhFallback), using Python with [pyatl](https://github.com/algotradelib/pyatl).

- Time series analytics (chapter 2)
  - [ ] Augmented Dickey-Fuller Test
  - [ ] Hurst Exponent and Variance Ratio Test
  - [ ] Half-life of Mean Reversion
  - [x] Cointegration and Johansen Test for Cointegration (`johansen-linear`: the eigenvector and the lookback are estimated on the whole file, so its figure looks ahead)
- Mean Reversion Strategies (chapter 3)
  - [x] Linear mean-reverting (`linear`, `linear-log`, `ratio`)
  - [x] Bollinger Bands (`bollinger`)
  - [x] Kalman Filter (`kalman`)
- Mean Reversion of Stocks and ETFs (chapter 4)
  - [x] Buy-on-Gap Model (`bog`)
  - [x] Cross-Sectional Linear Long-Short (`linear-longshort`/`longshort`)
  - [ ] Arbitrage between an ETF and Component Stocks
- Interday Momentum Strategies (chapter 6)
  - [x] Time series momentum (`tsmom`)
  - [ ] Cross-sectional momentum
- Mean Reversion of Currencies and Futures (chapter 5)
  - [ ] Trading Currency Cross-Rates
  - [ ] Trading Futures Calendar Spread
  - [ ] Volatility futures versus Equity index futures

## Install

It needs Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), and an installed algotradelib; see [its install guide](https://github.com/algotradelib/algotradelib#install).
`uv sync` builds pyatl from source against that library, so it also needs a C++17 compiler and CMake.

```bash
uv sync
```

If algotradelib is installed under a prefix CMake does not search, set `CMAKE_PREFIX_PATH` to that prefix for `uv sync`.

## Running

```bash
uv run atl-strategy -s <strategy_name> -d <data_path> ...
```

## Acknowledgements

This project was developed with the assistance of generative AI tools.

## Disclaimer

This project is released under the MIT license; see [LICENSE](LICENSE).
The software is provided as is, without warranty of any kind, and the authors are not liable for any loss, including trading losses, arising from its use.
Nothing in this repository is investment advice.
