"""Replications of E. P. Chan (2013), *Algorithmic Trading*, as pyatl strategies."""

from .momentum import TsMomentum
from .reversion import BollingerBand, KalmanReversion, LinearLongShort, LinearReversion, LinearSpread
from .reversion_intraday import BuyOnGap, IntradayLongShort

__all__ = [
    "BollingerBand",
    "BuyOnGap",
    "IntradayLongShort",
    "KalmanReversion",
    "LinearLongShort",
    "LinearReversion",
    "LinearSpread",
    "TsMomentum",
]
