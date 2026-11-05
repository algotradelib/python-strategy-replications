"""The regressions, the z-score, the Johansen test and the Kalman filter the strategies use, with NumPy and statsmodels."""

from __future__ import annotations

import math
import warnings

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def hedge_ratio(y, x) -> float:
    """The least-squares slope of y on x, with an intercept."""
    design = np.column_stack([x, np.ones(len(x))])
    return float(np.linalg.lstsq(design, np.asarray(y), rcond=None)[0][0])


def mean_sd(window) -> tuple[float, float]:
    """The mean and the sample deviation of a window."""
    values = np.asarray(window)
    return float(values.mean()), float(values.std(ddof=1))


def round_half_away(value):
    """Rounds to whole numbers, half away from zero, as C++'s `std::round` (Python's `round` and `np.rint` go to even)."""
    whole = np.trunc(value)
    return np.where(np.abs(value - whole) >= 0.5, whole + np.sign(value), whole)


def shares(value: float) -> int | None:
    """A position in whole shares, or None when the value is not finite (nothing is stated)."""
    return int(round_half_away(value)) if math.isfinite(value) else None


def half_life(series) -> float:
    """The half-life of mean reversion, `-ln 2 / b` from the least-squares fit of `p - p[-1] = a + b p[-1]`."""
    p = np.asarray(series)
    slope = sm.OLS(np.diff(p), sm.add_constant(p[:-1])).fit().params[1]
    return float(-math.log(2) / slope)


def johansen_portfolio(prices) -> tuple[np.ndarray, float]:
    """The first Johansen eigenvector (a constant, one lagged difference), as statsmodels returns it, and its half-life."""
    with warnings.catch_warnings():
        # statsmodels' eigenvectors come out complex with zero imaginary parts, and it warns as it drops them.
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        evec = coint_johansen(np.asarray(prices), det_order=0, k_ar_diff=1).evec
    weights = np.real_if_close(evec[:, 0])
    return weights, half_life(np.asarray(prices) @ weights)


class KalmanFilter:
    """A Kalman filter for the regression y = beta[0] x + beta[1], with beta a random walk (identity transition)."""

    def __init__(self, delta: float = 0.0001, v_e: float = 0.001) -> None:
        self.v_w = delta / (1 - delta) * np.eye(2)
        self.r = -self.v_w  # the first prediction adds v_w, so the first covariance is 0
        self.beta = np.zeros(2)
        self.v_e = v_e
        self.e = self.q = 0.0

    def forecast(self, y: float, x: float) -> None:
        """One prediction from x, then the correction by y."""
        x_t = np.array([x, 1.0])
        self.r = self.r + self.v_w
        self.q = float(x_t @ self.r @ x_t) + self.v_e
        self.e = y - float(x_t @ self.beta)
        gain = self.r @ x_t / self.q
        self.beta = self.beta + gain * self.e
        self.r = self.r - np.outer(gain, x_t) @ self.r

    def hedge_ratio(self) -> float:
        return float(self.beta[0])

    def measurement_resid(self) -> float:
        return self.e

    def measurement_sd(self) -> float:
        return math.sqrt(self.q)
