"""The regression, the z-score and the Kalman filter the strategies use, with NumPy."""

from __future__ import annotations

import math

import numpy as np


def hedge_ratio(y, x) -> float:
    """The least-squares slope of y on x, with an intercept."""
    design = np.column_stack([x, np.ones(len(x))])
    return float(np.linalg.lstsq(design, np.asarray(y), rcond=None)[0][0])


def mean_sd(window) -> tuple[float, float]:
    """The mean and the sample deviation of a window."""
    values = np.asarray(window)
    return float(values.mean()), float(values.std(ddof=1))


def shares(value: float) -> int | None:
    """A position in whole shares, or None when the value is not finite (nothing is stated)."""
    return round(value) if math.isfinite(value) else None


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
