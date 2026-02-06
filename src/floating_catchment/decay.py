"""
Distance-decay functions.

Every function in this module has the same signature::

    weights = fn(distances, threshold, **params)

where *distances* is an array-like of cost values and *threshold* is the
catchment size ``d₀``.  The return value is an array of weights in [0, 1].
Values beyond the threshold always receive weight 0.

References
----------
- Binary / step:  Luo & Wang (2003)
- Linear:         Simple linear interpolation
- Gaussian:       Kwan (1998), Joseph & Bantock (1982)
- Epanechnikov:   Silverman (1986)
- Butterworth:    Butterworth filter adapted for spatial analysis
- Power:          Generic inverse-power decay
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _to_array(d: ArrayLike) -> np.ndarray:
    return np.asarray(d, dtype=float)


# --------------------------------------------------------------------------- #
#  Decay functions
# --------------------------------------------------------------------------- #

def binary(d: ArrayLike, d0: float) -> np.ndarray:
    """Step / binary decay.  Weight is 1 inside catchment, 0 outside.

    W(d) = 1  if d ≤ d₀
           0  otherwise
    """
    d = _to_array(d)
    return np.where(d <= d0, 1.0, 0.0)


def linear(d: ArrayLike, d0: float) -> np.ndarray:
    """Linear decay from 1 at d=0 to 0 at d=d₀.

    W(d) = 1 − d/d₀   if d ≤ d₀
           0            otherwise
    """
    d = _to_array(d)
    if d0 == 0:
        return np.where(d == 0, 1.0, 0.0)
    w = 1.0 - d / d0
    return np.where(d <= d0, np.maximum(w, 0.0), 0.0)


def gaussian(d: ArrayLike, d0: float, beta: float = 1.0) -> np.ndarray:
    """Gaussian decay.

    W(d) = exp(−d² / (β · d₀²))   if d ≤ d₀
           0                        otherwise

    Parameters
    ----------
    beta : float
        Shape parameter controlling the steepness of decay.  A larger
        value produces a flatter curve.
    """
    d = _to_array(d)
    if d0 == 0:
        return np.where(d == 0, 1.0, 0.0)
    w = np.exp(-(d ** 2) / (beta * d0 ** 2))
    return np.where(d <= d0, w, 0.0)


def epanechnikov(d: ArrayLike, d0: float) -> np.ndarray:
    """Epanechnikov kernel decay.

    W(d) = ¾ · (1 − (d/d₀)²)   if d ≤ d₀
           0                     otherwise
    """
    d = _to_array(d)
    if d0 == 0:
        return np.where(d == 0, 0.75, 0.0)
    u = d / d0
    w = 0.75 * (1.0 - u ** 2)
    return np.where(d <= d0, np.maximum(w, 0.0), 0.0)


def butterworth(d: ArrayLike, d0: float, n: int = 2) -> np.ndarray:
    """Butterworth filter decay.

    W(d) = 1 / (1 + (d/d₀)^(2n))   if d ≤ d₀
           0                         otherwise

    Parameters
    ----------
    n : int
        Filter order.  Higher values produce a sharper cutoff.
    """
    d = _to_array(d)
    w = 1.0 / (1.0 + (d / d0) ** (2 * n))
    return np.where(d <= d0, w, 0.0)


def power(d: ArrayLike, d0: float, alpha: float = 1.5) -> np.ndarray:
    """Inverse-power decay.

    W(d) = 1 − (d/d₀)^α   if d ≤ d₀
           0                otherwise

    Parameters
    ----------
    alpha : float
        Exponent controlling the curvature.
    """
    d = _to_array(d)
    w = 1.0 - (d / d0) ** alpha
    return np.where(d <= d0, np.maximum(w, 0.0), 0.0)


def zonal(
    d: ArrayLike,
    zones: list[float],
    weights: list[float],
) -> np.ndarray:
    """Zonal / step-wise decay for E2SFCA.

    The catchment is divided into concentric zones defined by breakpoints.
    Each zone is assigned a constant weight.

    Parameters
    ----------
    d : array-like
        Distance / cost values.
    zones : list[float]
        Sorted zone boundaries, e.g. ``[10, 20, 30]`` for three zones
        (0–10, 10–20, 20–30).
    weights : list[float]
        Weight for each zone (same length as *zones*).  E.g.
        ``[1.0, 0.68, 0.22]``.

    Example
    -------
    >>> zonal([5, 15, 25, 35], zones=[10, 20, 30], weights=[1.0, 0.68, 0.22])
    array([1.  , 0.68, 0.22, 0.  ])
    """
    d = _to_array(d)
    if len(zones) != len(weights):
        raise ValueError("zones and weights must have the same length")

    result = np.zeros_like(d)
    lower = 0.0
    for upper, w in zip(zones, weights):
        mask = (d > lower) & (d <= upper)
        result[mask] = w
        lower = upper

    # Handle d == 0 (same location): assign first zone weight
    result[d == 0] = weights[0] if weights else 0.0

    return result
