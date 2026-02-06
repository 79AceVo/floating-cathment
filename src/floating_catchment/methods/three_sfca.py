"""
Three-Step Floating Catchment Area (3SFCA)
==========================================

Reference
---------
Wan, N., Zou, B., & Sternberg, T. (2012). A three-step floating
catchment area method for analyzing spatial access to health services.
*International Journal of Geographical Information Science*, 26(6),
1073-1089.

Method
------
Adds a **selection / competition weight** step before the standard 2SFCA.

**Step 0** — Compute selection weights for each demand-supply pair.
The probability that demand *i* selects supply *j* is proportional to
the supply capacity weighted by distance decay, relative to all other
reachable supply:

    Tᵢⱼ = G(dᵢⱼ) · Sⱼ / Σ_{j' ∈ {d(i,j') ≤ d₀}} G(dᵢⱼ') · Sⱼ'

**Step 1** — Supply-to-demand ratio:

    Rⱼ = Sⱼ / Σ_{k ∈ {d(k,j) ≤ d₀}} Dₖ · Tₖⱼ · G(dₖⱼ)

**Step 2** — Accessibility index:

    Aᵢ = Σ_{j ∈ {d(i,j) ≤ d₀}} Rⱼ · Tᵢⱼ · G(dᵢⱼ)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from floating_catchment import decay as decay_module


def three_sfca(
    supply: pd.Series,
    demand: pd.Series,
    cost_matrix: pd.DataFrame,
    threshold: float,
    decay_fn: str = "gaussian",
    **decay_kwargs,
) -> pd.Series:
    """Compute the 3SFCA accessibility index.

    Parameters
    ----------
    supply : Series
        Supply capacity indexed by supply location ID.
    demand : Series
        Demand (population) indexed by demand location ID.
    cost_matrix : DataFrame
        Cost / distance matrix (rows = demand, columns = supply).
    threshold : float
        Catchment size ``d₀``.
    decay_fn : str
        Name of a function in :mod:`floating_catchment.decay`.
    **decay_kwargs
        Extra keyword arguments for the decay function.

    Returns
    -------
    Series
        Accessibility index ``Aᵢ`` for each demand location.
    """
    fn = getattr(decay_module, decay_fn)

    demand_ids = cost_matrix.index
    supply_ids = cost_matrix.columns
    S = supply.reindex(supply_ids).values
    D = demand.reindex(demand_ids).values
    cost = cost_matrix.values  # (n_demand, n_supply)

    G = fn(cost, threshold, **decay_kwargs)  # decay weights

    # Step 0: selection weights T[i, j]
    # Numerator: G[i,j] * S[j]
    numerator = G * S[np.newaxis, :]
    # Denominator: for each demand i, sum over all reachable supply
    denom = numerator.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)  # avoid /0
    T = numerator / denom

    # Step 1: supply-to-demand ratio
    weighted_demand = (D[:, np.newaxis] * T * G).sum(axis=0)
    R = np.zeros_like(weighted_demand)
    mask = weighted_demand > 0
    R[mask] = S[mask] / weighted_demand[mask]

    # Step 2: accessibility
    A = (R[np.newaxis, :] * T * G).sum(axis=1)

    return pd.Series(A, index=demand_ids, name="accessibility")
