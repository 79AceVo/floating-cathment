"""
Modified Two-Step Floating Catchment Area (M2SFCA)
==================================================

Reference
---------
Delamater, P. L. (2013). Spatial accessibility in suboptimally
configured health care systems: A modified two-step floating catchment
area (M2SFCA) metric. *Health & Place*, 24, 30-43.

Method
------
Uses a **continuous** distance-decay function and accounts for
*suboptimal* configurations by normalising weights so that each demand
location's influence sums to 1 across all reachable supply.

**Step 1** — Supply-to-demand ratio with normalised demand weights:

    Rⱼ = Sⱼ / Σ_{k ∈ {d(k,j) ≤ d₀}} Dₖ · f(dₖⱼ) / Σ_{j'} f(dₖⱼ')

**Step 2** — Accessibility:

    Aᵢ = Σ_{j ∈ {d(i,j) ≤ d₀}} Rⱼ · f(dᵢⱼ) / Σ_{j'} f(dᵢⱼ')
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from floating_catchment import decay as decay_module


def m2sfca(
    supply: pd.Series,
    demand: pd.Series,
    cost_matrix: pd.DataFrame,
    threshold: float,
    decay_fn: str = "gaussian",
    **decay_kwargs,
) -> pd.Series:
    """Compute the M2SFCA accessibility index.

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
    cost = cost_matrix.values

    # Raw decay weights
    f = fn(cost, threshold, **decay_kwargs)  # (n_demand, n_supply)

    # Row-normalise: each demand location's weights sum to 1
    row_sums = f.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    f_norm = f / row_sums

    # Step 1
    weighted_demand = (D[:, np.newaxis] * f_norm).sum(axis=0)
    R = np.zeros_like(weighted_demand)
    mask = weighted_demand > 0
    R[mask] = S[mask] / weighted_demand[mask]

    # Step 2
    A = (f_norm * R[np.newaxis, :]).sum(axis=1)

    return pd.Series(A, index=demand_ids, name="accessibility")
