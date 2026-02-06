"""
Two-Step Floating Catchment Area (2SFCA)
========================================

Reference
---------
Luo, W., & Wang, F. (2003). Measures of spatial accessibility to health
care in a GIS environment: synthesis and a case study in the Chicago
region. *Environment and Planning B*, 30(6), 865-884.

Method
------
**Step 1** — For each supply location *j*, compute a supply-to-demand
ratio within its catchment:

    Rⱼ = Sⱼ / Σ_{k ∈ {d(k,j) ≤ d₀}} Dₖ

**Step 2** — For each demand location *i*, sum the ratios of all supply
locations whose catchment contains *i*:

    Aᵢ = Σ_{j ∈ {d(i,j) ≤ d₀}} Rⱼ
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from floating_catchment import decay as decay_module


def two_sfca(
    supply: pd.Series,
    demand: pd.Series,
    cost_matrix: pd.DataFrame,
    threshold: float,
    decay_fn: str = "binary",
    **decay_kwargs,
) -> pd.Series:
    """Compute the 2SFCA accessibility index.

    Parameters
    ----------
    supply : Series
        Supply capacity indexed by supply location ID.
    demand : Series
        Demand (population) indexed by demand location ID.
    cost_matrix : DataFrame
        Cost / distance matrix.  Rows = demand IDs, columns = supply IDs.
    threshold : float
        Catchment size ``d₀``.
    decay_fn : str
        Name of a function in :mod:`floating_catchment.decay`.
        Default ``"binary"`` gives the classic 2SFCA.
    **decay_kwargs
        Extra keyword arguments forwarded to the decay function.

    Returns
    -------
    Series
        Accessibility index ``Aᵢ`` for each demand location.
    """
    fn = getattr(decay_module, decay_fn)

    # Ensure alignment
    demand_ids = cost_matrix.index
    supply_ids = cost_matrix.columns
    S = supply.reindex(supply_ids)
    D = demand.reindex(demand_ids)
    cost = cost_matrix.values  # (n_demand, n_supply)

    # Step 1: supply-to-demand ratio per supply location
    weights = fn(cost, threshold, **decay_kwargs)  # (n_demand, n_supply)
    weighted_demand = (D.values[:, np.newaxis] * weights).sum(axis=0)
    # Avoid division by zero — set ratio to 0 where no demand in catchment
    R = np.zeros_like(weighted_demand)
    mask = weighted_demand > 0
    R[mask] = S.values[mask] / weighted_demand[mask]

    # Step 2: sum ratios for each demand location
    A = (weights * R[np.newaxis, :]).sum(axis=1)

    return pd.Series(A, index=demand_ids, name="accessibility")
