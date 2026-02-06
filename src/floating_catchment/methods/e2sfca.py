"""
Enhanced Two-Step Floating Catchment Area (E2SFCA)
==================================================

Reference
---------
Luo, W., & Qi, Y. (2009). An enhanced two-step floating catchment area
(E2SFCA) method for measuring spatial accessibility to primary care
physicians. *Health & Place*, 15(4), 1100-1107.

Method
------
The catchment is divided into *r* concentric zones, each with a weight
*Wᵣ* (e.g. zone 0–10 min = 1.0, 10–20 min = 0.68, 20–30 min = 0.22).

**Step 1** — Supply-to-demand ratio:

    Rⱼ = Sⱼ / Σᵣ Σ_{k ∈ Dᵣ} Dₖ · Wᵣ

**Step 2** — Accessibility index:

    Aᵢ = Σᵣ Σ_{j ∈ Dᵣ} Rⱼ · Wᵣ
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from floating_catchment.decay import zonal as zonal_decay


def e2sfca(
    supply: pd.Series,
    demand: pd.Series,
    cost_matrix: pd.DataFrame,
    zones: list[float],
    weights: list[float],
) -> pd.Series:
    """Compute the E2SFCA accessibility index.

    Parameters
    ----------
    supply : Series
        Supply capacity indexed by supply location ID.
    demand : Series
        Demand (population) indexed by demand location ID.
    cost_matrix : DataFrame
        Cost / distance matrix (rows = demand, columns = supply).
    zones : list[float]
        Sorted zone boundary breakpoints, e.g. ``[10, 20, 30]``.
    weights : list[float]
        Weight for each zone, e.g. ``[1.0, 0.68, 0.22]``.

    Returns
    -------
    Series
        Accessibility index ``Aᵢ`` for each demand location.
    """
    demand_ids = cost_matrix.index
    supply_ids = cost_matrix.columns
    S = supply.reindex(supply_ids)
    D = demand.reindex(demand_ids)
    cost = cost_matrix.values

    # Zonal weights matrix
    W = zonal_decay(cost, zones=zones, weights=weights)

    # Step 1
    weighted_demand = (D.values[:, np.newaxis] * W).sum(axis=0)
    R = np.zeros_like(weighted_demand)
    mask = weighted_demand > 0
    R[mask] = S.values[mask] / weighted_demand[mask]

    # Step 2
    A = (W * R[np.newaxis, :]).sum(axis=1)

    return pd.Series(A, index=demand_ids, name="accessibility")
