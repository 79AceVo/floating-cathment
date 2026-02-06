"""
Kernel Density Two-Step Floating Catchment Area (KD2SFCA)
=========================================================

Reference
---------
Dai, D. (2010). Black residential segregation, disparities in spatial
access to health care facilities, and late-stage breast cancer diagnosis
in metropolitan Detroit. *Health & Place*, 16(5), 1038-1052.

Method
------
Applies a **kernel density function** (e.g. Epanechnikov) as the
distance-decay weight within each catchment.

**Step 1** — Supply-to-demand ratio:

    Rⱼ = Sⱼ / Σ_{k ∈ {d(k,j) ≤ d₀}} Dₖ · K(dₖⱼ / d₀)

**Step 2** — Accessibility:

    Aᵢ = Σ_{j ∈ {d(i,j) ≤ d₀}} Rⱼ · K(dᵢⱼ / d₀)

The default kernel is the Epanechnikov kernel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from floating_catchment import decay as decay_module


def kd2sfca(
    supply: pd.Series,
    demand: pd.Series,
    cost_matrix: pd.DataFrame,
    threshold: float,
    kernel: str = "epanechnikov",
    **kernel_kwargs,
) -> pd.Series:
    """Compute the KD2SFCA accessibility index.

    Parameters
    ----------
    supply : Series
        Supply capacity indexed by supply location ID.
    demand : Series
        Demand (population) indexed by demand location ID.
    cost_matrix : DataFrame
        Cost / distance matrix (rows = demand, columns = supply).
    threshold : float
        Catchment bandwidth ``d₀``.
    kernel : str
        Kernel function name from :mod:`floating_catchment.decay`.
        Default ``"epanechnikov"``.
    **kernel_kwargs
        Extra keyword arguments for the kernel function.

    Returns
    -------
    Series
        Accessibility index ``Aᵢ`` for each demand location.
    """
    fn = getattr(decay_module, kernel)

    demand_ids = cost_matrix.index
    supply_ids = cost_matrix.columns
    S = supply.reindex(supply_ids).values
    D = demand.reindex(demand_ids).values
    cost = cost_matrix.values

    # Kernel weights
    K = fn(cost, threshold, **kernel_kwargs)

    # Step 1
    weighted_demand = (D[:, np.newaxis] * K).sum(axis=0)
    R = np.zeros_like(weighted_demand)
    mask = weighted_demand > 0
    R[mask] = S[mask] / weighted_demand[mask]

    # Step 2
    A = (K * R[np.newaxis, :]).sum(axis=1)

    return pd.Series(A, index=demand_ids, name="accessibility")
