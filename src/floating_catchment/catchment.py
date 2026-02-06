"""
Catchment area computation.

Provides two strategies for building a cost / distance matrix between
supply and demand locations:

1. **Euclidean buffer** – computed on-the-fly from point geometries.
2. **Pre-computed travel-time matrix** – loaded from an ESRI OD Cost
   Matrix export (or any compatible table) and filtered by a threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def euclidean_distance_matrix(
    supply: gpd.GeoDataFrame,
    demand: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Compute the pairwise Euclidean distance matrix.

    Parameters
    ----------
    supply : GeoDataFrame
        Supply locations. Index values become column labels.
    demand : GeoDataFrame
        Demand locations. Index values become row labels.

    Returns
    -------
    DataFrame
        Shape ``(len(demand), len(supply))``.  ``df.loc[i, j]`` is the
        Euclidean distance from demand *i* to supply *j*.

    Notes
    -----
    Distances are in the CRS unit of the input GeoDataFrames.  For
    meaningful results in metres / kilometres, project both inputs to a
    suitable projected CRS **before** calling this function.
    """
    supply_coords = np.array([(g.x, g.y) for g in supply.geometry])
    demand_coords = np.array([(g.x, g.y) for g in demand.geometry])

    # Broadcasting: (n_demand, 1, 2) - (1, n_supply, 2)
    diff = demand_coords[:, np.newaxis, :] - supply_coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    return pd.DataFrame(dist, index=demand.index, columns=supply.index)


def filter_cost_matrix(
    cost_matrix: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Mask entries in a cost matrix that exceed a threshold.

    Values above the threshold are replaced with ``np.inf`` so that
    decay functions naturally return 0 for unreachable pairs.

    Parameters
    ----------
    cost_matrix : DataFrame
        Cost / distance / travel-time matrix. Rows are demand, columns
        are supply.
    threshold : float
        Maximum cost to keep.

    Returns
    -------
    DataFrame
        Same shape, with out-of-catchment entries set to ``np.inf``.
    """
    result = cost_matrix.copy()
    result[result > threshold] = np.inf
    return result
