"""
Floating Catchment Area (FCA) — Spatial Accessibility Library
=============================================================

Modular implementations of FCA family methods:

- **2SFCA**   – Two-Step Floating Catchment Area (Luo & Wang, 2003)
- **E2SFCA**  – Enhanced 2SFCA with distance-decay zones (Luo & Qi, 2009)
- **3SFCA**   – Three-Step FCA with competition weights (Wan et al., 2012)
- **M2SFCA**  – Modified 2SFCA with continuous decay (Delamater, 2013)
- **KD2SFCA** – Kernel Density 2SFCA (Dai, 2010)

Quick start::

    from floating_catchment import two_sfca, euclidean_distance_matrix
    from floating_catchment.decay import gaussian

    cost = euclidean_distance_matrix(supply_gdf, demand_gdf)
    scores = two_sfca(supply, demand, cost, threshold=30)
"""

from floating_catchment.methods.two_sfca import two_sfca
from floating_catchment.methods.e2sfca import e2sfca
from floating_catchment.methods.three_sfca import three_sfca
from floating_catchment.methods.m2sfca import m2sfca
from floating_catchment.methods.kd2sfca import kd2sfca
from floating_catchment.catchment import (
    euclidean_distance_matrix,
    filter_cost_matrix,
)
from floating_catchment.io import (
    load_points,
    load_cost_matrix,
)
from floating_catchment.viz import plot_accessibility, interactive_map

__version__ = "0.1.0"

__all__ = [
    "two_sfca",
    "e2sfca",
    "three_sfca",
    "m2sfca",
    "kd2sfca",
    "euclidean_distance_matrix",
    "filter_cost_matrix",
    "load_points",
    "load_cost_matrix",
    "plot_accessibility",
    "interactive_map",
]
