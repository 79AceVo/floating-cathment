"""
Shared test fixtures — synthetic supply, demand, and cost data.

Layout (projected CRS units — think of metres):

    Supply locations (hospitals):
        H1 (100, 100)  capacity=10
        H2 (500, 500)  capacity=20
        H3 (900, 100)  capacity=15

    Demand locations (population centres):
        P1 (  50,  80)  pop=1000
        P2 ( 150, 120)  pop=2000
        P3 ( 200, 500)  pop=1500
        P4 ( 480, 520)  pop=3000
        P5 ( 600, 400)  pop= 800
        P6 ( 850, 150)  pop=2500
        P7 ( 920,  80)  pop=1200
        P8 ( 700, 700)  pop=1800
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pytest


@pytest.fixture()
def supply_gdf() -> gpd.GeoDataFrame:
    data = {
        "name": ["H1", "H2", "H3"],
        "capacity": [10, 20, 15],
    }
    geometry = [Point(100, 100), Point(500, 500), Point(900, 100)]
    return gpd.GeoDataFrame(data, geometry=geometry, index=["H1", "H2", "H3"])


@pytest.fixture()
def demand_gdf() -> gpd.GeoDataFrame:
    data = {
        "name": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
        "population": [1000, 2000, 1500, 3000, 800, 2500, 1200, 1800],
    }
    geometry = [
        Point(50, 80),
        Point(150, 120),
        Point(200, 500),
        Point(480, 520),
        Point(600, 400),
        Point(850, 150),
        Point(920, 80),
        Point(700, 700),
    ]
    return gpd.GeoDataFrame(
        data,
        geometry=geometry,
        index=["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
    )


@pytest.fixture()
def supply_series(supply_gdf) -> pd.Series:
    return supply_gdf["capacity"]


@pytest.fixture()
def demand_series(demand_gdf) -> pd.Series:
    return demand_gdf["population"]


@pytest.fixture()
def cost_matrix(supply_gdf, demand_gdf) -> pd.DataFrame:
    """Euclidean distance matrix computed from the fixtures."""
    from floating_catchment.catchment import euclidean_distance_matrix

    return euclidean_distance_matrix(supply_gdf, demand_gdf)


@pytest.fixture()
def sample_csv_supply(tmp_path, supply_gdf):
    """Write supply data to a temporary CSV and return the path."""
    path = tmp_path / "supply.csv"
    df = supply_gdf.copy()
    df["longitude"] = df.geometry.x
    df["latitude"] = df.geometry.y
    df.drop(columns="geometry").to_csv(path, index=True)
    return path


@pytest.fixture()
def sample_csv_demand(tmp_path, demand_gdf):
    """Write demand data to a temporary CSV and return the path."""
    path = tmp_path / "demand.csv"
    df = demand_gdf.copy()
    df["longitude"] = df.geometry.x
    df["latitude"] = df.geometry.y
    df.drop(columns="geometry").to_csv(path, index=True)
    return path


@pytest.fixture()
def sample_long_cost_csv(tmp_path, cost_matrix):
    """Write cost matrix in ESRI OD Cost Matrix long format."""
    path = tmp_path / "od_cost.csv"
    records = []
    for origin in cost_matrix.index:
        for dest in cost_matrix.columns:
            records.append({
                "OriginID": origin,
                "DestinationID": dest,
                "Total_Cost": cost_matrix.loc[origin, dest],
            })
    pd.DataFrame(records).to_csv(path, index=False)
    return path
