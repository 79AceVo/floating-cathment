"""Tests for floating_catchment.io."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest

from floating_catchment.io import load_points, load_cost_matrix


class TestLoadPoints:
    def test_from_geodataframe(self, supply_gdf):
        result = load_points(supply_gdf)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 3

    def test_from_dataframe(self):
        df = pd.DataFrame({
            "latitude": [40.0, 41.0],
            "longitude": [-74.0, -73.0],
            "value": [1, 2],
        })
        result = load_points(df)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        assert result.geometry.iloc[0].x == -74.0
        assert result.geometry.iloc[0].y == 40.0

    def test_from_csv(self, sample_csv_supply):
        result = load_points(sample_csv_supply)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 3

    def test_custom_columns(self):
        df = pd.DataFrame({
            "lat": [40.0],
            "lng": [-74.0],
        })
        result = load_points(df, lat_col="lat", lon_col="lng")
        assert result.geometry.iloc[0].x == -74.0

    def test_id_col(self):
        df = pd.DataFrame({
            "id": ["A", "B"],
            "latitude": [40.0, 41.0],
            "longitude": [-74.0, -73.0],
        })
        result = load_points(df, id_col="id")
        assert list(result.index) == ["A", "B"]


class TestLoadCostMatrix:
    def test_long_format(self, sample_long_cost_csv, cost_matrix):
        result = load_cost_matrix(sample_long_cost_csv, fmt="long")
        assert result.shape == cost_matrix.shape
        # Values should match the original cost matrix
        for oi in cost_matrix.index:
            for di in cost_matrix.columns:
                assert result.loc[oi, di] == pytest.approx(
                    cost_matrix.loc[oi, di]
                )

    def test_wide_format(self, cost_matrix, tmp_path):
        path = tmp_path / "wide.csv"
        cost_matrix.to_csv(path)
        result = load_cost_matrix(path, fmt="wide")
        assert result.shape[0] == cost_matrix.shape[0]

    def test_from_dataframe_long(self, cost_matrix):
        records = []
        for origin in cost_matrix.index:
            for dest in cost_matrix.columns:
                records.append({
                    "OriginID": origin,
                    "DestinationID": dest,
                    "Total_Cost": cost_matrix.loc[origin, dest],
                })
        df = pd.DataFrame(records)
        result = load_cost_matrix(df, fmt="long")
        assert result.shape == cost_matrix.shape

    def test_invalid_format_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Unsupported format"):
            load_cost_matrix(df, fmt="invalid")
