"""Tests for floating_catchment.viz."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI

import matplotlib.pyplot as plt
import folium
import pandas as pd
import geopandas as gpd
import pytest

from floating_catchment.viz import plot_accessibility, interactive_map
from floating_catchment.methods.two_sfca import two_sfca


@pytest.fixture()
def accessibility_gdf(supply_gdf, demand_gdf, supply_series, demand_series, cost_matrix):
    """Demand GeoDataFrame with an accessibility column."""
    scores = two_sfca(supply_series, demand_series, cost_matrix, threshold=500)
    gdf = demand_gdf.copy()
    gdf["accessibility"] = scores
    return gdf


class TestPlotAccessibility:
    def test_returns_figure(self, accessibility_gdf):
        fig = plot_accessibility(accessibility_gdf)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_supply_overlay(self, accessibility_gdf, supply_gdf):
        fig = plot_accessibility(accessibility_gdf, supply_gdf=supply_gdf)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_column(self, accessibility_gdf):
        accessibility_gdf["custom_score"] = accessibility_gdf["accessibility"] * 100
        fig = plot_accessibility(accessibility_gdf, column="custom_score")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestInteractiveMap:
    def test_returns_folium_map(self, accessibility_gdf):
        gdf = accessibility_gdf.set_crs("EPSG:4326", allow_override=True)
        m = interactive_map(gdf)
        assert isinstance(m, folium.Map)

    def test_with_supply(self, accessibility_gdf, supply_gdf):
        gdf = accessibility_gdf.set_crs("EPSG:4326", allow_override=True)
        sgdf = supply_gdf.set_crs("EPSG:4326", allow_override=True)
        m = interactive_map(gdf, supply_gdf=sgdf, supply_label_col="name")
        assert isinstance(m, folium.Map)
