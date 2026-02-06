"""Tests for floating_catchment.catchment."""

import numpy as np
import pandas as pd
import pytest

from floating_catchment.catchment import euclidean_distance_matrix, filter_cost_matrix


class TestEuclideanDistanceMatrix:
    def test_shape(self, supply_gdf, demand_gdf):
        result = euclidean_distance_matrix(supply_gdf, demand_gdf)
        assert result.shape == (len(demand_gdf), len(supply_gdf))

    def test_index_alignment(self, supply_gdf, demand_gdf):
        result = euclidean_distance_matrix(supply_gdf, demand_gdf)
        assert list(result.index) == list(demand_gdf.index)
        assert list(result.columns) == list(supply_gdf.index)

    def test_diagonal_like(self, supply_gdf):
        """Distance from a point to itself should be zero."""
        result = euclidean_distance_matrix(supply_gdf, supply_gdf)
        for idx in supply_gdf.index:
            assert result.loc[idx, idx] == pytest.approx(0.0)

    def test_symmetry(self, supply_gdf, demand_gdf):
        """d(i,j) should equal d(j,i) when computed both ways."""
        m1 = euclidean_distance_matrix(supply_gdf, demand_gdf)
        m2 = euclidean_distance_matrix(demand_gdf, supply_gdf)
        # m1 is (demand, supply), m2 is (supply, demand)
        for si in supply_gdf.index:
            for di in demand_gdf.index:
                assert m1.loc[di, si] == pytest.approx(m2.loc[si, di])

    def test_known_distance(self, supply_gdf, demand_gdf):
        """Spot-check a known distance."""
        # H1=(100,100), P1=(50,80) → sqrt(50²+20²) = sqrt(2900)
        result = euclidean_distance_matrix(supply_gdf, demand_gdf)
        expected = np.sqrt(50**2 + 20**2)
        assert result.loc["P1", "H1"] == pytest.approx(expected)

    def test_all_non_negative(self, supply_gdf, demand_gdf):
        result = euclidean_distance_matrix(supply_gdf, demand_gdf)
        assert (result.values >= 0).all()


class TestFilterCostMatrix:
    def test_within_threshold_unchanged(self, cost_matrix):
        threshold = 10000  # very large — everything stays
        result = filter_cost_matrix(cost_matrix, threshold)
        np.testing.assert_array_almost_equal(result.values, cost_matrix.values)

    def test_above_threshold_becomes_inf(self, cost_matrix):
        threshold = 200
        result = filter_cost_matrix(cost_matrix, threshold)
        mask = cost_matrix.values > threshold
        assert np.all(np.isinf(result.values[mask]))

    def test_does_not_modify_original(self, cost_matrix):
        original = cost_matrix.copy()
        filter_cost_matrix(cost_matrix, 200)
        pd.testing.assert_frame_equal(cost_matrix, original)
