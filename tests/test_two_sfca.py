"""Tests for the 2SFCA method."""

import numpy as np
import pandas as pd
import pytest

from floating_catchment.methods.two_sfca import two_sfca


class TestTwoSFCA:
    def test_returns_series(self, supply_series, demand_series, cost_matrix):
        result = two_sfca(supply_series, demand_series, cost_matrix, threshold=500)
        assert isinstance(result, pd.Series)
        assert result.name == "accessibility"

    def test_correct_length(self, supply_series, demand_series, cost_matrix):
        result = two_sfca(supply_series, demand_series, cost_matrix, threshold=500)
        assert len(result) == len(demand_series)

    def test_all_non_negative(self, supply_series, demand_series, cost_matrix):
        result = two_sfca(supply_series, demand_series, cost_matrix, threshold=500)
        assert (result >= 0).all()

    def test_zero_threshold_gives_zero(self, supply_series, demand_series, cost_matrix):
        """With threshold=0, no demand reaches any supply."""
        result = two_sfca(supply_series, demand_series, cost_matrix, threshold=0)
        assert (result == 0).all()

    def test_very_large_threshold(self, supply_series, demand_series, cost_matrix):
        """All demand reaches all supply — everyone gets the same score."""
        result = two_sfca(supply_series, demand_series, cost_matrix, threshold=10000)
        # With binary decay, all weights = 1, so every demand gets
        # total_supply / total_demand
        total_S = supply_series.sum()
        total_D = demand_series.sum()
        expected = total_S / total_D
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_closer_demand_has_higher_score(
        self, supply_series, demand_series, cost_matrix
    ):
        """P2 is close to H1 — should have >= P8 which is far from H1."""
        result = two_sfca(supply_series, demand_series, cost_matrix, threshold=300)
        # P2 is ~70 from H1; P8 is ~721 from H1 — P8 only reachable by H2
        assert result.loc["P2"] >= result.loc["P8"] or result.loc["P8"] == 0

    def test_with_gaussian_decay(self, supply_series, demand_series, cost_matrix):
        result = two_sfca(
            supply_series, demand_series, cost_matrix,
            threshold=500, decay_fn="gaussian",
        )
        assert (result >= 0).all()
        assert len(result) == len(demand_series)

    def test_binary_vs_gaussian(self, supply_series, demand_series, cost_matrix):
        """Gaussian decay should produce different scores than binary."""
        r_bin = two_sfca(
            supply_series, demand_series, cost_matrix,
            threshold=500, decay_fn="binary",
        )
        r_gau = two_sfca(
            supply_series, demand_series, cost_matrix,
            threshold=500, decay_fn="gaussian",
        )
        # They should not be identical (unless a degenerate case)
        assert not np.allclose(r_bin.values, r_gau.values)
