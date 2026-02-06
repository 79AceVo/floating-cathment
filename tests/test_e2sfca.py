"""Tests for the E2SFCA method."""

import numpy as np
import pandas as pd
import pytest

from floating_catchment.methods.e2sfca import e2sfca


class TestE2SFCA:
    def test_returns_series(self, supply_series, demand_series, cost_matrix):
        result = e2sfca(
            supply_series, demand_series, cost_matrix,
            zones=[200, 400, 600], weights=[1.0, 0.68, 0.22],
        )
        assert isinstance(result, pd.Series)
        assert result.name == "accessibility"

    def test_correct_length(self, supply_series, demand_series, cost_matrix):
        result = e2sfca(
            supply_series, demand_series, cost_matrix,
            zones=[200, 400, 600], weights=[1.0, 0.68, 0.22],
        )
        assert len(result) == len(demand_series)

    def test_all_non_negative(self, supply_series, demand_series, cost_matrix):
        result = e2sfca(
            supply_series, demand_series, cost_matrix,
            zones=[200, 400, 600], weights=[1.0, 0.68, 0.22],
        )
        assert (result >= 0).all()

    def test_single_zone_equals_binary_2sfca(
        self, supply_series, demand_series, cost_matrix
    ):
        """A single zone with weight 1.0 should match 2SFCA with binary decay."""
        from floating_catchment.methods.two_sfca import two_sfca

        threshold = 600
        r_e2sfca = e2sfca(
            supply_series, demand_series, cost_matrix,
            zones=[threshold], weights=[1.0],
        )
        r_2sfca = two_sfca(
            supply_series, demand_series, cost_matrix,
            threshold=threshold, decay_fn="binary",
        )
        np.testing.assert_array_almost_equal(r_e2sfca.values, r_2sfca.values)

    def test_more_zones_different_scores(
        self, supply_series, demand_series, cost_matrix
    ):
        r1 = e2sfca(
            supply_series, demand_series, cost_matrix,
            zones=[600], weights=[1.0],
        )
        r3 = e2sfca(
            supply_series, demand_series, cost_matrix,
            zones=[200, 400, 600], weights=[1.0, 0.68, 0.22],
        )
        assert not np.allclose(r1.values, r3.values)
