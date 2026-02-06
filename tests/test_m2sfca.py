"""Tests for the M2SFCA method."""

import numpy as np
import pandas as pd
import pytest

from floating_catchment.methods.m2sfca import m2sfca


class TestM2SFCA:
    def test_returns_series(self, supply_series, demand_series, cost_matrix):
        result = m2sfca(
            supply_series, demand_series, cost_matrix, threshold=500,
        )
        assert isinstance(result, pd.Series)
        assert result.name == "accessibility"

    def test_correct_length(self, supply_series, demand_series, cost_matrix):
        result = m2sfca(
            supply_series, demand_series, cost_matrix, threshold=500,
        )
        assert len(result) == len(demand_series)

    def test_all_non_negative(self, supply_series, demand_series, cost_matrix):
        result = m2sfca(
            supply_series, demand_series, cost_matrix, threshold=500,
        )
        assert (result >= 0).all()

    def test_normalisation_effect(self, supply_series, demand_series, cost_matrix):
        """M2SFCA normalises weights per demand location — should differ
        from plain 2SFCA with same decay."""
        from floating_catchment.methods.two_sfca import two_sfca

        r_m = m2sfca(
            supply_series, demand_series, cost_matrix,
            threshold=500, decay_fn="gaussian",
        )
        r_2 = two_sfca(
            supply_series, demand_series, cost_matrix,
            threshold=500, decay_fn="gaussian",
        )
        assert not np.allclose(r_m.values, r_2.values)

    def test_zero_threshold(self, supply_series, demand_series, cost_matrix):
        result = m2sfca(
            supply_series, demand_series, cost_matrix, threshold=0,
        )
        assert (result == 0).all()
