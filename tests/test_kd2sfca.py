"""Tests for the KD2SFCA method."""

import numpy as np
import pandas as pd
import pytest

from floating_catchment.methods.kd2sfca import kd2sfca


class TestKD2SFCA:
    def test_returns_series(self, supply_series, demand_series, cost_matrix):
        result = kd2sfca(
            supply_series, demand_series, cost_matrix, threshold=500,
        )
        assert isinstance(result, pd.Series)
        assert result.name == "accessibility"

    def test_correct_length(self, supply_series, demand_series, cost_matrix):
        result = kd2sfca(
            supply_series, demand_series, cost_matrix, threshold=500,
        )
        assert len(result) == len(demand_series)

    def test_all_non_negative(self, supply_series, demand_series, cost_matrix):
        result = kd2sfca(
            supply_series, demand_series, cost_matrix, threshold=500,
        )
        assert (result >= 0).all()

    def test_default_kernel_is_epanechnikov(
        self, supply_series, demand_series, cost_matrix
    ):
        """Default should use Epanechnikov kernel."""
        from floating_catchment.methods.two_sfca import two_sfca

        r_kd = kd2sfca(
            supply_series, demand_series, cost_matrix, threshold=500,
        )
        r_ep = two_sfca(
            supply_series, demand_series, cost_matrix,
            threshold=500, decay_fn="epanechnikov",
        )
        np.testing.assert_array_almost_equal(r_kd.values, r_ep.values)

    def test_different_kernels(self, supply_series, demand_series, cost_matrix):
        r_ep = kd2sfca(
            supply_series, demand_series, cost_matrix,
            threshold=500, kernel="epanechnikov",
        )
        r_ga = kd2sfca(
            supply_series, demand_series, cost_matrix,
            threshold=500, kernel="gaussian",
        )
        assert not np.allclose(r_ep.values, r_ga.values)

    def test_zero_threshold(self, supply_series, demand_series, cost_matrix):
        result = kd2sfca(
            supply_series, demand_series, cost_matrix, threshold=0,
        )
        assert (result == 0).all()
