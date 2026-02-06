"""Tests for the 3SFCA method."""

import numpy as np
import pandas as pd
import pytest

from floating_catchment.methods.three_sfca import three_sfca


class TestThreeSFCA:
    def test_returns_series(self, supply_series, demand_series, cost_matrix):
        result = three_sfca(
            supply_series, demand_series, cost_matrix, threshold=500,
        )
        assert isinstance(result, pd.Series)
        assert result.name == "accessibility"

    def test_correct_length(self, supply_series, demand_series, cost_matrix):
        result = three_sfca(
            supply_series, demand_series, cost_matrix, threshold=500,
        )
        assert len(result) == len(demand_series)

    def test_all_non_negative(self, supply_series, demand_series, cost_matrix):
        result = three_sfca(
            supply_series, demand_series, cost_matrix, threshold=500,
        )
        assert (result >= 0).all()

    def test_zero_threshold(self, supply_series, demand_series, cost_matrix):
        result = three_sfca(
            supply_series, demand_series, cost_matrix, threshold=0,
        )
        assert (result == 0).all()

    def test_competition_reduces_scores(
        self, supply_series, demand_series, cost_matrix
    ):
        """3SFCA accounts for competition — scores should generally differ
        from 2SFCA."""
        from floating_catchment.methods.two_sfca import two_sfca

        r_3 = three_sfca(
            supply_series, demand_series, cost_matrix,
            threshold=500, decay_fn="gaussian",
        )
        r_2 = two_sfca(
            supply_series, demand_series, cost_matrix,
            threshold=500, decay_fn="gaussian",
        )
        # They should produce different distributions
        assert not np.allclose(r_3.values, r_2.values)
