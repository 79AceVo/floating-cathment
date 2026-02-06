"""Tests for floating_catchment.decay."""

import numpy as np
import pytest

from floating_catchment import decay


class TestBinary:
    def test_inside_catchment(self):
        assert decay.binary(5, 10) == pytest.approx(1.0)

    def test_on_boundary(self):
        assert decay.binary(10, 10) == pytest.approx(1.0)

    def test_outside_catchment(self):
        assert decay.binary(15, 10) == pytest.approx(0.0)

    def test_array(self):
        result = decay.binary([0, 5, 10, 15], 10)
        np.testing.assert_array_almost_equal(result, [1, 1, 1, 0])


class TestLinear:
    def test_at_origin(self):
        assert decay.linear(0, 10) == pytest.approx(1.0)

    def test_midpoint(self):
        assert decay.linear(5, 10) == pytest.approx(0.5)

    def test_at_boundary(self):
        assert decay.linear(10, 10) == pytest.approx(0.0)

    def test_outside(self):
        assert decay.linear(15, 10) == pytest.approx(0.0)


class TestGaussian:
    def test_at_origin(self):
        assert decay.gaussian(0, 10) == pytest.approx(1.0)

    def test_decreases_with_distance(self):
        w1 = decay.gaussian(3, 10)
        w2 = decay.gaussian(7, 10)
        assert w1 > w2

    def test_outside(self):
        assert decay.gaussian(11, 10) == pytest.approx(0.0)

    def test_beta_parameter(self):
        # Higher beta → flatter curve → higher weight at same distance
        w_low = decay.gaussian(5, 10, beta=0.5)
        w_high = decay.gaussian(5, 10, beta=2.0)
        assert w_high > w_low


class TestEpanechnikov:
    def test_at_origin(self):
        assert decay.epanechnikov(0, 10) == pytest.approx(0.75)

    def test_at_boundary(self):
        assert decay.epanechnikov(10, 10) == pytest.approx(0.0)

    def test_outside(self):
        assert decay.epanechnikov(15, 10) == pytest.approx(0.0)

    def test_decreases(self):
        w1 = decay.epanechnikov(2, 10)
        w2 = decay.epanechnikov(8, 10)
        assert w1 > w2


class TestButterworth:
    def test_at_origin(self):
        assert decay.butterworth(0, 10) == pytest.approx(1.0)

    def test_outside(self):
        assert decay.butterworth(15, 10) == pytest.approx(0.0)

    def test_higher_order_sharper(self):
        w_low = decay.butterworth(8, 10, n=1)
        w_high = decay.butterworth(8, 10, n=4)
        # Higher order → sharper cutoff → at same point below threshold,
        # higher-order retains more weight
        assert w_high > w_low


class TestPower:
    def test_at_origin(self):
        assert decay.power(0, 10) == pytest.approx(1.0)

    def test_at_boundary(self):
        assert decay.power(10, 10) == pytest.approx(0.0)

    def test_outside(self):
        assert decay.power(15, 10) == pytest.approx(0.0)


class TestZonal:
    def test_basic_zones(self):
        result = decay.zonal(
            [5, 15, 25, 35],
            zones=[10, 20, 30],
            weights=[1.0, 0.68, 0.22],
        )
        np.testing.assert_array_almost_equal(result, [1.0, 0.68, 0.22, 0.0])

    def test_on_boundary(self):
        result = decay.zonal(
            [10, 20, 30],
            zones=[10, 20, 30],
            weights=[1.0, 0.68, 0.22],
        )
        np.testing.assert_array_almost_equal(result, [1.0, 0.68, 0.22])

    def test_zero_distance(self):
        result = decay.zonal([0], zones=[10, 20], weights=[1.0, 0.5])
        assert result[0] == pytest.approx(1.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            decay.zonal([5], zones=[10, 20], weights=[1.0])
