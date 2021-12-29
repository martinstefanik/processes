"""Tests for the time_grid method inherited by all stochastic processes."""

import numpy as np
import pytest

from processes.processes import StochasticProcess


def test_time_grid(T, n_time_grid):
    time = StochasticProcess.time_grid(T=T, n_time_grid=n_time_grid)
    assert isinstance(time, np.ndarray)
    assert time.shape == (n_time_grid,)
    assert time[0] == 0
    assert time[-1] == T


def test_time_grid_invalid_T(T_invalid, n_time_grid):
    if isinstance(T_invalid, (int, float)):
        with pytest.raises(ValueError):
            StochasticProcess.time_grid(T=T_invalid, n_time_grid=n_time_grid)
    else:
        with pytest.raises(TypeError):
            StochasticProcess.time_grid(T=T_invalid, n_time_grid=n_time_grid)


def test_time_grid_invalid_n_time_grid(T, n_time_grid_invalid):
    if isinstance(n_time_grid_invalid, int):
        with pytest.raises(ValueError):
            StochasticProcess.time_grid(T=T, n_time_grid=n_time_grid_invalid)
    else:
        with pytest.raises(TypeError):
            StochasticProcess.time_grid(T=T, n_time_grid=n_time_grid_invalid)
