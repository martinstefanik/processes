"""Tests for the TimeChangedBrownianMotion class."""


import numpy as np
import pytest

from processes.processes import TimeChangedBrownianMotion


@pytest.fixture(scope="module")
def process():
    return TimeChangedBrownianMotion(time_change=lambda t: np.tanh(t))


@pytest.mark.parametrize("time_change", [1, "a", lambda x, t: x])
def test_init_errors(time_change):
    if callable(time_change):
        with pytest.raises(ValueError):
            TimeChangedBrownianMotion(time_change=time_change)
    else:
        with pytest.raises(TypeError):
            TimeChangedBrownianMotion(time_change=time_change)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.time_change = lambda x, t: x


def test_sample(process, T, n_time_grid, n_paths):
    x0 = 1
    paths = process.sample(T=T, n_time_grid=n_time_grid, x0=x0, n_paths=n_paths)
    assert isinstance(paths, np.ndarray)
    if n_paths == 1:
        assert paths.shape == (n_time_grid,)
        assert paths[0] == x0
    else:
        assert paths.shape == (n_paths, n_time_grid)
        assert np.all(paths[:, 0] == x0)


@pytest.mark.parametrize(
    "time_change", [lambda t: np.exp(-t), lambda t: -np.exp(-t)]
)
def test_invalid_time_change(process, time_change, n_time_grid):
    process.time_change = time_change
    with pytest.raises(ValueError):
        process.sample(T=1, n_time_grid=n_time_grid, x0=1, n_paths=1)
