"""Tests for the BrownianMotion class."""


import numpy as np
import pytest

from processes.processes import BrownianMotion


@pytest.fixture(scope="module")
def process():
    return BrownianMotion(mu=0.2, sigma=0.1)


@pytest.mark.parametrize("mu, sigma", [(1, -2), (1, {}), ({}, 1.2)])
def test_init_errors(mu, sigma):
    if all(map(lambda p: isinstance(p, (int, float)), [mu, sigma])):
        with pytest.raises(ValueError):
            BrownianMotion(mu=mu, sigma=sigma)
    else:
        with pytest.raises(TypeError):
            BrownianMotion(mu=mu, sigma=sigma)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.sigma = -0.1


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
