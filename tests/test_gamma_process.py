"""Tests for the GammaProcess class."""


import numpy as np
import pytest

from processes.processes import GammaProcess


@pytest.fixture(scope="module")
def process():
    return GammaProcess(shape=0.3, scale=1.2)


@pytest.mark.parametrize("shape, scale", [(-1, 2), ("a", 1), (1, -1), (1, "2")])
def test_init_errors(shape, scale):
    if all(map(lambda p: isinstance(p, (int, float)), [shape, scale])):
        with pytest.raises(ValueError):
            GammaProcess(shape=shape, scale=scale)
    else:
        with pytest.raises(TypeError):
            GammaProcess(shape=shape, scale=scale)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.shape = -0.1
    with pytest.raises(ValueError):
        process.scale = -0.1


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
