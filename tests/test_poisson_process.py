"""Tests for the PoissonProcess class."""


import numpy as np
import pytest

from processes.processes import PoissonProcess


@pytest.fixture(scope="module")
def process():
    return PoissonProcess(intensity=5)


@pytest.mark.parametrize("intensity", [-2, {}])
def test_init_errors(intensity):
    if isinstance(intensity, (int, float)):
        with pytest.raises(ValueError):
            PoissonProcess(intensity=intensity)
    else:
        with pytest.raises(TypeError):
            PoissonProcess(intensity=intensity)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.intensity = -0.1


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
