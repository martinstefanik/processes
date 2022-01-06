"""Tests for the FractionalBrownianMotion class."""


import numpy as np
import pytest

from processes.processes import FractionalBrownianMotion


@pytest.fixture(scope="module")
def process():
    return FractionalBrownianMotion(hurst=0.15)


@pytest.mark.parametrize("hurst", [2, -2, "2", 1.2])
def test_init_errors(hurst):
    if isinstance(hurst, (int, float)):
        with pytest.raises(ValueError):
            FractionalBrownianMotion(hurst=hurst)
    else:
        with pytest.raises(TypeError):
            FractionalBrownianMotion(hurst=hurst)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.hurst = -0.1


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
