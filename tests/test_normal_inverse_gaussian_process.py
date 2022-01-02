"""Tests for the NormalInverseGaussianProcess class."""


import numpy as np
import pytest

from processes.processes import NormalInverseGaussianProcess


@pytest.fixture
def process():
    return NormalInverseGaussianProcess(alpha=0.2, beta=0.1, delta=2, mu=0)


@pytest.mark.parametrize(
    "alpha, beta, delta", [(-0.2, 0.1, 2), (2, 1, -0.2), (0.2, -0.7, 2)]
)
def test_init_errors(alpha, beta, delta):
    with pytest.raises(ValueError):
        NormalInverseGaussianProcess(alpha=alpha, beta=beta, delta=delta, mu=0)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.alpha = -0.1
    with pytest.raises(ValueError):
        process.alpha = 0.05
    with pytest.raises(ValueError):
        process.beta = -0.7
    with pytest.raises(ValueError):
        process.delta = -1


def test_sample(process, T, n_time_grid, n_paths):
    paths = process.sample(T=T, n_time_grid=n_time_grid, x0=1, n_paths=n_paths)
    assert isinstance(paths, np.ndarray)
    if n_paths == 1:
        assert paths.shape == (n_time_grid,)
    else:
        assert paths.shape == (n_paths, n_time_grid)
