"""Tests for the OrnsteinUhlenbeckProcess class."""


import numpy as np
import pytest

from processes.processes import OrnsteinUhlenbeckProcess


@pytest.fixture(scope="module")
def process():
    return OrnsteinUhlenbeckProcess(theta=1.2, mu=0.02, sigma=0.07)


@pytest.mark.parametrize(
    "theta, mu, sigma",
    [
        (-1.2, 0.01, 0.1),
        ("a", 0.01, 0.1),
        (1.2, "a", 0.1),
        (1.2, 0.01, -0.1),
        (1.2, 0.01, "a"),
    ],
)
def test_init_errors(theta, mu, sigma):
    if all(map(lambda p: isinstance(p, (int, float)), [theta, mu, sigma])):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeckProcess(theta=theta, mu=mu, sigma=sigma)
    else:
        with pytest.raises(TypeError):
            OrnsteinUhlenbeckProcess(theta=theta, mu=mu, sigma=sigma)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.theta = -0.1
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
