"""Tests for the MultidimensionalBrownianMotion class."""


import numpy as np
import pytest

from processes.processes import MultidimensionalBrownianMotion


@pytest.fixture(scope="module")
def process():
    mu = np.array([1, 0.7])
    sigma = np.array([[1, 0.5], [0.5, 1]])
    return MultidimensionalBrownianMotion(mu=mu, sigma=sigma)


@pytest.mark.parametrize(
    "mu, sigma",
    [
        ("a", np.eye(2)),
        (np.eye(2), np.eye(2)),
        (1, np.eye(2)),
        (np.ones(3), np.eye(5)),  # incompatible parameters
        (np.ones(2), {}),
        (np.ones(2), np.ones(shape=(2, 2, 2))),
        (np.ones(2), np.ones(2)),
        (np.ones(2), -np.ones(2)),  # sigma is not positive definite here
        (np.ones(2), -np.ones((2, 3))),  # sigma is not square here
    ],
)
def test_init_errors(mu, sigma):
    if isinstance(mu, np.ndarray) and isinstance(sigma, np.ndarray):
        with pytest.raises(ValueError):
            MultidimensionalBrownianMotion(mu=mu, sigma=sigma)
    else:
        with pytest.raises(TypeError):
            MultidimensionalBrownianMotion(mu=mu, sigma=sigma)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.sigma = -np.eye(2)  # sigma is not positive definite here
    with pytest.raises(ValueError):
        process.mu = np.ones(3)  # incompatible size


@pytest.mark.parametrize("x0", [2, np.array([0.2, 3])])
def test_sample(process, T, n_time_grid, x0, n_paths):
    paths = process.sample(T=T, n_time_grid=n_time_grid, x0=x0, n_paths=n_paths)
    assert isinstance(paths, np.ndarray)
    if n_paths == 1:
        assert paths.shape == (n_time_grid, process.dim)
        assert np.all(paths[0] == x0)
    else:
        assert paths.shape == (n_paths, n_time_grid, process.dim)
        assert np.all(paths[:, 0] == x0)


@pytest.mark.parametrize(
    "x0", ["start", np.ones(3), np.ones((2, 2, 2)), np.array([2])]
)
def test_invalid_x0(process, n_time_grid, x0):
    if isinstance(x0, np.ndarray):
        with pytest.raises(ValueError):
            process.sample(T=1, n_time_grid=n_time_grid, x0=x0, n_paths=1)
    else:
        with pytest.raises(TypeError):
            process.sample(T=1, n_time_grid=n_time_grid, x0=x0, n_paths=1)
