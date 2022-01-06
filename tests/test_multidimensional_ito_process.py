"""Tests for the MultidimensionalItoProcess class."""


import numpy as np
import pytest

from processes.processes import MultidimensionalItoProcess


@pytest.fixture(scope="module")
def process():
    mu = lambda x, t: np.array([np.abs(x[0]), x[1] - 0.1 * t])
    sigma = lambda x, t: np.array(
        [[np.tanh(np.abs(x[0] + x[1])), 0.5], [0, 0.2 * t]]
    )
    dim = 2
    return MultidimensionalItoProcess(mu=mu, sigma=sigma, dim=dim)


@pytest.mark.parametrize(
    "mu, sigma, dim",
    [
        (lambda x: np.array(x, x), lambda x, t: np.eye(2), 2),
        ("a", lambda x, t: np.eye(2), 2),
        (lambda x, t: np.array(x, x), lambda x: np.eye(2), 2),
        (lambda x, t: np.array(x, x), "a", 2),
        (lambda x, t: np.array(x, x), lambda x, t: np.eye(2), "a"),
        (lambda x, t: np.array(x, x), lambda x, t: np.eye(2), 1.2),
    ],
)
def test_init_errors(mu, sigma, dim):
    if callable(mu) and callable(sigma) and isinstance(dim, int):
        with pytest.raises(ValueError):
            MultidimensionalItoProcess(mu=mu, sigma=sigma, dim=dim)
    else:
        with pytest.raises(TypeError):
            MultidimensionalItoProcess(mu=mu, sigma=sigma, dim=dim)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.mu = lambda x: x
    with pytest.raises(ValueError):
        process.sigma = lambda x: x


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
