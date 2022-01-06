"""Tests for the WishartProcess class."""


import numpy as np
import pytest

from processes.processes import WishartProcess


@pytest.fixture(scope="module")
def process():
    Q = np.array([[2, 0.1], [0.5, 1]])
    K = np.array([[-1, 0.4], [-0.7, 1]])
    alpha = 0.3
    return WishartProcess(Q=Q, K=K, alpha=alpha)


@pytest.mark.parametrize(
    "Q, K, alpha",
    [
        ("a", np.eye(2), 0.3),
        (np.eye(2), {}, 0.3),
        (np.eye(2), np.eye(2), "a"),
        (np.eye(2), np.eye(3), 0.3),
    ],
)
def test_init_errors(Q, K, alpha):
    if (
        isinstance(Q, np.ndarray)
        and isinstance(K, np.ndarray)
        and isinstance(alpha, (float, int))
    ):
        with pytest.raises(ValueError):
            WishartProcess(Q=Q, K=K, alpha=alpha)
    else:
        with pytest.raises(TypeError):
            WishartProcess(Q=Q, K=K, alpha=alpha)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.Q = np.array([0, 1])
    with pytest.raises(ValueError):
        process.Q = np.eye(5)
    with pytest.raises(ValueError):
        process.K = np.array([0, 1])
    with pytest.raises(ValueError):
        process.alpha = -0.1


def test_sample(process, T, n_time_grid, n_paths):
    x0 = 0.1 * np.eye(2)
    dim = np.shape(x0)[0]
    paths = process.sample(T=T, n_time_grid=n_time_grid, x0=x0, n_paths=n_paths)
    assert isinstance(paths, np.ndarray)
    if n_paths == 1:
        assert paths.shape == (n_time_grid, dim, dim)
        assert np.all(paths[0] == x0)
    else:
        assert paths.shape == (n_paths, n_time_grid, dim, dim)
        assert np.all(paths[:, 0] == x0)


@pytest.mark.parametrize("x0", [-1, "start", np.ones(3), np.eye(3), -np.eye(2)])
def test_invalid_x0(process, n_time_grid, x0):
    if isinstance(x0, np.ndarray):
        with pytest.raises(ValueError):
            process.sample(T=1, n_time_grid=n_time_grid, x0=x0, n_paths=1)
    else:
        with pytest.raises(TypeError):
            process.sample(T=1, n_time_grid=n_time_grid, x0=x0, n_paths=1)
