"""Tests for the CoxIngersollRossProcess class."""


import numpy as np
import pytest

from processes.processes import CoxIngersollRossProcess


@pytest.fixture(scope="module")
def process():
    return CoxIngersollRossProcess(theta=1.2, mu=0.01, sigma=0.04)


@pytest.mark.parametrize(
    "theta, mu, sigma",
    [
        (-1.2, 0.01, 0.04),
        (1.2, -0.01, 0.04),
        (1.2, 0.01, -0.04),
        (0.1, 0.01, 0.7),  # Feller condition does not hold here
        ({}, 0.01, 0.04),
        (1.2, [], 0.04),
        (1.2, 0.01, []),
    ],
)
def test_init_errors(theta, mu, sigma):
    if all(map(lambda p: isinstance(p, (int, float)), [theta, mu, sigma])):
        with pytest.raises(ValueError):
            CoxIngersollRossProcess(theta=theta, mu=mu, sigma=sigma)
    else:
        with pytest.raises(TypeError):
            CoxIngersollRossProcess(theta=theta, mu=mu, sigma=sigma)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.theta = -0.4
    with pytest.raises(ValueError):  # violate the Feller condition
        process.theta = 0.00001
    with pytest.raises(ValueError):
        process.mu = -0.1
    with pytest.raises(ValueError):
        process.sigma = -0.1


@pytest.mark.parametrize(
    "algorithm",
    ["conditional", "alfonsi", "euler-maruyama", "milstein-sym"],
)
def test_sample(process, algorithm, T, n_time_grid, n_paths):
    x0 = 0.02
    paths = process.sample(
        T=T,
        n_time_grid=n_time_grid,
        x0=x0,
        n_paths=n_paths,
        algorithm=algorithm,
    )
    assert isinstance(paths, np.ndarray)
    if n_paths == 1:
        assert paths.shape == (n_time_grid,)
        assert paths[0] == x0
    else:
        assert paths.shape == (n_paths, n_time_grid)
        assert np.all(paths[:, 0] == x0)


@pytest.mark.parametrize("x0", [-1, "start"])
def test_invalid_x0(process, n_time_grid, x0):
    if isinstance(x0, (float, int)):
        with pytest.raises(ValueError):
            process.sample(T=1, n_time_grid=n_time_grid, x0=x0, n_paths=1)
    else:
        with pytest.raises(TypeError):
            process.sample(T=1, n_time_grid=n_time_grid, x0=x0, n_paths=1)


@pytest.mark.parametrize("algorithm", [-1, "start"])
def test_invalid_algorithm(process, algorithm, n_time_grid):
    if isinstance(algorithm, str):
        with pytest.raises(ValueError):
            process.sample(
                T=1,
                n_time_grid=n_time_grid,
                x0=0.02,
                n_paths=1,
                algorithm=algorithm,
            )
    else:
        with pytest.raises(TypeError):
            process.sample(
                T=1,
                n_time_grid=n_time_grid,
                x0=0.02,
                n_paths=1,
                algorithm=algorithm,
            )
