"""Tests for the InverseBesselProcess class."""


import numpy as np
import pytest

from processes.processes import InverseBesselProcess


@pytest.fixture(scope="module")
def process():
    return InverseBesselProcess(n=3.0)


@pytest.mark.parametrize("n", [-1, "a"])
def test_init_errors(n):
    if isinstance(n, (int, float)):
        with pytest.raises(ValueError):
            InverseBesselProcess(n=n)
    else:
        with pytest.raises(TypeError):
            InverseBesselProcess(n=n)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.n = -0.4


@pytest.mark.parametrize(
    "algorithm",
    [None, "radial", "alfonsi", "euler-maruyama", "milstein-sym"],
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


@pytest.mark.parametrize("x0", [-1, 0, "start"])
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


@pytest.mark.parametrize(
    "algorithm, n", [("radial", 1), ("radial", 2.5), ("alfonsi", 1.7)]
)
def test_invalid_radial(algorithm, n, n_time_grid):
    process = InverseBesselProcess(n=n)
    with pytest.raises(ValueError):
        process.sample(
            T=1,
            n_time_grid=1,
            x0=0.02,
            n_paths=n_time_grid,
            algorithm=algorithm,
        )
