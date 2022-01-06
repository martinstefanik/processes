"""Tests for the BrownianBridge class."""


import numpy as np
import pytest

from processes.processes import BrownianBridge


@pytest.fixture(scope="module")
def process():
    return BrownianBridge(a=1, b=3.5, T=2)


@pytest.mark.parametrize(
    "a, b, T", [([], 2, 1), (1, {}, 1), (1, 2, -3), (1, 2, {})]
)
def test_init_errors(a, b, T):
    if all(map(lambda p: isinstance(p, (int, float)), [a, b, T])):
        with pytest.raises(ValueError):
            BrownianBridge(a=a, b=b, T=T)
    else:
        with pytest.raises(TypeError):
            BrownianBridge(a=a, b=b, T=T)


def test_post_init_modification(process):
    with pytest.raises(ValueError):
        process.T = -1.2


def test_sample(process, T, n_time_grid, n_paths):
    paths = process.sample(T=T, n_time_grid=n_time_grid, n_paths=n_paths)
    assert isinstance(paths, np.ndarray)
    if n_paths == 1:
        assert paths.shape == (n_time_grid,)
        assert paths[0] == process.a
    else:
        assert paths.shape == (n_paths, n_time_grid)
        assert np.all(paths[:, 0] == process.a)
