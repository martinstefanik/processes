"""
Testing that passing invalid parameters to the sample method of each stochastic
process raises appropriate errors. This does not include the x0 parameter since
the restrictions on it depend on the specific process.
"""

import pytest


def test_sample_invalid_n_paths(any_process, n_time_grid, n_paths_invalid):
    process, x0 = any_process
    if isinstance(n_paths_invalid, int):
        with pytest.raises(ValueError):
            process.sample(
                T=1, n_time_grid=n_time_grid, x0=x0, n_paths=n_paths_invalid
            )
    else:
        with pytest.raises(TypeError):
            process.sample(
                T=1, n_time_grid=n_time_grid, x0=x0, n_paths=n_paths_invalid
            )


def test_sample_invalid_n_time_grid(any_process, n_time_grid_invalid):
    process, x0 = any_process
    if isinstance(n_time_grid_invalid, int):
        with pytest.raises(ValueError):
            process.sample(
                T=1, n_time_grid=n_time_grid_invalid, x0=x0, n_paths=1
            )
    else:
        with pytest.raises(TypeError):
            process.sample(
                T=1, n_time_grid=n_time_grid_invalid, x0=x0, n_paths=1
            )


def test_sample_invalid_T(any_process, T_invalid, n_time_grid):
    process, x0 = any_process
    if isinstance(T_invalid, (int, float)):
        with pytest.raises(ValueError):
            process.sample(
                T=T_invalid, n_time_grid=n_time_grid, x0=x0, n_paths=1
            )
    else:
        with pytest.raises(TypeError):
            process.sample(
                T=T_invalid, n_time_grid=n_time_grid, x0=x0, n_paths=1
            )
