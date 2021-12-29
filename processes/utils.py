#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities."""

from inspect import signature
from numbers import Real
from typing import Any

import numpy as np


def validate_number(value: Any, name: str) -> None:
    """Check that a given object is a real number."""
    if not isinstance(value, Real):
        raise TypeError(f"'{name}' must be a number.")


def validate_positive_number(value: Any, name: str) -> None:
    """Check that a given number is positive."""
    validate_number(value, name)
    if not value > 0:
        raise ValueError(f"'{name}' must be positive.")


def validate_nonnegative_number(value: Any, name: str) -> None:
    """Check that a given number is nonnegative."""
    validate_number(value, name)
    if not value >= 0:
        raise ValueError(f"'{name}' must be nonnegative.")


def validate_integer(value: Any, name: str) -> None:
    """Check that a given object is an integer."""
    if not isinstance(value, int):
        raise TypeError(f"'{name}' must be an integer.")


def validate_positive_integer(value: Any, name: str) -> None:
    """Check that a given number is a positive integer."""
    if not isinstance(value, int):
        raise TypeError(f"'{name}' must be an integer.")
    if not value > 0:
        raise ValueError(f"'{name}' must be positive.")


def validate_matrix(value: Any, name: str) -> None:
    """Check that a given matrix is a numpy array."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy array.")


def validate_square_matrix(value: Any, name: str) -> None:
    """Check that a given matrix is square."""
    validate_matrix(value, name)
    if value.shape[0] != value.shape[1]:
        raise ValueError(f"'{name}' must be square. Shape: {value.shape}")


def validate_possemdef_matrix(value: Any, name: str) -> None:
    """Check that a given matrix is positive semi-definite definite."""
    validate_square_matrix(value, name)
    if not np.all(np.linalg.eigvals(value) >= 0):
        raise ValueError(f"'{name}' must be positive semi-definite.")


def validate_callable_args(value: Any, n_args: int, name: str) -> None:
    """Check that an object is a callable taking a given number of arguments."""
    if not callable(value):
        raise TypeError(f"'{name}' must be callable.")
    if len(signature(value).parameters) != n_args:
        raise ValueError(f"'{name}' must take {n_args} arguments.")


def validate_common_sampling_parameters(
    T: Any, n_time_grid: Any, n_paths: Any
) -> None:
    """Validate parameters for sampling paths of a stochastic process."""
    validate_positive_number(T, "T")
    validate_positive_integer(n_time_grid, "n_time_grid")
    validate_positive_integer(n_paths, "n_paths")


def validate_1d_array(array: Any, name: str) -> None:
    """Check that a given object is a one-dimensional numpy array."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy array.")
    if not np.squeeze(array).shape != (array.shape,):
        raise ValueError(f"'{name}' must be one-dimensional.")


def validate_nonnegative_1d_array(array: Any, name: str) -> None:
    """Check that a given object is a non-negative, 1D numpy array."""
    validate_1d_array(array, name)
    if not np.all(array >= 0):
        raise ValueError(f"'{name}' must be non-negative.")


def get_time_increments(time_grid: np.ndarray) -> np.ndarray:
    """Get time increments from a given time grid."""
    dt = np.diff(time_grid)
    if not np.all(dt > 0):
        raise ValueError("'time_grid' must be strictly increasing.")
    return dt
