#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continuous time stochastic processes."""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.core.umath_tests import matrix_multiply

from .utils import (
    validate_callable_args,
    validate_common_sampling_parameters,
    validate_posdef_matrix,
    validate_positive_integer,
    validate_positive_number,
)


class StochasticProcess(ABC):
    """Abstract base class for a stochastic processes."""

    @abstractmethod
    def sample(
        self, T: float, n_time_grid: int, x0: float, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the stochastic process."""
        raise NotImplementedError("'sample' method is not implemented.")

    @classmethod
    def time_grid(cls, T: float, n_time_grid: int) -> np.ndarray:
        """Generate a time grid for sample paths."""
        # Sanity check for input parameters
        validate_positive_number(T, "T")
        validate_positive_integer(n_time_grid, "n_time_grid")
        return np.linspace(0, T, num=n_time_grid)


class BrownianMotion(StochasticProcess):
    """Brownian motion."""

    def __init__(self, mu: float = 0, sigma: float = 1) -> None:
        """Initialize a Brownian motion."""
        self.mu = mu
        self.sigma = sigma

    @property
    def sigma(self) -> float:
        """Sigma getter."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        """Sigma setter."""
        validate_positive_number(value, "sigma")
        self._sigma = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: float = 1
    ) -> np.ndarray:
        """Generate sample paths of the Brownian motion."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)

        dt = T / n_time_grid
        increments = np.random.normal(
            loc=self.mu * dt,
            scale=self.sigma * np.sqrt(dt),
            size=(n_paths, n_time_grid - 1),
        )
        paths = np.cumsum(increments, axis=1)
        paths = np.insert(paths, 0, x0, axis=1)
        paths = np.squeeze(paths)

        return paths


class GeometricBrownianMotion(BrownianMotion):
    """Geometric Brownian motion."""

    def sample(
        self, T: float, n_time_grid: int, x0: float = 1, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the geometric Brownian motion."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)

        bm_paths = super().sample(
            T=T, n_time_grid=n_time_grid, x0=0, n_paths=n_paths
        )
        time_grid = self.time_grid(T, n_time_grid)
        paths = x0 * np.exp(
            (self.mu - 1 / 2 * self.sigma ** 2) * time_grid
            + self.sigma * bm_paths
        )

        return paths


class MultiDimensionalBrownianMotion(StochasticProcess):
    """Multi-dimensional Brownian motion."""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        """Initialize a multi-dimensional Brownian motion."""
        self.mu = mu
        self.sigma = sigma

    @property
    def mu(self) -> np.ndarray:
        """Mu getter."""
        return self._mu

    @mu.setter
    def mu(self, value: np.ndarray) -> None:
        """Mu setter."""
        if hasattr(self, "_sigma"):
            self._check_parameter_compatibility(value, self._sigma)
        self._mu = value

    @property
    def sigma(self) -> np.ndarray:
        """Sigma getter."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: np.ndarray) -> None:
        """Sigma setter."""
        validate_posdef_matrix(value, "sigma")
        if hasattr(self, "_mu"):
            self._check_parameter_compatibility(self._mu, value)
        self._sigma = value

    @staticmethod
    def _check_parameter_compatibility(mu, sigma) -> None:
        """Check the compatibility of the input parameters."""
        if len(mu) != sigma.shape[0]:
            raise ValueError("Incompatible dimensions of 'mu' and 'sigma'.")

    def sample(
        self, x0: np.ndarray, T: float, n_time_grid: int, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the multi-dimensional Brownian motion."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)

        dt = T / n_time_grid
        std_increments = np.random.normal(
            size=(n_paths, n_time_grid - 1, len(self.mu))
        )
        increments = np.sqrt(dt) * np.tensordot(
            std_increments, self.sigma.T, axes=1
        )
        paths = np.cumsum(increments, axis=1)
        paths = np.insert(paths, 0, x0, axis=1)
        paths = np.squeeze(paths)

        return paths


class TimeChangedBrownianMotion(StochasticProcess):
    """Time-changed Brownian motion."""

    def __init__(self, time_change: Callable) -> None:
        """Initialize a time-changed Brownian motion."""
        self.time_change = time_change

    @property
    def time_change(self) -> Callable:
        """Time change getter."""
        return self._time_change

    @time_change.setter
    def time_change(self, value: Callable) -> None:
        """Time change setter."""
        validate_callable_args(value, 1, "time_change")
        self._time_change = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: float = 1
    ) -> np.ndarray:
        """Generate sample paths of the time-changed Brownian motion."""
        # Sanity check for input parameters
        validate_common_sampling_parameters()

        tc_grid = self.time_change(np.linspace(0, T, num=n_time_grid))
        tc_dt = np.diff(tc_grid)
        std_increments = np.random.normal(size=(n_paths, n_time_grid - 1))
        increments = np.sqrt(tc_dt) * std_increments
        paths = np.cumsum(increments, axis=1)
        paths = np.insert(paths, 0, x0, axis=1)
        paths = np.squeeze(paths)

        return paths


class OrnsteinUhlenbeckProcess(StochasticProcess):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, theta: float, mu: float, sigma: float) -> None:
        """Initialize an Ornstein-Uhlenbeck process."""
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    @property
    def theta(self) -> float:
        """Theta getter."""
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        """Theta setter."""
        validate_positive_number(value, "theta")
        self._theta = value

    @property
    def sigma(self) -> float:
        """Sigma getter."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        """Sigma setter."""
        validate_positive_number(value, "sigma")
        self._sigma = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Ornstein-Uhlenbeck process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters()

        # Generate the random part of the process
        time_changed_bm = TimeChangedBrownianMotion(
            time_change=lambda t: np.exp(2 * self.theta * t) - 1
        )
        time_grid = self.time_grid(T, n_time_grid)
        scale = (
            self.sigma
            / np.sqrt(2 * self.theta)
            * np.exp(-self.theta * time_grid)
        )
        bm_part = scale * time_changed_bm.sample(
            x0=0, T=T, n_time_grid=n_time_grid, n_paths=n_paths
        )

        # Generate the drift / deterministic part of the process
        drift_part = x0 * np.exp(-self.theta * time_grid) + self.mu * (
            1 - np.exp(-self.theta * time_grid)
        )

        # Put together the drift and Brownian motion parts to form paths
        paths = drift_part + bm_part

        return paths


class MultiDimensionalItoProcess(StochasticProcess):
    """Ito process."""

    def __init__(self, mu: Callable, sigma: Callable) -> None:
        """Initialize an Ito process."""
        self.mu = mu
        self.sigma = sigma

    @property
    def mu(self) -> Callable:
        """Mu getter."""
        return self._mu

    @mu.setter
    def mu(self, value: Callable) -> None:
        """Mu setter."""
        validate_callable_args(value, 2, "mu")
        self._mu = value

    @property
    def sigma(self) -> Callable:
        """Sigma getter."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: Callable) -> None:
        """Sigma setter."""
        validate_callable_args(value, 2, "sigma")
        self._sigma = value

    def sample(
        self, T: float, n_time_grid: int, x0: float, n_paths: int = 1
    ) -> np.ndarray:
        """
        Generate sample paths of the Ito process. Sampling is done using the
        Euler-Maruyama scheme.
        """
        # Sanity check for input parameters
        validate_common_sampling_parameters()

        # Run the Euler-Maruyama scheme
        dt = T / n_time_grid
        paths = np.zeros(shape=(n_paths, n_time_grid, self.d))
        paths[:, 0, :] = x0
        noise_cov = dt * np.eye(self.d)
        for i in range(1, n_time_grid):
            t = (i - 1) * dt
            drift_coeffs = np.apply_along_axis(
                lambda x: self.mu(x, t), axis=1, arr=paths[:, i - 1, :]
            )  # t is the same for all paths in the current iteration
            diffusion_coeffs = np.apply_along_axis(
                lambda x: self.sigma(x, t), axis=1, arr=paths[:, i - 1, :]
            )  # t is the same for all paths in the current iteration
            noise = np.random.multivariate_normal(
                mean=np.zeros(self.d), cov=noise_cov, size=n_paths
            )
            paths[:, i, :] = (
                paths[:, i - 1, :]
                + drift_coeffs * dt
                + matrix_multiply(
                    diffusion_coeffs, np.expand_dims(noise, 2)
                ).squeeze()
            )
        paths = np.squeeze(paths)

        return paths


class PoissonProcess(StochasticProcess):
    """Poisson process."""

    def __init__(self, intensity: float) -> None:
        """Initialize a Poisson process."""
        self.intensity = intensity

    @property
    def intensity(self) -> float:
        """Intensity getter."""
        return self._intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        """Intensity setter."""
        validate_positive_number(value, "intensity")
        self._intensity = value

    def sample(
        self, T: float, n_time_grid: int, x0: int = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Poisson process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        if not isinstance(x0, int):
            raise ValueError("'x0' must be an integer.")

        dt = T / n_time_grid
        increments = np.random.poisson(
            lam=self.intensity * dt, size=(n_paths, n_time_grid)
        )
        paths = np.cumsum(increments, axis=1)
        paths = np.insert(paths, 0, x0, axis=1)
        paths = np.squeeze(paths)

        return paths
