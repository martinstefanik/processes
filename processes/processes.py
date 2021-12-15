#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continuous time stochastic processes."""

from abc import ABC, abstractmethod
from math import ceil
from typing import Callable, Type, Union

import numpy as np
from numpy.core.umath_tests import matrix_multiply

from .distributions import Distribution
from .utils import (
    get_time_increments,
    validate_1d_array,
    validate_callable_args,
    validate_common_sampling_parameters,
    validate_nonnegative_1d_array,
    validate_nonnegative_number,
    validate_number,
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
        self.mu = mu
        self.sigma = sigma

    @property
    def mu(self) -> float:  # noqa: D102
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        validate_number(value, "mu")
        self._mu = value

    @property
    def sigma(self) -> float:  # noqa: D102
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        validate_positive_number(value, "sigma")
        self._sigma = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Brownian motion."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_number(x0, "x0")

        dt = T / n_time_grid
        increments = np.random.normal(
            loc=self.mu * dt,
            scale=self.sigma * np.sqrt(dt),
            size=(n_paths, n_time_grid - 1),
        )
        paths = np.cumsum(increments, axis=1)
        paths = np.insert(paths, 0, 0, axis=1)
        paths = np.squeeze(paths)
        if x0 != 0:
            paths += x0

        return paths

    def _sample_at(
        self, times: np.ndarray, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Brownian motion at given times."""
        # Sanity check for input parameters
        validate_nonnegative_1d_array(times, "times")
        validate_number(x0, "x0")

        if times[0] != 0:  # we need an increment from 0 in any case
            times = np.insert(times, 0, 0)
        dt = get_time_increments(times)
        increments = np.random.normal(
            loc=self.mu * dt,
            scale=self.sigma * np.sqrt(dt),
            size=(n_paths, len(dt)),
        )
        paths = np.cumsum(increments, axis=1)
        if x0 != 0:
            paths += x0
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
        validate_number(x0, "x0")

        bm_paths = super().sample(
            T=T, n_time_grid=n_time_grid, x0=0, n_paths=n_paths
        )
        time_grid = self.time_grid(T, n_time_grid)
        paths = x0 * np.exp(
            (self.mu - 1 / 2 * self.sigma ** 2) * time_grid
            + self.sigma * bm_paths
        )

        return paths


class MultidimensionalBrownianMotion(StochasticProcess):
    """Multi-dimensional Brownian motion."""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        self.mu = mu
        self.sigma = sigma

    @property
    def mu(self) -> np.ndarray:  # noqa: D102
        return self._mu

    @mu.setter
    def mu(self, value: np.ndarray) -> None:
        if hasattr(self, "_sigma"):
            self._check_parameter_compatibility(value, self._sigma)
        self._mu = value

    @property
    def sigma(self) -> np.ndarray:  # noqa: D102
        return self._sigma

    @sigma.setter
    def sigma(self, value: np.ndarray) -> None:
        validate_posdef_matrix(value, "sigma")
        if hasattr(self, "_mu"):
            self._check_parameter_compatibility(self._mu, value)
        self._sigma = value

    @staticmethod
    def _check_parameter_compatibility(
        mu: np.ndarray, sigma: np.ndarray
    ) -> None:
        """Check the compatibility of the input parameters."""
        if len(mu) != sigma.shape[0]:
            raise ValueError("Incompatible dimensions of 'mu' and 'sigma'.")

    def sample(
        self,
        x0: Union[np.ndarray, float],
        T: float,
        n_time_grid: int,
        n_paths: int = 1,
    ) -> np.ndarray:
        """Generate sample paths of the multi-dimensional Brownian motion."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        if isinstance(x0, float):
            x0 = np.array([x0] * len(self.mu))
        else:
            validate_1d_array(x0, "x0")
            if len(x0) != len(self.mu):
                raise ValueError(f"'x0' of unexpected length: {len(x0)}.")

        dt = T / n_time_grid
        increments = np.random.multivariate_normal(
            mean=self.mu * dt,
            cov=self.sigma * dt,
            size=(n_paths, n_time_grid - 1),
        )
        paths = np.cumsum(increments, axis=1)
        paths = np.insert(paths, 0, 0, axis=1)
        if not np.all(x0 == 0):
            paths += np.reshape(x0, (1, 1, len(x0)))
        paths = np.squeeze(paths)

        return paths


class TimeChangedBrownianMotion(StochasticProcess):
    """Time-changed Brownian motion."""

    def __init__(self, time_change: Callable) -> None:
        self.time_change = time_change

    @property
    def time_change(self) -> Callable:  # noqa: D102
        return self._time_change

    @time_change.setter
    def time_change(self, value: Callable) -> None:
        validate_callable_args(value, 1, "time_change")
        self._time_change = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the time-changed Brownian motion."""
        # Sanity check for input parameters
        validate_common_sampling_parameters()
        validate_number(x0, "x0")

        times = self.time_change(np.linspace(0, T, num=n_time_grid))
        paths = BrownianMotion()._sample_at(times, 0, n_paths)
        if x0 != 0:
            paths += x0

        return paths


class OrnsteinUhlenbeckProcess(StochasticProcess):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, theta: float, mu: float, sigma: float) -> None:
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    @property
    def theta(self) -> float:  # noqa: D102
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        validate_positive_number(value, "theta")
        self._theta = value

    @property
    def mu(self) -> float:  # noqa: D102
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        validate_number(value, "mu")
        self._mu = value

    @property
    def sigma(self) -> float:  # noqa: D102
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        validate_positive_number(value, "sigma")
        self._sigma = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Ornstein-Uhlenbeck process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters()
        validate_number(x0, "x0")

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


class ItoProcess(StochasticProcess):
    """Ito process."""

    def __init__(
        self, mu: Callable, sigma: Callable, positive: bool = False
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.positive = positive

    @property
    def mu(self) -> Callable:  # noqa: D102
        return self._mu

    @mu.setter
    def mu(self, value: Callable) -> None:
        validate_callable_args(value, 2, "mu")
        self._mu = value

    @property
    def sigma(self) -> Callable:  # noqa: D102
        return self._sigma

    @sigma.setter
    def sigma(self, value: Callable) -> None:
        validate_callable_args(value, 2, "sigma")
        self._sigma = value

    @property
    def positive(self) -> bool:  # noqa: D102
        return self._positive

    @positive.setter
    def positive(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("'positive' must be a boolean.")
        self._positive = value

    def sample(
        self, T: float, n_time_grid: int, x0: float, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Ito process."""
        # TODO: Add a generic scheme to ensure positivity if positive=True.
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_number(x0, "x0")

        # Run the Euler-Maruyama scheme
        dt = T / n_time_grid
        paths = np.zeros(shape=(n_paths, n_time_grid))
        paths[:, 0] = x0
        noise_scale = np.sqrt(dt)
        for i in range(1, n_time_grid):
            t = (i - 1) * dt
            drift_coeffs = np.array(
                list(map(lambda x: self.mu(x, t), paths[:, i - 1]))
            )  # t is the same for all paths in the current iteration
            diffusion_coeffs = np.array(
                list(map(lambda x: self.sigma(x, t), paths[:, i - 1]))
            )  # t is the same for all paths in the current iteration
            noise = np.random.normal(loc=0, scale=noise_scale, size=n_paths)
            paths[:, i] = (
                paths[:, i - 1] + drift_coeffs * dt + diffusion_coeffs * noise
            )
        paths = np.squeeze(paths)
        return paths


class MultidimensionalItoProcess(ItoProcess):
    """Multi-dimensional Ito process."""

    def sample(
        self, T: float, n_time_grid: int, x0: float, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the multi-dimensional Ito process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        if isinstance(x0, float):
            x0 = np.array([x0] * len(self.mu))
        else:
            validate_1d_array(x0, "x0")
            if len(x0) != len(self.mu):
                raise ValueError(f"'x0' of unexpected length: {len(x0)}.")

        # Run the Euler-Maruyama scheme
        dt = T / n_time_grid
        paths = np.zeros(shape=(n_paths, n_time_grid, self.d))
        paths[:, 0, :] = x0
        noise_cov = dt * np.eye(self.d)

        # TODO: Check whether this cannot be improved both in terms of speed as
        #       well as in terms of readability.
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
        self.intensity = intensity

    @property
    def intensity(self) -> float:  # noqa: D102
        return self._intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        validate_positive_number(value, "intensity")
        self._intensity = value

    def sample(
        self, T: float, n_time_grid: int, x0: int = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Poisson process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_number(x0, "x0")

        dt = T / n_time_grid
        increments = np.random.poisson(
            lam=self.intensity * dt, size=(n_paths, n_time_grid - 1)
        )
        paths = np.cumsum(increments, axis=1)
        paths = np.insert(paths, 0, 0, axis=1)
        paths = np.squeeze(paths)
        if x0 != 0:
            paths += x0

        return paths


class CompoundPoissonProcess(PoissonProcess):
    """Compound Poisson process."""

    def __init__(
        self, intensity: float, increment_dist: Type[Distribution]
    ) -> None:
        self.intensity = intensity
        self.increment_dist = increment_dist

    @property
    def increment_dist(self) -> Type[Distribution]:  # noqa: D102
        return self._increment_dist

    @increment_dist.setter
    def increment_dist(self, value: Type[Distribution]) -> None:
        if not isinstance(value, Distribution):
            raise ValueError("'increment_dist' must be of type Distribution.")
        self._increment_dist = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the compound Poisson process."""
        pp_paths = super().sample(T, n_time_grid, 0, n_paths)
        paths = []
        for i in range(n_paths):
            terminal = pp_paths[i, -1]
            increments = self.increment_dist.sample(size=terminal)
            path = np.cumsum(increments)
            path = np.insert(path, 0, 0)[pp_paths[i]]
            paths.append(path)
        paths = np.vstack(paths)
        if x0 != 0:
            paths += x0

        return paths


class CoxIngersollRossProcess(StochasticProcess):
    """Cox-Ingersoll-Ross process."""

    def __init__(self, theta: float, mu: float, sigma: float) -> None:
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    @property
    def theta(self) -> float:  # noqa: D102
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        validate_positive_number(value, "theta")
        if hasattr(self, "_mu") and hasattr(self, "_sigma"):
            self._check_feller_condition(value, self._mu, self._sigma)
        self._theta = value

    @property
    def mu(self) -> float:  # noqa: D102
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        validate_positive_number(value, "mu")
        if hasattr(self, "_theta") and hasattr(self, "_sigma"):
            self._check_feller_condition(self._theta, value, self._sigma)
        self._mu = value

    @property
    def sigma(self) -> float:  # noqa: D102
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        validate_positive_number(value, "sigma")
        if hasattr(self, "_theta") and hasattr(self, "_mu"):
            self._check_feller_condition(self._theta, self._mu, value)
        self._sigma = value

    @staticmethod
    def _check_feller_condition(theta: float, mu: float, sigma: float) -> None:
        """Check that the Feller condition holds."""
        if not 2 * theta * mu >= sigma ** 2:
            raise ValueError(
                "The Feller condition for the parameters "
                "'theta', 'mu' and 'sigma' does not hold."
            )

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Cox-Ingersoll-Ross process."""
        # TODO: Use a scheme that ensures positivity.
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_positive_number(x0, "x0")

        cir = ItoProcess(
            lambda x, t: self.theta * (self.mu - x),
            lambda x, t: self.sigma * np.sqrt(np.abs(x)),
        )
        paths = cir.sample(T, n_time_grid, x0, n_paths)

        return paths


class BesselProcess(StochasticProcess):
    """Bessel process."""

    def __init__(self, n: float) -> None:
        self.n = n

    @property
    def n(self) -> float:  # noqa: D102
        return self._n

    @n.setter
    def n(self, value: float) -> None:
        validate_nonnegative_number(value, "n")
        self._n = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Bessel process."""
        # TODO: Use a scheme that ensures positivity.
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_nonnegative_number(x0, "x0")

        # Sample paths of the squared Bessel process for numerical stability
        squared_bessel = ItoProcess(
            lambda x, t: self.n,
            lambda x, t: 2 * np.sqrt(np.abs(x)),
        )
        paths = squared_bessel.sample(T, n_time_grid, x0, n_paths)

        return np.sqrt(paths)


class InverseBesselProcess(BesselProcess):
    """Inverse Bessel process."""

    def sample(
        self, T: float, n_time_grid: int, x0: float = 1, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the inverse Bessel process."""
        paths = super().sample(T, n_time_grid, x0, n_paths)
        paths = 1 / paths
        return paths


class FractionalBrownianMotion(StochasticProcess):
    """Fractional Brownian motion."""

    def __init__(self, hurst: float) -> None:
        self.hurst = hurst

    @property
    def hurst(self) -> float:  # noqa: D102
        return self._hurst

    @hurst.setter
    def hurst(self, value: float) -> None:
        validate_number(value, "hurst")
        if not value > 0 and not value < 1:
            raise ValueError("'hurst' must be in (0, 1).")
        self._hurst = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the fractional Brownian motion."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_number(x0, "x0")

        dt = T / n_time_grid
        size = 2 ** ceil(np.log2(n_time_grid - 2)) + 1
        sqrt_eigenvalues = np.sqrt(
            np.fft.irfft(self._acf_fractional_gaussian_noise(self.hurst, size))[
                :size
            ]
        )
        scale = dt ** (2 * self.hurst) * 2 ** (1 / 2) * (size - 1)

        z = np.random.normal(scale=scale, size=2 * size).view(complex)
        z[0] = z[0].real * 2 ** (1 / 2)
        z[-1] = z[-1].real * 2 ** (1 / 2)
        increments = np.fft.irfft(sqrt_eigenvalues * z)[:n_time_grid]
        paths = np.cumsum(increments)
        paths = np.squeeze(paths)

        return paths

    @staticmethod
    def _acf_fractional_gaussian_noise(hurst: float, n: float):
        """Autocovariance function of fractional Gaussian noise."""
        rho = np.arange(n + 1) ** (2 * hurst)
        rho = 1 / 2 * (rho[2:] - 2 * rho[1:-1] + rho[:-2])
        rho = np.insert(rho, 0, 1)
        return rho
