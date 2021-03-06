"""Classes for stochastic processes."""

from abc import ABC, abstractmethod
from functools import lru_cache
from math import ceil
from typing import Callable, Literal, Optional, Union

import numpy as np
from scipy.stats import norminvgauss

from processes.utils import (
    get_time_increments,
    validate_1d_array,
    validate_callable_args,
    validate_common_sampling_parameters,
    validate_integer,
    validate_nonnegative_1d_array,
    validate_nonnegative_number,
    validate_number,
    validate_positive_integer,
    validate_positive_number,
    validate_possemdef_matrix,
    validate_square_matrix,
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


class BaseLevyProcess(StochasticProcess):
    """
    Base class for Levy processes that can all be sampled in the same way using
    the given the distribution of their increments.
    """

    @staticmethod
    def _sample_from_increment_dist(
        increments_sampler: Callable[[Union[tuple, list, float]], np.ndarray],
        n_time_grid: int,
        x0: float,
        n_paths: int,
    ):
        """
        Generate sample paths of a Levy process from the distribution of its
        increments.
        """
        increments = increments_sampler((n_paths, n_time_grid - 1))
        paths = np.cumsum(increments, axis=1)
        paths = np.insert(paths, 0, 0, axis=1)
        paths = np.squeeze(paths)
        if x0 != 0:
            paths += x0

        return paths


class BrownianMotion(BaseLevyProcess):
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
        sampler = lambda size: np.random.normal(
            loc=self.mu * dt, scale=self.sigma * np.sqrt(dt), size=size
        )
        paths = self._sample_from_increment_dist(
            sampler, n_time_grid, x0, n_paths
        )

        return paths

    def _sample_at(
        self, times: np.ndarray, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Brownian motion at given times."""
        # Sanity check for input parameters
        validate_nonnegative_1d_array(times, "times")
        validate_number(x0, "x0")

        if times[0] != 0:
            times = np.insert(times, 0, 0)
        dt = get_time_increments(times)
        increments = np.random.normal(
            loc=self.mu * dt,
            scale=self.sigma * np.sqrt(dt),
            size=(n_paths, len(dt)),
        )
        paths = np.cumsum(increments, axis=1)
        if times[0] == 0:
            paths = np.insert(paths, 0, 0, axis=1)
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
        validate_positive_number(x0, "x0")

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
        validate_1d_array(value, "mu")
        if hasattr(self, "_sigma"):
            self._check_parameter_compatibility(value, self._sigma)
        self._mu = value

    @property
    def sigma(self) -> np.ndarray:  # noqa: D102
        return self._sigma

    @sigma.setter
    def sigma(self, value: np.ndarray) -> None:
        validate_possemdef_matrix(value, "sigma")
        if hasattr(self, "_mu"):
            self._check_parameter_compatibility(self._mu, value)
        self._sigma = value

    @property
    def dim(self) -> int:  # noqa: D102
        return len(self.mu)

    @staticmethod
    def _check_parameter_compatibility(
        mu: np.ndarray, sigma: np.ndarray
    ) -> None:
        """Check the compatibility of the input parameters."""
        if len(mu) != sigma.shape[0]:
            raise ValueError("Incompatible dimensions of 'mu' and 'sigma'.")

    def sample(
        self,
        T: float,
        n_time_grid: int,
        x0: Union[np.ndarray, float],
        n_paths: int = 1,
    ) -> np.ndarray:
        """Generate sample paths of the multi-dimensional Brownian motion."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        if isinstance(x0, (float, int)):
            x0 = np.array([x0] * self.dim)
        else:
            validate_1d_array(x0, "x0")
            if len(x0) != self.dim:
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
            paths += np.reshape(x0, (1, 1, self.dim))
        paths = np.squeeze(paths)

        return paths


class TimeChangedBrownianMotion(StochasticProcess):
    """Time-changed Brownian motion."""

    def __init__(self, time_change: Callable[[np.ndarray], np.ndarray]) -> None:
        self.time_change = time_change

    @property
    def time_change(self) -> Callable[[np.ndarray], np.ndarray]:  # noqa: D102
        return self._time_change

    @time_change.setter
    def time_change(self, value: Callable[[np.ndarray], np.ndarray]) -> None:
        validate_callable_args(value, 1, "time_change")
        self._time_change = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the time-changed Brownian motion."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_number(x0, "x0")

        times = self.time_change(np.linspace(0, T, num=n_time_grid))
        paths = BrownianMotion()._sample_at(times=times, x0=0, n_paths=n_paths)
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
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
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


class VasicekProcess(OrnsteinUhlenbeckProcess):
    """
    Vasicek process. This is just an alternative name for the Ornstein-Uhlenbeck
    process and their implementations are the same.
    """

    pass


class ItoProcess(StochasticProcess):
    """Ito process."""

    def __init__(self, mu: Callable, sigma: Callable) -> None:
        self.mu = mu
        self.sigma = sigma

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

    def sample(
        self, T: float, n_time_grid: int, x0: float, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Ito process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_number(x0, "x0")

        # Run the Euler-Maruyama scheme
        dt = T / n_time_grid
        paths = np.zeros(shape=(n_paths, n_time_grid))
        paths[:, 0] = x0
        dW_scale = np.sqrt(dt)
        for i in range(1, n_time_grid):
            t = (i - 1) * dt
            drift_coeffs = np.array(
                list(map(lambda x: self.mu(x, t), paths[:, i - 1]))
            )  # t is the same for all paths in the current iteration
            diffusion_coeffs = np.array(
                list(map(lambda x: self.sigma(x, t), paths[:, i - 1]))
            )  # t is the same for all paths in the current iteration
            dW = np.random.normal(loc=0, scale=dW_scale, size=n_paths)
            paths[:, i] = (
                paths[:, i - 1] + drift_coeffs * dt + diffusion_coeffs * dW
            )
        paths = np.squeeze(paths)
        return paths


class MultidimensionalItoProcess(ItoProcess):
    """Multi-dimensional Ito process."""

    def __init__(self, mu: Callable, sigma: Callable, dim: int) -> None:
        super().__init__(mu, sigma)
        self.dim = dim

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
    def dim(self) -> int:  # noqa: D102
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        validate_integer(value, "dim")
        if not value >= 2:
            raise ValueError("'dim' must be greater or equal to 2.")
        self._dim = value

    def sample(
        self,
        T: float,
        n_time_grid: int,
        x0: Union[np.ndarray, float],
        n_paths: int = 1,
    ) -> np.ndarray:
        """Generate sample paths of the multi-dimensional Ito process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        if isinstance(x0, (float, int)):
            x0 = np.array([x0] * self.dim)
        else:
            validate_1d_array(x0, "x0")
            if len(x0) != self.dim:
                raise ValueError(f"'x0' of unexpected length: {len(x0)}.")

        # Run the Euler-Maruyama scheme
        dt = T / n_time_grid
        dW_cov = dt * np.eye(self.dim)
        times = np.linspace(0, T, n_time_grid)
        paths = np.zeros(shape=(n_paths, n_time_grid, self.dim))
        paths[:, 0] = x0

        # TODO: Check whether this cannot be improved both in terms of speed as
        #       well as in terms of readability.
        for i in range(1, n_time_grid):
            t = times[i - 1]
            drift_coeffs = np.apply_along_axis(
                lambda x: self.mu(x, t), axis=1, arr=paths[:, i - 1]
            )  # t is the same for all paths in the current iteration
            diff_coeffs = np.apply_along_axis(
                lambda x: self.sigma(x, t), axis=1, arr=paths[:, i - 1]
            )  # t is the same for all paths in the current iteration
            dW = np.random.multivariate_normal(
                mean=np.zeros(self.dim), cov=dW_cov, size=n_paths
            )
            paths[:, i] = (
                paths[:, i - 1]
                + drift_coeffs * dt
                + np.stack([diff_coeffs[i] @ dW[i] for i in range(n_paths)])
            )
        paths = np.squeeze(paths)

        return paths


class PoissonProcess(BaseLevyProcess):
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
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Poisson process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_number(x0, "x0")

        dt = T / n_time_grid
        sampler = lambda size: np.random.poisson(self.intensity * dt, size=size)
        paths = self._sample_from_increment_dist(
            sampler, n_time_grid, x0, n_paths
        )

        return paths


class CompoundPoissonProcess(PoissonProcess):
    """Compound Poisson process."""

    def __init__(
        self,
        intensity: float,
        jump_sampler: Callable[[Union[list, tuple, float]], np.ndarray],
    ) -> None:
        self.intensity = intensity
        self.jump_sampler = jump_sampler

    @property
    def jump_sampler(
        self,
    ) -> Callable[[Union[list, tuple, float]], np.ndarray]:  # noqa: D102
        return self._jump_sampler

    @jump_sampler.setter
    def jump_sampler(
        self, value: Callable[[Union[list, tuple, float]], np.ndarray]
    ) -> None:
        validate_callable_args(value, n_args=1, name="jump_sampler")
        self._jump_sampler = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the compound Poisson process."""
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_number(x0, "x0")

        pp_paths = super().sample(T, n_time_grid, 0, n_paths)
        if n_paths == 1:
            pp_paths = np.expand_dims(pp_paths, axis=0)
        paths = []
        for i in range(n_paths):
            n_jumps = pp_paths[i, -1]
            jumps = self.jump_sampler(n_jumps)
            path = np.cumsum(jumps)
            path = np.insert(path, 0, 0)[pp_paths[i]]
            paths.append(path)
        paths = np.vstack(paths)
        if x0 != 0:
            paths = paths + x0

        return np.squeeze(paths)


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
        if not 2 * theta * mu > sigma ** 2:
            raise ValueError(
                "The Feller condition for the parameters "
                "'theta', 'mu' and 'sigma' does not hold."
            )

    def sample(
        self,
        T: float,
        n_time_grid: int,
        x0: float,
        n_paths: int = 1,
        algorithm: Literal[
            "alfonsi", "euler-maruyama", "milstein-sym", "conditional"
        ] = "conditional",
    ) -> np.ndarray:
        """Generate sample paths of the Cox-Ingersoll-Ross process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_positive_number(x0, "x0")
        if not isinstance(algorithm, str):
            raise TypeError("'algorithm' must be of type str.")

        if algorithm == "alfonsi":
            paths = self._alfonsi(T, n_time_grid, x0, n_paths)
        elif algorithm == "euler-maruyama":
            paths = self._euler_maruyama(T, n_time_grid, x0, n_paths)
        elif algorithm == "milstein-sym":
            paths = self._milstein_sym(T, n_time_grid, x0, n_paths)
        elif algorithm == "conditional":
            paths = self._conditional(T, n_time_grid, x0, n_paths)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return paths

    def _euler_maruyama(
        self, T: float, n_time_grid: int, x0: float, n_paths: int
    ) -> np.ndarray:
        """Euler-Maruyama scheme for the Cox-Ingersoll-Ross process."""
        cir = ItoProcess(
            lambda x, t: self.theta * (self.mu - x),
            lambda x, t: self.sigma * np.sqrt(np.abs(x)),
        )
        paths = cir.sample(T, n_time_grid, x0, n_paths)
        return paths

    def _alfonsi(
        self, T: float, n_time_grid: int, x0: float, n_paths: int
    ) -> np.ndarray:
        """
        Alfonsi scheme for the Cox-Ingersoll-Ross process. See On the
        discretization schemes for the CIR (and Bessel squared) processes
        (2005).
        """
        dt = T / n_time_grid
        dW_scale = np.sqrt(dt)
        xi = 1 + self.theta / 2 * dt
        two_xi = 2 * xi
        eight_xi = 8 * xi
        four_xi_squared = 4 * xi ** 2
        rho = 4 * self.theta * self.mu - self.sigma ** 2

        paths = np.zeros(shape=(n_paths, n_time_grid))
        paths[:, 0] = x0
        for i in range(1, n_time_grid):
            dW = np.random.normal(scale=dW_scale, size=n_paths)
            z = np.sqrt(paths[:, i - 1]) + self.sigma / 2 * dW
            paths[:, i] = (
                z / two_xi
                + np.sqrt(z ** 2 / four_xi_squared + rho / eight_xi * dt)
            ) ** 2
        paths = np.squeeze(paths)

        return paths

    def _milstein_sym(
        self, T: float, n_time_grid: int, x0: float, n_paths: int
    ) -> np.ndarray:
        """
        Symmetrized Milstein scheme for the Cox-Ingersoll-Ross process from the
        paper Strong convergence of the symmetrized Milstein scheme for some
        CEV-like SDEs.
        """
        dt = T / n_time_grid
        dW_scale = np.sqrt(dt)

        paths = np.zeros(shape=(n_paths, n_time_grid))
        paths[:, 0] = x0
        for i in range(1, n_time_grid):
            dW = np.random.normal(scale=dW_scale, size=n_paths)
            paths[:, i] = np.abs(
                paths[:, i - 1]
                + self.theta * (self.mu - paths[:, i - 1]) * dt
                + self.sigma * np.sqrt(np.abs(paths[:, i - 1])) * dW
                + self.sigma ** 2 / 4 * (dW ** 2 - dt)
            )
        paths = np.squeeze(paths)

        return paths

    def _conditional(
        self, T: float, n_time_grid: int, x0: float, n_paths: int
    ) -> np.ndarray:
        """
        Exact sampling scheme based on the knowledge of the transition density
        of the process.
        """
        dt = T / n_time_grid
        d = 4 * self.mu * self.theta / self.sigma ** 2
        c = self.sigma ** 2 * (1 - np.exp(-self.theta * dt)) / (4 * self.theta)

        paths = np.zeros(shape=(n_paths, n_time_grid))
        paths[:, 0] = x0
        for i in range(1, n_time_grid):
            lam = paths[:, i - 1] * np.exp(-self.theta * dt) / c
            Z = np.random.normal(size=n_paths)
            X = np.random.chisquare(df=d - 1, size=n_paths)
            paths[:, i] = c * ((Z + np.sqrt(lam)) ** 2 + X)

        return np.squeeze(paths)


class SquaredBesselProcess(StochasticProcess):
    """Squared Bessel process."""

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
        self,
        T: float,
        n_time_grid: int,
        x0: float = 0,
        n_paths: int = 1,
        algorithm: Optional[
            Literal["alfonsi", "euler-maruyama", "milstein-sym", "radial"]
        ] = None,
    ) -> np.ndarray:
        """Generate sample paths of the squared Bessel process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_nonnegative_number(x0, "x0")
        if not isinstance(algorithm, (str, type(None))):
            raise TypeError("'algorithm' must be of type str.")

        if algorithm is None:  # pragma: no cover
            if self.n >= 2:  # alfonsi seems better when possible
                paths = self._alfonsi(T, n_time_grid, x0, n_paths)
            else:
                paths = self._milstein_sym(T, n_time_grid, x0, n_paths)
        elif algorithm == "euler-maruyama":
            paths = self._euler_maruyama(T, n_time_grid, x0, n_paths)
        elif algorithm == "milstein-sym":
            paths = self._milstein_sym(T, n_time_grid, x0, n_paths)
        elif algorithm == "alfonsi":
            paths = self._alfonsi(T, n_time_grid, x0, n_paths)
        elif algorithm == "radial":
            paths = self._radial(T, n_time_grid, x0, n_paths)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return paths

    def _euler_maruyama(
        self, T: float, n_time_grid: int, x0: float, n_paths: int
    ) -> np.ndarray:
        """Euler-Maruyama scheme for the squared Bessel process."""
        squared_bessel = ItoProcess(
            lambda x, t: self.n,
            lambda x, t: 2 * np.sqrt(np.abs(x)),
        )
        paths = squared_bessel.sample(T, n_time_grid, x0, n_paths)
        return paths

    def _alfonsi(
        self, T: float, n_time_grid: int, x0: float, n_paths: int
    ) -> np.ndarray:
        """
        Alfonsi scheme for the squared Bessel process. See On the discretization
        schemes for the CIR (and Bessel squared) processes (2005).
        """
        if self.n < 2:
            raise ValueError("'alfonsi' not applicable for n < 2.")
        dt = T / n_time_grid
        dW_scale = np.sqrt(dt)
        rho = (self.n - 2) * dt

        paths = np.zeros(shape=(n_paths, n_time_grid))
        paths[:, 0] = x0
        for i in range(1, n_time_grid):
            dW = np.random.normal(scale=dW_scale, size=n_paths)
            paths[:, i] = (dW + np.sqrt(dW ** 2 + paths[:, i - 1] + rho)) ** 2
        paths = np.squeeze(paths)

        return paths

    def _radial(
        self, T: float, n_time_grid: int, x0: float, n_paths: int
    ) -> np.ndarray:
        """
        Exact sampling based on the representation of a squared Bessel process
        as the squared L2 norm of standard n-dimensional Brownian motion.
        """
        if not self.n == int(self.n) or not self.n >= 2:
            raise ValueError(
                "'radial' algorithm can only be used for integer 'n'."
            )
        n = int(self.n)  # this is to also work for n = 3.0 for instance
        bm = MultidimensionalBrownianMotion(mu=np.zeros(n), sigma=np.eye(n))
        bm_x0 = np.zeros(n)
        bm_x0[0] = np.sqrt(x0)
        paths = bm.sample(T, n_time_grid, bm_x0, n_paths)
        paths = np.linalg.norm(paths, ord=2, axis=-1) ** 2

        return paths

    def _milstein_sym(
        self, T: float, n_time_grid: int, x0: float, n_paths: int
    ) -> np.ndarray:
        """
        Symmetrized Milstein scheme for the squared Bessel process from the
        paper Strong convergence of the symmetrized Milstein scheme for some
        CEV-like SDEs.
        """
        dt = T / n_time_grid
        dW_scale = np.sqrt(dt)
        n_dt = self.n * dt

        # Sample paths of the squared Bessel process for numerical stability
        paths = np.zeros(shape=(n_paths, n_time_grid))
        paths[:, 0] = x0
        for i in range(1, n_time_grid):
            dW = np.random.normal(scale=dW_scale, size=n_paths)
            paths[:, i] = np.abs(
                paths[:, i - 1]
                + n_dt
                + 2 * np.sqrt(np.abs(paths[:, i - 1])) * dW
                + (dW ** 2 - dt)
            )
        paths = np.squeeze(paths)

        return paths


class BesselProcess(SquaredBesselProcess):
    """Bessel process."""

    def sample(
        self,
        T: float,
        n_time_grid: int,
        x0: float = 0,
        n_paths: int = 1,
        algorithm: Optional[
            Literal["alfonsi", "euler-maruyama", "milstein-sym", "radial"]
        ] = None,
    ) -> np.ndarray:
        """Generate sample paths of the Bessel process."""
        validate_nonnegative_number(x0, "x0")
        paths = super().sample(T, n_time_grid, x0 ** 2, n_paths, algorithm)
        if algorithm == "euler-maruyama":
            return np.sqrt(np.abs(paths))
        else:
            return np.sqrt(paths)


class InverseBesselProcess(BesselProcess):
    """Inverse Bessel process."""

    def sample(
        self,
        T: float,
        n_time_grid: int,
        x0: float,
        n_paths: int = 1,
        algorithm: Optional[
            Literal["alfonsi", "euler-maruyama", "milstein-sym", "radial"]
        ] = None,
    ) -> np.ndarray:
        """Generate sample paths of the inverse Bessel process."""
        validate_positive_number(x0, "x0")
        paths = super().sample(T, n_time_grid, 1 / x0, n_paths, algorithm)
        if algorithm == "euler-maruyama":
            return 1 / np.abs(paths)
        else:
            return 1 / paths


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
        if not value > 0 or not value < 1:
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
        size = 2 ** ceil(np.log2(n_time_grid - 3)) + 1
        sqrt_eigenvalues = self._sqrt_eigenvalues(size)
        scale = (dt ** self.hurst) * 2 ** (1 / 2) * (size - 1)
        paths = []
        for i in range(n_paths):
            z = np.random.normal(scale=scale, size=2 * size).view(complex)
            z[0] = z[0].real * 2 ** (1 / 2)
            z[-1] = z[-1].real * 2 ** (1 / 2)
            fBm_increments = np.fft.irfft(sqrt_eigenvalues * z)[
                : (n_time_grid - 1)
            ]
            path = np.cumsum(fBm_increments)
            paths.append(path)
        paths = np.vstack(paths)
        paths = np.insert(paths, 0, 0, axis=1)
        if x0 != 0:
            paths = paths + x0
        paths = np.squeeze(paths)

        return paths

    @lru_cache(maxsize=1)
    def _sqrt_eigenvalues(self, size: int) -> np.ndarray:
        """
        Compute the square root of the eigenvalues of circulant matrix in which
        the covariance matrix of the increments of the fBm is embedded.
        """
        sqrt_eigenvalues = np.sqrt(
            np.fft.irfft(self._acf_fractional_gaussian_noise(self.hurst, size))[
                :size
            ]
        )
        return sqrt_eigenvalues

    @staticmethod
    def _acf_fractional_gaussian_noise(hurst: float, n: float) -> np.ndarray:
        """Autocovariance function of fractional Gaussian noise."""
        rho = np.arange(n + 1) ** (2 * hurst)
        rho = 1 / 2 * (rho[2:] - 2 * rho[1:-1] + rho[:-2])
        rho = np.insert(rho, 0, 1)
        return rho


class GammaProcess(BaseLevyProcess):
    """Gamma process."""

    def __init__(self, shape: float, scale: float) -> None:
        self.shape = shape
        self.scale = scale

    @property
    def shape(self) -> float:  # noqa: D102
        return self._shape

    @shape.setter
    def shape(self, value: float) -> None:
        validate_positive_number(value, "shape")
        self._shape = value

    @property
    def scale(self) -> float:  # noqa: D102
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        validate_positive_number(value, "scale")
        self._scale = value

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Gamma process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_number(x0, "x0")

        dt = T / n_time_grid
        sampler = lambda size: np.random.gamma(
            shape=self.shape * dt, scale=self.scale, size=size
        )
        paths = self._sample_from_increment_dist(
            sampler, n_time_grid, x0, n_paths
        )

        return paths


class BrownianBridge(StochasticProcess):
    """Brownian bridge."""

    def __init__(self, a: float, b: float, T: float) -> None:
        self.a = a
        self.b = b
        self.T = T

    @property
    def a(self) -> float:  # noqa: D102
        return self._a

    @a.setter
    def a(self, value: float) -> None:
        validate_number(value, "a")
        self._a = value

    @property
    def b(self) -> float:  # noqa: D102
        return self._b

    @b.setter
    def b(self, value: float) -> None:
        validate_number(value, "b")
        self._b = value

    @property
    def T(self) -> float:  # noqa: D102
        return self._T

    @T.setter
    def T(self, value: float) -> None:
        validate_positive_number(value, "T")
        self._T = value

    def sample(
        self,
        T: float,
        n_time_grid: int,
        x0: Optional[float] = None,
        n_paths: int = 1,
    ) -> np.ndarray:
        """Generate sample paths of the Brownian bridge."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        a = x0 if x0 is not None else self.a

        time = self.time_grid(self.T, n_time_grid)
        time_bm = time[time <= T]
        bm_paths = BrownianMotion()._sample_at(
            times=time_bm, x0=0, n_paths=n_paths
        )
        if n_paths == 1:  # unify the dimensionality regardless of n_paths
            bm_paths = np.expand_dims(bm_paths, axis=0)
        paths = (
            a
            + bm_paths
            + time_bm
            * (self.b - a - np.reshape(bm_paths[:, -1], (-1, 1)))
            / time_bm[-1]
        )
        n_constant_part = len(time) - len(time_bm)
        if n_constant_part > 0:
            constant_part = self.b * np.ones((n_paths, n_constant_part))
            paths = np.concatenate((paths, constant_part), axis=1)
        paths = np.squeeze(paths)
        return paths


class WishartProcess(StochasticProcess):
    """Wishart process."""

    def __init__(self, Q: np.ndarray, K: np.ndarray, alpha: float) -> None:
        self.Q = Q
        self.K = K
        self.alpha = alpha

    @property
    def Q(self) -> np.ndarray:  # noqa: D102
        return self._Q

    @Q.setter
    def Q(self, value: np.ndarray) -> None:
        validate_square_matrix(value, "Q")
        if hasattr(self, "_K"):
            self._check_parameter_compatibility(value, self._K)
        self._Q = value

    @property
    def K(self) -> np.ndarray:  # noqa: D102
        return self._K

    @K.setter
    def K(self, value: np.ndarray) -> None:
        validate_square_matrix(value, "K")
        if hasattr(self, "_Q"):
            self._check_parameter_compatibility(self._Q, value)
        self._K = value

    @property
    def alpha(self) -> float:  # noqa: D102
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        validate_nonnegative_number(value, "alpha")
        self._alpha = value

    @property
    def dim(self) -> int:  # noqa: D102
        return self.Q.shape[0]

    @staticmethod
    def _check_parameter_compatibility(Q: np.ndarray, K: np.ndarray) -> None:
        """Check if the parameters Q and K are compatible."""
        if Q.shape[0] != K.shape[0]:
            raise ValueError("Incompatible dimension of 'Q' and 'K'.")

    def sample(
        self, T: float, n_time_grid: int, x0: np.ndarray, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Wishart process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_possemdef_matrix(x0, "x0")

        dt = T / n_time_grid
        dW_scale = np.sqrt(dt)
        dim = self.dim
        Q_transposed = self.Q.T
        rho = self.alpha * np.matmul(Q_transposed, self.Q)

        paths = np.zeros(shape=(n_paths, n_time_grid, dim, dim))
        paths[:, 0] = x0
        eigval, right_eigvec = np.linalg.eigh(x0)
        for i in range(n_paths):
            for j in range(1, n_time_grid):
                dW = np.random.normal(scale=dW_scale, size=(dim, dim))
                sqrt_X_t_m_1 = (
                    right_eigvec @ np.diag(np.sqrt(eigval)) @ right_eigvec.T
                )
                drift = paths[i, j - 1] @ self.K
                vol = sqrt_X_t_m_1 @ dW @ self.Q
                X_t = (
                    paths[i, j - 1] + vol + vol.T + (drift + drift.T + rho) * dt
                )
                paths[i, j], eigval, right_eigvec = self._project_onto_PSD_cone(
                    X_t
                )
        paths = np.squeeze(paths)

        return paths

    @staticmethod
    def _project_onto_PSD_cone(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project a *symmetric* matrix onto the code of symmetric positive
        semidefinite matrices using the Frobenius norm. This is used in order
        to guarantee that the values of the Wishart processe are always positive
        semidefinite matrices.
        """
        eigval, right_eigvec = np.linalg.eigh(matrix)
        eigval = np.maximum(eigval, 0)
        if np.any(eigval == 0):
            matrix = right_eigvec @ np.diag(eigval) @ right_eigvec.T
        return matrix, eigval, right_eigvec


class NormalInverseGaussianProcess(BaseLevyProcess):
    """Normal inverse Gaussian process."""

    def __init__(
        self, alpha: float, beta: float, delta: float, mu: float
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.mu = mu

    @property
    def alpha(self) -> float:  # noqa: D102
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        validate_positive_number(value, "alpha")
        if hasattr(self, "_beta"):
            self._check_parameter_compatibility(value, self._beta)
        self._alpha = value

    @property
    def beta(self) -> float:  # noqa: D102
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        validate_number(value, "beta")
        if hasattr(self, "_alpha"):
            self._check_parameter_compatibility(self._alpha, value)
        self._beta = value

    @property
    def delta(self) -> float:  # noqa: D102
        return self._delta

    @delta.setter
    def delta(self, value: float) -> None:
        validate_positive_number(value, "delta")
        self._delta = value

    @property
    def mu(self) -> float:  # noqa: D102
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        validate_number(value, "mu")
        self._mu = value

    @staticmethod
    def _check_parameter_compatibility(alpha: float, beta: float) -> None:
        """Check if the parameters Q and K are compatible."""
        if not alpha > np.abs(beta):
            raise ValueError("alpha > |beta| is required.")

    def sample(
        self, T: float, n_time_grid: int, x0: float = 0, n_paths: int = 1
    ) -> np.ndarray:
        """Generate sample paths of the Normal inverse Gaussian process."""
        # Sanity check for input parameters
        validate_common_sampling_parameters(T, n_time_grid, n_paths)
        validate_number(x0, "x0")

        dt = T / n_time_grid
        delta_dt = self.delta * dt
        mu_dt = self.mu * dt
        inc_dist = norminvgauss(
            a=self.alpha * delta_dt,
            b=self.beta * delta_dt,
            loc=mu_dt,
            scale=delta_dt,
        )
        sampler = lambda size: inc_dist.rvs(size=size)
        paths = self._sample_from_increment_dist(
            sampler, n_time_grid, x0, n_paths
        )

        return paths
