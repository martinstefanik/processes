#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probability distributions."""


from abc import ABC, abstractmethod
from typing import Sequence, Union

import numpy as np

from processes.utils import validate_positive_number


class Distribution(ABC):
    """Abstract base class for a probability distributions."""

    @abstractmethod
    def sample(self, size: Sequence[int]) -> Union[np.ndarray, float]:
        """Generate a sample from the probability distribution."""
        raise NotImplementedError("'sample' method not implemented.")


class NormalDistribution(Distribution):
    """Normal distribution."""

    def __init__(self, mu: float = 0, sigma: float = 1) -> None:
        self.mu = mu
        self.sigma = sigma

    @property
    def sigma(self) -> float:  # noqa: D102
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:  # noqa: D102
        validate_positive_number(sigma, "sigma")
        self._sigma = sigma

    def sample(self, size: Sequence[int] = 1) -> Union[np.ndarray, float]:
        """Generate a sample from the normal distribution."""
        return np.random.normal(self.mu, self.sigma, size)


class LogNormalDistribution(NormalDistribution):
    """Log-normal distribution."""

    def sample(self, size: Sequence[int] = 1) -> Union[np.ndarray, float]:
        """Generate a sample from the log-normal distribution."""
        return np.random.lognormal(self.mu, self.sigma, size)


class ExponentialDistribution(Distribution):
    """Exponential distribution."""

    def __init__(self, lam: float = 1) -> None:
        self.lam = lam

    @property
    def lam(self) -> float:  # noqa: D102
        return self._lam

    @lam.setter
    def lam(self, value: float) -> None:  # noqa: D102
        validate_positive_number(value, "mu")
        self._lam = value

    def sample(self, size: Sequence[int] = 1) -> Union[np.ndarray, float]:
        """Generate a sample from the exponential distribution."""
        return np.random.exponential(self.lam, size)


class ChisquaredDistribution(Distribution):
    """Chi-squared distribution."""

    def __init__(self, nu: float = 1) -> None:
        self.nu = nu

    @property
    def nu(self) -> float:  # noqa: D102
        return self._nu

    @nu.setter
    def nu(self, value: float) -> None:  # noqa: D102
        validate_positive_number(value, "nu")
        self._nu = value

    def sample(self, size: Sequence[int] = 1) -> Union[np.ndarray, float]:
        """Generate a sample from the chi-squared distribution."""
        return np.random.chisquare(self.nu, size)


class GammaDistribution(Distribution):
    """Gamma distribution."""

    def __init__(self, shape: float = 1, scale: float = 1) -> None:
        self.shape = shape
        self.scale = scale

    @property
    def shape(self) -> float:  # noqa: D102
        return self._shape

    @shape.setter
    def shape(self, value: float) -> None:  # noqa: D102
        validate_positive_number(value, "shape")
        self._shape = value

    @property
    def scale(self) -> float:  # noqa: D102
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:  # noqa: D102
        validate_positive_number(value, "scale")
        self._scale = value

    def sample(self, size: Sequence[int] = 1) -> Union[np.ndarray, float]:
        """Generate a sample from the gamma distribution."""
        return np.random.gamma(self.shape, self.scale, size)
