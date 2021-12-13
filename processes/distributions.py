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
        """Generate a sample from the probability distribution."""
        return np.random.lognormal(self.mu, self.sigma, size)
