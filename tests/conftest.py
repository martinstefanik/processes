"""Global fixtures."""

import pytest

from processes.distributions import NormalDistribution
from processes.processes import *

processes = (
    {
        "class": BesselProcess,
        "parameters": {"n": 2.5},
        "x0": 1.5,
    },
    {
        "class": BrownianBridge,
        "parameters": {"a": 1, "b": 5, "T": 2},
        "x0": 1,
    },
    {
        "class": BrownianMotion,
        "parameters": {"mu": 0, "sigma": 1},
        "x0": 1,
    },
    {
        "class": CoxIngersollRossProcess,
        "parameters": {"theta": 1.2, "mu": 0.01, "sigma": 0.03},
        "x0": 0.02,
    },
    {
        "class": GammaProcess,
        "parameters": {"shape": 0.5, "scale": 0.1},
        "x0": 1,
    },
    {
        "class": GeometricBrownianMotion,
        "parameters": {"mu": 0.01, "sigma": 0.02},
        "x0": 2.5,
    },
    {
        "class": InverseBesselProcess,
        "parameters": {"n": 5},
        "x0": 1.5,
    },
    {
        "class": MultidimensionalBrownianMotion,
        "parameters": {"mu": np.repeat(0.2, 3), "sigma": 2 * np.eye(3)},
        "x0": np.array([4, -1, 0.2]),
    },
    {
        "class": MultidimensionalItoProcess,
        "parameters": {
            "mu": lambda x, t: np.array([np.tanh(x[0]), np.log(1 + x[1] ** 2)]),
            "sigma": lambda x, t: np.array(
                [[2, x[0] + x[1]], [np.tanh(x[1]), 1]]
            ),
            "dim": 2,
        },
        "x0": np.array([1, -1]),
    },
    {
        "class": OrnsteinUhlenbeckProcess,
        "parameters": {"theta": 1.2, "mu": 0.01, "sigma": 0.02},
        "x0": 0.01,
    },
    {
        "class": ItoProcess,
        "parameters": {
            "mu": lambda x, t: np.sin(x),
            "sigma": lambda x, t: np.tanh(x + t),
        },
        "x0": 0.25,
    },
    {
        "class": PoissonProcess,
        "parameters": {"intensity": 4.5},
        "x0": 4,
    },
    {
        "class": CompoundPoissonProcess,
        "parameters": {"intensity": 7, "jump_dist": NormalDistribution()},
        "x0": 0.4,
    },
    {
        "class": SquaredBesselProcess,
        "parameters": {"n": 3.5},
        "x0": 0.2,
    },
    {
        "class": TimeChangedBrownianMotion,
        "parameters": {"time_change": lambda t: np.log(1 + t)},
        "x0": 1,
    },
    {
        "class": WishartProcess,
        "parameters": {
            "Q": 0.1 * np.eye(2),
            "K": np.array([[1, 3], [0.2, 1]]),
            "alpha": 0.3,
        },
        "x0": np.eye(2),
    },
)


@pytest.fixture(params=processes)
def any_process(request):
    process = request.param["class"](**request.param["parameters"])
    x0 = request.param["x0"]
    return process, x0


@pytest.fixture(params=[1, 2])
def n_paths(request):
    return request.param


@pytest.fixture(params=[-1, 1.2, list()])
def n_paths_invalid(request):
    return request.param


@pytest.fixture
def n_time_grid():
    return 100


@pytest.fixture(params=[-1, 1.2, list()])
def n_time_grid_invalid(request):
    return request.param


@pytest.fixture(params=[1, 5])
def T(request):
    return request.param


@pytest.fixture(params=[-1, -1.2, list()])
def T_invalid(request):
    return request.param
