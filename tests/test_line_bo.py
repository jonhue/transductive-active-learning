import jax.numpy as jnp
import jax.random as jr
from jax._src.config import config
from pytest import approx
from lib.algorithms import DiscreteGreedyAlgorithm
from lib.algorithms.line_bo import (
    CoordinateLineBO,
    DescentLineBO,
    RandomLineBO,
    alg_constructor,
)
from lib.function.unknown import UnknownFunction
from lib.gp import kernels
from lib.model.continuous import ContinuousModel
from lib.noise import HomoscedasticNoise

config.update("jax_enable_x64", True)


class DummyAlg(DiscreteGreedyAlgorithm):
    def F(self):
        pass


noise_rate = HomoscedasticNoise(q=1, noise_rates=jnp.array([0.1]))

kernel = kernels.stationary.Gaussian()

D = 2
key = jr.PRNGKey(0)
model = ContinuousModel(
    key=key,
    d=D,
    prior_kernels=[kernel],
    beta=jnp.array([1.0]),
    noise_rate=noise_rate,
    use_objective_as_constraint=True,
)

bounds = jnp.array([[-1.0, 1.0], [0.0, 1.0]])
n = 10

f = UnknownFunction(q=1, f=lambda x: jnp.array([jnp.linalg.norm(x)]))
eps = 0.01


def test_random():
    random_lbo = RandomLineBO(
        alg=alg_constructor(DummyAlg), model=model, bounds=bounds, n=n, eps=eps
    )
    assert jnp.linalg.norm(random_lbo._normalized_direction(f)) == approx(1, abs=1e-5)


def test_coordinate():
    coordinate_lbo = CoordinateLineBO(
        alg=alg_constructor(DummyAlg), model=model, bounds=bounds, n=n, eps=eps
    )
    assert jnp.linalg.norm(coordinate_lbo._normalized_direction(f)) == approx(
        1, abs=1e-5
    )


def test_descent():
    descent_lbo = DescentLineBO(
        alg=alg_constructor(DummyAlg),
        model=model,
        bounds=bounds,
        n=n,
        eps=eps,
        m=2 * D,
        alpha=0.1,
    )
    assert jnp.linalg.norm(descent_lbo._normalized_direction(f)) == approx(1, abs=1e-5)
