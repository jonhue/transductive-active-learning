import jax.numpy as jnp
import jax.random as jr
from jax._src.config import config
from lib.gp import kernels
from lib.model.continuous import ContinuousModel
from lib.noise import HomoscedasticNoise

config.update("jax_enable_x64", True)

noise_rate = HomoscedasticNoise(q=1, noise_rates=jnp.array([0.1]))

kernel = kernels.stationary.Gaussian(lengthscale=1)

key = jr.PRNGKey(0)
model = ContinuousModel(
    key=key,
    d=1,
    prior_kernels=[kernel],
    beta=jnp.array([1.0]),
    noise_rate=noise_rate,
    use_objective_as_constraint=True,
)

X = jnp.array([[-1], [-0.5], [0], [0.5], [1]])
N = X.shape[0]

marginalized_model = model.marginalize(X)


def test_mean():
    assert jnp.all(marginalized_model.means == jnp.zeros((1, N)))


def test_stddev():
    assert jnp.all(marginalized_model.stddevs == jnp.ones((1, N)))


def test_confidence_bounds():
    assert jnp.all(marginalized_model.l == -jnp.ones((1, N)))
    assert jnp.all(marginalized_model.u == jnp.ones((1, N)))


def test_sample():
    assert marginalized_model.sample(k=10).shape == (1, 10, N)


def test_step():
    y = jnp.sin(X).T
    model.step(X, y)
    marginalized_model = model.marginalize(X)
    assert jnp.any(marginalized_model.means != jnp.zeros((1, N)))
