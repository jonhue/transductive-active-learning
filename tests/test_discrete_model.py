import jax.numpy as jnp
import jax.random as jr
from jax._src.config import config
from lib.gp.gaussian_distribution import GaussianDistribution
from lib.gp import kernels
from lib.model.marginal import MarginalModel
from lib.noise import HomoscedasticNoise

config.update("jax_enable_x64", True)

noise_rate = HomoscedasticNoise(q=1, noise_rates=jnp.array([0.1]))

domain = jnp.linspace(-1, 1, 100).reshape(-1, 1)
N = domain.shape[0]
kernel = kernels.stationary.Gaussian(lengthscale=1)
distr = GaussianDistribution(mean=jnp.zeros(N), covariance=kernel.covariance(domain))

key = jr.PRNGKey(0)
model = MarginalModel(
    key=key,
    domain=domain,
    distrs=[distr],
    beta=jnp.array([1.0]),
    noise_rate=noise_rate,
    use_objective_as_constraint=True,
)


def test_mean():
    assert jnp.all(model.means == jnp.zeros((1, N)))


def test_stddev():
    assert jnp.all(model.stddevs == jnp.ones((1, N)))


def test_confidence_bounds():
    assert jnp.all(model.l == -jnp.ones((1, N)))
    assert jnp.all(model.u == jnp.ones((1, N)))


def test_sample():
    assert model.sample(k=10).shape == (1, 10, N)
