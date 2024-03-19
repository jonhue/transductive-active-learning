import jax.numpy as jnp
from jax._src.config import config
import optax
from pytest import approx
from lib.gp.hyperopt import optimize_theta
from lib.gp import kernels
from lib.noise import HomoscedasticNoise

config.update("jax_enable_x64", True)

noise_rate = HomoscedasticNoise(q=1, noise_rates=jnp.array([0.1]))
X = jnp.array([[-1], [-0.5], [0], [0.5], [1]])
y = jnp.sin(X).T[0]
noise_std = noise_rate.at(X)[0]


def test_gaussian():
    params, nlls = optimize_theta(
        K=kernels.stationary.Gaussian,
        params=kernels.stationary.Gaussian.default_params,
        X=X,
        y=y,
        noise_std=noise_std,
        optimizer=optax.adam(learning_rate=0.01),
        num_iters=500,
    )
    assert params.get("variance") == approx(1.012, abs=1e-3)
    assert params.get("lengthscale") == approx(1.761, abs=1e-3)
    assert nlls[0] == approx(1.552, abs=1e-3)
    assert nlls[-1] == approx(0.509, abs=1e-3)


def test_laplace():
    params, nlls = optimize_theta(
        K=kernels.stationary.Laplace,
        params=kernels.stationary.Laplace.default_params,
        X=X,
        y=y,
        noise_std=noise_std,
        optimizer=optax.adam(learning_rate=0.01),
        num_iters=500,
    )
    assert params.get("variance") == approx(0.478, abs=1e-3)
    assert params.get("lengthscale") == approx(2.112, abs=1e-3)
    assert nlls[0] == approx(4.560, abs=1e-3)
    assert nlls[-1] == approx(3.376, abs=1e-3)
