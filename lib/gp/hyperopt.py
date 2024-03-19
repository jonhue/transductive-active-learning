from typing import List, Tuple, Type
import jax.numpy as jnp
from jax import jit, value_and_grad
from jaxtyping import Array, Float
import optax
from tqdm import tqdm

from lib.gp import kernels
from lib.gp.gaussian_distribution import GaussianDistribution
from lib.typing import ScalarFloat
from lib.utils import noise_covariance_matrix


def negative_log_likelihood(
    theta: kernels.P,
    K: Type[kernels.Parameterized[kernels.P]],
    X: Float[Array, "n d"],
    y: Float[Array, "n"],
    noise_std: Float[Array, "n"] | float,
    mean: Float[Array, "n"] | float | None = None,
) -> ScalarFloat:
    """
    Computes the negative log likelihood (NLL) associated with kernel parameters `theta`.
    If `theta` includes a `"noise_std"` key, its value will be prioritized over the `noise_std` argument.
    """
    n = X.shape[0]
    if isinstance(mean, float) or isinstance(mean, int):
        mean = jnp.full(n, mean)
    elif mean is None:
        mean = jnp.zeros(n)
    if "noise_std" in theta.keys():
        noise_std = theta["noise_std"]  # type: ignore
    K_theta = K(**theta).covariance(X) + noise_covariance_matrix(n, noise_std)
    return -GaussianDistribution(mean=mean, covariance=K_theta).log_prob(y)


def optimize_theta(
    K: Type[kernels.Parameterized[kernels.P]],
    params: kernels.P,
    X: Float[Array, "n d"],
    y: Float[Array, "n"],
    noise_std: Float[Array, "n"] | float,
    optimizer: optax.GradientTransformation,
    num_iters=1_000,
    tol=1e-3,
    mean: Float[Array, "n"] | float | None = None,
    fit_noise_std=False,
) -> Tuple[kernels.P, List[ScalarFloat]]:
    """
    Minimizes the NLL to improve upon the initial kernel hyperparameters `params`.
    """
    if fit_noise_std:
        params = {**params, "noise_std": noise_std}  # type: ignore
    opt_state = optimizer.init(params)  # type: ignore

    @jit
    def step(params, opt_state):
        nll, grads = value_and_grad(negative_log_likelihood)(
            params, K, X, y, noise_std, mean
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, nll

    nlls = []
    pbar = tqdm(range(num_iters))
    for i in pbar:
        params, opt_state, nll = step(params, opt_state)
        nlls.append(nll)
        pbar.set_description(f"{nll}")
        if i % 100 == 0 and i > 0 and nll + tol > nlls[-100]:
            break

    return params, nlls
