from typing import List, Tuple
from jaxtyping import Array, Float
import jax.random as jr
from lib.function import Function
from lib.gp.gaussian_distribution import GaussianDistribution
from lib.gp.kernels import Kernel
from lib.gp.means import Mean, ZeroMean
from lib.noise import Noise
from lib.utils import get_indices


def initial_observations(
    key: jr.KeyArray,
    f: Function,
    n_samples: int,
    region: Float[Array, "n d"],
) -> Tuple[Float[Array, "n_samples d"], Float[Array, "q n_samples"]]:
    r"""
    Observes `n_samples` points within `region` which are chosen uniformly at random (with replacement).

    :param key: Randomization key.
    :param f: Wrapper around unknown function.
    :param n_samples: Number of samples.
    :param region: Subset of the domain $\mathcal{X}$ in which to sample (uniformly). Must be non-empty.
    """
    n = region.shape[0]
    indices = jr.randint(key=key, minval=0, maxval=n, shape=(n_samples,)).sort()
    X = region[indices]
    y = f.observe(X)
    return X, y


def prior_distr(
    noise_rate: Noise,
    domain: Float[Array, "n d"],
    X: Float[Array, "m d"],
    y: Float[Array, "q m"],
    kernels: List[Kernel],
    means: List[Mean],
) -> List[GaussianDistribution]:
    r"""
    For each $i \in \mathcal{I}$, computes a GP prior. Assumes that observation noise is Gaussian.

    :param noise_rate: (Assumed) noise standard deviations.
    :param domain: Finite domain.
    :param X: Locations of observations. Must be a subset of `domain`.
    :param y: Observations.
    :param kernels: Prior kernel functions for each $i \in \mathcal{I}$.
    :param means: Prior mean functions for each $i \in \mathcal{I}$.
    """
    q = y.shape[0]
    indices = get_indices(domain, X)
    noise_rates = noise_rate.at(X)

    posterior_distrs = []
    for i in range(q):
        covariance = kernels[i].covariance(domain)
        prior_distr = GaussianDistribution(
            mean=means[i].vector(domain), covariance=covariance
        )
        posterior_distr = prior_distr.posterior(
            obs_idx=indices, obs=y[i], obs_noise_std=noise_rates[i]
        )
        posterior_distrs.append(posterior_distr)
    return posterior_distrs
