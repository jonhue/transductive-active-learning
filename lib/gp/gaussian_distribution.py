from __future__ import annotations
from typing import Tuple
from chex import dataclass
from jaxtyping import Array, Float, Int
from jax import jit
import jax.numpy as jnp
import jax.random as jr
from lib.gp.kernels import Kernel
from lib.gp.means import Mean
from lib.typing import ScalarFloat
from lib.utils import noise_covariance_matrix, solve_linear_system

DEFAULT_JITTER = 1e-5


@jit
def _compute_posterior_covariance(
    prior_covariance: Float[Array, "n n"],
    obs_idx: Float[Array, "m"],
    obs_noise_std: Float[Array, "m"] | float | None = None,
    jitter: float = DEFAULT_JITTER,
) -> Tuple[Float[Array, "n n"], Float[Array, "m n"]]:
    if obs_idx.shape[0] == 0:
        return prior_covariance, jnp.array([])

    n = prior_covariance.shape[0]
    full_idx = jnp.arange(n)

    obs_noise_mat = noise_covariance_matrix(
        k=obs_idx.shape[0], noise_std=obs_noise_std, default=jitter
    )
    Sigma = prior_covariance[jnp.ix_(obs_idx, obs_idx)] + obs_noise_mat
    Kxt = prior_covariance[jnp.ix_(obs_idx, full_idx)]
    Sigma_inv_Kxt, _ = solve_linear_system(Sigma, Kxt)

    covariance = prior_covariance - Kxt.T @ Sigma_inv_Kxt
    return covariance, Sigma_inv_Kxt


@jit
def compute_posterior_covariance(
    prior_covariance: Float[Array, "n n"],
    obs_idx: Float[Array, "m"],
    obs_noise_std: Float[Array, "m"] | float | None = None,
    jitter: float = DEFAULT_JITTER,
) -> Float[Array, "n n"]:
    """
    Computes the Gaussian posterior covariance matrix conditional on $m$ observations at locations `obs_idx` with noise `obs_noise_std`.
    """
    return _compute_posterior_covariance(
        prior_covariance=prior_covariance,
        obs_idx=obs_idx,
        obs_noise_std=obs_noise_std,
        jitter=jitter,
    )[0]


@dataclass
class GaussianDistribution:
    r"""
    **Multivariate Gaussian** $\mathcal{N}(\boldsymbol{\mu}; \boldsymbol{\Sigma})$ with dimension $n$.
    """

    mean: Float[Array, "n"]
    """Mean vector."""
    covariance: Float[Array, "n n"]
    """Covariance matrix."""

    @classmethod
    def initialize(cls, mean: Mean, kernel: Kernel, X: Float[Array, "n d"]):
        return cls(mean=mean.vector(X), covariance=kernel.covariance(X))

    @property
    def n(self) -> int:
        """Dimension."""
        return self.mean.shape[0]

    @property
    @jit
    def variance(self) -> Float[Array, "n"]:
        """Vectors of variances of all one-dimensional marginals."""
        return jnp.diagonal(self.covariance)

    @property
    @jit
    def stddev(self) -> Float[Array, "n"]:
        """Vectors of standard deviations of all one-dimensional marginals."""
        return jnp.sqrt(self.variance)

    def sample(self, key: jr.KeyArray, sample_shape: tuple) -> Float[Array, "n"]:
        """Sample independent functions from the Gaussian."""
        return jr.multivariate_normal(
            key=key,
            mean=self.mean,
            cov=self.covariance,
            shape=sample_shape,
            method="svd",
        )

    @jit
    def posterior(
        self,
        obs_idx: Int[Array, "m"],
        obs: Float[Array, "m"],
        obs_noise_std: Float[Array, "m"] | float | None = None,
        jitter: float = DEFAULT_JITTER,
    ) -> GaussianDistribution:
        """
        Computes the Gaussian posterior distribution conditional on $m$ observations `obs` at locations `obs_idx` with noise `obs_noise_std`.
        """
        prior_mean = self.mean
        prior_covariance = self.covariance

        covariance, Sigma_inv_Kxt = _compute_posterior_covariance(
            prior_covariance=prior_covariance,
            obs_idx=obs_idx,
            obs_noise_std=obs_noise_std,
            jitter=jitter,
        )
        mean = prior_mean + Sigma_inv_Kxt.T @ (obs - prior_mean[obs_idx])
        return GaussianDistribution(mean=mean, covariance=covariance)

    @property
    @jit
    def total_variance(self) -> ScalarFloat:
        """Computes the total variance of the distribution."""
        return jnp.sum(self.variance)

    @jit
    def log_prob(self, y: Float[Array, "n"]) -> ScalarFloat:
        """Log probability of observation `y`."""
        delta = y - self.mean
        alpha, L = solve_linear_system(self.covariance, delta)
        return -(
            0.5 * delta.T @ alpha
            + jnp.sum(jnp.log(jnp.diag(L)))
            + 0.5 * self.n * jnp.log(2 * jnp.pi)
        )

    @jit
    def entropy(self, jitter: float = 0.0) -> ScalarFloat:
        """Computes the entropy of the distribution."""
        return 0.5 * (
            self.n * jnp.log(2 * jnp.pi * jnp.e)
            + jnp.linalg.slogdet(
                self.covariance + jitter * jnp.eye(self.covariance.shape[0])
            )[1]
        )

    @jit
    def information_gain(
        self,
        roi_idx: Int[Array, "k"],
        obs_idx: Int[Array, "m"],
        obs_noise_std: Float[Array, "m"] | None = None,
        jitter: float = DEFAULT_JITTER,
    ) -> ScalarFloat:
        """
        Computes the mutual information of `roi_idx` and `obs_idx` where the points in `obs_idx` are perturbed by i.i.d. Gaussian noise with variances `obs_noise_std`.
        """
        prior_covariance = self.covariance[jnp.ix_(obs_idx, obs_idx)]
        posterior_covariance = compute_posterior_covariance(
            prior_covariance=self.covariance,
            obs_idx=roi_idx,
            obs_noise_std=None,
            jitter=jitter,
        )[jnp.ix_(obs_idx, obs_idx)]

        obs_noise_mat = noise_covariance_matrix(
            k=obs_idx.shape[0], noise_std=obs_noise_std
        )
        predictive_covariance = prior_covariance + obs_noise_mat
        posterior_predictive_covariance = posterior_covariance + obs_noise_mat
        predictive_covariance_log_det = jnp.linalg.slogdet(predictive_covariance)[1]
        posterior_predictive_covariance_log_det = jnp.nan_to_num(
            jnp.linalg.slogdet(posterior_predictive_covariance)[1], nan=-jnp.inf
        )
        return 0.5 * jnp.maximum(
            predictive_covariance_log_det - posterior_predictive_covariance_log_det, 0
        )

    @jit
    def sub_distr(self, idx: Int[Array, "m"]) -> GaussianDistribution:
        """Returns Gaussian limited to the points in `idx`."""
        return GaussianDistribution(
            mean=self.mean[idx], covariance=self.covariance[jnp.ix_(idx, idx)]
        )
