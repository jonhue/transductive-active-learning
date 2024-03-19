from jax import vmap
import jax.numpy as jnp
from jaxtyping import Array, Float
from lib.algorithms import DirectedDiscreteGreedyAlgorithm
from lib.gp.gaussian_distribution import compute_posterior_covariance
from lib.model.marginal import MarginalModel
from lib.typing import ScalarInt
from lib.utils import get_indices, noise_covariance_matrix


class ITL(DirectedDiscreteGreedyAlgorithm[MarginalModel]):
    r"""
    **Information-based Transductive Learning (ITL)**

    Chooses the point which minimizes the posterior uncertainty about the region of interest:     $$\DeclareMathOperator*{\argmax}{arg max} \argmax_{\mathbf{x} \in \mathcal{S}_n}\ I(\mathbf{f}(\mathcal{A}_n); \mathbf{y}(\mathbf{x}) \mid \mathcal{D}_n).$$

    Is equivalent to `uncertainty_sampling` if the sample domain (i.e., `model.operating_safe_set(domain)`) is a subset of the region of interest and the noise is homoscedastic.
    """

    def F(self) -> Float[Array, "N"]:
        roi = self.roi
        if self.model.safe_set_is_subset_of(A=roi):
            return _undirected(model=self.model)
        else:
            return _directed(model=self.model, roi=roi)


def _undirected(model: MarginalModel) -> Float[Array, "N"]:
    obs_noise_var = jnp.square(model.noise_rate.at(model.domain))
    return jnp.sum(jnp.log(1 + model.variances / obs_noise_var), axis=0)


def _directed(
    model: MarginalModel,
    roi: Float[Array, "m d"],
) -> Float[Array, "N"]:
    roi_idx = model.get_indices(roi)
    covariances = model.covariances
    obs_noise_stds = model.noise_rate.at(model.domain)

    def engine(i: ScalarInt) -> Float[Array, "N"]:
        prior_variance = jnp.diag(covariances[i])
        posterior_covariance = compute_posterior_covariance(
            prior_covariance=covariances[i],
            obs_idx=roi_idx,
            obs_noise_std=None,
        )
        posterior_variance = jnp.diag(posterior_covariance)

        obs_noise_std = obs_noise_stds[i]
        obs_noise = jnp.diag(
            noise_covariance_matrix(k=obs_noise_std.shape[0], noise_std=obs_noise_std)
        )
        predictive_variance = prior_variance + obs_noise
        posterior_predictive_variance = posterior_variance + obs_noise
        return 0.5 * jnp.maximum(
            jnp.log(predictive_variance / posterior_predictive_variance), 0
        )

    return jnp.sum(vmap(engine)(jnp.arange(model.q)), axis=0)
