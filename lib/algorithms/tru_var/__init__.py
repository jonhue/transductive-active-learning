from jax import vmap
import jax.numpy as jnp
from jaxtyping import Array, Float
from lib.algorithms import DirectedDiscreteGreedyAlgorithm
from lib.algorithms.tru_var.model import TruVarModel
from lib.typing import ScalarFloat, ScalarFloatArray, ScalarInt
from lib.utils import get_indices


class TruVar(DirectedDiscreteGreedyAlgorithm[TruVarModel]):
    """
    **Truncated Variance Reduction** [1]

    [1] Bogunovic, Ilija, et al. "Truncated variance reduction: A unified approach to bayesian optimization and level-set estimation." Advances in neural information processing systems 29 (2016).
    """

    def F(self) -> Float[Array, "N"]:
        assert self.model.q == 1 and self.model.first_constraint == 1

        roi = self.roi
        self.model.update_variance_threshold(roi)

        beta = self.model.beta(self.model.t)[0]
        covariance = self.model.covariances[0]
        variance = self.model.variances[0]
        noise_var = jnp.square(self.model.noise_rate.at(self.model.domain)[0])

        def compute_posterior_variance(
            idx: ScalarInt, obs_idx: ScalarInt
        ) -> ScalarFloat:
            return variance[idx] - covariance[idx, obs_idx] ** 2 / (
                variance[obs_idx] + noise_var[obs_idx]
            )

        def compute_truncated_variance(
            idx: ScalarInt, obs_idx: ScalarInt
        ) -> ScalarFloat:
            return jnp.maximum(
                beta * compute_posterior_variance(idx, obs_idx),
                self.model.eta**2,
            )

        def compute_total_posterior_variance(obs_idx: ScalarInt) -> ScalarFloatArray:
            return jnp.sum(
                vmap(lambda idx: compute_truncated_variance(idx, obs_idx))(
                    get_indices(self.model.domain, roi)
                )
            )

        return -vmap(compute_total_posterior_variance)(jnp.arange(self.n))
