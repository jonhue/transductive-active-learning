from jax import vmap
import jax.numpy as jnp
from jaxtyping import Array, Float
from lib.algorithms import DirectedDiscreteGreedyAlgorithm
from lib.model.marginal import MarginalModel
from lib.typing import ScalarFloat, ScalarInt


class VTL(DirectedDiscreteGreedyAlgorithm[MarginalModel]):
    r"""
    **Variance-based Transductive Learning (VTL)**

    $$\DeclareMathOperator*{\argmin}{arg min} \argmin_{\mathbf{x} \in \mathcal{S}_n}\ \mathrm{tr}\, \mathrm{Var}[\mathbf{f}(\mathcal{A}_n) \mid \mathcal{D}_n, \mathbf{y}(\mathbf{x})].$$
    """

    def F(self) -> Float[Array, "N"]:
        roi = self.roi
        covariances = self.model.covariances
        noise_var = jnp.square(self.model.noise_rate.at(self.model.domain))

        def engine(i: ScalarInt) -> ScalarFloat:
            covariance = covariances[i]
            variance = jnp.diag(covariance)

            def compute_posterior_variance(
                idx: ScalarInt, obs_idx: ScalarInt
            ) -> ScalarFloat:
                return variance[idx] - covariance[idx, obs_idx] ** 2 / (
                    variance[obs_idx] + noise_var[i, obs_idx]
                )

            def compute_total_posterior_variance(obs_idx: ScalarInt) -> ScalarFloat:
                return jnp.sum(
                    vmap(lambda idx: compute_posterior_variance(idx, obs_idx))(
                        self.model.get_indices(roi)
                    )
                )

            return -vmap(compute_total_posterior_variance)(jnp.arange(self.n))

        return jnp.sum(vmap(engine)(jnp.arange(self.model.q)), axis=0)
