import jax.numpy as jnp
from jaxtyping import Array, Float
from lib.algorithms import DirectedDiscreteGreedyAlgorithm
from lib.model.marginal import MarginalModel


class MMITL(DirectedDiscreteGreedyAlgorithm[MarginalModel]):
    r"""
    **Mean-Marginal Information-based Transductive Learning (MM-ITL)**

    $$\DeclareMathOperator*{\argmax}{arg max} \argmax_{\mathbf{x} \in \mathcal{S}_n} \sum_{\mathbf{x'} \in \mathcal{A}_n}\ I(\mathbf{f}(\mathbf{x'}); \mathbf{y}(\mathbf{x}) \mid \mathcal{D}_n).$$
    """

    def F(self) -> Float[Array, "N"]:
        sqd_cor = self.model.sqd_correlation(X=self.model.domain, A=self.roi)
        marginal_info_gain = -0.5 * jnp.log(1 - sqd_cor)
        return jnp.sum(marginal_info_gain, axis=(0, 1))
