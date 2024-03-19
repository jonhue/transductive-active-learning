import jax.numpy as jnp
from jaxtyping import Array, Float
from lib.algorithms import DirectedDiscreteGreedyAlgorithm
from lib.model.marginal import MarginalModel


class CTL(DirectedDiscreteGreedyAlgorithm[MarginalModel]):
    r"""
    **Correlation-based Transductive Learning (CTL)**

    $$\DeclareMathOperator*{\argmax}{arg max} \argmax_{\mathbf{x} \in \mathcal{S}_n}\ \sum_{\mathbf{x'} \in \mathcal{A}_n} \sum_{i \in \mathcal{I}} \mathrm{Cor}[f_i(\mathbf{x'}), y_i(\mathbf{x}) \mid \mathcal{D}_n]^2.$$
    """

    def F(self) -> Float[Array, "N"]:
        return jnp.sum(
            self.model.sqd_correlation(X=self.model.domain, A=self.roi), axis=(0, 1)
        )
