from jax import jit, vmap
import jax.numpy as jnp
from jaxtyping import Array, Float
import jax.random as jr
import jax.scipy.stats as jstats
from lib.algorithms import DiscreteGreedyAlgorithm
from lib.model.marginal import MarginalModel
from lib.typing import ScalarFloat, ScalarInt


class UniformSampling(DiscreteGreedyAlgorithm):
    """Chooses the next point uniformly at random."""

    def F(self) -> Float[Array, "N"]:
        key = self.acquire_key()
        return jr.uniform(key=key, shape=(self.n,))


class UncertaintySampling(DiscreteGreedyAlgorithm):
    """Chooses the point with the largest (prior) uncertainty."""

    def F(self) -> Float[Array, "N"]:
        return jnp.sum(self.model.stddevs, axis=0)


class UCB(DiscreteGreedyAlgorithm):
    """Chooses the point with the largest upper confidence bound."""

    def F(self) -> Float[Array, "N"]:
        return self.model.u[0]


@jit
def _mes(
    mean: Float[Array, "n"],
    stddev: Float[Array, "n"],
    optima: Float[Array, "k n"],
) -> Float[Array, "n"]:
    def engine(opt: ScalarFloat) -> ScalarFloat:
        gamma = (opt - mean) / stddev
        return gamma * jstats.norm.pdf(gamma) / (2 * jstats.norm.cdf(gamma)) - jnp.log(
            jstats.norm.cdf(gamma)
        )

    return jnp.mean(vmap(engine)(optima), axis=0)


class MES(DiscreteGreedyAlgorithm):
    r"""Max-value Entropy Search. Choosing $\mathbf{x}$ based on $\argmax_{\mathbf{x} \in \mathcal{S}_n}\ I(y^\star; \mathbf{y}(\mathbf{x}) \mid \mathcal{D}_n).$"""

    k: ScalarInt

    def __init__(
        self,
        model: MarginalModel,
        k: ScalarInt,
        sample_region: Float[Array, "m d"] | None = None,
    ):
        super().__init__(model, sample_region)
        self.k = k

    def F(self) -> Float[Array, "N"]:
        J = jnp.arange(self.model.first_constraint, self.model.q)
        optima = self.model.thompson_sampling_values(k=self.k, J=J)
        return _mes(
            mean=self.model.means[0], stddev=self.model.stddevs[0], optima=optima
        )


class EI(DiscreteGreedyAlgorithm):
    """Expected improvement."""

    def F(self) -> Float[Array, "N"]:
        x_opt = self.model.argmax
        opt_idx = self.model.get_indices(x_opt.reshape(1, -1))[0]
        opt = self.model.means[0, opt_idx]
        gamma = (opt - self.model.means[0]) / self.model.stddevs[0]
        return (self.model.means[0] - opt) * jstats.norm.cdf(
            gamma
        ) + self.model.stddevs[0] * jstats.norm.pdf(gamma)


class EIC(DiscreteGreedyAlgorithm):
    """Expected improvement with constraints."""

    def F(self) -> Float[Array, "N"]:
        assert self.model.q == self.model.first_constraint + 1

        ei = EI(model=self.model, sample_region=self.sample_region).F()
        gamma = (
            -self.model.means[self.model.first_constraint]
            / self.model.stddevs[self.model.first_constraint]
        )
        return jstats.norm.cdf(gamma) * ei
