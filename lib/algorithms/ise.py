from jax import jit, vmap
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from lib.algorithms import DiscreteGreedyAlgorithm
from lib.algorithms.baselines import _mes
from lib.model.marginal import MarginalModel
from lib.noise import HomoscedasticNoise
from lib.typing import ScalarFloat, ScalarInt
from lib.utils import set_to_idx


@jit
def _ise(
    mean: Float[Array, "n"],
    variance: Float[Array, "n"],
    noise_std: ScalarFloat,
    sqd_cor: Float[Array, "n n"],
    pessimistic_safe_set_idx: Int[Array, "m"],
) -> Float[Array, "n"]:
    c1 = 1 / (jnp.log(2) * jnp.pi)
    c2 = 2 * c1 - 1

    def engine(j: ScalarInt) -> ScalarFloat:
        variance_j = variance[j]
        sqd_cor_j = sqd_cor[j, :]

        ent = jnp.log(2) * jnp.exp(
            -1 / (jnp.pi * jnp.log(2)) * (mean / jnp.sqrt(variance)) ** 2
        )

        cond_ent = (
            jnp.log(2)
            * jnp.sqrt(
                (noise_std**2 + variance_j * (1 - sqd_cor_j))
                / (noise_std**2 + variance_j * (1 + c2 * sqd_cor_j))
            )
            * jnp.exp(
                -c1
                * (mean / jnp.sqrt(variance)) ** 2
                * (noise_std**2 + variance_j)
                / (noise_std**2 + variance_j * (1 + c2 * sqd_cor_j))
            )
        )

        mi = ent - cond_ent
        return jnp.max(mi)

    result = jnp.zeros_like(mean)
    result_within_safe_set = vmap(lambda j: engine(j))(pessimistic_safe_set_idx)
    return result.at[pessimistic_safe_set_idx].set(result_within_safe_set)


class ISE(DiscreteGreedyAlgorithm[MarginalModel]):
    r"""
    **Information-theoretic Safe Exploration (ISE)** [1]

    $$\DeclareMathOperator*{\argmax}{arg max} \mathbf{x}_{n+1} \approx \argmax_{\mathbf{x} \in \mathcal{S}_n} \max_{\mathbf{z} \in \mathcal{X}} I(\mathbb{1}\\{f(\mathbf{z}) \geq 0\\}; y(\mathbf{x}) \mid \mathcal{D}_n).$$

    [1] Bottero, Alessandro, et al. "Information-Theoretic Safe Exploration with Gaussian Processes." Advances in Neural Information Processing Systems 35 (2022): 30707-30719.
    """

    def F(self) -> Float[Array, "N"]:
        assert self.model.q == self.model.first_constraint + 1
        assert isinstance(self.model.noise_rate, HomoscedasticNoise)

        return _ise(
            mean=self.model.means[self.model.first_constraint],
            variance=self.model.variances[self.model.first_constraint],
            noise_std=self.model.noise_rate._noise_rates[self.model.first_constraint],
            sqd_cor=self.model.sqd_correlation(self.model.domain)[
                self.model.first_constraint
            ],
            pessimistic_safe_set_idx=set_to_idx(self.model.pessimistic_safe_set),
        )


class ISEBO(DiscreteGreedyAlgorithm[MarginalModel]):
    r"""
    **Information-theoretic Safe Exploration and Optimization (ISE-BO)** [1]

    [1] Bottero, Alessandro, et al. "Information-Theoretic Safe Bayesian Optimization." Preprint (2024).
    """

    k: int

    def __init__(
        self,
        model: MarginalModel,
        k: ScalarInt,
        sample_region: Float[Array, "m d"] | None = None,
    ):
        super().__init__(model, sample_region)
        self.k = int(k)

    def F(self) -> Float[Array, "N"]:
        assert self.model.q == self.model.first_constraint + 1
        assert isinstance(self.model.noise_rate, HomoscedasticNoise)

        ise = _ise(
            mean=self.model.means[self.model.first_constraint],
            variance=self.model.variances[self.model.first_constraint],
            noise_std=self.model.noise_rate._noise_rates[self.model.first_constraint],
            sqd_cor=self.model.sqd_correlation(self.model.domain)[
                self.model.first_constraint
            ],
            pessimistic_safe_set_idx=set_to_idx(self.model.pessimistic_safe_set),
        )

        J = jnp.arange(self.model.first_constraint, self.model.q)
        optima = self.model.thompson_sampling_values(k=self.k, J=J)
        mes = _mes(
            mean=self.model.means[0], stddev=self.model.stddevs[0], optima=optima
        )

        return jnp.maximum(ise, mes)
