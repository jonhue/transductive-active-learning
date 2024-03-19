import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from lib.algorithms import DiscreteGreedyAlgorithm
from lib.algorithms.baselines import UniformSampling
from lib.algorithms.safe_opt.model import HeuristicSafeOptModel, SafeOptModel
from lib.model.marginal import MarginalModel


class SafeOpt(DiscreteGreedyAlgorithm[SafeOptModel | HeuristicSafeOptModel]):
    r"""
    **SafeOpt** [1]

    $$\DeclareMathOperator*{\argmax}{arg max} \mathbf{x}_{n+1} = \argmax_{\mathbf{x} \in \mathcal{G}_n \cup \mathcal{M}_n}\ \sigma_n(\mathbf{x})$$ where $\mathcal{G}_n \subseteq \mathcal{S}_n$ is a set of "expanders" and $\mathcal{M}_n \subseteq \mathcal{S}_n$ is a set of "maximizers".
    If $\mathcal{G}_n \cup \mathcal{M}_n = \emptyset$, selects $\mathbf{x} \in \mathcal{S}_n$ uniformly at random.

    [1] Berkenkamp, Felix, Angela P. Schoellig, and Andreas Krause. "Safe controller optimization for quadrotors with Gaussian processes." 2016 IEEE international conference on robotics and automation (ICRA). IEEE, 2016.
    """

    def F(self) -> Float[Array, "N"]:
        sample_region = self.model.safeopt_expanders | self.model.safeopt_maximizers
        return _safe_opt(model=self.model, sample_region=sample_region)


class SafeOptExploration(DiscreteGreedyAlgorithm[SafeOptModel | HeuristicSafeOptModel]):
    r"""
    Variant of **SafeOpt** focusing on expansion only:

    $$\DeclareMathOperator*{\argmax}{arg max} \mathbf{x}_{n+1} = \argmax_{\mathbf{x} \in \mathcal{G}_n}\ \sigma_n(\mathbf{x}).$$
    If $\mathcal{G}_n = \emptyset$, selects $\mathbf{x} \in \mathcal{S}_n$ uniformly at random.
    """

    def F(self) -> Float[Array, "N"]:
        sample_region = self.model.safeopt_expanders
        return _safe_opt(model=self.model, sample_region=sample_region)


def _safe_opt(model: MarginalModel, sample_region: Bool[Array, "N"]):
    if jnp.sum(sample_region) > 0:
        return jnp.sum(model.stddevs.T.at[~sample_region].set(-jnp.inf), axis=1)
    else:
        return UniformSampling(model=model).F()
