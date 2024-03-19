from functools import partial
from typing import Callable, List
from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
from lib.gp.gaussian_distribution import GaussianDistribution
from lib.gp.kernels import Kernel
from lib.gp.means import Mean, ZeroMean
from lib.model import Model
from lib.model.marginal import MarginalModel
from lib.noise import Noise


class ContinuousModel(Model):
    r"""
    **Gaussian process** statistical model of $\mathbf{f}$ when the domain $\mathcal{X}$ is continuous.

    Finite multivariate Gaussians are recomputed dynamically, conditioning on the entire history of observations.

    Assumes that observation noise is Gaussian.
    The lower/upper confidence bounds are not guaranteed to behave monotonically over time.
    """

    X: Float[Array, "t d"]
    """Observed points."""
    y: Float[Array, "q t"]
    """Observations."""

    def __init__(
        self,
        key: jr.KeyArray,
        d: int,
        prior_kernels: List[Kernel],
        beta: Float[Array, "q"] | Callable[[int], Float[Array, "q"]],
        noise_rate: Noise,
        use_objective_as_constraint: bool = False,
        X: Float[Array, "t d"] | None = None,
        y: Float[Array, "q t"] | None = None,
        prior_means: List[Mean] | None = None,
    ):
        q = len(prior_kernels)
        super().__init__(key, q, beta, noise_rate, use_objective_as_constraint)

        if prior_means is None:
            prior_means = [ZeroMean() for i in range(q)]
        self._prior_means = prior_means
        self._prior_kernels = prior_kernels

        self.X = X if X is not None else jnp.empty((0, d))
        self.y = y if y is not None else jnp.empty((q, 0))

    def marginalize(self, X: Float[Array, "n d"]) -> MarginalModel:
        model = MarginalModel(
            key=self.acquire_key(),
            domain=X,
            distrs=self.distrs(X),
            beta=self.beta,
            noise_rate=self.noise_rate,
            use_objective_as_constraint=self.first_constraint == 0,
            t=self.t,
        )
        return model

    @property
    def t(self) -> int:
        return self.X.shape[0]

    def distr(self, X: Float[Array, "n d"], i: int) -> GaussianDistribution:
        """Distribution of function `f_i` limited to points in `X`."""

        def engine(t: int, X: Float[Array, "n d"]) -> GaussianDistribution:
            n = X.shape[0]
            X_train = self.X[:t]
            y_train = self.y[i, :t]
            X_dom = jnp.concatenate((X, X_train), axis=0)
            prior = GaussianDistribution.initialize(
                mean=self._prior_means[i], kernel=self._prior_kernels[i], X=X_dom
            )
            posterior = prior.posterior(
                obs_idx=n + jnp.arange(t),
                obs=y_train,
                obs_noise_std=self.noise_rate.at(X_train)[i],
            )
            return posterior.sub_distr(jnp.arange(n))

        return engine(t=self.t, X=X)

    def distrs(self, X: Float[Array, "n d"]) -> List[GaussianDistribution]:
        """Distributions limited to points in `X`."""
        return [self.distr(X, i) for i in range(self.q)]

    def step(self, X: Float[Array, "m d"], y: Float[Array, "q m"]):
        self.X = jnp.concatenate((self.X, X), axis=0)
        self.y = jnp.concatenate((self.y, y), axis=1)
