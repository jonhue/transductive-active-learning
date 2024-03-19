import jax.numpy as jnp
from jaxtyping import Array, Float
from lib.model.continuous import ContinuousModel
from lib.model.marginal import MarginalModel
from lib.typing import ScalarFloat


class TruVarModel(MarginalModel):
    eta: ScalarFloat
    r"""$\eta > 0$. Variance threshold."""
    delta: ScalarFloat
    r"""$\delta > 0$. When the maximum variance in ROI is smaller than $(1 + \delta) \eta$, the variance threshold is decreased."""
    r: ScalarFloat
    r"""$r \in (0,1)$. Variance decrease ratio."""

    def __init__(
        self,
        eta: ScalarFloat,
        delta: ScalarFloat,
        r: ScalarFloat,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eta = eta
        self.delta = delta
        self.r = r

    def update_variance_threshold(self, roi: Float[Array, "m d"]):
        if jnp.all(
            jnp.sqrt(self.beta(self.t)[0]) * self.stddevs[:, self.get_indices(roi)]
            <= (1 + self.delta) * self.eta
        ):
            self.eta = self.r * self.eta


class TruVarContinuousModel(ContinuousModel):
    eta: ScalarFloat
    delta: ScalarFloat
    r: ScalarFloat

    def __init__(
        self,
        eta: ScalarFloat,
        delta: ScalarFloat,
        r: ScalarFloat,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eta = eta
        self.delta = delta
        self.r = r

    def marginalize(self, X: Float[Array, "n d"]) -> TruVarModel:
        model = TruVarModel(
            eta=self.eta,
            delta=self.delta,
            r=self.r,
            key=self.acquire_key(),
            domain=X,
            distrs=self.distrs(X),
            beta=self.beta,
            noise_rate=self.noise_rate,
            use_objective_as_constraint=self.first_constraint == 0,
            t=self.t,
        )
        return model
