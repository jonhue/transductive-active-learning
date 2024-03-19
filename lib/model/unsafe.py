from functools import partial
from jax import jit
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from lib.model.continuous import ContinuousModel
from lib.model.marginal import MarginalModel


class UnsafeModel(MarginalModel):
    @property
    @partial(jit, static_argnums=0)
    def operating_safe_set(self) -> Bool[Array, "n"]:
        return jnp.ones((self.n,), dtype=bool)


class UnsafeContinuousModel(ContinuousModel):
    def marginalize(self, X: Float[Array, "n d"]) -> UnsafeModel:
        model = UnsafeModel(
            key=self.acquire_key(),
            domain=X,
            distrs=self.distrs(X),
            beta=self.beta,
            noise_rate=self.noise_rate,
            use_objective_as_constraint=self.first_constraint == 0,
        )
        model._t = self.t
        return model
