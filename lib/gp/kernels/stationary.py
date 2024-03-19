from typing import NotRequired
import jax.numpy as jnp
from jaxtyping import Array, Float
from lib.gp.kernels.base import Parameterized, Parameters
from lib.typing import ScalarFloat


class StationaryParameters(Parameters):
    variance: NotRequired[ScalarFloat]
    lengthscale: NotRequired[ScalarFloat]


class Stationary(Parameterized[StationaryParameters]):
    @classmethod
    @property
    def default_params(cls) -> StationaryParameters:
        return {
            "variance": 1.0,
            "lengthscale": 1.0,
        }


class Gaussian(Stationary):
    def __call__(self, x: Float[Array, "d"], y: Float[Array, "d"]) -> ScalarFloat:
        return self.params["variance"] * jnp.exp(
            -0.5
            * jnp.square(jnp.linalg.norm((x - y), ord=2) / self.params["lengthscale"])
        )


class Laplace(Stationary):
    def __call__(self, x: Float[Array, "d"], y: Float[Array, "d"]) -> ScalarFloat:
        return self.params["variance"] * jnp.exp(
            -jnp.linalg.norm((x - y), ord=1) / self.params["lengthscale"]
        )


class Matern32(Stationary):
    def __call__(self, x: Float[Array, "d"], y: Float[Array, "d"]) -> ScalarFloat:
        tau = jnp.linalg.norm((x - y), ord=2) / self.params["lengthscale"]
        return (
            self.params["variance"]
            * (1.0 + jnp.sqrt(3.0) * tau)
            * jnp.exp(-jnp.sqrt(3.0) * tau)
        )


class Matern52(Stationary):
    def __call__(self, x: Float[Array, "d"], y: Float[Array, "d"]) -> ScalarFloat:
        tau = jnp.linalg.norm((x - y), ord=2) / self.params["lengthscale"]
        return (
            self.params["variance"]
            * (1.0 + jnp.sqrt(5.0) * tau + 5.0 / 3.0 * jnp.square(tau))
            * jnp.exp(-jnp.sqrt(5.0) * tau)
        )
