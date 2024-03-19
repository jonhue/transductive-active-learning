from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TypedDict
from jax import vmap
from jaxtyping import Array, Float
import jax.random as jr

from lib.gp import means
from lib.typing import KeyArray, ScalarFloat
from lib.utils import estimate_L


class Kernel(ABC):
    @abstractmethod
    def __call__(self, x: Float[Array, "d"], y: Float[Array, "d"]) -> ScalarFloat:
        pass

    def cross_covariance(
        self, X: Float[Array, "n d"], Y: Float[Array, "m d"]
    ) -> Float[Array, "n m"]:
        f_vmap1 = vmap(self, in_axes=(0, None))
        f_vmap2 = vmap(f_vmap1, in_axes=(None, 0))
        return f_vmap2(X, Y)  # type: ignore

    def covariance(self, X: Float[Array, "n d"]) -> Float[Array, "n n"]:
        return self.cross_covariance(X, X)

    def estimate_L(
        self, key: KeyArray, n_samples: int, X: Float[Array, "n d"]
    ) -> ScalarFloat:
        subkey1, subkey2 = jr.split(key)
        mean = means.ZeroMean()
        vals = jr.multivariate_normal(
            subkey1, mean.vector(X), self.covariance(X), method="svd"
        )
        return estimate_L(subkey2, n_samples=n_samples, domain=X, f=vals)


class Parameters(TypedDict):
    pass


P = TypeVar("P", bound=Parameters)


class Parameterized(Kernel, Generic[P]):
    def __init__(self, **params):
        self.params = {
            **self.__class__.default_params,  # type: ignore
            **(params if params is not None else {}),
        }

    @classmethod
    @property
    @abstractmethod
    def default_params(cls) -> P:
        pass
