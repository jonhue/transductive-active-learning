from abc import ABC, abstractmethod
from jax import vmap
from jaxtyping import Array, Float

from lib.typing import ScalarFloat


class Mean(ABC):
    @abstractmethod
    def __call__(self, x: Float[Array, "d"]) -> ScalarFloat:
        pass

    def vector(self, X: Float[Array, "n d"]) -> Float[Array, "n"]:
        return vmap(self)(X)  # type: ignore


class ZeroMean(Mean):
    def __call__(self, x: Float[Array, "d"]) -> ScalarFloat:
        return 0


class ConstantMean(Mean):
    value: ScalarFloat

    def __init__(self, value: ScalarFloat):
        self.value = value

    def __call__(self, x: Float[Array, "d"]) -> ScalarFloat:
        return self.value
