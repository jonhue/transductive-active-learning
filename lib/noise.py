from abc import ABC, abstractmethod
from typing import Callable, List, Sequence
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from lib.typing import KeyArray


NoiseOracle = Callable[[KeyArray, Sequence[int]], Array]
"""Function returning (unscaled) noise with a provided shape."""

gaussian_noise: NoiseOracle = lambda key, shape: jr.normal(
    key, shape, dtype=jnp.float64
)
"""I.i.d. standard Gaussian noise, returned with shape `shape`."""


class Noise(ABC):
    @abstractmethod
    def at(self, X: Float[Array, "n d"]) -> Float[Array, "q n"]:
        """
        Noise rate at every point in `X` (e.g., the noise standard deviation if noise is Gaussian).
        """
        pass


class HomoscedasticNoise(Noise):
    def __init__(self, q: int, noise_rates: Float[Array, "q"]):
        self.q = q
        self._noise_rates = noise_rates

    def at(self, X: Float[Array, "n d"]) -> Float[Array, "q n"]:
        n = X.shape[0]
        return jnp.tile(self._noise_rates, (n, 1)).T


class HeteroscedasticNoise(Noise):
    def __init__(
        self, noise_rates: List[Callable[[Float[Array, "n d"]], Float[Array, "n"]]]
    ):
        self.q = len(noise_rates)
        self._noise_rates = noise_rates

    def at(self, X: Float[Array, "n d"]) -> Float[Array, "q n"]:
        return jnp.array([self._noise_rates[i](X) for i in range(self.q)])
