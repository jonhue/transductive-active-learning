from typing import NotRequired
from jaxtyping import Array, Float
from lib.gp.kernels.base import Parameterized, Parameters
from lib.typing import ScalarFloat


class LinearParameters(Parameters):
    variance: NotRequired[ScalarFloat]


class Linear(Parameterized[LinearParameters]):
    def __call__(self, x: Float[Array, "d"], y: Float[Array, "d"]) -> ScalarFloat:
        return self.params["variance"] * x.T @ y

    @classmethod
    @property
    def default_params(cls) -> LinearParameters:
        return {
            "variance": 1.0,
        }
