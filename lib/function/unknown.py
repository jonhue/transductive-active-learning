from typing import Callable
from jax import vmap
import jax.numpy as jnp
from jaxtyping import Array, Float
from lib.function import Function


class UnknownFunction(Function):
    def __init__(
        self,
        q: int,
        f: Callable[[Float[Array, "d"]], Float[Array, "q"]],
        use_objective_as_constraint: bool = False,
    ):
        r"""
        :param q: Number of unknown functions.
        :param f: Function returning (noisy) evaluations of $f_i \;\colon \mathcal{X} \to \mathbb{R}$.
        :param use_objective_as_constraint: If `true`, also treats $f_1$ as a constraint.
        """
        self.q = q
        self.first_constraint = 0 if use_objective_as_constraint else 1
        self._f = f

    def observe(self, X: Float[Array, "m d"]) -> Float[Array, "q m"]:
        m = X.shape[0]
        return vmap(lambda j: self._f(X[j]))(jnp.arange(m)).T
