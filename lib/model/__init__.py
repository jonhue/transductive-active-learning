from abc import ABC, abstractmethod
from typing import Callable
from jaxtyping import Array, Bool, Float
import jax.numpy as jnp
import jax.random as jr
from lib.noise import Noise


class Model(ABC):
    r"""
    **Gaussian process** statistical model of $\mathbf{f}$.

    Assumes that observation noise is Gaussian.
    """

    q: int
    r"""Number of unknown functions: $|\mathcal{I}|$."""
    first_constraint: int
    """Index of function acting as the first constraint."""
    beta: Callable[[int], Float[Array, "q"]]
    """Scaling factor of confidence bounds. Is a function of $t$."""
    noise_rate: Noise
    """(Assumed) noise standard deviations."""

    def __init__(
        self,
        key: jr.KeyArray,
        q: int,
        beta: Float[Array, "q"] | Callable[[int], Float[Array, "q"]],
        noise_rate: Noise,
        use_objective_as_constraint: bool = False,
    ):
        r"""
        :param key: Randomization key.
        :param q: Number of unknown functions.
        :param beta: Scaling factor of confidence bounds. Either a constant or a function of $t$.
        :param noise_rate: (Assumed) noise standard deviations.
        :param use_objective_as_constraint: If `true`, also treats $f_1$ as a constraint.
        """
        self._key = key
        self.q = q
        self.first_constraint = 0 if use_objective_as_constraint else 1
        self.beta = (lambda t: beta) if not callable(beta) else beta
        self.noise_rate = noise_rate

    def acquire_key(self) -> jr.KeyArray:
        self._key, key = jr.split(self._key)
        return key

    @property
    def constraint_set(self) -> Bool[Array, "q"]:
        r"""Set of constraints: $\mathcal{I}_s$."""
        return jnp.arange(self.q) >= self.first_constraint

    @property
    @abstractmethod
    def t(self) -> int:
        """Time step. Initially, $t=0$."""
        pass

    @abstractmethod
    def step(
        self,
        X: Float[Array, "m d"],
        y: Float[Array, "q m"],
    ):
        r"""Updating the statistical model with $m$ new observations `y` of $\mathbf{f}$ at points `X`."""
        pass
