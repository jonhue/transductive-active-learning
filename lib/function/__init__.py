from abc import ABC, abstractmethod
from jaxtyping import Array, Float


class Function(ABC):
    r"""Wrapper around the unknown function $\mathbf{f} = [f_1, \dots, f_{q}]^\top$."""

    q: int
    r"""Number of unknown functions: $|\mathcal{I}|$."""
    first_constraint: int
    """Index of function acting as the first constraint."""

    @abstractmethod
    def observe(self, X: Float[Array, "m d"]) -> Float[Array, "q m"]:
        r"""Noisy observation of $\mathbf{f}$ at `X`."""
        pass
