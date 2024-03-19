from typing import Callable, Tuple
from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, Int
from tqdm import tqdm
from lib.typing import ScalarBool, ScalarFloat


@jit
def solve_linear_system(
    A: Float[Array, "n m"], B: Float[Array, "n k"]
) -> Tuple[Float[Array, "m k"], Float[Array, "m n"]]:
    r"""
    Solves the linear system $\mathbf{A} \mathbf{X} = \mathbf{B}$ for $\mathbf{X} \in \mathbb{R}^{m \times k}$ where $\mathbf{A} \in \mathbb{R}^{n \times m}$ and $\mathbf{B} \in \mathbb{R}^{n \times k}$.
    Assumes that $\mathbf{A}$ is positive definite.

    Returns the solution $\mathbf{X}$ and the Cholesky factor $\mathbf{L}$.

    More stable than `jnp.linalg.solve(A, B)`.
    """
    L = jnp.linalg.cholesky(A)
    X = jnp.linalg.solve(L.T, jnp.linalg.solve(L, B))
    return X, L


def grad_approx(
    x: Float[Array, "d"],
    f: Callable[[Float[Array, "n d"]], Float[Array, "n"]],
    h: float = 1e-3,
):
    r"""
    Approximate the gradient of `f` at `x` using finite differences.
    """
    d = x.shape[0]
    I = jnp.eye(d)
    X = jnp.concatenate((x + h * I, x - h * I, x.reshape(1, -1)), axis=0)
    sample = f(X)

    grad = (sample[:d] - sample[d : 2 * d]) / (2.0 * h)
    return grad


def noise_covariance_matrix(
    k: int, noise_std: Float[Array, "k"] | float | None, default: float = 0
) -> Float[Array, "k k"]:
    """Return a diagonal matrix of noise variances."""
    if noise_std is None:
        noise_std = default
    return jnp.identity(k) * jnp.square(noise_std)


def get_indices(X: Float[Array, "n d"], A: Float[Array, "m d"]) -> Int[Array, "m"]:
    """Returns indices of points in `A` within `X`. Assumes that `A` is a subset of `X`."""
    distances = jnp.linalg.norm(X[:, None] - A, axis=2)
    return jnp.argmin(distances, axis=0)


def get_mask(X: Float[Array, "n d"], A: Float[Array, "m d"]) -> Bool[Array, "n"]:
    """Returns boolean mask of points in `A` within `X`. Assumes that `A` is a subset of `X`."""
    n = X.shape[0]
    return jnp.zeros((n,), dtype=bool).at[get_indices(X, A)].set(True)


def is_subset_of(X: Float[Array, "n d"], A: Float[Array, "m d"]) -> ScalarBool:
    """Returns `True` iff `X` is a subset of `A`."""
    if X.shape[0] == 0:
        return A.shape[0] == 0
    if A.shape[0] == 0:
        return False
    return jnp.all(jnp.isin(X, A))


def set_to_idx(arr: Float[Array, "N"]) -> Float[Array, "M"]:
    return jnp.where(arr)[0]


def std_error(arr: Array, axis: int) -> Array:
    return jnp.std(arr, axis=axis) / jnp.sqrt(arr.shape[axis])


class Dataset:
    X: Float[Array, "n d"] | None
    y: Float[Array, "n q"] | None

    def __init__(
        self, X: Float[Array, "n d"] | None = None, y: Float[Array, "n q"] | None = None
    ):
        self.X = X
        self.y = y


def estimate_L(
    key,
    n_samples: int,
    domain: Float[Array, "n d"],
    func: Callable[[Float[Array, "m d"]], Float[Array, "m"]] | None = None,
    f: Float[Array, "n"] | None = None,
    progress=True,
) -> ScalarFloat:
    """Estimates the Lipschitz constant of `func` on `domain` using `n_samples`."""
    n = domain.shape[0]
    if func is not None:
        f = func(domain)
    assert f is not None

    result = jnp.array(0)
    subkeys = jr.split(key, num=n_samples)
    for t in tqdm(range(n_samples), disable=not progress):
        i, j = jr.shuffle(subkeys[t], jnp.arange(n))[:2]
        L = jnp.abs(f[i] - f[j]) / jnp.sum(jnp.abs(domain[i] - domain[j]))
        if result < L:
            result = L
    return result
