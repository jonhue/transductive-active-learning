import math
from typing import Callable
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float
from matplotlib import gridspec
import matplotlib.pyplot as plt
from lib.function import Function
from lib.noise import Noise, NoiseOracle
from lib.typing import KeyArray, ScalarFloat
from lib.utils import Dataset, estimate_L


class SyntheticFunction(Function):
    def __init__(
        self,
        key: KeyArray,
        q: int,
        f: Callable[[Float[Array, "d"]], Float[Array, "q"]],
        noise_rate: Noise,
        noise_oracle: NoiseOracle,
        use_objective_as_constraint: bool = False,
    ):
        r"""
        :param key: Randomization key.
        :param q: Number of unknown functions.
        :param f: Function returning (noisy) evaluations of $f_i \;\colon \mathcal{X} \to \mathbb{R}$.
        :param noise_rate: Noise rates.
        :param noise_oracle: Function that given a `key` and a `shape` returns an `ndarray` of independent (unscaled) noise variables $\epsilon_j$.
        :param use_objective_as_constraint: If `true`, also treats $f_1$ as a constraint.
        """
        self._key = key
        self.q = q
        self.first_constraint = 0 if use_objective_as_constraint else 1
        self._f = f
        self._noise_rate = noise_rate
        self._noise_oracle = noise_oracle

    def acquire_key(self) -> KeyArray:
        self._key, key = jr.split(self._key)
        return key

    def evaluate(self, X: Float[Array, "m d"]) -> Float[Array, "q m"]:
        r"""Evaluates $\mathbf{f}$ at `X`."""
        return vmap(self._f)(X).T

    def observe(self, X: Float[Array, "m d"]) -> Float[Array, "q m"]:
        key = self.acquire_key()
        m = X.shape[0]
        noise = self._noise_oracle(key, (self.q, m))
        return self.evaluate(X) + self._noise_rate.at(X) * noise

    def safe_set(self, X: Float[Array, "n d"]) -> Bool[Array, "n"]:
        """Boolean vector indicating which points of `X` are safe."""
        return jnp.all(self.evaluate(X)[self.first_constraint :] >= 0, axis=0)

    def max_within(self, X: Float[Array, "n d"]) -> ScalarFloat:
        """Maximum of objective function $f_1$ within `X`."""
        return jnp.max(self.evaluate(X)[0], initial=-jnp.inf)

    def max(self, X: Float[Array, "n d"]) -> ScalarFloat:
        """Safe maximum of objective function $f_1$ within `X`."""
        return self.max_within(X[self.safe_set(X)])

    def argmax_within(self, X: Float[Array, "n d"]) -> Float[Array, "d"]:
        """Maximum of objective function $f_1$ within `X`."""
        idx = jnp.argmax(self.evaluate(X))
        if len(idx.shape) > 1:
            idx = idx[0]
        return X[idx]

    def argmax(self, X: Float[Array, "n d"]) -> Float[Array, "d"]:
        """Safe maximum of objective function $f_1$ within `X`."""
        return self.argmax_within(X[self.safe_set(X)])

    def estimate_L(
        self, domain: Float[Array, "n d"], sample_frac=0.01
    ) -> Float[Array, "q"]:
        """Estimates Lipschitz constants of objective function and constraint functions. Lower bounds of the true Lipschitz constants."""
        n = domain.shape[0]
        n_samples = math.ceil(sample_frac * n**2)
        key = self.acquire_key()
        self.L_est = jnp.array(
            [
                estimate_L(key, n_samples, domain=domain, f=self.evaluate(domain)[i])
                for i in range(self.q)
            ]
        )
        return self.L_est

    def plot_1d(self, domain: Float[Array, "n d"], D: Dataset | None = None):
        assert domain.shape[1] == 1, "Dimension has to be 1."

        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])

        for i in range(self.q):
            label = "Objective" if i == 0 else f"Constraint {i}"
            y = self.evaluate(domain)[i]
            ax.plot(domain, y, label=label)
        ax.set_title("Ground Truth")

        if D is not None and D.X is not None and D.y is not None:
            ax.scatter(D.X, D.y, color="tab:orange")

        plt.close()
        return fig

    def plot_2d(self, domain: Float[Array, "n d"]):
        assert domain.shape[1] == 2, "Dimension has to be 2."
        n = jnp.power(domain.shape[0], 1 / domain.shape[1]).astype(int)

        n_rows = math.ceil(self.q / 2)
        fig = plt.figure(tight_layout=True, figsize=(12, 6 * n_rows))
        gs = gridspec.GridSpec(n_rows, 2)

        xx, yy = domain.T.reshape(2, n, n)
        for i in range(self.q):
            ax = fig.add_subplot(gs[i])
            ax.set_title("Objective" if i == 0 else f"Constraint {i}")
            zz = self.evaluate(domain)[i].reshape(n, n)
            ax.contourf(xx, yy, zz)
            mesh = ax.pcolormesh(xx, yy, zz)
            plt.colorbar(mesh, ax=ax)
            if i >= self.first_constraint:
                boundary = ax.contour(xx, yy, zz, [0], cmap="Greys")
                ax.clabel(boundary)

        plt.close()
        return fig

    def plot_sigma_1d(self, domain: Float[Array, "n d"], no_title=True):
        assert domain.shape[1] == 1, "Dimension has to be 1."

        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(domain, self._noise_rate.at(domain)[0])
        if not no_title:
            ax.set_title("Noise Rate")
        plt.close()
        return fig

    def plot_sigma_2d(self, domain: Float[Array, "n d"], no_title=True):
        assert domain.shape[1] == 2, "Dimension has to be 2."
        n = jnp.power(domain.shape[0], 1 / domain.shape[1]).astype(int)

        n_rows = math.ceil(self.q / 2)
        fig = plt.figure(tight_layout=True, figsize=(12, 6 * n_rows))
        gs = gridspec.GridSpec(n_rows, 2)

        xx, yy = domain.T.reshape(2, n, n)
        for i in range(self.q):
            ax = fig.add_subplot(gs[i])
            if not no_title:
                ax.set_title("Objective" if i == 0 else f"Constraint {i}")
            zz = self._noise_rate.at(domain)[i].reshape(n, n)
            ax.contourf(xx, yy, zz)
            mesh = ax.pcolormesh(xx, yy, zz)
            plt.colorbar(mesh, ax=ax)

        plt.close()
        return fig
