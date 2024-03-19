from typing import List
from warnings import warn
from jax import vmap
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
import jax.random as jr
from matplotlib.patches import Patch
from scipy.spatial import distance
from lib.algorithms.safe_opt.model import SafeOptModel
from lib.function.synthetic import SyntheticFunction
from lib.gp.prior import prior_distr
from lib.model.continuous import ContinuousModel
from lib.plotting import plot_region
from lib.typing import ScalarInt
from lib.utils import set_to_idx


class GoOSEModel(SafeOptModel):
    oracle_i: int | None
    r"""Index of point $\mathbf{x}^\ast$ proposed by the oracle."""
    epsilon: float
    """Convergence threshold for uncertainty."""
    safeopt_optimistic_safe_set: Bool[Array, "N"]
    """Optimistic safe set maintained by GoOSE."""

    def __init__(self, epsilon: float, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.safeopt_optimistic_safe_set = jnp.ones_like(
            self.safeopt_safe_set, dtype=bool
        )
        assert (
            self.q == self.first_constraint + 1
        ), "GoOSE can only handle a single constraint."
        self.oracle_i = None

    @property
    def unconverged_safe_set(self) -> Bool[Array, "N"]:
        r"""Set of points which are in `safeopt_safe_set` and whose confidence interval has size at least $\epsilon$."""
        return self.safeopt_safe_set & (
            self.u[self.first_constraint] - self.l[self.first_constraint] > self.epsilon
        )

    def priorities(self) -> Float[Array, "N"]:
        """Priority of points with respect to oracle point."""
        assert self.oracle_i is not None
        dist = jnp.array(
            distance.cdist(
                self.domain[self.oracle_i].reshape(1, -1),
                self.domain,
                metric="cityblock",
            )
        )
        return dist[0]

    @property
    def safeopt_potential_expanders(self) -> Bool[Array, "N"]:
        """Set of points which are in `safeopt_optimistic_safe_set` but not in `safeopt_safe_set`."""
        return self.safeopt_optimistic_safe_set & ~self.safeopt_safe_set

    def update_safeopt_optimistic_safe_set(self):
        # pairwise distances of points within safe set with all points
        dist = jnp.array(
            distance.cdist(
                self.domain[self.safeopt_optimistic_safe_set],
                self.domain,
                metric="cityblock",
            )
        )

        # lower confidence bounds of all points within safe set, repeated `dist.shape[1]`-times.
        U = vmap(
            lambda i: jnp.tile(
                self.u[i, self.safeopt_optimistic_safe_set].reshape(1, -1).T,
                (1, dist.shape[1]),
            )
        )(jnp.arange(self.first_constraint, self.q))

        # self.L * dist for each constraint
        Lip = jnp.repeat(
            jnp.repeat(
                self.L[self.first_constraint :].reshape(-1, 1, 1),
                dist.shape[-1],
                axis=-1,
            ),
            dist.shape[-2],
            axis=-2,
        )
        Ldist = Lip * dist

        # matrix denoting the pairs of points permitting expansion of the safe set
        cond = jnp.all(U - Ldist >= 0, axis=0)

        self.safeopt_optimistic_safe_set = (
            jnp.any(cond, axis=0) & self.safeopt_optimistic_safe_set
        )

    def potential_immediate_expanders(self) -> Bool[Array, "m k"]:
        # pairwise distances of all points with potential expanders
        dist = jnp.array(
            distance.cdist(
                self.domain,
                self.domain[self.safeopt_potential_expanders],
                metric="cityblock",
            )
        )

        # upper confidence bounds of all points, repeated `dist.shape[1]`-times.
        U = jnp.tile(
            self.u[self.first_constraint, self.safeopt_potential_expanders]
            .reshape(1, -1)
            .T,
            (1, dist.shape[0]),
        ).T

        # matrix denoting the pairs of points constituting an expander
        cond = U - self.L[self.first_constraint] * dist >= 0

        return cond[self.unconverged_safe_set, :]

    def objective(self, oracle: Float[Array, "N"]) -> Float[Array, "N"]:
        if (
            self.oracle_i is None or not self.safeopt_optimistic_safe_set[self.oracle_i]
        ):  # there is no oracle point or the oracle point is not safe
            self.oracle_i = jnp.argmax(
                oracle.at[~self.safeopt_optimistic_safe_set].set(-jnp.inf)
            ).item()
        if self.safeopt_safe_set[self.oracle_i]:  # the oracle point is in the safe set
            return jnp.eye(self.n, dtype=jnp.float64)[self.oracle_i]

        # safe expansion
        idx = self.priorities()[self.safeopt_potential_expanders].argsort()[
            ::-1
        ]  # descending order
        potential_immediate_expanders = self.potential_immediate_expanders()
        sorted_potential_immediate_expanders = potential_immediate_expanders[:, idx]
        i = first_true_index(sorted_potential_immediate_expanders)
        if i is None:  # fallback when no informative points exist
            warn("GoOSE could not find any informative point.")
            return jr.uniform(key=self.acquire_key(), shape=(self.n,))
        return jnp.eye(self.n, dtype=jnp.float64)[
            set_to_idx(self.unconverged_safe_set)[i]
        ]

    def add_new_points(self, X: Float[Array, "m d"], y: Float[Array, "q m"]):
        super().step(X, y)
        self.update_safeopt_optimistic_safe_set()

    def _plot_safe_set_1d(self, ax):
        assert self.d == 1, "Dimension has to be 1."
        ax.fill_between(
            self.domain.flatten(),
            jnp.min(self.l),
            jnp.max(self.u),
            where=self.safeopt_safe_set,
            facecolor="tab:grey",
            alpha=0.1,
            linewidth=0,
        )
        ax.fill_between(
            self.domain.flatten(),
            jnp.min(self.l),
            jnp.max(self.u),
            where=self.safeopt_optimistic_safe_set,
            facecolor="tab:grey",
            alpha=0.1,
            linewidth=0,
        )

    def _plot_safe_set_2d(
        self, ax, f: SyntheticFunction, interpolation: str
    ) -> List[Patch]:
        assert self.d == 2, "Dimension has to be 2."
        n = jnp.power(self.n, 1 / self.d).astype(int)

        colors = ["#606060", "#939393", "#C7C7C7", "#FFFFFF"]
        colors.reverse()
        plot_region(
            ax,
            self.domain,
            jnp.ones_like(self.domain, dtype=bool),
            color=colors[-1],
            interpolation=interpolation,
        )
        plot_region(
            ax,
            self.domain,
            self.safeopt_optimistic_safe_set.reshape(n, n).T,
            color=colors[2],
            background_color=colors[-1],
            interpolation=interpolation,
        )
        plot_region(
            ax,
            self.domain,
            f.safe_set(self.domain).reshape(n, n).T,
            color=colors[1],
            background_color=colors[2],
            interpolation=interpolation,
        )
        plot_region(
            ax,
            self.domain,
            self.safeopt_safe_set.reshape(n, n).T,
            color=colors[0],
            background_color=colors[1],
            interpolation=interpolation,
        )

        return [
            Patch(facecolor=colors[0], label="(Pessimistic) Safe Set"),
            Patch(facecolor=colors[1], label="True Safe Region"),
            Patch(facecolor=colors[2], label="Optimistic Safe Set"),
        ]


def first_true_index(mat: Bool[Array, "N M"]) -> ScalarInt | None:
    """Returns the row-index of the first `True` value in any column."""
    # Find the index of the first True value in each column
    row_idx = jnp.argmax(mat, axis=0)

    # Find the first column with a True value
    col_idx = jnp.argmax(jnp.diag(mat[row_idx]))

    if not mat[row_idx[col_idx], col_idx]:
        return None
    return row_idx[col_idx]


class GoOSEContinuousModel(ContinuousModel):
    L: Float[Array, "q"]
    epsilon: float
    prior_n_samples: int

    def __init__(
        self,
        L: Float[Array, "q"],
        epsilon: float,
        prior_n_samples: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.L = L
        self.epsilon = epsilon
        self.prior_n_samples = prior_n_samples

    def marginalize(self, X: Float[Array, "n d"]) -> GoOSEModel:
        """Requires that `self.X` is a subset of `X`."""
        model = GoOSEModel(
            key=self.acquire_key(),
            domain=X,
            distrs=prior_distr(
                noise_rate=self.noise_rate,
                domain=X,
                X=self.X[: self.prior_n_samples],
                y=self.y[:, : self.prior_n_samples],
                kernels=self._prior_kernels,
                means=self._prior_means,
            ),
            beta=self.beta,
            noise_rate=self.noise_rate,
            use_objective_as_constraint=self.first_constraint == 0,
            L=self.L,
            epsilon=self.epsilon,
        )
        model.step(self.X[self.prior_n_samples :], self.y[:, self.prior_n_samples :])
        return model
