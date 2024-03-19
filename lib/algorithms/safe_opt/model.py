from typing import List
import numpy as np
from jax import vmap
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from matplotlib.patches import Patch
from scipy.spatial import distance
from lib.function.synthetic import SyntheticFunction
from lib.gp.gaussian_distribution import GaussianDistribution
from lib.gp.prior import prior_distr
from lib.model.continuous import ContinuousModel
from lib.model.marginal import MarginalModel
from lib.plotting import plot_region
from lib.typing import ScalarBool, ScalarInt
from lib.utils import set_to_idx


class SafeOptModel(MarginalModel):
    L: Float[Array, "q"]
    r"""Lipschitz constants of $h_i, \; i \in [q]$."""
    safeopt_safe_set: Bool[Array, "N"]
    """(Pessimistic) safe set maintained by SafeOpt."""

    def __init__(
        self,
        L: Float[Array, "q"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.L = L
        self.safeopt_safe_set = self.pessimistic_safe_set
        self.update_safeopt_expanders()

    @property
    def operating_safe_set(self) -> Bool[Array, "n"]:
        """The safe set used for greedily selecting the next point."""
        return self.safeopt_safe_set

    @property
    def safeopt_maximizers(self) -> Bool[Array, "N"]:
        """Set of points within `X` which are pessimistically safe and potentially better than the worst-case (pessimistically safe) maximum."""
        return self.safeopt_safe_set & (self.u[0] >= self.max_l)

    def update_safeopt_expanders(self):
        # pairwise distances of all points with points outside safe set
        dist = jnp.array(
            distance.cdist(
                self.domain, self.domain[~self.safeopt_safe_set], metric="cityblock"
            )
        )

        # upper confidence bounds of all points, repeated `dist.shape[1]`-times.
        U = vmap(lambda i: jnp.tile(self.u[i].reshape(1, -1).T, (1, dist.shape[1])))(
            jnp.arange(self.first_constraint, self.q)
        )

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

        # matrix denoting the pairs of points constituting an expander
        cond = jnp.all(U - Ldist >= 0, axis=0)

        self.safeopt_expanders = jnp.any(cond, axis=1) & self.safeopt_safe_set

    def update_safeopt_safe_set(self):
        # pairwise distances of points within safe set with all points
        dist = jnp.array(
            distance.cdist(
                self.domain[self.safeopt_safe_set], self.domain, metric="cityblock"
            )
        )

        # lower confidence bounds of all points within safe set, repeated `dist.shape[1]`-times.
        L = vmap(
            lambda i: jnp.tile(
                self.l[i, self.safeopt_safe_set].reshape(1, -1).T,
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
        cond = jnp.all(L - Ldist >= 0, axis=0)

        self.safeopt_safe_set = jnp.any(cond, axis=0) | self.safeopt_safe_set

    def step(self, X: Float[Array, "m d"], y: Float[Array, "q m"]):
        super().step(X, y)
        self.update_safeopt_safe_set()
        self.update_safeopt_expanders()

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

    def _plot_safe_set_2d(
        self, ax, f: SyntheticFunction, interpolation: str
    ) -> List[Patch]:
        assert self.d == 2, "Dimension has to be 2."
        n = jnp.power(self.n, 1 / self.d).astype(int)

        colors = ["#606060", "#939393", "#FFFFFF"]
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
            f.safe_set(self.domain).reshape(n, n).T,
            color=colors[1],
            background_color=colors[-1],
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
        ]


class SafeOptContinuousModel(ContinuousModel):
    L: Float[Array, "q"]
    prior_n_samples: int

    def __init__(
        self,
        L: Float[Array, "q"],
        prior_n_samples: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.L = L
        self.prior_n_samples = prior_n_samples

    def marginalize(self, X: Float[Array, "n d"]) -> SafeOptModel:
        """Requires that `self.X` is a subset of `X`."""
        model = SafeOptModel(
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
        )
        model.step(self.X[self.prior_n_samples :], self.y[:, self.prior_n_samples :])
        return model


class HeuristicSafeOptModel(MarginalModel):
    """
    **Heuristic SafeOpt**

    Variant of SafeOpt which does not depend on the Lipschitz constant.
    The safe set is expanded based on the confidence intervals and the set of maximizers defined accordingly as the set of points which are (pessimistically) safe and potentially optimal.
    The set of expanders is defined as the set of pessimistically safe points who, if added with their largest possible value to the GP, lead to the expansion of the safe set.
    """

    @property
    def safeopt_maximizers(self) -> Bool[Array, "N"]:
        """Set of points within domain which are pessimistically safe and potentially better than the worst-case (pessimistically safe) maximum."""
        return self.operating_safe_set & (self.u[0] >= self.max_l)

    @property
    def safeopt_expanders(self) -> Bool[Array, "N"]:
        safe_set = self.operating_safe_set
        u = self.u
        result = np.zeros(self.n, dtype=bool)

        # check for each safe point if an expander
        for idx in set_to_idx(safe_set):
            # check if expander for all constraints
            for i in range(self.first_constraint, self.q):
                # Add safe point with its max possible value to the gp
                obs_idx = idx.reshape(-1)
                posterior_distr = self.distr(i).posterior(
                    obs_idx=obs_idx,
                    obs=u[i, obs_idx],
                    obs_noise_std=self.noise_rate.at(obs_idx)[i],
                )

                # Prediction of previously unsafe points based on that
                mean = posterior_distr.mean[~safe_set]
                std = posterior_distr.stddev[~safe_set]
                l = mean - self.beta(self.t)[i] * std

                # If any unsafe lower bound is suddenly above fmin then the point is an expander
                result[idx] = jnp.any(l >= 0)

                # Break if one safety GP is not an expander
                if not result[idx]:
                    break
        return jnp.array(result).reshape(-1)


class HeuristicSafeOptContinuousModel(ContinuousModel):
    def marginalize(self, X: Float[Array, "n d"]) -> HeuristicSafeOptModel:
        model = HeuristicSafeOptModel(
            key=self.acquire_key(),
            domain=X,
            distrs=self.distrs(X),
            beta=self.beta,
            noise_rate=self.noise_rate,
            use_objective_as_constraint=self.first_constraint == 0,
            t=self.t,
        )
        return model
