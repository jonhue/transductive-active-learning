from functools import partial
import math
from typing import Callable, List
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, Int
from matplotlib import gridspec
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from tqdm import tqdm
from lib.function.synthetic import SyntheticFunction
from lib.gp.gaussian_distribution import GaussianDistribution
from lib.model import Model
from lib.noise import Noise
from lib.typing import KeyArray, ScalarBool, ScalarFloat, ScalarFloatArray, ScalarInt
from lib.utils import Dataset, estimate_L, get_indices, is_subset_of, set_to_idx
from lib.plotting import plot_region


class MarginalModel(Model):
    r"""
    **Gaussian process** statistical model of $\mathbf{f}$ when the domain $\mathcal{X}$ is finite.

    The corresponding (finite) multivariate Gaussians is being kept in memory as observations are made.

    Assumes that observation noise is Gaussian.
    """

    domain: Float[Array, "n d"]
    """Finite domain."""

    def __init__(
        self,
        key: KeyArray,
        domain: Float[Array, "n d"],
        distrs: List[GaussianDistribution],
        beta: Float[Array, "q"] | Callable[[int], Float[Array, "q"]],
        noise_rate: Noise,
        use_objective_as_constraint: bool = False,
        t: int = 0,
    ):
        q = len(distrs)
        super().__init__(key, q, beta, noise_rate, use_objective_as_constraint)

        self.domain = domain
        self._means = jnp.array([distr.mean for distr in distrs])
        self._covariances = jnp.array([distr.covariance for distr in distrs])
        self._t = t
        self._l = self._recompute_l()
        self._u = self._recompute_u()
        self._X = jnp.array([]).reshape(0, domain.shape[1])

    @property
    def n(self) -> int:
        return self.domain.shape[0]

    @property
    def d(self) -> int:
        return self.domain.shape[1]

    @property
    def t(self) -> int:
        return self._t

    @partial(jit, static_argnums=0)
    def get_indices(self, X: Float[Array, "n d"]) -> Int[Array, "n"]:
        return get_indices(self.domain, X)

    def distr(self, i: ScalarInt) -> GaussianDistribution:
        return GaussianDistribution(
            mean=self._means[i], covariance=self._covariances[i]
        )

    def masked_distr(
        self, i: ScalarInt, mask: Bool[Array, "n"]
    ) -> GaussianDistribution:
        mask_idx = set_to_idx(mask)
        return GaussianDistribution(
            mean=self._means[i, mask],
            covariance=self._covariances[i][jnp.ix_(mask_idx, mask_idx)],
        )

    @property
    def l(self) -> Float[Array, "q n"]:
        """Lower confidence bounds."""
        return self._l

    @property
    def u(self) -> Float[Array, "q n"]:
        """Upper confidence bounds."""
        return self._u

    @property
    def means(self) -> Float[Array, "q n"]:
        """Mean vectors."""
        return self._means

    @property
    def covariances(self) -> Float[Array, "q n n"]:
        """Covariance matrices."""
        return self._covariances

    @property
    def variances(self) -> Float[Array, "q n"]:
        """Variances."""
        return vmap(jnp.diag)(self.covariances)

    @property
    def stddevs(self) -> Float[Array, "q n"]:
        """Standard deviations."""
        return jnp.sqrt(self.variances)

    def entropy(self, jitter: float = 0.0) -> ScalarFloat:
        """Entropy, treating all functions as independent."""
        return jnp.sum(
            vmap(lambda i: self.distr(i).entropy(jitter=jitter))(jnp.arange(self.q))
        )

    def masked_entropy(
        self, mask: Bool[Array, "n"], jitter: float = 0.0
    ) -> ScalarFloat:
        """Entropy within `mask`, treating all functions as independent."""
        return jnp.sum(
            vmap(lambda i: self.masked_distr(i, mask).entropy(jitter=jitter))(
                jnp.arange(self.q)
            )
        )

    def sqd_correlation(
        self, X: Float[Array, "n d"], A: Float[Array, "m d"] | None = None
    ) -> Float[Array, "q m n"]:
        r"""
        Matrix of squared correlations of observations in `X` with points in `A` where the $k,l$-th entry is $\mathrm{Cor}[f_i(\mathbf{x}_k), y_i(\mathbf{x}_l) \mid \mathcal{D}_n]^2$ with $\mathbf{x}_k$ in `A` and $\mathbf{x}_l$ in `X`.

        If `A` is `None`, then `A` falls back to `X`.
        """
        if A is None:
            A = X

        n = X.shape[0]
        m = A.shape[0]
        X_idx = self.get_indices(X)
        A_idx = self.get_indices(A)
        XA_idx = jnp.concatenate((X_idx, A_idx))
        covariances_XA = self.covariances[jnp.ix_(jnp.arange(self.q), XA_idx, XA_idx)]
        sqd_cross_covariance = jnp.square(covariances_XA[:, :n, n:])
        obs_noise_var = jnp.square(self.noise_rate.at(X))
        variance_X = self.variances[:, X_idx]
        variance_A = self.variances[:, A_idx]

        def compute_sqd_correlation(i: ScalarInt, k: ScalarInt) -> Float[Array, "n"]:
            return sqd_cross_covariance[i, :, k] / (
                (variance_X[i] + obs_noise_var[i]) * variance_A[i, k]
            )

        return vmap(
            lambda i: vmap(lambda k: compute_sqd_correlation(i, k))(jnp.arange(m))
        )(jnp.arange(self.q))

    def _recompute_l(self) -> Float[Array, "q n"]:
        return self.means - jnp.expand_dims(self.beta(self.t), axis=1) * self.stddevs

    def _recompute_u(self) -> Float[Array, "q n"]:
        return self.means + jnp.expand_dims(self.beta(self.t), axis=1) * self.stddevs

    def calibration(self, f: SyntheticFunction) -> Bool[Array, "q n"]:
        """Determines where the current statistical model is well-calibrated."""
        f_X = f.evaluate(self.domain)
        return (self.l <= f_X) & (f_X <= self.u)

    @property
    def optimistic_safe_set(self) -> Bool[Array, "n"]:
        """Optimistic safe set according to the current model."""
        return jnp.all(self.u[self.first_constraint :] >= 0, axis=0)

    @property
    def pessimistic_safe_set(self) -> Bool[Array, "n"]:
        """Pessimistic safe set according to the current model."""
        return jnp.all(self.l[self.first_constraint :] >= 0, axis=0)

    @property
    def operating_safe_set(self) -> Bool[Array, "n"]:
        """The safe set used for greedily selecting the next point. Defaults to the pessimistic safe set."""
        return self.pessimistic_safe_set

    def safe_set_calibration(self, f: SyntheticFunction) -> Bool[Array, "n"]:
        """Determines where the optimistic/pessimistic safe sets are well-calibrated."""
        safe_set = f.safe_set(self.domain)
        return (~safe_set & ~self.pessimistic_safe_set) | (
            safe_set & self.optimistic_safe_set
        )

    def safe_set_is_subset_of(self, A: Float[Array, "m d"]) -> ScalarBool:
        """Returns `True` iff `operating_safe_set` is a subset of the points in `A`."""
        return is_subset_of(self.domain[self.operating_safe_set], A)

    @property
    def potential_expanders(self) -> Bool[Array, "n"]:
        """Set of points which are optimistically safe and pessimistically unsafe."""
        return self.optimistic_safe_set & ~self.pessimistic_safe_set

    def argmax_within(self, mask: Bool[Array, "n"]) -> Float[Array, "d"]:
        """Point maximizing the lower confidence bound within `mask`."""
        assert jnp.sum(mask) > 0
        return self.domain[mask][jnp.argmax(self.l[0, mask])]

    @property
    def argmax(self) -> Float[Array, "d"]:
        """Pessimistically safe point maximizing the lower confidence bound."""
        return self.argmax_within(self.operating_safe_set)

    def convergence_gap_within(self, mask: Bool[Array, "n"]) -> ScalarInt:
        """Returns the number of points within `mask` where the highest upper confidence bound is larger than the best known lower bound. Equals `1` at convergence."""
        return jnp.sum(self.u[0, mask] > jnp.max(self.l[0, mask], initial=-jnp.inf))

    @property
    def max_l(self) -> ScalarFloat:
        """Pessimistically safe maximum of lower confidence bounds."""
        return jnp.max(self.l[0, self.operating_safe_set], initial=-jnp.inf)

    @property
    def potential_maximizers(self) -> Bool[Array, "n"]:
        """Set of points which are optimistically safe and potentially better than the worst-case (pessimistically safe) maximum."""
        return self.optimistic_safe_set & (self.u[0] >= self.max_l)

    def sample(self, k: int, keys: KeyArray | None = None) -> Float[Array, "q k n"]:
        """Sample $k$ independent functions from the statistical model."""
        if keys is None:
            self._key, *keys = jr.split(self._key, num=1 + self.q)  # type: ignore
            assert keys is not None
        return jnp.array(
            [
                self.distr(i).sample(key=keys[i], sample_shape=(k,))
                for i in range(self.q)
            ]
        )

    def thompson_sampling(
        self,
        k: int,
        I: Int[Array, "r"] | None = None,
        J: Int[Array, "s"] | None = None,
        strict: bool = False,
    ) -> Bool[Array, "n"]:
        r"""
        Set of up to $k$ points each of which is a "safe maximum" of an independent sample from the current model.

        For each sample $\mathbf{f}^{(j)}$ with induced safe set $\mathcal{S}^{(j)}(J)$ (with respect to the constraint set `J`), we select $$\DeclareMathOperator*{\argmax}{arg max} \mathbf{x}_j = \argmax_{\mathbf{x} \in \mathcal{S}^{(j)}(J)} \min_{i \in I} f_i^{(j)}(\mathbf{x}).$$

        :param k: Number of samples.
        :param X: Domain.
        :param I: Set of objective functions. Must have length at least `1`. If `None`, defaults to `[0]`.
        :param J: Set of constraint functions. If `None`, defaults to `[]`.
        :param strict: If `True`, raises an error if the resulting set is empty (i.e., because the constraints were not satisfied by any of the samples). Defaults to `False`.
        """
        self._key, *subkeys = jr.split(self._key, num=1 + k * self.q)
        all_keys = jnp.array(subkeys).reshape(k, self.q, 2)
        if I is None:
            I = jnp.array([0])
        assert I.shape[0] > 0
        if J is None:
            J = jnp.zeros(self.q, dtype=bool)

        def engine(keys: Int[Array, "q 2"]) -> Bool[Array, "n"]:
            samples = self.sample(k=1, keys=keys)[:, 0, :]
            safety = jnp.all(samples[J] >= 0, axis=0)
            safe_optimum = jnp.max(
                jnp.min(jnp.where(safety, samples[I], -jnp.inf), axis=0)
            )
            optimality = jnp.min(samples[I], axis=0) >= jnp.repeat(
                safe_optimum, repeats=self.n
            )
            return safety & optimality

        result = jnp.any(vmap(engine)(all_keys), axis=0)
        if strict and jnp.sum(result) == 0:
            raise Exception("Thompson Sampling: No sample has a safe point.")
        return result

    def thompson_sampling_values(
        self,
        k: int,
        I: Int[Array, "r"] | None = None,
        J: Int[Array, "s"] | None = None,
    ) -> Float[Array, "k"]:
        r"""
        Safe maxima of $k$ independent samples from the current model.

        For each sample $\mathbf{f}^{(j)}$ with induced safe set $\mathcal{S}^{(j)}(J)$ (with respect to the constraint set `J`), we select $$f_j = \max_{\mathbf{x} \in \mathcal{S}^{(j)}(J)} \min_{i \in I} f_i^{(j)}(\mathbf{x}).$$

        :param k: Number of samples.
        :param X: Domain.
        :param I: Set of objective functions. Must have length at least `1`. If `None`, defaults to `[0]`.
        :param J: Set of constraint functions. If `None`, defaults to `[]`.
        """
        self._key, *subkeys = jr.split(self._key, num=1 + k * self.q)
        all_keys = jnp.array(subkeys).reshape(k, self.q, 2)
        if I is None:
            I = jnp.array([0])
        assert I.shape[0] > 0
        if J is None:
            J = jnp.zeros(self.q, dtype=bool)

        def engine(keys: Int[Array, "q 2"]) -> ScalarFloatArray:
            samples = self.sample(k=1, keys=keys)[:, 0, :]
            safety = jnp.all(samples[J] >= 0, axis=0)
            safe_optimum = jnp.max(
                jnp.min(jnp.where(safety, samples[I], -jnp.inf), axis=0)
            )
            return safe_optimum

        return vmap(engine)(all_keys)

    @property
    def ucb(self) -> Bool[Array, "n"]:
        """Indicator vector (optimistically safe) for UCB action."""
        safe_set_mask = self.optimistic_safe_set
        return safe_set_mask & (self.u[0] == jnp.max(self.u[0, safe_set_mask]))

    @property
    def max_u(self) -> ScalarFloat:
        """Upper bound to the maximum objective value within the true safe set."""
        return jnp.max(self.u[0, self.optimistic_safe_set], initial=-jnp.inf)

    @property
    def pessimistic_max_u(self) -> ScalarFloat:
        """Upper bound to the maximum objective value within the pessimistic safe set."""
        return jnp.max(self.u[0, self.pessimistic_safe_set], initial=-jnp.inf)

    def objective_argmax(self, F: Float[Array, "n"]) -> Float[Array, "d"]:
        """
        Returns safe point (according to `operating_safe_set`) maximizing `F`.
        In case multiple safe points maximize the objective, returns one of them uniformly at random.
        """
        safe_set = self.operating_safe_set
        if jnp.sum(safe_set) == 0:
            raise Exception("Operating safe set is empty. Cannot make observations.")
        if jnp.all(F == -jnp.inf):
            raise Exception("Objective is negative infinity everywhere.")
        safe_F = F.at[~safe_set].set(-jnp.inf)
        if jnp.all(jnp.isnan(safe_F[safe_set])):
            raise Exception("Objective is NaN everywhere.")
        safe_max = jnp.nanmax(safe_F)
        indices = jnp.argwhere(safe_F == safe_max).reshape(-1)

        key = self.acquire_key()
        index = jr.choice(key, indices)
        return self.domain[index]

    def step(self, X: Float[Array, "m d"], y: Float[Array, "q m"]):
        m = X.shape[0]
        X_idx = self.get_indices(X)
        noise_rate = self.noise_rate.at(X)

        def engine(i: ScalarInt) -> GaussianDistribution:
            return self.distr(i).posterior(
                obs_idx=X_idx,
                obs=y[i],
                obs_noise_std=noise_rate[i],
            )

        posterior_distrs = vmap(engine)(jnp.arange(self.q))
        self._means = posterior_distrs.mean
        self._covariances = posterior_distrs.covariance

        self._l = jnp.maximum(self._l, self._recompute_l())
        self._u = jnp.minimum(self._u, self._recompute_u())
        self._t += m
        self._X = jnp.concatenate([self._X, X], axis=0)

    def estimate_L(self, sample_frac=0.01, n_samples=100) -> Float[Array, "q"]:
        r"""Estimate Lipschitz constants based on `n_samples` from statistical model of $\mathbf{f}$."""
        L_est = []
        for i in range(self.q):
            print("Running for function", i, "...")
            L = 0
            key = self.acquire_key()
            samples = self.distr(i).sample(key=key, sample_shape=(n_samples,))
            for j in tqdm(range(n_samples)):
                n_samples_L = math.ceil(sample_frac * self.n**2)
                key = self.acquire_key()
                new_L = estimate_L(
                    key, n_samples_L, domain=self.domain, f=samples[j], progress=False
                )
                if new_L > L:
                    L = new_L
            L_est.append(L)
        return jnp.array(L_est)

    def _plot_safe_set_1d(self, ax):
        assert self.domain.shape[1] == 1, "Dimension has to be 1."
        ax.fill_between(
            self.domain.flatten(),
            jnp.min(self.l),
            jnp.max(self.u),
            where=self.pessimistic_safe_set,
            facecolor="tab:grey",
            alpha=0.1,
            linewidth=0,
        )
        ax.fill_between(
            self.domain.flatten(),
            jnp.min(self.l),
            jnp.max(self.u),
            where=self.optimistic_safe_set,
            color="tab:grey",
            alpha=0.1,
            linewidth=0,
        )

    def plot_1d(
        self,
        D: Dataset | None = None,
        n_samples: int = 0,
        f: SyntheticFunction | None = None,
        show_safe_set=True,
        title: str | None = None,
        legend: bool = True,
    ):
        assert self.domain.shape[1] == 1, "Dimension has to be 1."

        n_rows = math.ceil(self.q / 2)
        fig = plt.figure(tight_layout=True, figsize=(12, 6 * n_rows))
        gs = gridspec.GridSpec(n_rows, 2)

        for i in range(self.q):
            ax = fig.add_subplot(gs[i])
            if self.q > 1:
                subtitle = "Objective" if i == 0 else f"Constraint {i}"
                ax.set_title(f"{title} [{subtitle}]" if title is not None else subtitle)

            key = self.acquire_key()
            samples = self.distr(i).sample(key=key, sample_shape=(n_samples,)).T

            if show_safe_set:
                self._plot_safe_set_1d(ax)

            ax.plot(
                self.domain,
                samples,
                label="Samples",
                color="black",
                linestyle="--",
                alpha=0.5,
            )
            ax.plot(self.domain, self.means[i], label="Mean", color="tab:blue")
            if f is not None:
                ax.plot(
                    self.domain,
                    f.evaluate(self.domain).T,
                    label="Ground Truth",
                    color="black",
                    linestyle="--",
                )
            ax.fill_between(
                self.domain.flatten(),
                self.l.reshape(-1),  # type: ignore
                self.u.reshape(-1),  # type: ignore
                label="Uncertainty",
                color="tab:blue",
                alpha=0.3,
            )
            if i >= self.first_constraint:
                ax.axhline(y=0, color="black")
            if f is not None and i >= self.first_constraint:
                ax.plot(
                    self.domain,
                    -self.beta(self.t)[i] * self.noise_rate.at(self.domain)[i],
                    label="Safety Boundary",
                    color="black",
                    linestyle="dotted",
                )
            if D is not None and D.X is not None and D.y is not None:
                ax.scatter(D.X, D.y, color="tab:orange")

        if legend:
            fig.legend()
        plt.close()
        return fig

    def _plot_safe_set_2d(
        self, ax, f: SyntheticFunction, interpolation: str
    ) -> List[Patch]:
        assert self.domain.shape[1] == 2, "Dimension has to be 2."
        n = jnp.power(self.domain.shape[0], 1 / self.domain.shape[1]).astype(int)

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
            self.optimistic_safe_set.reshape(n, n).T,
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
            self.pessimistic_safe_set.reshape(n, n).T,
            color=colors[0],
            background_color=colors[1],
            interpolation=interpolation,
        )

        return [
            Patch(facecolor=colors[0], label="Pessimistic Safe Region"),
            Patch(facecolor=colors[1], label="True Safe Region"),
            Patch(facecolor=colors[2], label="Optimistic Safe Region"),
        ]

    def _plot_roi_2d(
        self,
        ax,
        roi: Bool[Array, "n"],
        sample_region: Bool[Array, "n"] | None,
        interpolation: str,
    ) -> List[Patch]:
        assert self.domain.shape[1] == 2, "Dimension has to be 2."
        n = jnp.power(self.domain.shape[0], 1 / self.domain.shape[1]).astype(int)

        colors = ["#74B3EB", "#606060", "#FFFFFF"]
        plot_region(
            ax,
            self.domain,
            roi.reshape(n, n).T,
            color=colors[0],
            background_color=colors[2],
            interpolation=interpolation,
        )
        if sample_region is not None:
            plot_region(
                ax,
                self.domain,
                sample_region.reshape(n, n).T,
                color=colors[1],
                background_color=colors[2],
                interpolation=interpolation,
                alpha=0.3,
            )

        lgd = [
            Patch(facecolor=colors[0], label="Region of Interest"),
        ]
        if sample_region is not None:
            lgd.append(Patch(facecolor="#D0D0D0", label="Sample Region"))
        return lgd

    def plot_2d(
        self,
        f: SyntheticFunction,
        X: Float[Array, "k 2"] | None = None,
        roi: Bool[Array, "n"] | None = None,
        sample_region: Bool[Array, "n"] | None = None,
        title="Safe Regions",
        interpolation: str = "bicubic",
        legend: bool = True,
    ):
        assert self.domain.shape[1] == 2, "Dimension has to be 2."

        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])

        if roi is not None:
            if jnp.sum(roi) == 1:
                ax.scatter(
                    self.domain[roi, 0],
                    self.domain[roi, 1],
                    marker="+",
                    color="#74B3EB",
                )
            handles = self._plot_roi_2d(ax, roi, sample_region, interpolation)
        else:
            handles = self._plot_safe_set_2d(ax, f, interpolation)

        if X is not None:
            ax.scatter(X[:, 0], X[:, 1], marker="+", color="black")  # type: ignore

        ax.invert_yaxis()
        ax.set_title(title)
        if legend:
            fig.legend(handles=handles)
        plt.close()
        return fig
