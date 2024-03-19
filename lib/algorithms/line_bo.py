from abc import abstractmethod
from jax import grad, vmap
import jax.numpy as jnp
import jax.random as jr
from typing import Callable, Tuple, Type
from jaxtyping import Array, Float
from lib.algorithms import (
    DirectedDiscreteGreedyAlgorithm,
    DiscreteGreedyAlgorithm,
    GreedyAlgorithm,
    ROIConstructor,
)
from lib.function import Function
from lib.model.continuous import ContinuousModel
from lib.model.marginal import MarginalModel
from lib.typing import ScalarFloat, ScalarInt
from lib.utils import grad_approx

AlgConstructor = Callable[[MarginalModel], DiscreteGreedyAlgorithm]


def alg_constructor(Alg: Type[DiscreteGreedyAlgorithm], **args) -> AlgConstructor:
    return lambda model: Alg(model=model, **args)


def directed_alg_constructor(
    Alg: Type[DirectedDiscreteGreedyAlgorithm], roi_constructor: ROIConstructor
) -> AlgConstructor:
    return lambda model: Alg(model=model, roi_constructor=roi_constructor)


class LineBO(GreedyAlgorithm):
    """
    **Line Bayesian Optimization (LineBO)** [1]

    Adaptively selects linear subspaces, discretizes them, and executes a given algorithm on this finite subdomain.

    [1] Kirschner, Johannes, et al. "Adaptive and safe Bayesian optimization in high dimensions via one-dimensional subspaces." International Conference on Machine Learning. PMLR, 2019.
    """

    alg: AlgConstructor
    """Given sub-domain and model, constructs an instance of the nested discrete algorithm."""
    model: ContinuousModel
    """Statistical model."""
    bounds: Float[Array, "d 2"]
    """Bounds of the continuous domain."""
    n: int
    """Size of the discretization of the linear subspace per unit."""
    eps: ScalarFloat
    """If the given norm of the provided direction is smaller than `eps`, falls back to CoordinateLineBO."""
    normalized_direction: Float[Array, "d"] | None = None
    """Normalized direction which is recomputed during every iteration."""
    encountered_max: Float[Array, "t d"]
    """Points associated to the highest encountered lower confidence bounds."""

    def __init__(
        self,
        alg: AlgConstructor,
        model: ContinuousModel,
        bounds: Float[Array, "d 2"],
        n: int,
        eps: ScalarFloat,
        x_init: Float[Array, "d"] | None = None,
    ):
        self.alg = alg
        self.model = model
        self.bounds = bounds
        self.n = n
        self.eps = eps
        self._x_init = x_init if x_init is not None else self.origin

        d = bounds.shape[0]
        self.encountered_max = jnp.empty(shape=(0, d))

    @property
    def d(self) -> int:
        return self.bounds.shape[0]

    @abstractmethod
    def _direction(self, f: Function) -> Float[Array, "d"]:
        """Returns a vector denoting the direction of the linear subspace."""
        pass

    def _normalized_direction(self, f: Function) -> Float[Array, "d"]:
        """Returns a *unit-length* vector denoting the direction of the linear subspace."""
        v = self._direction(f)
        norm = jnp.linalg.norm(v)
        if norm < self.eps or jnp.isinf(norm):
            return _coordinate_lbo(self)
        return v / norm

    @property
    def origin(self) -> Float[Array, "d"]:
        return (self.bounds[:, 0] + self.bounds[:, 1]) / 2

    @property
    def x_init(self) -> Float[Array, "d"]:
        if self.model.t == 0:
            return self._x_init
        return self.model.X[-1]

    def subspace_bounds(self) -> Tuple[ScalarFloat, ScalarFloat]:
        assert self.normalized_direction is not None
        normalized_direction = self.normalized_direction

        def engine(l: ScalarInt) -> Float[Array, "2"]:
            return (self.bounds[l, :] - self.x_init[l]) / normalized_direction[l]

        maximum_displacements_from_origin = vmap(engine)(jnp.arange(self.d)).reshape(-1)

        positive_displacements = maximum_displacements_from_origin[
            maximum_displacements_from_origin > 0
        ]
        negative_displacements = maximum_displacements_from_origin[
            maximum_displacements_from_origin <= 0
        ]
        if jnp.isinf(jnp.min(positive_displacements, initial=jnp.inf)):
            positive_displacements = maximum_displacements_from_origin[
                maximum_displacements_from_origin >= 0
            ]
            negative_displacements = maximum_displacements_from_origin[
                maximum_displacements_from_origin < 0
            ]

        return (jnp.max(negative_displacements), jnp.min(positive_displacements))

    def project(self, X: Float[Array, "n"]) -> Float[Array, "n d"]:
        assert self.normalized_direction is not None
        return self.x_init + jnp.outer(X, self.normalized_direction)

    def domain(self) -> Float[Array, "n d"]:
        subspace_bounds = self.subspace_bounds()
        subspace_length = subspace_bounds[1] - subspace_bounds[0]
        subspace_domain = jnp.linspace(
            subspace_bounds[0], subspace_bounds[1], num=int(subspace_length * self.n)
        )
        domain = self.project(subspace_domain)
        return domain

    def extended_domain(self) -> Float[Array, "n d"]:
        """
        Includes previously observed points to ensure that the safe set is non-empty.
        Includes previous guesses for the optimum to reevaluate them.
        """
        domain = self.domain()
        return jnp.concatenate([self.model.X, domain, self.encountered_max])

    def step(self, f: Function) -> dict:
        self.normalized_direction = self._normalized_direction(f)
        domain = self.extended_domain()
        model = self.model.marginalize(domain)
        alg = self.alg(model)
        X, y, result = alg._step(f, update_model=False)
        self.model.step(X, y)

        argmax = model.argmax
        self.encountered_max = jnp.concatenate(
            [self.encountered_max, argmax.reshape(1, -1)]
        )

        result["marginal_model"] = model
        result["argmax"] = argmax
        return result


class RandomLineBO(LineBO):
    """Variant of LineBO which chooses a direction uniformly at random."""

    def _direction(self, f: Function) -> Float[Array, "d"]:
        key = self.acquire_key()
        return jr.uniform(key, (self.d,))


class CoordinateLineBO(LineBO):
    """Variant of LineBO which chooses the direction uniformly at random from the set of basis vectors."""

    def _direction(self, f: Function) -> Float[Array, "d"]:
        return _coordinate_lbo(self)


class DescentLineBO(LineBO):
    """Variant of LineBO which chooses the direction based on the gradient of $f_0$."""

    m: int
    """Number of iterations to estimate the gradient."""
    alpha: ScalarFloat
    """Step size."""

    def __init__(
        self,
        alg: AlgConstructor,
        model: ContinuousModel,
        bounds: Array,
        n: int,
        eps: ScalarFloat,
        m: int,
        alpha: ScalarFloat,
        x_init: Float[Array, "d"] | None = None,
    ):
        super().__init__(alg, model, bounds, n, eps, x_init)
        self.m = m
        self.alpha = alpha

    def _direction(self, f: Function) -> Float[Array, "d"]:
        for i in range(self.m):
            # if self.grad_from_sample:
            key = self.model.acquire_key()
            g = grad_approx(
                x=self.x_init,
                f=lambda X: self.model.distr(X=X, i=0).sample(
                    key=key, sample_shape=(1,)
                )[0],
            )
            # else:
            #     mean_grad = grad(
            #         lambda x: self.model.mean(X=x.reshape(1, -1), i=0, use_cache=False)[
            #             0
            #         ]
            #     )
            #     stddev_grad = grad(
            #         lambda x: self.model.stddev(
            #             X=x.reshape(1, -1), i=0, use_cache=False
            #         )[0]
            #     )
            #     key = self.acquire_key()
            #     g = mean_grad(self.x_init)# + stddev_grad(self.x_init) * jr.normal(key=key, shape=self.x_init.shape)
            if jnp.linalg.norm(g) < self.eps:
                return _coordinate_lbo(self)
            x = self.x_init - self.alpha * g
            X = jnp.clip(x.reshape(1, -1), self.bounds[:, 0], self.bounds[:, 1])
            y = f.observe(X)
            self.model.step(X=X, y=y)
        return grad_approx(x=self.x_init, f=lambda X: self.model.distr(X=X, i=0).mean)


def _coordinate_lbo(lbo: LineBO) -> Float[Array, "d"]:
    key = lbo.acquire_key()
    i = jr.randint(key, shape=(), minval=0, maxval=lbo.d)
    return jnp.eye(lbo.d)[i]
