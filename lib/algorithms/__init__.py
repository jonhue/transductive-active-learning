r"""
Implementation of acquisition functions.

* `lib.algorithms.itl` implements (directed and undirected) information-based transductive learning (ITL).
* `lib.algorithms.ctl` implements correlation-based transductive learning (CTL).
* `lib.algorithms.vtl` implements variance-based transductive learning (VTL).
* `lib.algorithms.ids` implements information-directed sampling (IDS).
* `lib.algorithms.eps_greedy` implements $\epsilon$-greedy algorithms for no-regret.
* `lib.algorithms.baselines` implements baselines such as uniform sampling, uncertainty sampling, and UCB.
* `lib.algorithms.tru_var` implements truncated variance reduction (TruVar).
* `lib.algorithms.ise` implements information-theoretic safe exploration (ISE) for boundary exploration.
* `lib.algorithms.safe_opt` implements SafeOpt.
* `lib.algorithms.goose` implements goal-oriented safe exploration (GoOSE).

* `lib.algorithms.line_bo` implements LineBO to generalize to high-dimensional continuous domains.
"""

from abc import ABC, abstractmethod
from typing import Callable, Generic, NewType, Tuple, TypeVar
from jax import vmap
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from lib.function import Function
from lib.model import Model
from lib.model.marginal import MarginalModel
from lib.typing import KeyArray, ScalarInt
from lib.utils import get_mask


class Algorithm(ABC):
    """Acquisition function."""

    @abstractmethod
    def step(self, f: Function) -> dict:
        pass


class GreedyAlgorithm(Algorithm):
    """Algorithm which takes steps greedily based on a statistical model."""

    model: Model

    def __init__(self, model: Model):
        self.model = model

    def acquire_key(self) -> KeyArray:
        return self.model.acquire_key()


ModelBase = TypeVar("ModelBase", bound=MarginalModel)


class DiscreteGreedyAlgorithm(GreedyAlgorithm, Generic[ModelBase]):
    """Greedy algorithm over a fixed finite domain."""

    sample_region: Float[Array, "m d"]
    model: ModelBase

    def __init__(
        self,
        model: ModelBase,
        sample_region: Float[Array, "m d"] | None = None,
    ):
        self.model = model
        self.sample_region = (
            sample_region if sample_region is not None else model.domain
        )

    @property
    def n(self) -> int:
        return self.model.domain.shape[0]

    @abstractmethod
    def F(self) -> Float[Array, "n"]:
        pass

    def _step(
        self, f: Function, update_model: bool = True
    ) -> Tuple[Float[Array, "1 d"], Float[Array, "q 1"], dict]:
        F = self.F().at[~get_mask(self.model.domain, self.sample_region)].set(-jnp.inf)
        X = self.model.objective_argmax(F=F).reshape(1, -1)
        y = f.observe(X)
        if update_model:
            self.model.step(X=X, y=y)
        return (
            X,
            y,
            {
                "x": X[0],
                "y": y[:, 0],
                "F": F,
                "F_max": F[self.model.get_indices(X)][0],
            },
        )

    def step(self, f: Function) -> dict:
        X, y, result = self._step(f)
        return result


ROI = NewType("ROI", Float[Array, "m d"])
"""Region of interest (ROI)"""

ROIConstructor = Callable[[MarginalModel], ROI]
"""
Function computing a region of interest based on a given model.
Requires that the returned ROI is a subset of the (finite) domain of the given model.
"""

ROIDescription = NewType("ROIDescription", Float[Array, "k d 2"])
"""Characterization of ROI through a finite union of $k$ sets, each described by lower and upper bounds per dimension."""


def fixed_roi_constructor(roi_description: ROIDescription) -> ROIConstructor:
    return lambda model: compute_roi(model.domain, roi_description)


f_roi_constructor: ROIConstructor = lambda model: ROI(model.domain)
pe_roi_constructor: ROIConstructor = lambda model: ROI(
    model.domain[model.potential_expanders]
)
pm_roi_constructor: ROIConstructor = lambda model: ROI(
    model.domain[model.potential_maximizers]
)


def ts_roi_constructor(k: int) -> ROIConstructor:
    return lambda model: ROI(
        model.domain[
            model.thompson_sampling(k=k, J=jnp.arange(model.first_constraint, model.q))
        ]
    )


ucb_roi_constructor: ROIConstructor = lambda model: ROI(model.domain[model.ucb])


def compute_roi_mask(
    domain: Float[Array, "n d"], roi_description: ROIDescription
) -> Bool[Array, "n"]:
    D = domain.shape[1]
    k = roi_description.shape[0]

    def engine(d: ScalarInt, l: ScalarInt) -> Bool[Array, "n"]:
        lower_bound = roi_description[l, d, 0]
        upper_bound = roi_description[l, d, 1]
        return (domain[:, d] >= lower_bound) & (domain[:, d] <= upper_bound)

    return jnp.any(
        vmap(lambda l: jnp.all(vmap(lambda d: engine(d, l))(jnp.arange(D)), axis=0))(
            jnp.arange(k)
        ),
        axis=0,
    )


def compute_roi(domain: Float[Array, "n d"], roi_description: ROIDescription) -> ROI:
    return ROI(domain[compute_roi_mask(domain, roi_description)])


class DirectedDiscreteGreedyAlgorithm(
    DiscreteGreedyAlgorithm[ModelBase], Generic[ModelBase]
):
    """Greedy algorithm over a finite domain which uses a (dynamically computed) region of interest."""

    roi_constructor: ROIConstructor

    def __init__(
        self,
        model: ModelBase,
        roi_constructor: ROIConstructor,
        sample_region: Float[Array, "m d"] | None = None,
    ):
        super().__init__(model, sample_region)
        self.roi_constructor = roi_constructor

    @property
    def roi(self) -> ROI:
        return self.roi_constructor(self.model)
