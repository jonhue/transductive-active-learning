from typing import NotRequired, TypedDict
import jax.numpy as jnp
from jaxtyping import Array, Float
from lib.algorithms.goose.model import GoOSEModel
from lib.algorithms.safe_opt.model import SafeOptModel
from lib.function import Function
from lib.model.marginal import MarginalModel
from lib.typing import ScalarFloat, ScalarInt


class BoundaryExplorationMetrics(TypedDict):
    pessimistic_safe_set_size: ScalarInt
    optimistic_safe_set_size: ScalarInt
    potential_expanders_size: ScalarInt
    max_stddev_in_potential_expanders: ScalarFloat
    safeopt_expanders_size: NotRequired[ScalarInt]
    safeopt_safe_set_size: NotRequired[ScalarInt]
    safeopt_optimistic_safe_set_size: NotRequired[ScalarInt]
    safeopt_potential_expanders_size: NotRequired[ScalarInt]


def boundary_exploration_metrics(
    model: MarginalModel, f: Function, x: Float[Array, "d"] | None
):
    potential_expanders = model.domain[model.potential_expanders]
    result = BoundaryExplorationMetrics(
        pessimistic_safe_set_size=jnp.sum(model.operating_safe_set),
        optimistic_safe_set_size=jnp.sum(model.optimistic_safe_set),
        potential_expanders_size=jnp.sum(model.potential_expanders),
        max_stddev_in_potential_expanders=jnp.max(
            model.stddevs[:, potential_expanders], initial=0
        ),
    )

    if isinstance(model, SafeOptModel):
        result["safeopt_expanders_size"] = jnp.sum(model.safeopt_expanders)
        result["safeopt_safe_set_size"] = jnp.sum(model.safeopt_safe_set)

    if isinstance(model, GoOSEModel):
        result["safeopt_optimistic_safe_set_size"] = jnp.sum(
            model.safeopt_optimistic_safe_set
        )
        result["safeopt_potential_expanders_size"] = jnp.sum(
            model.safeopt_potential_expanders
        )

    return result
