from typing import NotRequired, TypedDict
import jax.numpy as jnp
from jaxtyping import Array, Float
from lib.function import Function
from lib.function.synthetic import SyntheticFunction
from lib.model.marginal import MarginalModel
from lib.typing import ScalarFloat


class ActiveLearningMetrics(TypedDict):
    entropy: ScalarFloat
    max_stddev: ScalarFloat
    avg_stddev: ScalarFloat
    entropy_in_safe_set: NotRequired[ScalarFloat]
    max_stddev_in_safe_set: NotRequired[ScalarFloat]
    avg_stddev_in_safe_set: NotRequired[ScalarFloat]


def active_learning_metrics(
    model: MarginalModel, f: Function, x: Float[Array, "d"] | None
) -> ActiveLearningMetrics:
    result = ActiveLearningMetrics(
        entropy=model.entropy(),
        max_stddev=jnp.max(model.stddevs, initial=0),
        avg_stddev=jnp.mean(model.stddevs),
    )

    if isinstance(f, SyntheticFunction):
        safe_set = model.domain[f.safe_set(model.domain)]
        result["entropy_in_safe_set"] = model.masked_entropy(safe_set)
        result["max_stddev_in_safe_set"] = jnp.max(
            model.stddevs[:, safe_set], initial=0
        )
        result["avg_stddev_in_safe_set"] = jnp.mean(model.stddevs[:, safe_set])

    return result
