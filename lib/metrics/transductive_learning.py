from typing import TypedDict
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float
from lib.function import Function
from lib.model.marginal import MarginalModel
from lib.typing import ScalarFloat


class TransductiveLearningMetrics(TypedDict):
    entropy_in_roi: ScalarFloat
    max_stddev_in_roi: ScalarFloat
    avg_stddev_in_roi: ScalarFloat


def transductive_learning_metrics_generator(
    roi_mask: Bool[Array, "n"], entropy_jitter=None
):
    def transductive_learning_metrics(
        model: MarginalModel, f: Function, x: Float[Array, "d"] | None
    ) -> TransductiveLearningMetrics:
        if entropy_jitter is not None:
            entropy_jitter["key"], key = jr.split(entropy_jitter["key"])
            jitter = entropy_jitter["amount"] * jr.uniform(key)
        else:
            jitter = 0.0
        return TransductiveLearningMetrics(
            entropy_in_roi=model.masked_entropy(roi_mask, jitter=jitter),
            max_stddev_in_roi=jnp.max(model.stddevs[:, roi_mask], initial=0),
            avg_stddev_in_roi=jnp.mean(model.stddevs[:, roi_mask]),
        )

    return transductive_learning_metrics
