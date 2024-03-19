from typing import NotRequired, TypedDict
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from lib.function import Function
from lib.function.synthetic import SyntheticFunction
from lib.model.marginal import MarginalModel
from lib.typing import ScalarFloat, ScalarInt


class TransductiveBanditsMetrics(TypedDict):
    entropy_in_roi: ScalarFloat
    max_stddev_in_roi: ScalarFloat
    avg_stddev_in_roi: ScalarFloat
    transductive_convergence_gap: ScalarInt
    transductive_regret: NotRequired[ScalarFloat]
    stddev_at_transductive_optimum: NotRequired[ScalarFloat]


def transductive_bandits_metrics_generator(roi_mask: Bool[Array, "n"]):
    def transductive_bandits_metrics(
        model: MarginalModel, f: Function, x: Float[Array, "d"] | None
    ) -> TransductiveBanditsMetrics:
        roi = model.domain[roi_mask]
        result = TransductiveBanditsMetrics(
            entropy_in_roi=model.masked_entropy(roi_mask),
            max_stddev_in_roi=jnp.max(model.stddevs[:, roi_mask], initial=0),
            avg_stddev_in_roi=jnp.mean(model.stddevs[:, roi_mask]),
            transductive_convergence_gap=model.convergence_gap_within(roi_mask),
        )

        if isinstance(f, SyntheticFunction):
            f_opt_idx = model.get_indices(f.argmax_within(roi_mask).reshape(1, -1))[0]

            result["transductive_regret"] = (
                f.max_within(roi)
                - f.evaluate(model.argmax_within(roi_mask).reshape(1, -1))[0, 0]
            )
            result["stddev_at_transductive_optimum"] = jnp.max(
                model.stddevs[:, f_opt_idx]
            )

        return result

    return transductive_bandits_metrics
