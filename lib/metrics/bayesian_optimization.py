from typing import NotRequired, TypedDict
import jax.numpy as jnp
from jaxtyping import Array, Float
from lib.algorithms.safe_opt.model import SafeOptModel
from lib.function import Function
from lib.function.synthetic import SyntheticFunction
from lib.model.continuous import ContinuousModel
from lib.model.marginal import MarginalModel
from lib.typing import ScalarFloat, ScalarInt


class BayesianOptimizationMetrics(TypedDict):
    pessimistic_safe_set_size: ScalarInt
    potential_maximizers_size: ScalarInt
    max_stddev_in_potential_maximizers: ScalarFloat
    surr_instantaneous_regret: ScalarFloat
    """Surrogate to the instantaneous regret based on upper/lower confidence bounds."""
    surr_simple_regret: ScalarFloat
    """Surrogate to the simple regret based on upper/lower confidence bounds."""
    stddev_at_safe_maximum: NotRequired[ScalarFloat]
    simple_regret: NotRequired[ScalarFloat]
    instantaneous_regret: NotRequired[ScalarFloat]
    constraint_min: NotRequired[ScalarFloat]
    safeopt_maximizers_size: NotRequired[ScalarFloat]


def bayesian_optimization_metrics(
    model: MarginalModel, f: Function, x: Float[Array, "d"] | None
) -> BayesianOptimizationMetrics:
    x_idx = model.get_indices(x.reshape(1, -1))[0] if x is not None else jnp.nan
    result = BayesianOptimizationMetrics(
        pessimistic_safe_set_size=jnp.sum(model.operating_safe_set),
        potential_maximizers_size=jnp.sum(model.potential_maximizers),
        max_stddev_in_potential_maximizers=jnp.max(
            model.stddevs[:, model.potential_maximizers],
            initial=0,
        ),
        surr_instantaneous_regret=(
            model.max_u - model.l[0, x_idx] if not jnp.isnan(x_idx) else jnp.nan
        ),
        surr_simple_regret=model.max_u - model.max_l,
    )

    if isinstance(f, SyntheticFunction):
        f_opt = f.max(model.domain)
        f_opt_idx = model.get_indices(f.argmax(model.domain).reshape(1, -1))[0]
        fs_x = f.evaluate(x.reshape(1, -1)) if x is not None else None

        result["stddev_at_safe_maximum"] = jnp.max(model.stddevs[:, f_opt_idx])
        result["simple_regret"] = f_opt - f.evaluate(model.argmax.reshape(1, -1))[0, 0]
        result["instantaneous_regret"] = (
            f_opt - fs_x[0, 0] if fs_x is not None else jnp.nan
        )
        result["constraint_min"] = (
            jnp.min(fs_x[f.first_constraint :, 0], initial=jnp.inf)
            if fs_x is not None
            else jnp.nan
        )

    if isinstance(model, SafeOptModel):
        result["safeopt_maximizers_size"] = jnp.sum(model.safeopt_maximizers)

    return result


class ContinuousBayesianOptimizationMetrics(TypedDict):
    pessimistic_safe_set_size: NotRequired[ScalarInt]
    stddev_at_safe_maximum: NotRequired[ScalarFloat]
    simple_regret: NotRequired[ScalarFloat]
    instantaneous_regret: NotRequired[ScalarFloat]
    constraint_min: NotRequired[ScalarFloat]


def continuous_bayesian_optimization_metrics_generator(
    x_opt: Float[Array, "d"], f_opt: ScalarFloat
):
    def continuous_bayesian_optimization_metrics(
        model: ContinuousModel,
        f: Function,
        x: Float[Array, "d"] | None,
        marginal_model: MarginalModel | None = None,
        argmax: Float[Array, "d"] | None = None,
    ) -> ContinuousBayesianOptimizationMetrics:
        result = ContinuousBayesianOptimizationMetrics()
        result["pessimistic_safe_set_size"] = (
            jnp.sum(marginal_model.operating_safe_set)
            if marginal_model is not None
            else 0
        )
        if isinstance(f, SyntheticFunction):
            fs_x = f.evaluate(x.reshape(1, -1)) if x is not None else None

            result["simple_regret"] = (
                (f_opt - f.evaluate(argmax.reshape(1, -1))[0, 0])
                if argmax is not None
                else jnp.inf
            )
            result["instantaneous_regret"] = (
                f_opt - fs_x[0, 0] if fs_x is not None else jnp.inf
            )
            result["constraint_min"] = (
                jnp.min(fs_x[f.first_constraint :, 0]) if fs_x is not None else jnp.inf
            )
        return result

    return continuous_bayesian_optimization_metrics
