from typing import List
import jax.numpy as jnp
from jaxtyping import Array, Int
from lib.algorithms import itl
from lib.model.marginal import MarginalModel
from lib.noise import Noise
from lib.typing import ScalarFloat


def compute_strong_information_ratio(
    F_max: List[ScalarFloat], prev: ScalarFloat | None = None
) -> ScalarFloat:
    r"""
    Greedily computes the strong information ratio: $$\min_n \min_{j \leq n} \frac{\max_{\mathbf{x}} I(\mathbf{h}(\mathcal{A}); \mathbf{y}(\mathbf{x}) \mid \mathcal{D}_j)}{\max_{\mathbf{x}} I(\mathbf{h}(\mathcal{A}); \mathbf{y}(\mathbf{x}) \mid \mathcal{D}_n)}.$$

    :param F_max: For each iteration, the objective value of the queried point.
    :param prev: The strong information ratio of the previous iteration.
    """
    if len(F_max) < 1:
        return jnp.array(jnp.inf)

    F_max_latest = F_max[-1]
    new_strong_information_ratio = jnp.nanmin(jnp.array(F_max) / F_max_latest)
    return (
        jnp.minimum(new_strong_information_ratio, prev)
        if prev is not None and not jnp.isnan(prev)
        else new_strong_information_ratio
    )


def compute_information_ratio(
    model: MarginalModel,
    roi_idx: Int[Array, "m"] | None,
    sample_region_idx: Int[Array, "m"] | None,
) -> ScalarFloat:
    r"""
    Computes the information ratio: $$\frac{\sum_{\mathbf{x} \in B} I(\mathbf{h}(\mathcal{A}); \mathbf{y}(\mathbf{x}) \mid \mathcal{D}_n)}{I(\mathbf{h}(\mathcal{A}); \mathbf{y}(B) \mid \mathcal{D}_n)}.$$

    To simplify, it lets $B = \mathcal{S}_n$.
    """
    if roi_idx is None:
        roi_idx = jnp.arange(model.n)
    if sample_region_idx is None:
        sample_region_idx = jnp.arange(model.n)

    boundary_idx = sample_region_idx

    joint_info_gain = 0
    noise_rates = model.noise_rate.at(model.domain)
    for i in range(model.q):
        joint_info_gain += model.distr(i).information_gain(
            roi_idx=roi_idx,
            obs_idx=boundary_idx,
            obs_noise_std=noise_rates[i],
        )
    indiv_info_gains = itl._directed(model=model, roi=model.domain[roi_idx])
    return jnp.sum(indiv_info_gains[boundary_idx]) / joint_info_gain
