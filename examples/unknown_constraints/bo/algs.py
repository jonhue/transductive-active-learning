import jax.numpy as jnp
from examples.experiment import Algorithm
from lib.algorithms.baselines import (
    eps_ucb,
    ucb,
    uncertainty_sampling,
    uniform_sampling,
)
from lib.algorithms.correlation_sampling import correlation_sampling
from lib.algorithms.goose import goose
from lib.algorithms.ids import eps_ids
from lib.algorithms.info_opt import info_opt
from lib.algorithms.safe_opt import safe_opt
from lib.algorithms.total_variance_reduction import total_variance_reduction


def _get_algs(T: int):
    return {
        "Uniform Sampling": Algorithm(
            F=lambda key, model, *_: uniform_sampling(key=key, n=model.n),
            prior="base",
            T=T,
        ),
        "Uncertainty Sampling = InfoOpt-F": Algorithm(
            F=lambda key, model, *_: uncertainty_sampling(model),
            prior="base",
            T=T,
        ),
        "InfoOpt-F": Algorithm(
            F=lambda key, model, roi_idx, gt, t: info_opt(model, roi_idx, gt),
            prior="base",
            T=T,
        ),
        "InfoOpt-PM [ours]": Algorithm(
            F=lambda key, model, roi_idx, gt, _: info_opt(model, roi_idx, gt),
            prior="base",
            region_of_interest=lambda model: model.potential_maximizers,
            T=T,
        ),
        "InfoOpt-TS [ours]": Algorithm(
            F=lambda key, model, roi_idx, gt, _: info_opt(model, roi_idx, gt),
            prior="base",
            region_of_interest=lambda model: model.ts(k=10),
            T=T,
        ),
        "InfoOpt-UCB [ours]": Algorithm(
            F=lambda key, model, roi_idx, gt, _: info_opt(model, roi_idx, gt),
            prior="base",
            region_of_interest=lambda model: model.ucb,
            T=T,
        ),
        "Correlation Sampling PM [ours]": Algorithm(
            F=lambda key, model, roi_idx, gt, t: correlation_sampling(
                model, roi_idx, gt
            ),
            prior="base",
            region_of_interest=lambda model: model.potential_maximizers,
            T=T,
        ),
        "Total Variance Reduction PM [ours]": Algorithm(
            F=lambda key, model, roi_idx, gt, t: total_variance_reduction(
                model, roi_idx, gt
            ),
            prior="base",
            region_of_interest=lambda model: model.potential_maximizers,
            T=T,
        ),
        "eps-IDS-PM [ours]": Algorithm(
            F=lambda key, model, roi_idx, gt, t: eps_ids(
                key, model, roi_idx, gt, eps=1 / t
            ),
            prior="base",
            region_of_interest=lambda model: model.potential_maximizers,
            T=T,
        ),
        "eps-IDS-PM (pessimistic) [ours]": Algorithm(
            F=lambda key, model, roi_idx, gt, t: eps_ids(
                key, model, roi_idx, gt, eps=1 / t, pessimistic=True
            ),
            prior="base",
            region_of_interest=lambda model: model.potential_maximizers,
            T=T,
        ),
        "eps-IDS-TS [ours]": Algorithm(
            F=lambda key, model, roi_idx, gt, t: eps_ids(
                key, model, roi_idx, gt, eps=1 / t
            ),
            prior="base",
            region_of_interest=lambda model: model.ts(k=10),
            T=T,
        ),
        "eps-IDS-TS (pessimistic) [ours]": Algorithm(
            F=lambda key, model, roi_idx, gt, t: eps_ids(
                key, model, roi_idx, gt, eps=1 / t, pessimistic=True
            ),
            prior="base",
            region_of_interest=lambda model: model.ts(k=10),
            T=T,
        ),
        "eps-IDS-F": Algorithm(
            F=lambda key, model, _, gt, t: eps_ids(
                key, model, roi_idx=None, gt=gt, eps=1 / t
            ),
            prior="base",
            T=T,
        ),
        "eps-IDS-UCB": Algorithm(
            F=lambda key, model, roi_idx, gt, t: eps_ids(
                key, model, roi_idx, gt, eps=1 / t
            ),
            prior="base",
            region_of_interest=lambda model: model.ucb,
            T=T,
        ),
        "eps-UCB": Algorithm(
            F=lambda key, model, roi_idx, gt, t: eps_ucb(
                key, model, roi_idx, gt, eps=1 / t
            ),
            prior="base",
            region_of_interest=lambda model: model.potential_maximizers,
            T=T,
        ),
        "SafeOpt": Algorithm(
            F=lambda key, model, *_: safe_opt(key, model), prior="safeopt", T=T
        ),
        "SafeOpt oracle-L": Algorithm(
            F=lambda key, model, *_: safe_opt(key, model),
            prior="safeopt_oracle-L",
            T=T,
        ),
        "SafeOpt no-L": Algorithm(
            F=lambda key, model, *_: safe_opt(key, model),
            prior="safeopt_no-L",
            T=T,
        ),
        "GoOSE-UCB": Algorithm(
            F=lambda key, model, *_: goose(model), prior="goose", T=T
        ),
        "GoOSE-UCB oracle-L": Algorithm(
            F=lambda key, model, *_: goose(model), prior="goose_oracle-L", T=T
        ),
        "UCB": Algorithm(F=lambda key, model, *_: ucb(model), prior="base", T=T),
    }


def get_algs(alg: str | None, T):
    if alg is not None:
        return {alg: _get_algs(T)[alg]}
    else:
        return _get_algs(T)
