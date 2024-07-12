from typing import Dict, Type
import jax.numpy as jnp
from jaxtyping import Array, Float
from examples.experiment import (
    AlgorithmConfig,
    ContinuousExperiment,
)
from lib.algorithms import (
    f_roi_constructor,
    pm_roi_constructor,
    ts_roi_constructor,
    ucb_roi_constructor,
)
from lib.algorithms.baselines import (
    EI,
    EIC,
    MES,
    UCB,
    UncertaintySampling,
    UniformSampling,
)
from lib.algorithms.ctl import CTL
from lib.algorithms.goose import GoOSE
from lib.algorithms.itl import ITL
from lib.algorithms.mm_itl import MMITL
from lib.algorithms.ise import ISE, ISEBO
from lib.algorithms.line_bo import (
    LineBO,
    alg_constructor,
    directed_alg_constructor,
)
from lib.algorithms.mvtl import MVTL
from lib.algorithms.safe_opt import SafeOpt
from lib.algorithms.vtl import VTL
from lib.model.continuous import ContinuousModel


class ContinuousBOExperiment(ContinuousExperiment):
    def __init__(
        self,
        n: int,
        bounds: Float[Array, "d 2"],
        x_init: Float[Array, "d"],
        LineBOClass: Type[LineBO],
        line_bo_params: dict = {},
        **kwargs
    ):
        super().__init__(prior_sample_region=jnp.array([x_init]), **kwargs)
        self.n = n
        self.bounds = bounds
        self.x_init = x_init
        self.LineBOClass = LineBOClass
        self.line_bo_params = line_bo_params

    @property
    def _prior_key_map(self) -> Dict[str, str]:
        return {
            "Uniform Sampling": "base",
            "Uncertainty Sampling": "base",
            "ITL-F [ours]": "base",
            "ITL-PM [ours]": "base",
            "Unsafe ITL-PM [ours]": "unsafe_base",
            "ITL-TS [ours]": "base",
            "VTL-TS [ours]": "base",
            "Unsafe ITL-TS [ours]": "unsafe_base",
            "Unsafe VTL-TS [ours]": "unsafe_base",
            "ITL-UCB [ours]": "base",
            "MM-ITL-PM [ours]": "base",
            "CTL-PM [ours]": "base",
            "VTL-PM [ours]": "base",
            "MVTL-PM [ours]": "base",
            "SafeOpt": "SafeOptModel",
            "Oracle SafeOpt": "OracleSafeOptModel",
            "Heuristic SafeOpt": "HeuristicSafeOptModel",
            "GoOSE": "GoOSEModel",
            "Oracle GoOSE": "OracleGoOSEModel",
            "ISE": "base",
            "ISE-BO": "base",
            "Unsafe UCB": "unsafe_base",
            "Unsafe EI": "unsafe_base",
            "EIC": "unsafe_base",
            "Unsafe MES": "unsafe_base",
        }

    def _prepare_alg_config(
        self, key: str, model: ContinuousModel, T: int
    ) -> AlgorithmConfig:
        track_information_ratios = False
        if key == "Uniform Sampling":
            alg_constr = alg_constructor(UniformSampling)
        elif key == "Uncertainty Sampling":
            alg_constr = alg_constructor(UncertaintySampling)
        elif key == "ITL-F [ours]":
            alg_constr = directed_alg_constructor(ITL, f_roi_constructor)
        elif key == "ITL-PM [ours]" or key == "Unsafe ITL-PM [ours]":
            alg_constr = directed_alg_constructor(ITL, pm_roi_constructor)
        elif key == "ITL-TS [ours]" or key == "Unsafe ITL-TS [ours]":
            alg_constr = directed_alg_constructor(ITL, ts_roi_constructor(k=10))
        elif key == "VTL-TS [ours]" or key == "Unsafe VTL-TS [ours]":
            alg_constr = directed_alg_constructor(VTL, ts_roi_constructor(k=10))
        elif key == "ITL-UCB [ours]":
            alg_constr = directed_alg_constructor(ITL, ucb_roi_constructor)
        elif key == "MM-ITL-PM [ours]":
            alg_constr = directed_alg_constructor(MMITL, pm_roi_constructor)
        elif key == "CTL-PM [ours]":
            alg_constr = directed_alg_constructor(CTL, pm_roi_constructor)
        elif key == "VTL-PM [ours]":
            alg_constr = directed_alg_constructor(VTL, pm_roi_constructor)
        elif key == "MVTL-PM [ours]":
            alg_constr = directed_alg_constructor(MVTL, pm_roi_constructor)
        elif key == "SafeOpt":
            alg_constr = alg_constructor(SafeOpt)
        elif key == "Oracle SafeOpt":
            alg_constr = alg_constructor(SafeOpt)
        elif key == "Heuristic SafeOpt":
            alg_constr = alg_constructor(SafeOpt)
        elif key == "GoOSE":
            alg_constr = alg_constructor(GoOSE)
        elif key == "Oracle GoOSE":
            alg_constr = alg_constructor(GoOSE)
        elif key == "ISE":
            alg_constr = alg_constructor(ISE)
        elif key == "ISE-BO":
            alg_constr = alg_constructor(ISEBO, k=10)
        elif key == "Unsafe UCB":
            alg_constr = alg_constructor(UCB)
        elif key == "Unsafe EI":
            alg_constr = alg_constructor(EI)
        elif key == "EIC":
            alg_constr = alg_constructor(EIC)
        elif key == "Unsafe MES":
            alg_constr = alg_constructor(MES, k=10)
        else:
            raise NotImplementedError
        alg = self.LineBOClass(
            alg=alg_constr,
            model=model,
            bounds=self.bounds,
            n=self.n,
            eps=1e-3,
            x_init=self.x_init,
            **self.line_bo_params,
        )
        return AlgorithmConfig(
            name=key, alg=alg, T=T, track_information_ratios=track_information_ratios
        )
