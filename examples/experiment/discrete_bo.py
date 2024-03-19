from typing import Dict
from examples.experiment import (
    AlgorithmConfig,
    DiscreteExperiment,
)
from lib.algorithms import (
    f_roi_constructor,
    pe_roi_constructor,
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
from lib.algorithms.goose.model import GoOSEModel
from lib.algorithms.itl import ITL
from lib.algorithms.mm_itl import MMITL
from lib.algorithms.ise import ISE, ISEBO
from lib.algorithms.safe_opt import SafeOpt
from lib.algorithms.safe_opt.model import HeuristicSafeOptModel, SafeOptModel
from lib.algorithms.vtl import VTL
from lib.model.marginal import MarginalModel


class DiscreteBOExperiment(DiscreteExperiment):
    @property
    def _prior_key_map(self) -> Dict[str, str]:
        return {
            "Uniform Sampling": "base",
            "Uncertainty Sampling": "base",
            "ITL-F [ours]": "base",
            "ITL-PM [ours]": "base",
            "ITL-PE [ours]": "base",
            "Unsafe ITL-PM [ours]": "unsafe_base",
            "ITL-TS [ours]": "base",
            "VTL-TS [ours]": "base",
            "Unsafe ITL-TS [ours]": "unsafe_base",
            "Unsafe VTL-TS [ours]": "unsafe_base",
            "ITL-UCB [ours]": "base",
            "MM-ITL-PM [ours]": "base",
            "CTL-PM [ours]": "base",
            "VTL-PM [ours]": "base",
            "VTL-PE [ours]": "base",
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
        self, key: str, model: MarginalModel, T: int
    ) -> AlgorithmConfig:
        track_information_ratios = False
        if key == "Uniform Sampling":
            alg = UniformSampling(model=model)
        elif key == "Uncertainty Sampling":
            alg = UncertaintySampling(model=model)
        elif key == "ITL-F [ours]":
            alg = ITL(model=model, roi_constructor=f_roi_constructor)
        elif key == "ITL-PM [ours]" or key == "Unsafe ITL-PM [ours]":
            alg = ITL(model=model, roi_constructor=pm_roi_constructor)
        elif key == "ITL-PE [ours]":
            alg = ITL(model=model, roi_constructor=pe_roi_constructor)
        elif key == "ITL-TS [ours]" or key == "Unsafe ITL-TS [ours]":
            alg = ITL(model=model, roi_constructor=ts_roi_constructor(k=10))
        elif key == "VTL-TS [ours]" or key == "Unsafe VTL-TS [ours]":
            alg = VTL(model=model, roi_constructor=ts_roi_constructor(k=10))
        elif key == "ITL-UCB [ours]":
            alg = ITL(model=model, roi_constructor=ucb_roi_constructor)
        elif key == "MM-ITL-PM [ours]":
            alg = MMITL(model=model, roi_constructor=pm_roi_constructor)
        elif key == "CTL-PM [ours]":
            alg = CTL(model=model, roi_constructor=pm_roi_constructor)
        elif key == "VTL-PM [ours]":
            alg = VTL(model=model, roi_constructor=pm_roi_constructor)
        elif key == "VTL-PE [ours]":
            alg = VTL(model=model, roi_constructor=pe_roi_constructor)
        elif key == "SafeOpt" and isinstance(model, SafeOptModel):
            alg = SafeOpt(model=model)
        elif key == "Oracle SafeOpt" and isinstance(model, SafeOptModel):
            alg = SafeOpt(model=model)
        elif key == "Heuristic SafeOpt" and isinstance(model, HeuristicSafeOptModel):
            alg = SafeOpt(model=model)
        elif key == "GoOSE" and isinstance(model, GoOSEModel):
            alg = GoOSE(model=model)
        elif key == "Oracle GoOSE" and isinstance(model, GoOSEModel):
            alg = GoOSE(model=model)
        elif key == "ISE":
            alg = ISE(model=model)
        elif key == "ISE-BO":
            alg = ISEBO(model=model, k=10)
        elif key == "Unsafe UCB":
            alg = UCB(model=model)
        elif key == "Unsafe EI":
            alg = EI(model=model)
        elif key == "EIC":
            alg = EIC(model=model)
        elif key == "Unsafe MES":
            alg = MES(model=model, k=10)
        else:
            raise NotImplementedError
        return AlgorithmConfig(
            name=key, alg=alg, T=T, track_information_ratios=track_information_ratios
        )
