from typing import Dict
from jaxtyping import Array, Float
from examples.experiment import (
    AlgorithmConfig,
    ContinuousExperiment,
    DiscreteExperiment,
)
from lib.algorithms import (
    ROIDescription,
    compute_roi,
    f_roi_constructor,
    fixed_roi_constructor,
)
from lib.algorithms.baselines import UncertaintySampling, UniformSampling
from lib.algorithms.ctl import CTL
from lib.algorithms.itl import ITL
from lib.algorithms.line_bo import CoordinateLineBO, directed_alg_constructor
from lib.algorithms.tru_var import TruVar
from lib.algorithms.vtl import VTL
from lib.model.continuous import ContinuousModel
from lib.model.marginal import MarginalModel
from lib.utils import get_mask


class DiscreteTransductiveLearningExperiment(DiscreteExperiment):
    def __init__(
        self,
        roi_description: ROIDescription,
        sample_region: Float[Array, "m d"] | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.roi_description = roi_description
        self.sample_region = sample_region

    @property
    def _prior_key_map(self) -> Dict[str, str]:
        return {
            "Uniform Sampling": "base",
            "Uncertainty Sampling": "base",
            "Uncertainty Sampling (constrained)": "base",
            "ITL (undirected) [ours]": "base",
            "ITL (directed) [ours]": "base",
            "CTL [ours]": "base",
            "VTL [ours]": "base",
            "TruVar": "TruVarModel",
        }

    def _prepare_alg_config(
        self, key: str, model: MarginalModel, T: int
    ) -> AlgorithmConfig:
        track_information_ratios = False
        roi_constructor = fixed_roi_constructor(self.roi_description)
        roi = compute_roi(self.domain, self.roi_description)
        if key == "Uniform Sampling":
            alg = UniformSampling(model=model, sample_region=self.sample_region)
        elif key == "Uncertainty Sampling":
            alg = UncertaintySampling(model=model, sample_region=self.sample_region)
        elif key == "Uncertainty Sampling (constrained)":
            sample_region = (
                roi
                if self.sample_region is None
                else model.domain[
                    get_mask(model.domain, self.sample_region)
                    & get_mask(model.domain, roi)
                ]
            )
            alg = UncertaintySampling(model=model, sample_region=sample_region)
        elif key == "ITL (undirected) [ours]":
            alg = ITL(
                model=model,
                roi_constructor=f_roi_constructor,
                sample_region=self.sample_region,
            )
        elif key == "ITL (directed) [ours]":
            alg = ITL(
                model=model,
                roi_constructor=roi_constructor,
                sample_region=self.sample_region,
            )
        elif key == "CTL [ours]":
            alg = CTL(
                model=model,
                roi_constructor=roi_constructor,
                sample_region=self.sample_region,
            )
        elif key == "VTL [ours]":
            alg = VTL(
                model=model,
                roi_constructor=roi_constructor,
                sample_region=self.sample_region,
            )
        elif key == "TruVar":
            alg = TruVar(
                model=model,  # type: ignore
                roi_constructor=roi_constructor,
                sample_region=self.sample_region,
            )
        else:
            raise NotImplementedError
        return AlgorithmConfig(
            name=key, alg=alg, T=T, track_information_ratios=track_information_ratios
        )


class ContinuousTransductiveLearningExperiment(ContinuousExperiment):
    def __init__(
        self,
        n: int,
        bounds: Float[Array, "d 2"],
        roi_description: ROIDescription,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n = n
        self.bounds = bounds
        self.roi_description = roi_description

    @property
    def _prior_key_map(self) -> Dict[str, str]:
        return {
            "ITL (continuous) [ours]": "base",
            "CTL (continuous) [ours]": "base",
            "VTL (continuous) [ours]": "base",
            "TruVar (continuous)": "TruVarModel",
        }

    def _prepare_alg_config(
        self, key: str, model: ContinuousModel, T: int
    ) -> AlgorithmConfig:
        track_information_ratios = False
        roi_constructor = fixed_roi_constructor(self.roi_description)
        if key == "ITL (continuous) [ours]":
            alg_constr = directed_alg_constructor(ITL, roi_constructor)
        elif key == "CTL (continuous) [ours]":
            alg_constr = directed_alg_constructor(CTL, roi_constructor)
        elif key == "VTL (continuous) [ours]":
            alg_constr = directed_alg_constructor(VTL, roi_constructor)
        elif key == "TruVar (continuous)":
            alg_constr = directed_alg_constructor(TruVar, roi_constructor)
        else:
            raise NotImplementedError
        alg = CoordinateLineBO(
            alg=alg_constr, model=model, bounds=self.bounds, n=self.n, eps=1e-1
        )
        return AlgorithmConfig(
            name=key, alg=alg, T=T, track_information_ratios=track_information_ratios
        )
