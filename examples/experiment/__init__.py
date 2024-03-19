from abc import abstractmethod
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Tuple
from warnings import warn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
import pandas as pd
from tqdm import tqdm
import wandb
from lib.algorithms import Algorithm, DirectedDiscreteGreedyAlgorithm
from lib.algorithms.goose.model import GoOSEContinuousModel, GoOSEModel
from lib.algorithms.tru_var.model import TruVarContinuousModel, TruVarModel
from lib.function import Function
from lib.function.synthetic import SyntheticFunction
from lib.gp.kernels.base import Kernel
from lib.gp.means import Mean
from lib.metrics import (
    compute_information_ratio,
    compute_strong_information_ratio,
)
from lib.model import Model
from lib.model.continuous import ContinuousModel
from lib.model.marginal import MarginalModel
from lib.model.unsafe import UnsafeContinuousModel, UnsafeModel
from lib.noise import Noise
from lib.plotting import store_and_show_fig
from lib.gp.prior import initial_observations, prior_distr
from lib.algorithms.safe_opt.model import (
    HeuristicSafeOptContinuousModel,
    HeuristicSafeOptModel,
    SafeOptContinuousModel,
    SafeOptModel,
)
from lib.typing import KeyArray, ScalarBool, ScalarFloat
from lib.utils import Dataset


class AlgorithmConfig:
    def __init__(
        self,
        name: str,
        alg: Algorithm,
        T: int,
        track_information_ratios: bool = False,
    ):
        self.name = name
        self.alg = alg
        self.T = T
        self.track_information_ratios = track_information_ratios


class ExperimentData(dict):
    X: List[Float[Array, "d"]] = []
    r"""For each iteration $t$, the queried point $\mathbf{x}_t$."""
    Y: List[Float[Array, "q"]] = []
    r"""For each iteration $t$, the observation $\mathbf{y}_t$."""
    F: List[Float[Array, "n"]] = []
    """For each iteration, the objective values for all $n$ points in the domain."""
    F_max: List[ScalarFloat] = []
    """For each iteration, the objective value of the queried point."""
    use_undirected: List[bool] = []
    """For each iteration, whether the InfoOpt algorithm has switched to undirected."""
    iter_time: List[float] = []
    """For each iteration, the time to evaluate the acquisition function, make the observation, and update the model."""

    def init(self, d: int, q: int, n: int, metrics: dict):
        self.add(
            x=jnp.full((d,), jnp.nan),
            y=jnp.full((q,), jnp.nan),
            F=jnp.full(n, jnp.nan),
            F_max=jnp.nan,
            metrics=metrics,
            iter_time=jnp.nan,
        )

    def add(
        self,
        x: Float[Array, "d"],
        y: Float[Array, "q"],
        metrics: dict,
        iter_time: float,
        F: Float[Array, "n"],
        F_max: float,
        log=True,
    ):
        self.X.append(x)
        self.Y.append(y)
        self.F.append(F)
        self.F_max.append(F_max)
        self.iter_time.append(iter_time)
        for k, v in metrics.items():
            if k in self.keys():
                self[k].append(v)
            else:
                self[k] = [v]

        if log:
            wandb.log(
                {"x": x, "y": y, "F_max": F_max, "iter_time": iter_time, **metrics}
            )

    def __getattr__(self, item):
        return super().__getitem__(item)

    def __setattr__(self, item, value):
        return super().__setitem__(item, value)

    def to_dict(self) -> dict:
        return {
            **dict(self),
            "F_max": self.F_max,
            "iter_time": self.iter_time,
        }


class Experiment:
    def __init__(
        self,
        key: KeyArray,
        name: str,
        d: int,
        f: Function,
        noise_rate: Noise,
        use_objective_as_constraint: bool,
        metrics_fn: Callable,
        plot_fn: Callable | None,
        beta: Float[Array, "q"] | Callable[[int], Float[Array, "q"]],
        T: int,
        alg_key: str,
        prior_means: List[Mean],
        prior_kernels: List[Kernel],
        prior_n_samples: int,
        prior_sample_region: Float[Array, "n d"],
        prior_sample_key: KeyArray | None = None,
        safe_opt_config: Dict[str, Any] | None = None,
        tru_var_config: Dict[str, Any] | None = None,
        track_figs: bool = False,
        check_calibration: bool = False,
    ):
        self._key = key
        self.name = name
        self.d = d
        self.f = f
        self.noise_rate = noise_rate
        self.use_objective_as_constraint = use_objective_as_constraint
        self.metrics_fn = metrics_fn
        self.plot_fn = plot_fn
        self.beta = beta
        self.T = T
        self.alg_key = alg_key
        self.prior_n_samples = prior_n_samples
        self.prior_means = prior_means
        self.prior_kernels = prior_kernels
        self.prior_sample_region = prior_sample_region
        self.prior_sample_key = prior_sample_key
        self.safe_opt_config = safe_opt_config
        self.tru_var_config = tru_var_config
        self.track_figs = track_figs
        self.check_calibration = check_calibration

    def acquire_key(self) -> KeyArray:
        self._key, key = jr.split(self._key)
        return key

    @abstractmethod
    def _prepare_model(
        self, key: str, X: Float[Array, "m d"], y: Float[Array, "q m"]
    ) -> Model:
        pass

    @abstractmethod
    def _check_calibration(self, model: Model) -> ScalarBool:
        pass

    @property
    @abstractmethod
    def _prior_key_map(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def _prepare_alg_config(self, key: str, model: Model, T: int) -> AlgorithmConfig:
        pass

    def _prepare_priors(self, key: str) -> Model:
        X, y = initial_observations(
            key=(
                self.prior_sample_key
                if self.prior_sample_key is not None
                else self.acquire_key()
            ),
            f=self.f,
            n_samples=self.prior_n_samples,
            region=self.prior_sample_region,
        )
        model = self._prepare_model(key, X, y)

        if not self._check_calibration(model):
            warn("The prior is not well-calibrated.")

        if isinstance(model, MarginalModel):
            f = self.f if isinstance(self.f, SyntheticFunction) else None
            if model.d == 1:
                store_and_show_fig(
                    "",
                    model.plot_1d(f=f, D=Dataset(X, y.T), legend=False),
                    self.name,
                    "prior",
                )
            elif model.d == 2 and isinstance(f, SyntheticFunction):
                store_and_show_fig(
                    "",
                    model.plot_2d(f=f, X=X, legend=False),
                    self.name,
                    "prior",
                )

        return model

    def _run(self) -> Tuple[pd.DataFrame, List]:
        model = self._prepare_priors(key=self._prior_key_map[self.alg_key])
        alg_config = self._prepare_alg_config(key=self.alg_key, model=model, T=self.T)

        # prepare model and objects
        result = {}
        figs = []
        data = ExperimentData()
        data.init(
            d=self.d,
            q=model.q,
            n=model.n if isinstance(model, MarginalModel) else 0,
            metrics=self.metrics_fn(model=model, f=self.f, x=None),
        )
        if self.track_figs and self.plot_fn is not None:
            figs.append(
                self.plot_fn(
                    model=model,
                    D=Dataset(),
                    f=self.f,
                    title=f"{alg_config.name} after 0 iterations",
                )
            )
            store_and_show_fig("", figs[-1], self.name, f"{alg_config.name}__0_iters")
        if alg_config.track_information_ratios:
            if isinstance(model, MarginalModel) and isinstance(
                alg_config.alg, DirectedDiscreteGreedyAlgorithm
            ):
                data.strong_information_ratio = [jnp.inf]
                data.information_ratio = [jnp.inf]
            else:
                raise NotImplementedError

        # execute algorithm
        for t in tqdm(range(alg_config.T)):
            # execute step
            start_time = time.process_time()
            step_data = alg_config.alg.step(f=self.f)
            iter_time = time.process_time() - start_time

            if self.check_calibration:
                # check if updated model is well-calibrated
                if not self._check_calibration(model) or (
                    "marginal_model" in step_data.keys()
                    and not self._check_calibration(step_data["marginal_model"])
                ):
                    warning = f"ITER {t}: The model is not well-calibrated."
                    warn(warning)

            # store data
            if "marginal_model" in step_data.keys() and "argmax" in step_data.keys():
                metrics = self.metrics_fn(
                    model=model,
                    f=self.f,
                    x=step_data["x"],
                    marginal_model=step_data["marginal_model"],
                    argmax=step_data["argmax"],
                )
            else:
                metrics = self.metrics_fn(model=model, f=self.f, x=step_data["x"])
            data.add(
                x=step_data["x"],
                y=step_data["y"],
                F=step_data["F"],
                F_max=step_data["F_max"],
                metrics=metrics,
                iter_time=iter_time,
            )
            if self.plot_fn is not None and (self.track_figs or t == alg_config.T - 1):
                D = Dataset(
                    X=jnp.array(data.X),
                    y=jnp.array(
                        data.Y
                    ),  # TODO: handle multiple unknown functions properly
                )
                fig = self.plot_fn(
                    model,
                    D=D,
                    f=self.f,
                    title=f"{alg_config.name} after {t+1} Iterations",
                )
                figs.append(fig)
                if fig is not None:
                    store_and_show_fig(
                        "", fig, self.name, f"{alg_config.name}__{t+1}_iters"
                    )

            # recompute information ratios
            if (
                alg_config.track_information_ratios
                and isinstance(model, MarginalModel)
                and isinstance(alg_config.alg, DirectedDiscreteGreedyAlgorithm)
            ):
                data.strong_information_ratio.append(
                    compute_strong_information_ratio(
                        F_max=data.F_max[1:], prev=data.strong_information_ratio[-1]
                    )  # type: ignore
                )
                data.information_ratio.append(
                    compute_information_ratio(
                        model=model,
                        roi_idx=model.get_indices(alg_config.alg.roi),
                        sample_region_idx=model.get_indices(model.operating_safe_set),
                    )  # type: ignore
                )

        result = data.to_dict()
        # with open(f"cache/{self.name}/results.pkl", "wb") as f:
        #     pickle.dump(result, f)
        return (
            pd.DataFrame(result).add_prefix(
                f"{alg_config.name}."
            ),  # .astype("float64"),
            figs,
        )

    def run(self):
        df, figs = self._run()
        Path(f"cache/{self.name}").mkdir(parents=True, exist_ok=True)
        df.to_pickle(f"cache/{self.name}/results")


class DiscreteExperiment(Experiment):
    def __init__(self, domain: Float[Array, "n d"], **kwargs):
        super().__init__(**kwargs)
        self.domain = domain

    def _prepare_model(
        self, key: str, X: Float[Array, "m d"], y: Float[Array, "q m"]
    ) -> Model:
        prior_distrs = prior_distr(
            noise_rate=self.noise_rate,
            domain=self.domain,
            X=X,
            y=y,
            kernels=self.prior_kernels,
            means=self.prior_means,
        )

        kwargs = {
            "key": self.acquire_key(),
            "domain": self.domain,
            "distrs": prior_distrs,
            "beta": self.beta,
            "noise_rate": self.noise_rate,
            "use_objective_as_constraint": self.use_objective_as_constraint,
        }

        if key == "base":
            return MarginalModel(**kwargs)
        elif key == "unsafe_base":
            return UnsafeModel(**kwargs)
        elif key == "SafeOptModel":
            assert self.safe_opt_config is not None
            return SafeOptModel(**kwargs, L=self.safe_opt_config["prior_L"])
        elif key == "OracleSafeOptModel":
            assert self.safe_opt_config is not None
            return SafeOptModel(**kwargs, L=self.safe_opt_config["oracle_L"])
        elif key == "HeuristicSafeOptModel":
            return HeuristicSafeOptModel(**kwargs)
        elif key == "GoOSEModel":
            assert self.safe_opt_config is not None
            return GoOSEModel(**kwargs, L=self.safe_opt_config["prior_L"], epsilon=1e-3)
        elif key == "OracleGoOSEModel":
            assert self.safe_opt_config is not None
            return GoOSEModel(
                **kwargs, L=self.safe_opt_config["oracle_L"], epsilon=1e-3
            )
        elif key == "TruVarModel":
            assert self.tru_var_config is not None
            return TruVarModel(
                **kwargs,
                eta=self.tru_var_config["eta"],
                delta=self.tru_var_config["delta"],
                r=self.tru_var_config["r"],
            )
        else:
            raise NotImplementedError

    def _check_calibration(self, model: MarginalModel) -> ScalarBool:
        return not isinstance(self.f, SyntheticFunction) or jnp.all(
            model.calibration(f=self.f)
        )


class ContinuousExperiment(Experiment):
    def _prepare_model(
        self, key: str, X: Float[Array, "m d"], y: Float[Array, "q m"]
    ) -> Model:
        kwargs = {
            "key": self.acquire_key(),
            "d": self.d,
            "prior_kernels": self.prior_kernels,
            "prior_means": self.prior_means,
            "X": X,
            "y": y,
            "beta": self.beta,
            "noise_rate": self.noise_rate,
            "use_objective_as_constraint": self.use_objective_as_constraint,
        }

        if key == "base":
            return ContinuousModel(**kwargs)
        elif key == "unsafe_base":
            return UnsafeContinuousModel(**kwargs)
        elif key == "SafeOptModel":
            assert self.safe_opt_config is not None
            return SafeOptContinuousModel(
                **kwargs,
                L=self.safe_opt_config["prior_L"],
                prior_n_samples=self.prior_n_samples,
            )
        elif key == "OracleSafeOptModel":
            assert self.safe_opt_config is not None
            return SafeOptContinuousModel(
                **kwargs,
                L=self.safe_opt_config["oracle_L"],
                prior_n_samples=self.prior_n_samples,
            )
        elif key == "HeuristicSafeOptModel":
            return HeuristicSafeOptContinuousModel(**kwargs)
        elif key == "GoOSEModel":
            assert self.safe_opt_config is not None
            return GoOSEContinuousModel(
                **kwargs,
                L=self.safe_opt_config["prior_L"],
                epsilon=1e-3,
                prior_n_samples=self.prior_n_samples,
            )
        elif key == "OracleGoOSEModel":
            assert self.safe_opt_config is not None
            return GoOSEContinuousModel(
                **kwargs,
                L=self.safe_opt_config["oracle_L"],
                epsilon=1e-3,
                prior_n_samples=self.prior_n_samples,
            )
        elif key == "TruVarModel":
            assert self.tru_var_config is not None
            return TruVarContinuousModel(
                **kwargs,
                eta=self.tru_var_config["eta"],
                delta=self.tru_var_config["delta"],
                r=self.tru_var_config["r"],
            )
        else:
            raise NotImplementedError

    def _check_calibration(self, model: Model) -> ScalarBool:
        return (
            not isinstance(self.f, SyntheticFunction)
            or not isinstance(model, MarginalModel)
            or jnp.all(model.calibration(f=self.f))
        )
