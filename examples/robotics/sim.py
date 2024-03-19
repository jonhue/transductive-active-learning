from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, NamedTuple, Type
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
from matplotlib import gridspec
import matplotlib.pyplot as plt
from jax import jit, vmap
from trajax import optimizers

from lib.typing import ScalarFloat, ScalarInt
from lib.utils import std_error
from lib.plotting import store_and_show_fig


class ModelFeedBackParams(NamedTuple):
    proportional_params: Float[Array, "u_dim"]
    derivative_params: Float[Array, "u_dim"]


class OptimalCost(ABC):
    x_dim: int
    u_dim: int
    num_nodes: int
    time_horizon: Tuple[float, float]
    dt: float
    ts: Float[Array, "N"]
    ilqr_params: optimizers.ILQRHyperparams
    ilqr: optimizers.ILQR

    def __init__(
        self,
        time_horizon: Tuple[float, float],
        num_nodes: int = 50,
        sim_params: dict = {},
    ):
        self.num_nodes = num_nodes
        self.time_horizon = time_horizon

        self.dt = (self.time_horizon[1] - self.time_horizon[0]) / num_nodes
        self.ts = jnp.linspace(
            self.time_horizon[0], self.time_horizon[1], num_nodes + 1
        )
        self.ilqr_params = optimizers.ILQRHyperparams(
            maxiter=1000, make_psd=False, psd_delta=1e0
        )
        self.ilqr = optimizers.ILQR(self.cost, self.next_step)

    @abstractmethod
    def cost(
        self,
        x: Float[Array, "x_dim"],
        u: Float[Array, "u_dim"],
        t: ScalarFloat,
        params: dict | None = None,
    ):
        pass

    @abstractmethod
    def next_step(
        self,
        x: Float[Array, "x_dim"],
        u: Float[Array, "u_dim"],
        t: ScalarFloat,
        params: dict | None = None,
    ) -> Float[Array, "x_dim"]:
        pass

    @partial(jit, static_argnums=0)
    def solve(self, x0: Float[Array, "x_dim"]) -> optimizers.ILQRResult:
        initial_U = jnp.zeros(
            shape=(
                self.num_nodes,
                self.u_dim,
            )
        )
        out = self.ilqr.solve(
            cost_params=None,
            dynamics_params=None,
            x0=x0,
            U=initial_U,
            hyperparams=self.ilqr_params,
        )
        return out

    @partial(jit, static_argnums=0)
    def evaluate(
        self,
        X: Float[Array, "N x_dim"],
        U: Float[Array, "N u_dim"],
    ) -> ScalarFloat:
        last_row = U[-1:]
        U_repeated = jnp.concatenate([U, last_row], axis=0)
        cost = optimizers.evaluate(cost=self.cost, X=X, U=U_repeated)
        return jnp.sum(cost)

    @partial(jit, static_argnums=0)
    def rollout(
        self,
        x0: Float[Array, "x_dim"],
        U: Float[Array, "N u_dim"],
        disturbance_params: ModelFeedBackParams,
        feedback_params: ModelFeedBackParams,
    ) -> Tuple[ScalarFloat, Float[Array, "N x_dim"]]:
        dynamics_params = {
            "disturbance_params": disturbance_params,
            "feedback_params": feedback_params,
        }
        X = self.ilqr._rollout(dynamics_params=dynamics_params, U=U, x0=x0)
        cost = self.evaluate(X=X, U=U)
        return cost, X

    @classmethod
    @abstractmethod
    def _constraint(cls, x: Float[Array, "x_dim"]) -> ScalarFloat:
        pass

    @classmethod
    @partial(jit, static_argnums=0)
    def constraint(cls, X: Float[Array, "N x_dim"]) -> ScalarFloat:
        fs = vmap(cls._constraint)(X)
        return jnp.min(fs)


class Simulator:
    x0: Float[Array, "x_dim"]
    disturbance_params: ModelFeedBackParams
    oc: OptimalCost
    U: Float[Array, "N u_dim"]

    def __init__(
        self,
        OC: Type[OptimalCost],
        x0: Float[Array, "x_dim"],
        state_target: Float[Array, "x_dim"],
        action_target: Float[Array, "u_dim"],
        disturbance_params: ModelFeedBackParams,
        T=3,
        N=100,
    ):
        self.x0 = x0
        self.disturbance_params = disturbance_params

        sim_params = {
            "state_target": state_target,
            "action_target": action_target,
        }
        self.oc = OC(time_horizon=(0, T), num_nodes=N, sim_params=sim_params)
        out = self.oc.solve(x0)
        self.U = out.us

    @partial(jit, static_argnums=0)
    def rollout(
        self, feedback_params: ModelFeedBackParams
    ) -> Tuple[ScalarFloat, Float[Array, "N x_dim"]]:
        return self.oc.rollout(
            x0=self.x0,
            U=self.U,
            disturbance_params=self.disturbance_params,
            feedback_params=feedback_params,
        )

    @classmethod
    def plot_feedback_params_safety(
        cls,
        OC: Type[OptimalCost],
        name: str,
        x0: Float[Array, "x_dim"],
        state_target: Float[Array, "x_dim"],
        action_target: Float[Array, "u_dim"],
        fmin: ScalarFloat,
        disturbance_scale: ScalarFloat,
        n_samples=100,
        P=jnp.linspace(0.0, 100.0, 100),
        ylabel="Constraint",
    ):
        def engine(k_p: ScalarFloat, seed: ScalarInt) -> ScalarFloat:
            key = jr.PRNGKey(seed=seed)
            disturbance_params = sample_disturbance_params(
                key=key, disturbance_scale=disturbance_scale
            )
            feedback_params = critical_damping(
                proportional_params=jnp.full((OC.u_dim,), k_p)
            )
            cost, X = cls(
                OC=OC,
                x0=x0,
                disturbance_params=disturbance_params,
                state_target=state_target,
                action_target=action_target,
            ).rollout(feedback_params)
            return OC.constraint(X)

        results = vmap(
            lambda k_p: vmap(lambda seed: engine(k_p, seed))(jnp.arange(n_samples))
        )(P)
        mean = jnp.mean(results, axis=1)
        std = std_error(results, axis=1)  # type: ignore

        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])

        ax.set_xlabel("$k_p$")
        ax.set_ylabel(ylabel)
        ax.axhline(y=fmin, linestyle="--", color="black")  # type: ignore
        ax.plot(P, mean, color="blue")
        ax.fill_between(P, mean - std, mean + std, color="blue", alpha=0.3)  # type: ignore
        store_and_show_fig("", fig, f"robotics/{name}", "Safety per k_p")
        plt.close()


def critical_damping(proportional_params: Float[Array, "u_dim"]) -> ModelFeedBackParams:
    derivative_params = 2 * jnp.sqrt(proportional_params)
    return ModelFeedBackParams(
        proportional_params=proportional_params, derivative_params=derivative_params
    )


def sample_disturbance_params(
    key: jr.KeyArray, disturbance_scale: ScalarFloat
) -> ModelFeedBackParams:
    proportional_params = disturbance_scale * jnp.square(jr.normal(key, shape=(4,)))
    disturbance_params = critical_damping(proportional_params)
    return disturbance_params
