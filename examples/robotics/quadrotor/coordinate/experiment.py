import argparse
import time
from jax import jit
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax._src.config import config
import jax.random as jr
import wandb

from examples.jax_test import jax_has_gpu
from examples.robotics.algs import ContinuousBOExperiment
from examples.robotics.quadrotor.sim import ACTION_TARGET, QuadrotorOptimalCost
from examples.robotics.sim import Simulator, critical_damping, sample_disturbance_params
from lib.algorithms.line_bo import CoordinateLineBO
from lib.function.synthetic import SyntheticFunction
from lib.gp import kernels, means
from lib.metrics.bayesian_optimization import (
    continuous_bayesian_optimization_metrics_generator,
)
from lib.noise import HomoscedasticNoise, gaussian_noise
from lib.typing import ScalarFloat

config.update("jax_enable_x64", True)

BETA = jnp.array([1, 1])

D = 4
N = 100
T = 100

X0 = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
STATE_TARGET = jnp.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
Z_MIN = 0.5

LOW = 0.0
HIGH = 20.0

COST_THRESHOLD = 200.0

LENGTHSCALES = jnp.array([0.1, 0.1])
NOISE_STD = 1e-1

L = jnp.array([5.877, 5.514])


def experiment(seed: int, alg: str | None, name: str):
    wandb.init(
        name="robotics/quadrotor/coordinate",
        dir="/cluster/scratch/jhuebotter/wandb/idl",
        project="IDL",
        config={
            "T": T,
            "beta": BETA,
            "lengthscale": LENGTHSCALES,
            "noise_std": NOISE_STD,
            "seed": seed,
            "alg": alg,
        },
        # mode="offline"
    )

    jax_has_gpu()
    print("SEED:", seed, "ALG:", alg)

    key = jr.PRNGKey(seed)

    # key, subkey = jr.split(key)
    disturbance_params = sample_disturbance_params(
        key=jr.PRNGKey(0), disturbance_scale=1.0
    )

    # x_safe_seed = 2 * jnp.ones((D,))
    x_safe_seed = jnp.array([0.0, 0.0, 0.0, 10.0])
    # key, subkey = jr.split(key)
    # x_safe_seed = jnp.maximum(
    #     0, disturbance_params.proportional_params + jr.normal(key=key, shape=(D,))
    # )  # TOO EASY

    sim = Simulator(
        OC=QuadrotorOptimalCost,
        x0=X0,
        disturbance_params=disturbance_params,
        state_target=STATE_TARGET,
        action_target=ACTION_TARGET,
    )

    @jit
    def smoothed_objective(cost: ScalarFloat) -> ScalarFloat:
        """Transforms cost into smoothed objective, ensuring that the objective value is upper bounded by $1$ and lower bounded by $-1$ for costs below `COST_THRESHOLD`."""
        hct = COST_THRESHOLD / 2
        normalized_cost = (cost - hct) / hct
        smoothed_cost = jnp.tanh(normalized_cost)
        return -smoothed_cost

    @jit
    def f_oracle(x: Float[Array, "d"]) -> Float[Array, "2"]:
        feedback_params = critical_damping(proportional_params=x)
        cost, X = sim.rollout(feedback_params)
        objective = smoothed_objective(jnp.nan_to_num(cost, nan=jnp.inf))
        constraint = QuadrotorOptimalCost.constraint(X) - Z_MIN
        return jnp.array([objective, constraint])

    noise_rate = HomoscedasticNoise(q=2, noise_rates=jnp.array([NOISE_STD, NOISE_STD]))
    key, subkey = jr.split(key)
    f = SyntheticFunction(
        key=subkey,
        q=2,
        f=f_oracle,
        noise_rate=noise_rate,
        noise_oracle=gaussian_noise,
        use_objective_as_constraint=False,
    )

    x_opt = disturbance_params.proportional_params
    f_opt = f.evaluate(x_opt.reshape(1, -1))[0, 0]
    assert jnp.all(f.safe_set(x_opt.reshape(1, -1))), "Optimum is unsafe"
    assert jnp.all(f.safe_set(x_safe_seed.reshape(1, -1))), "Safe seed is unsafe"

    X = jnp.linspace(jnp.array([0.0, 0, 0, 0.0]), jnp.array([0.0, 0, 0, 10.0]))
    key, subkey = jr.split(key)
    k1 = kernels.stationary.Matern52(lengthscale=LENGTHSCALES[0])
    L1 = k1.estimate_L(key, n_samples=1_000, X=X)
    key, subkey = jr.split(key)
    k2 = kernels.stationary.Matern52(lengthscale=LENGTHSCALES[1])
    L2 = k2.estimate_L(key, n_samples=1_000, X=X)

    key, subkey = jr.split(key)
    exp = ContinuousBOExperiment(
        n=N,
        bounds=jnp.tile(jnp.array([LOW, HIGH]), (D, 1)),
        x_init=x_safe_seed,
        LineBOClass=CoordinateLineBO,
        key=subkey,
        name=name,
        d=D,
        f=f,
        noise_rate=noise_rate,
        use_objective_as_constraint=False,
        metrics_fn=continuous_bayesian_optimization_metrics_generator(x_opt, f_opt),
        plot_fn=lambda model, D, f, title: None,
        beta=BETA,
        T=T,
        alg_key=alg,
        prior_means=[means.ZeroMean(), means.ConstantMean(-Z_MIN)],
        prior_kernels=[k1, k2],
        prior_n_samples=10,
        safe_opt_config={"prior_L": jnp.array([L1, L2]), "oracle_L": L},
        track_figs=False,
    )
    exp.run()


def main(args):
    t_start = time.process_time()
    assert args.alg is not None
    experiment(seed=args.seed, alg=args.alg, name=args.name)
    print("Total time taken:", time.process_time() - t_start, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alg", type=str, default="ITL-PM [ours]")
    parser.add_argument("--name", type=str, default="robotics/quadrotor/coordinate")
    args = parser.parse_args()
    main(args)
