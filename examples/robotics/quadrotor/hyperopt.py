import argparse
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
from jax._src.config import config
import optax

from lib.gp import kernels
from lib.gp.hyperopt import optimize_theta, negative_log_likelihood
from lib.noise import gaussian_noise
from examples.robotics.algs import ContinuousBOExperiment
from examples.robotics.quadrotor.sim import ACTION_TARGET, QuadrotorOptimalCost
from examples.robotics.sim import Simulator, critical_damping, sample_disturbance_params
from lib.typing import ScalarFloat

config.update("jax_enable_x64", True)

X0 = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
STATE_TARGET = jnp.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
Z_MIN = 0.8

LOW = 0.0
HIGH = 100.0

COST_THRESHOLD = 200.0

NOISE_STD = 1e-1
MEAN = -0.4


def main(kernel, fn, n, o=True):
    print("KERNEL", kernel, "FN", fn, "N", n)

    key = jr.PRNGKey(0)
    key, subkey = jr.split(key)
    disturbance_params = sample_disturbance_params(key=subkey, disturbance_scale=1.0)

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

    key, subkey = jr.split(key)
    X = jr.uniform(subkey, (n, 4), minval=LOW, maxval=HIGH)
    f = vmap(f_oracle)(X)[:, fn]
    y = f + NOISE_STD * gaussian_noise(key, (n,))

    if kernel == "Gaussian":
        K = kernels.stationary.Gaussian
    elif kernel == "Laplace":
        K = kernels.stationary.Laplace
    elif kernel == "Matern32":
        K = kernels.stationary.Matern32
    elif kernel == "Matern52":
        K = kernels.stationary.Matern52
    else:
        raise NotImplementedError

    params, nlls = optimize_theta(
        K=K,
        params={"lengthscale": 1.0},
        X=X,
        y=y,
        noise_std=NOISE_STD,
        mean=MEAN,
        optimizer=optax.sgd(learning_rate=0.1),
        num_iters=300,
        fit_noise_std=o,
    )
    print(params)

    # num_subsets = 100
    # subset_indices = jr.choice(key, X_full.shape[0], shape=(num_subsets, n), replace=True)
    # def test(indices):
    #   X = jnp.take(X_full, indices, axis=0)
    #   y = jnp.take(y_full, indices, axis=0)
    #   return negative_log_likelihood(theta=params, K=K, X=X, y=y, noise_std=NOISE_STD, mean=MEAN)
    # result = vmap(lambda indices: test(indices))(subset_indices)
    # print("NLL", jnp.mean(result), "±", jnp.std(result) / jnp.sqrt(num_subsets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for stationary kernels."
    )
    parser.add_argument("kernel", help="Stationary kernel to fit.")
    parser.add_argument("-f", "--fn", help="Function.", default=0, type=int)
    parser.add_argument(
        "-n", "--n", help="Number of training samples.", default=1_000, type=int
    )
    parser.add_argument(
        "-o",
        "--obs-noise-fit",
        help="Fit observation noise.",
        nargs="?",
        const=True,
        default=False,
        type=bool,
    )

    args = parser.parse_args()

    main(args.kernel, args.fn, args.n, args.obs_noise_fit)
