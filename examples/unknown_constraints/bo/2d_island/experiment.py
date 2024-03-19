import argparse
import time
from jax import jit
import jax.numpy as jnp
from jaxtyping import Array, Float
import jax.scipy.stats as jstats
from jax._src.config import config
import jax.random as jr
import wandb

from examples.jax_test import jax_has_gpu
from examples.experiment.discrete_bo import DiscreteBOExperiment
from lib.function.synthetic import SyntheticFunction
from lib.gp import kernels, means
from lib.metrics.bayesian_optimization import bayesian_optimization_metrics
from lib.noise import HomoscedasticNoise, gaussian_noise
from lib.plotting import store_and_show_fig

config.update("jax_enable_x64", True)

BETA = jnp.array([15])

LOW = -3
HIGH = 3
N = 50
DENSITY = (HIGH - LOW) / N
DOMAIN = jnp.mgrid[LOW:HIGH:DENSITY, LOW:HIGH:DENSITY].reshape(2, -1).T

PRIOR_SAMPLE_REGION = DOMAIN[
    (
        (DOMAIN[:, 0] >= -0.5)
        & (DOMAIN[:, 0] <= 0.5)
        & (DOMAIN[:, 1] >= -0.5)
        & (DOMAIN[:, 1] <= 0.5)
    )
]
LENGTHSCALE = 1
NOISE_STD = 1e-1

ORACLE_L = jnp.array([11.66914771, 7.8560165], dtype=jnp.float64)
PRIOR_L = jnp.array([63.15282346, 80.97604292], dtype=jnp.float64)

T = 100


def experiment(seed: int, alg: str | None, name: str):
    wandb.init(
        name="unknown_constraints/bo/2d_island",
        dir="/cluster/scratch/jhuebotter/wandb/idl",
        project="IDL",
        config={
            "T": T,
            "beta": BETA,
            "lengthscale": LENGTHSCALE,
            "noise_std": NOISE_STD,
            "seed": seed,
            "alg": alg,
        },
        # mode="offline"
    )

    jax_has_gpu()
    print("SEED:", seed, "ALG:", alg)

    key = jr.PRNGKey(seed)

    @jit
    def f_oracle(x: Float[Array, "d"]) -> Float[Array, "2"]:
        d = x.shape[0]
        f = 100 * jstats.multivariate_normal.pdf(
            x,
            jnp.array([2, 2], dtype=jnp.float64),
            jnp.array([[1, 0.5], [0.5, 1]], dtype=jnp.float64),
        )
        g = 100 * jstats.multivariate_normal.pdf(x, jnp.zeros(d), jnp.identity(d)) - 2.5
        return jnp.array([f, g])

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
    store_and_show_fig("", f.plot_2d(domain=DOMAIN), name, "ground_truth")

    mean = means.ZeroMean()
    kernel = kernels.stationary.Gaussian(lengthscale=LENGTHSCALE)

    key, subkey = jr.split(key)
    exp = DiscreteBOExperiment(
        domain=DOMAIN,
        key=subkey,
        name=name,
        d=2,
        f=f,
        noise_rate=noise_rate,
        use_objective_as_constraint=False,
        metrics_fn=bayesian_optimization_metrics,
        plot_fn=lambda model, D, f, title: model.plot_2d(X=D.X, f=f, title=title),
        beta=BETA,
        T=T,
        alg_key=alg,
        prior_means=[mean, mean],
        prior_kernels=[kernel, kernel],
        prior_n_samples=1,
        prior_sample_region=PRIOR_SAMPLE_REGION,
        safe_opt_config={"prior_L": PRIOR_L, "oracle_L": ORACLE_L},
        track_figs=False,
    )
    exp.run()


def main(args):
    t_start = time.process_time()
    experiment(seed=args.seed, alg=args.alg, name=args.name)
    print("Total time taken:", time.process_time() - t_start, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alg", type=str, default="Oracle SafeOpt")
    parser.add_argument("--name", type=str, default="unknown_constraints/bo/2d_island")
    args = parser.parse_args()
    main(args)
