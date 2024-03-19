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

BETA = jnp.array([5])

LOW = -2
HIGH = 8
DOMAIN = jnp.linspace(LOW, HIGH, 500).reshape(-1, 1)

PRIOR_SAMPLE_REGION = DOMAIN[(DOMAIN[:, 0] > 4.5) & (DOMAIN[:, 0] < 6)]
LENGTHSCALE = 1
NOISE_STD = 1e-1

ORACLE_L = jnp.array([1.54725186], dtype=jnp.float64)
PRIOR_L = jnp.array([2.50495534], dtype=jnp.float64)

T = 1_000


def experiment(seed: int, alg: str | None, name: str):
    wandb.init(
        name="unknown_constraints/bo/1d_hard",
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
    mean = means.ZeroMean()
    kernel = kernels.stationary.Gaussian(lengthscale=LENGTHSCALE)

    @jit
    def f_oracle(x: Float[Array, "1"]) -> Float[Array, "1"]:
        return (
            5 * jstats.norm.pdf(x, 0.4, 0.85)
            + 1.45 * jstats.norm.pdf(x, 2.5, 0.9)
            + 1.8 * jstats.norm.pdf(x, 3.2, 0.55)
            + 4.5 * jstats.norm.pdf(x, 5.5, 0.93)
            - 0.95
        )

    noise_rate = HomoscedasticNoise(q=1, noise_rates=jnp.array([NOISE_STD]))
    key, subkey = jr.split(key)
    f = SyntheticFunction(
        key=subkey,
        q=1,
        f=f_oracle,
        noise_rate=noise_rate,
        noise_oracle=gaussian_noise,
        use_objective_as_constraint=True,
    )
    store_and_show_fig("", f.plot_1d(domain=DOMAIN), name, "ground_truth")
    store_and_show_fig("", f.plot_sigma_1d(domain=DOMAIN), name, "noise")

    key, subkey = jr.split(key)
    exp = DiscreteBOExperiment(
        domain=DOMAIN,
        key=subkey,
        name=name,
        d=1,
        f=f,
        noise_rate=noise_rate,
        use_objective_as_constraint=True,
        metrics_fn=bayesian_optimization_metrics,
        plot_fn=lambda model, D, f, title: model.plot_1d(D=D, f=f, title=title),
        beta=BETA,
        T=T,
        alg_key=alg,
        prior_means=[mean],
        prior_kernels=[kernel],
        prior_n_samples=8,
        prior_sample_key=jr.PRNGKey(8),
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
    parser.add_argument("--alg", type=str, default="SafeOpt")
    parser.add_argument("--name", type=str, default="unknown_constraints/bo/1d_hard")
    args = parser.parse_args()
    main(args)
