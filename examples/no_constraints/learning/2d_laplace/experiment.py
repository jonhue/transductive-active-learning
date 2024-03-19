import argparse
import time
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax._src.config import config
import jax.random as jr
import wandb

from examples.jax_test import jax_has_gpu
from examples.experiment.transductive_learning import (
    DiscreteTransductiveLearningExperiment,
)
from lib.algorithms import ROIDescription, compute_roi_mask
from lib.function.synthetic import SyntheticFunction
from lib.gp import kernels, means
from lib.metrics import transductive_learning_metrics_generator
from lib.noise import HomoscedasticNoise, gaussian_noise
from lib.plotting import store_and_show_fig
from lib.utils import get_indices

config.update("jax_enable_x64", True)

BETA = jnp.array([1.0])

LOW = -3
HIGH = 3
N = 50
DENSITY = (HIGH - LOW) / N
DOMAIN = jnp.mgrid[LOW:HIGH:DENSITY, LOW:HIGH:DENSITY].reshape(2, -1).T

PRIOR_SAMPLE_REGION = DOMAIN
LENGTHSCALE = 10
NOISE_STD = 1e-1

ROI_DESCRIPTION = ROIDescription(jnp.array([[[-1, 1], [-1, 1]]]))

T = 500


def experiment(seed: int, alg: str | None, name: str):
    wandb.init(
        name="no_constraints/learning/2d_laplace",
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
    key, subkey = jr.split(key)
    mean = means.ZeroMean()
    kernel = kernels.stationary.Laplace(lengthscale=LENGTHSCALE)
    vals = jr.multivariate_normal(
        subkey, mean.vector(DOMAIN), kernel.covariance(DOMAIN), method="svd"
    )

    def f_oracle(x: Float[Array, "d"]) -> Float[Array, "1"]:
        return vals[get_indices(DOMAIN, x.reshape(1, -1))].reshape(-1)

    noise_rate = HomoscedasticNoise(q=1, noise_rates=jnp.array([NOISE_STD]))
    key, subkey = jr.split(key)
    f = SyntheticFunction(
        key=subkey,
        q=1,
        f=f_oracle,
        noise_rate=noise_rate,
        noise_oracle=gaussian_noise,
        use_objective_as_constraint=False,
    )
    store_and_show_fig("", f.plot_2d(DOMAIN), name, "ground_truth")

    roi_mask = compute_roi_mask(DOMAIN, ROI_DESCRIPTION)
    key, subkey = jr.split(key)
    exp = DiscreteTransductiveLearningExperiment(
        domain=DOMAIN,
        roi_description=ROI_DESCRIPTION,
        key=subkey,
        name=name,
        d=2,
        f=f,
        noise_rate=noise_rate,
        use_objective_as_constraint=False,
        metrics_fn=transductive_learning_metrics_generator(roi_mask),
        plot_fn=lambda model, D, f, title: model.plot_2d(
            domain=DOMAIN, X=D.X, f=f, roi=roi_mask, title=title
        ),
        beta=BETA,
        T=T,
        alg_key=alg,
        prior_means=[mean],
        prior_kernels=[kernel],
        prior_n_samples=0,
        prior_sample_region=PRIOR_SAMPLE_REGION,
        # safe_opt_config={"prior_L": PRIOR_L, "oracle_L": ORACLE_L},
        tru_var_config={"eta": 1, "delta": 0, "r": 0.1},
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
    parser.add_argument("--alg", type=str, default="Uniform Sampling")
    parser.add_argument(
        "--name", type=str, default="no_constraints/learning/2d_laplace"
    )
    args = parser.parse_args()
    main(args)
