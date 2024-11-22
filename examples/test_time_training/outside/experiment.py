import argparse
import time
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
from jax._src.config import config
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
NOISE_STD = 1
ENTROPY_JITTER = 9e-17

ROI_DESCRIPTION = ROIDescription(jnp.array([[[-1.1, -1], [-0.05, 0.05]]]))

SAMPLE_REGION_MASK = (
    (DOMAIN[:, 0] >= 0)
    & (DOMAIN[:, 0] <= 3)
    # & (DOMAIN[:, 1] >= -1)
    # & (DOMAIN[:, 1] <= 3)
)
SAMPLE_REGION = DOMAIN[SAMPLE_REGION_MASK]

T = 500


def experiment(
    seed: int,
    alg: str | None,
    noise_std: float,
    kernel_name: str,
    lengthscale: float,
    name: str,
):
    wandb.init(
        name="test_time_training/outside-2",
        dir="/cluster/scratch/jhuebotter/wandb/idl",
        project="IDL",
        config={
            "T": T,
            "beta": BETA,
            "lengthscale": lengthscale,
            "noise_std": noise_std,
            "seed": seed,
            "alg": alg,
            "kernel": kernel_name,
            "jitter": ENTROPY_JITTER,
        },
        # mode="offline"
    )

    jax_has_gpu()
    print(
        "SEED:",
        seed,
        "ALG:",
        alg,
        "NOISE_STD:",
        noise_std,
        "KERNEL:",
        kernel_name,
        "LENGTHSCALE:",
        lengthscale,
    )

    if kernel_name == "Gaussian":
        kernel = kernels.stationary.Gaussian(lengthscale=lengthscale)
    elif kernel_name == "Laplace":
        kernel = kernels.stationary.Laplace(lengthscale=lengthscale)
    else:
        raise NotImplementedError

    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    mean = means.ZeroMean()
    vals = jr.multivariate_normal(
        subkey, mean.vector(DOMAIN), kernel.covariance(DOMAIN), method="svd"
    )

    def f_oracle(x: Float[Array, "d"]) -> Float[Array, "1"]:
        return vals[get_indices(DOMAIN, x.reshape(1, -1))].reshape(-1)

    noise_rate = HomoscedasticNoise(q=1, noise_rates=jnp.array([noise_std]))
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
    key, jitter_key = jr.split(key)
    exp = DiscreteTransductiveLearningExperiment(
        domain=DOMAIN,
        roi_description=ROI_DESCRIPTION,
        sample_region=SAMPLE_REGION,
        key=subkey,
        name=name,
        d=2,
        f=f,
        noise_rate=noise_rate,
        use_objective_as_constraint=False,
        metrics_fn=transductive_learning_metrics_generator(
            roi_mask, entropy_jitter={"amount": ENTROPY_JITTER, "key": jitter_key}
        ),
        plot_fn=lambda model, D, f, title: model.plot_2d(
            X=D.X,
            f=f,
            roi=roi_mask,
            sample_region=SAMPLE_REGION_MASK,
            title=title,
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
    experiment(
        seed=args.seed,
        alg=args.alg,
        noise_std=args.noise_std,
        kernel_name=args.kernel,
        lengthscale=args.lengthscale,
        name=args.name,
    )
    print("Total time taken:", time.process_time() - t_start, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alg", type=str, default="MM-ITL [ours]")
    parser.add_argument("--noise-std", type=float, default=NOISE_STD)
    parser.add_argument("--kernel", type=str, default="Gaussian")
    parser.add_argument("--lengthscale", type=float, default=1.0)
    parser.add_argument("--name", type=str, default="test_time_training/outside")
    args = parser.parse_args()
    main(args)
