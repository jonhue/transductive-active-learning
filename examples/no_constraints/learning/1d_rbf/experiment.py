import argparse
import time
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax._src.config import config
import jax.random as jr

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

BETA = jnp.array([5.0])

LOW = -5
HIGH = 5
DOMAIN = jnp.linspace(LOW, HIGH, 500).reshape(-1, 1)

PRIOR_SAMPLE_REGION = DOMAIN

ROI_DESCRIPTION = ROIDescription(jnp.array([[[-4, -2]], [[2, 4]]]))

T = 100


def experiment(seed: int, alg: str, name: str):
    jax_has_gpu()
    print("SEED:", seed, "ALG:", alg)

    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    mean = means.ZeroMean()
    kernel = kernels.stationary.Gaussian(lengthscale=1)
    vals = jr.multivariate_normal(
        subkey, mean.vector(DOMAIN), kernel.covariance(DOMAIN), method="svd"
    )

    def f_oracle(x: Float[Array, "d"]) -> Float[Array, "1"]:
        return vals[get_indices(DOMAIN, x.reshape(1, -1))].reshape(-1)

    noise_rate = HomoscedasticNoise(q=1, noise_rates=jnp.array([1e-1]))
    key, subkey = jr.split(key)
    f = SyntheticFunction(
        key=subkey,
        q=1,
        f=f_oracle,
        noise_rate=noise_rate,
        noise_oracle=gaussian_noise,
        use_objective_as_constraint=False,
    )
    store_and_show_fig("", f.plot_1d(DOMAIN), name, "ground_truth")
    store_and_show_fig("", f.plot_sigma_1d(DOMAIN), name, "noise")

    roi_mask = compute_roi_mask(DOMAIN, ROI_DESCRIPTION)
    key, subkey = jr.split(key)
    exp = DiscreteTransductiveLearningExperiment(
        domain=DOMAIN,
        roi_description=ROI_DESCRIPTION,
        key=subkey,
        name=name,
        d=1,
        f=f,
        noise_rate=noise_rate,
        use_objective_as_constraint=False,
        metrics_fn=transductive_learning_metrics_generator(roi_mask),
        plot_fn=lambda model, D, f, title: model.plot_1d(
            domain=DOMAIN, D=D, f=f, title=title
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
    parser.add_argument("--name", type=str, default="no_constraints/learning/1d_rbf")
    args = parser.parse_args()
    main(args)
