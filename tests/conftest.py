import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange, reduce

from feetgp.gp import gp_loglikelihood, kernel

PROFILES = ["rbf", "matern52"]


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.uniform(size=(24, 4, 3)))
    y = jnp.asarray(rng.normal(size=(24, 12)))
    return x, y


@pytest.fixture(params=PROFILES)
def profile(request):
    return request.param


def group_norms(x) -> np.ndarray:
    return np.asarray(jnp.sqrt(reduce(x**2, "... g -> g", "sum")))


def flat_design(x):
    return rearrange(x, "n d g -> n (d g)")


def x_update_objective(profile, design, y, theta_target, rho):
    """The augmented Lagrangian one output minimises inside a fit's x update."""

    def objective(theta_and_log_nugget):
        theta = theta_and_log_nugget[:-1]
        nugget = jnp.exp(theta_and_log_nugget[-1])
        Koo = kernel(profile, theta, design, design) + nugget * jnp.eye(len(y))
        loglik, _, _ = gp_loglikelihood(Koo, y)
        return -loglik + rho * 0.5 * jnp.sum((theta - theta_target) ** 2)

    return objective


def negative_loglikelihood_grad(profile, design, y):
    """Value and gradient of the unpenalized nll in theta, as the fit computes it."""

    def loss(theta, log_nugget):
        Koo = kernel(profile, theta, design, design)
        Koo = Koo + jnp.exp(log_nugget) * jnp.eye(len(y))
        loglik, _, _ = gp_loglikelihood(Koo, y)
        return -loglik

    return jax.value_and_grad(loss)
