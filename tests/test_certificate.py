import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feetgp import admm
from feetgp.glassogp import (
    GroupLassoGaussianProcess,
    hetgpy_auto_bounds,
    kkt_certificate,
    penalized_objective,
    theta_box,
)

jax.config.update("jax_enable_x64", True)

G_MIN, G_MAX = jnp.array(1e-4), jnp.array(100.0)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.uniform(size=(24, 12)))
    y = jnp.asarray(rng.normal(size=(24, 12)))
    return x, y


def test_certificate_degrades_away_from_the_fit(toy_data):
    x_train, y_train = toy_data
    group_size = 3
    model, _, state, info = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(0.0),
        group_size=group_size,
        autoregressive=False,
        max_iterations=200,
        chunk_size=4,
    )
    certificate = info["certificate"]
    assert np.isfinite(certificate["max_live_kkt"])
    assert np.isfinite(certificate["nugget_grad"])

    # shrinking theta moves off the optimum while staying inside the box
    lower, _ = hetgpy_auto_bounds(x_train)
    o, d_times_g = model.theta.shape
    bounds = theta_box(lower, o, d_times_g, group_size)
    perturbed = kkt_certificate(
        0.5 * model.theta,
        state.aux,
        jnp.array(0.0),
        x_train,
        y_train,
        group_size,
        bounds,
        g_min=G_MIN,
        g_max=G_MAX,
        chunk_size=4,
    )
    assert perturbed["max_live_kkt"] > 5 * certificate["max_live_kkt"]


def test_certificate_dead_groups_carry_trivial_slack(toy_data):
    x_train, y_train = toy_data
    l1 = 50.0
    model, _, _, info = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(l1),
        group_size=3,
        autoregressive=False,
        max_iterations=200,
        chunk_size=4,
    )
    assert np.allclose(model.theta, 0.0)

    # evenness forces grad f(0) = 0, so the slack is exactly l1: dead groups are
    # certified vacuously, which is the documented reason multi-start exists
    certificate = info["certificate"]
    assert np.all(np.isnan(certificate["live_kkt"]))
    assert np.allclose(certificate["dead_slack"], l1)
    assert certificate["max_live_kkt"] == 0.0


def test_penalized_objective_matches_fit_loglik(toy_data):
    x_train, y_train = toy_data
    group_size = 3
    model, llk, _, _ = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(0.0),
        group_size=group_size,
        autoregressive=False,
        max_iterations=50,
        chunk_size=4,
    )
    at_zero = penalized_objective(
        model.theta, model.g, jnp.array(0.0), x_train, y_train, group_size
    )
    assert np.isclose(at_zero, -llk)

    norms = np.linalg.norm(np.asarray(admm.to_groups(model.theta, group_size)), axis=-1)
    at_one = penalized_objective(
        model.theta, model.g, jnp.array(1.0), x_train, y_train, group_size
    )
    assert np.isclose(at_one - at_zero, norms.sum())
