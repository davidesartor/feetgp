import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange, repeat

from feetgp.gp import (
    GroupLassoGaussianProcess,
    hetgpy_auto_bounds,
    kkt_certificate,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.uniform(size=(24, 4, 3)))
    y = jnp.asarray(rng.normal(size=(24, 12)))
    return x, y


def test_certificate_degrades_away_from_the_fit(toy_data):
    x_train, y_train = toy_data
    model, _, state, info = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(0.0),
        autoregressive=False,
        max_iterations=200,
        chunk_size=4,
    )
    certificate = info["certificate"]
    assert np.isfinite(certificate["max_live_kkt"])
    assert np.isfinite(certificate["nugget_grad"])

    lower, _ = hetgpy_auto_bounds(rearrange(x_train, "n d g -> n (d g)"))
    theta_max = repeat(
        jnp.sqrt(2.0 / lower),
        "(d g) -> d (o g)",
        g=x_train.shape[2],
        o=y_train.shape[1],
    )
    bounds = jnp.stack([jnp.zeros_like(theta_max), theta_max], axis=0)
    perturbed = kkt_certificate(
        0.5 * model.theta,
        state.aux,
        jnp.array(0.0),
        x_train,
        y_train,
        bounds,
        chunk_size=4,
    )
    assert perturbed["max_live_kkt"] > 5 * certificate["max_live_kkt"]


def test_certificate_dead_groups_carry_trivial_slack(toy_data):
    x_train, y_train = toy_data
    l1 = 100.0
    model, _, _, info = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(l1),
        autoregressive=False,
        max_iterations=200,
        chunk_size=4,
    )
    assert np.allclose(model.theta, 0.0)

    certificate = info["certificate"]
    assert np.all(np.isnan(certificate["live_kkt"]))
    assert np.allclose(certificate["dead_slack"], l1)
    assert certificate["max_live_kkt"] == 0.0
