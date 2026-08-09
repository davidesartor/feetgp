"""Sanity checks that a GP still fits, and stays numerically sane, in float32."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange

from feetgp.gp import (  # noqa: F401
    GaussianProcess,
    gp_loglikelihood,
    kernel,
)


@pytest.fixture
def smooth_data():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.uniform(size=(96, 2, 3)))
    flat = rearrange(x, "n d g -> n (d g)")
    y = jnp.stack(
        [
            jnp.sin(2.0 * flat[:, 0]) + flat[:, 3] ** 2,
            jnp.cos(3.0 * flat[:, 1]) - 0.5 * flat[:, 4],
        ],
        axis=1,
    )
    return x[:64], y[:64], x[64:], y[64:]


def test_default_precision_is_float32():
    assert not jax.config.jax_enable_x64
    assert jnp.zeros(1).dtype == jnp.float32


def test_kernel_is_psd_on_near_duplicate_points(profile):
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.uniform(size=(1, 8))).repeat(64, axis=0)
    x = x + jnp.asarray(rng.normal(scale=1e-4, size=x.shape))
    theta = jnp.asarray(rng.uniform(0.5, 2.0, size=(8,)))

    # the expanded distance cancels here, so an unclamped kernel exceeds one
    K = kernel(profile, theta, x, x)
    assert K.dtype == jnp.float32
    assert K.max() <= 1.0
    assert np.allclose(np.diag(K), 1.0)
    assert np.isfinite(np.linalg.cholesky(np.asarray(K + 1e-3 * jnp.eye(len(x))))).all()


def test_loglikelihood_is_finite_at_the_smallest_nugget(profile, smooth_data):
    x_train, y_train, _, _ = smooth_data
    flat = rearrange(x_train, "n d g -> n (d g)")

    # theta at zero is the group lasso limit: an all ones kernel plus the nugget
    for theta in (jnp.zeros(flat.shape[1]), jnp.ones(flat.shape[1])):
        Koo = kernel(profile, theta, flat, flat) + 1e-3 * jnp.eye(len(flat))
        loglik, b, nu = gp_loglikelihood(Koo, y_train[:, 0])
        assert np.isfinite([loglik, b, nu]).all(), theta
        assert nu > 0.0


def test_fit_predicts_held_out_data(profile, smooth_data):
    x_train, y_train, x_test, y_test = smooth_data
    model, nll, _, _ = GaussianProcess.fit(
        x_train, y_train, profile=profile, l1_penalty=jnp.array(0.0), max_iterations=20
    )
    assert np.isfinite(nll)

    mean, cov = model.predict(x_test)
    mean = rearrange(mean, "o m -> m o")
    residual = jnp.sum((y_test - mean) ** 2, axis=0)
    total = jnp.sum((y_test - y_train.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - residual / total
    assert (r2 > 0.9).all(), r2

    # a float32 posterior can round the predictive variance below zero
    variance = jnp.diagonal(cov, axis1=-2, axis2=-1)
    assert (variance > 0.0).all()


def test_fit_stays_in_float32(profile, smooth_data):
    x_train, y_train, _, _ = smooth_data
    model, nll, state, certificate = GaussianProcess.fit(
        x_train, y_train, profile=profile, l1_penalty=jnp.array(0.1), max_iterations=5
    )
    leaves = jax.tree.leaves((model, nll, state, certificate))
    arrays = [leaf for leaf in leaves if eqx.is_array(leaf)]
    floats = [leaf for leaf in arrays if jnp.issubdtype(leaf.dtype, jnp.floating)]
    assert floats and all(leaf.dtype == jnp.float32 for leaf in floats)


def test_fit_respects_its_bounds_and_kills_groups(profile, smooth_data):
    x_train, y_train, _, _ = smooth_data
    model, _, _, certificate = GaussianProcess.fit(
        x_train, y_train, profile=profile, l1_penalty=jnp.array(50.0), max_iterations=40
    )
    assert (model.theta >= 0.0).all()
    assert np.allclose(model.nugget.clip(1e-3, 100.0), model.nugget, rtol=1e-6)
    assert np.isfinite(certificate).all()

    # a heavy penalty has to zero out whole groups
    norms = jnp.sqrt(jnp.sum(model.theta**2, axis=(0, 1)))
    assert (norms == 0.0).any(), norms


@pytest.mark.skip("AutoregressiveGaussianProcess is gone, targets come flat now")
def test_autoregressive_fit_is_finite(profile, smooth_data):
    x_train, _, x_test, _ = smooth_data
    model, nll, _, certificate = AutoregressiveGaussianProcess.fit(  # noqa: F821
        x_train, profile=profile, l1_penalty=jnp.array(0.1), max_iterations=5
    )
    assert np.isfinite(nll)
    assert np.isfinite(certificate).all()
    assert np.isfinite(model.theta).all() and (model.theta >= 0.0).all()

    mean, cov = model.predict(x_test)
    assert np.isfinite(mean).all()
    assert (jnp.diagonal(cov, axis1=-2, axis2=-1) > 0.0).all()
