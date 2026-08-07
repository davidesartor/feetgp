import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange

from feetgp import glasso_admm
from feetgp.gp import (
    GroupLassoGaussianProcess,
    admm_x_update_loss,
    gp_posterior,
    kernel,
)
from feetgp.linear import GroupLassoLinear

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.uniform(size=(24, 12)))
    y = jnp.asarray(rng.normal(size=(24, 12)))
    return x, y


def test_linear_fit_satisfies_group_lasso_kkt(toy_data):
    x_train, y_train = toy_data
    group_size, l1 = 2, 0.5
    model, _, _, _ = GroupLassoLinear.fit(
        x_train, y_train, jnp.array(l1), group_size, max_iterations=4000, tol=1e-10
    )

    residual = y_train - x_train @ model.theta.T
    grad = -x_train.T @ residual
    grad = rearrange(grad, "(d g) o -> d (g o)", g=group_size)
    theta = rearrange(model.theta, "o (d g) -> d (g o)", g=group_size)

    for group, (grad_group, theta_group) in enumerate(zip(grad, theta)):
        norm = np.linalg.norm(theta_group)
        if norm < 1e-6:
            assert np.linalg.norm(grad_group) <= l1 + 1e-3, group
        else:
            optimality = grad_group + l1 * theta_group / norm
            assert np.linalg.norm(optimality) < 1e-3, group


def test_gp_posterior_matches_naive_reference():
    rng = np.random.default_rng(2)
    n, m = 15, 4
    a = rng.normal(size=(n, n))
    Koo = jnp.asarray(a @ a.T + n * np.eye(n))
    Kox = jnp.asarray(rng.normal(size=(n, m)))
    b = jnp.asarray(rng.normal(size=(m, m)))
    Kxx = jnp.asarray(b @ b.T + m * np.eye(m))
    ys = jnp.asarray(rng.normal(size=(n,)))
    trend = jnp.array(0.4)

    gain = jnp.linalg.solve(Koo, Kox).T
    expected_mean = trend + gain @ (ys - trend)
    Kbx = jnp.ones((1, n)) @ gain.T
    expected_cov = Kxx - gain @ Kox
    expected_cov += (1 - Kbx).T @ (1 - Kbx) / jnp.linalg.inv(Koo).sum()

    assert np.allclose(gp_posterior(Kox, Koo, ys, trend), expected_mean)
    mean, cov = gp_posterior(Kox, Koo, ys, trend, Kxx)
    assert np.allclose(mean, expected_mean)
    assert np.allclose(cov, expected_cov)


def test_gp_predict_matches_per_output_posterior(toy_data):
    x_train, y_train = toy_data
    rng = np.random.default_rng(3)
    o = y_train.shape[1]
    xs = jnp.asarray(rng.uniform(size=(5, x_train.shape[1])))
    model = GroupLassoGaussianProcess(
        theta=jnp.asarray(rng.uniform(0.5, 2.0, size=(o, x_train.shape[1]))),
        g=jnp.full((o,), 0.1),
        b=jnp.asarray(rng.normal(size=(o,))),
        nu=jnp.full((o,), 1.3),
        x_train=x_train,
        y_train=y_train,
    )

    mean = model.predict(xs)
    assert mean.shape == (o, len(xs))

    for i in range(o):
        nu, g = model.nu[i], model.g[i]
        Koo = nu * (
            kernel(model.theta[i], x_train, x_train) + g * jnp.eye(len(x_train))
        )
        Kox = nu * kernel(model.theta[i], x_train, xs)
        expected = gp_posterior(Kox, Koo, y_train[:, i], model.b[i])
        assert np.allclose(mean[i], expected)

    mean_with_cov, cov = model.predict(xs, covariance=True)
    assert cov.shape == (o, len(xs), len(xs))
    assert np.allclose(mean_with_cov, mean)


def test_admm_x_update_grad_matches_finite_differences(toy_data):
    x_train, y_train = toy_data
    rng = np.random.default_rng(4)
    d = x_train.shape[1]
    x = jnp.asarray(rng.uniform(0.5, 1.5, size=(d + 1,)))
    target_theta = jnp.asarray(rng.uniform(0.5, 1.5, size=(d,)))
    args = (
        target_theta,
        jnp.array(2.0),
        x_train,
        y_train[:, 0],
        jnp.array(1e-4),
        jnp.array(100.0),
    )

    grad = jax.grad(admm_x_update_loss)(x, args)
    step = 1e-6
    for i in range(d + 1):
        shift = jnp.zeros(d + 1).at[i].set(step)
        up = admm_x_update_loss(x + shift, args)
        down = admm_x_update_loss(x - shift, args)
        assert np.isclose(grad[i], (up - down) / (2 * step), rtol=1e-4, atol=1e-6)


def test_x_update_objective_is_flat_below_zero(toy_data):
    x_train, y_train = toy_data
    rng = np.random.default_rng(5)
    d = x_train.shape[1]
    x = jnp.asarray(rng.uniform(0.5, 1.5, size=(d + 1,)))
    args = (
        jnp.zeros(d),
        jnp.array(0.0),
        x_train,
        y_train[:, 0],
        jnp.array(1e-4),
        jnp.array(100.0),
    )

    negative = x.at[: d // 2].set(-1.0)
    at_zero = x.at[: d // 2].set(0.0)
    assert admm_x_update_loss(negative, args) == admm_x_update_loss(at_zero, args)
    assert admm_x_update_loss(negative, args) != admm_x_update_loss(x, args)

    grad = jax.grad(admm_x_update_loss)(at_zero, args)
    assert np.allclose(grad[: d // 2], 0.0)


def test_autoregressive_output_cannot_see_its_own_group(toy_data):
    x_train, y_train = toy_data
    group_size = 3
    model, _, state, _ = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(0.0),
        group_size=group_size,
        autoregressive=True,
        max_iterations=2,
        chunk_size=1,
    )
    state_theta = glasso_admm.to_outputs(state.x, group_size)
    for i in range(y_train.shape[1]):
        group_start = (i // group_size) * group_size
        own_group = slice(group_start, group_start + group_size)
        assert np.allclose(model.theta[i, own_group], 0.0), i
        assert np.allclose(state_theta[i, own_group], 0.0), i


def test_inner_solver_stops_before_its_cap(toy_data):
    from vlse.optim import minimise

    x_train, y_train = toy_data
    rng = np.random.default_rng(6)
    d = x_train.shape[1]
    x0 = jnp.asarray(rng.uniform(0.5, 1.5, size=(d + 1,)))
    args = (
        jnp.zeros(d),
        jnp.array(1.0),
        x_train,
        y_train[:, 0],
        jnp.array(1e-4),
        jnp.array(100.0),
    )
    bounds = (
        jnp.concat([jnp.zeros(d), jnp.array([-jnp.inf])]),
        jnp.full(d + 1, jnp.inf),
    )

    cap = 400
    solution = minimise(
        admm_x_update_loss,
        x0,
        bounds,
        args=(args,),
        tol=1e-2,
        max_iterations=cap,
        history_length=40,
    )
    assert int(solution.iteration) < cap


def test_fit_converges_within_its_iteration_budget(toy_data):
    x_train, y_train = toy_data
    _, _, _, info = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(0.5),
        group_size=3,
        autoregressive=False,
        max_iterations=200,
        inner_maxiter=50,
        chunk_size=4,
    )
    assert info["converged"], info


def test_warmstart_from_state_reuses_dual_and_rho(toy_data):
    x_train, y_train = toy_data
    _, _, state, _ = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(0.1),
        group_size=3,
        max_iterations=2,
        chunk_size=1,
    )
    assert not np.allclose(state.u, 0.0)

    def one_iteration(warmstart):
        _, _, next_state, _ = GroupLassoGaussianProcess.fit(
            x_train,
            y_train,
            l1_penalty=jnp.array(0.2),
            group_size=3,
            max_iterations=1,
            chunk_size=1,
            warmstart=warmstart,
        )
        return next_state

    warm, cold = one_iteration(state), one_iteration(None)
    assert not np.allclose(warm.u, cold.u)
    assert not np.allclose(warm.x, cold.x)
    assert not np.allclose(warm.aux, cold.aux)
