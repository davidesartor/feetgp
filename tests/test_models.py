import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange
from vlse.optim import minimise

from feetgp.gp import GaussianProcess, gp_posterior, hetgpy_auto_bounds, kernel
from feetgp.linear import Linear

from conftest import flat_design, group_norms, x_update_objective


def test_linear_fit_satisfies_group_lasso_kkt(toy_data):
    x_train, y_train = toy_data
    l1 = 0.5
    model, _, _, _ = Linear.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(l1),
        max_iterations=4000,
        tol=jnp.array(1e-6),
    )

    design = flat_design(x_train)
    residual = y_train - model.predict(x_train)
    grad = rearrange(-design.T @ residual, "(d g) o -> o d g", g=x_train.shape[2])

    for group, norm in enumerate(group_norms(model.theta)):
        grad_group = np.asarray(grad[..., group])
        theta_group = np.asarray(model.theta[..., group])
        if norm < 1e-6:
            assert np.linalg.norm(grad_group) <= l1 + 1e-3, group
        else:
            optimality = grad_group + l1 * theta_group / norm
            assert np.linalg.norm(optimality) < 1e-3, group


def test_kernel_is_a_correlation_matrix(profile, toy_data):
    x_train, _ = toy_data
    design = flat_design(x_train)
    rng = np.random.default_rng(1)
    theta = jnp.asarray(rng.uniform(0.5, 2.0, size=design.shape[1]))

    K = kernel(profile, theta, design, design)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert K.max() <= 1.0 and K.min() >= 0.0
    assert np.isfinite(np.linalg.cholesky(np.asarray(K + 1e-3 * jnp.eye(len(K))))).all()

    # the kernel is stationary and decreasing, so a longer lengthscale correlates more
    assert (kernel(profile, 0.5 * theta, design, design) >= K - 1e-6).all()


def test_kernel_gradient_is_finite_on_the_diagonal(profile, toy_data):
    x_train, _ = toy_data
    design = flat_design(x_train)
    theta = jnp.ones(design.shape[1])

    # matern52 differentiates a square root that hits zero at every repeated point
    grad = jax.grad(lambda t: jnp.sum(kernel(profile, t, design, design)))(theta)
    assert np.isfinite(grad).all()


def test_unknown_kernel_profile_is_rejected(toy_data):
    x_train, _ = toy_data
    design = flat_design(x_train)
    with pytest.raises(ValueError, match="unknown kernel profile"):
        kernel("cubic", jnp.ones(design.shape[1]), design, design)


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

    mean, cov = gp_posterior(Kxx, Kox, Koo, ys, trend)
    assert np.allclose(mean, expected_mean, rtol=1e-4, atol=1e-5)
    assert np.allclose(cov, expected_cov, rtol=1e-4, atol=1e-4)


def test_gp_predict_matches_per_output_posterior(profile, toy_data):
    x_train, y_train = toy_data
    rng = np.random.default_rng(3)
    o = y_train.shape[1]
    _, d, g = x_train.shape
    xs = jnp.asarray(rng.uniform(size=(5, d, g)))
    model = GaussianProcess(
        theta=jnp.asarray(rng.uniform(0.5, 2.0, size=(o, d, g))),
        nugget=jnp.full((o,), 0.1),
        b=jnp.asarray(rng.normal(size=(o,))),
        nu=jnp.full((o,), 1.3),
        x_train=x_train,
        y_train=y_train,
        profile=profile,
    )

    mean, cov = model.predict(xs)
    assert mean.shape == (o, len(xs))
    assert cov.shape == (o, len(xs), len(xs))

    design, xs_flat = flat_design(x_train), rearrange(xs, "m d g -> m (d g)")
    for i in range(o):
        nu, nugget = model.nu[i], model.nugget[i]
        theta_flat = rearrange(model.theta[i], "d g -> (d g)")
        Koo = nu * kernel(profile, theta_flat, design, design)
        Koo = Koo + nugget * jnp.eye(len(x_train))
        Kox = nu * kernel(profile, theta_flat, design, xs_flat)
        Kxx = nu * kernel(profile, theta_flat, xs_flat, xs_flat)
        expected, expected_cov = gp_posterior(Kxx, Kox, Koo, y_train[:, i], model.b[i])
        assert np.allclose(mean[i], expected, rtol=1e-4, atol=1e-5)
        assert np.allclose(cov[i], expected_cov, rtol=1e-4, atol=1e-5)

    # a mock query axis vectorizes to per-point 1x1 covariances
    batched_mean, batched_cov = model.predict(xs[:, None])
    assert batched_cov.shape == (len(xs), o, 1, 1)
    assert np.allclose(
        rearrange(batched_mean, "m o 1 -> o m"), mean, rtol=1e-4, atol=1e-5
    )


def test_x_update_objective_grad_matches_finite_differences(toy_data):
    x_train, y_train = toy_data
    rng = np.random.default_rng(4)
    design = flat_design(x_train)
    d = design.shape[1]
    x = jnp.asarray(rng.uniform(0.5, 1.5, size=(d + 1,)))
    target = jnp.asarray(rng.uniform(0.5, 1.5, size=(d,)))
    objective = x_update_objective("rbf", design, y_train[:, 0], target, jnp.array(2.0))

    # float32 cancels a tighter step, so the difference is taken over a wide one
    grad = jax.grad(objective)(x)
    step = 1e-2
    for i in range(d + 1):
        shift = jnp.zeros(d + 1).at[i].set(step)
        difference = (objective(x + shift) - objective(x - shift)) / (2 * step)
        assert np.isclose(grad[i], difference, rtol=2e-2, atol=1e-2), i


def test_x_update_objective_pulls_theta_towards_its_target(toy_data):
    x_train, y_train = toy_data
    design = flat_design(x_train)
    d = design.shape[1]
    target = jnp.full((d,), 1.5)
    x = jnp.concat([target, jnp.zeros(1)])
    penalized = x_update_objective("rbf", design, y_train[:, 0], target, jnp.array(2.0))
    unpenalized = x_update_objective(
        "rbf", design, y_train[:, 0], target, jnp.array(0.0)
    )

    # at the target the proximal term is flat, so it changes neither value nor gradient
    assert np.allclose(penalized(x), unpenalized(x), rtol=1e-5)
    assert np.allclose(jax.grad(penalized)(x), jax.grad(unpenalized)(x), atol=1e-4)

    # anywhere else it charges for the distance
    away = x.at[: d // 2].set(0.0)
    assert penalized(away) > unpenalized(away)


def test_inner_solver_stops_before_its_cap(toy_data):
    x_train, y_train = toy_data
    rng = np.random.default_rng(6)
    design = flat_design(x_train)
    d = design.shape[1]
    x0 = jnp.asarray(rng.uniform(0.5, 1.5, size=(d + 1,)))
    objective = x_update_objective(
        "rbf", design, y_train[:, 0], jnp.zeros(d), jnp.array(1.0)
    )
    bounds = (
        jnp.concat([jnp.zeros(d), jnp.array([-jnp.inf])]),
        jnp.full(d + 1, jnp.inf),
    )

    cap = 400
    solution = minimise(
        objective, x0, bounds, tol=1e-2, max_iterations=cap, history_length=40
    )
    assert int(solution.iteration) < cap


def test_fit_converges_within_its_iteration_budget(toy_data):
    x_train, y_train = toy_data
    tol = jnp.array(1e-3)
    _, _, state, _ = GaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(0.5),
        max_iterations=300,
        tol=tol,
        profile="rbf",
    )
    assert state.converged(tol), (state.primal_residual, state.dual_residual)


def test_fit_respects_the_auto_bounds(toy_data):
    x_train, y_train = toy_data
    nugget_range = jnp.array([0.001, 100.0])
    _, _, upper = hetgpy_auto_bounds(flat_design(x_train))
    model, _, state, _ = GaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(0.1),
        max_iterations=20,
        nugget_range=nugget_range,
        profile="rbf",
    )

    # l-bfgs-b keeps the primal iterate in the box, the prox output only follows it
    upper = rearrange(upper, "(d g) -> d g", g=x_train.shape[2])
    assert (state.x >= 0.0).all()
    assert (state.x <= upper + 1e-5).all()
    assert np.allclose(model.nugget.clip(*nugget_range), model.nugget, rtol=1e-6)


def test_fit_carries_its_profile_into_the_model(profile, toy_data):
    x_train, y_train = toy_data
    model, nll, _, _ = GaussianProcess.fit(
        x_train, y_train, l1_penalty=jnp.array(0.1), max_iterations=5, profile=profile
    )
    assert model.profile == profile
    assert np.isfinite(nll)
    assert np.isfinite(model.predict(x_train[:3])[0]).all()


def test_warmstart_keeps_the_primal_iterate_and_restarts_the_dual(toy_data):
    x_train, y_train = toy_data
    _, _, state, _ = GaussianProcess.fit(
        x_train, y_train, l1_penalty=jnp.array(0.1), max_iterations=4, profile="rbf"
    )
    assert not np.allclose(state.u, 0.0)

    def one_iteration(warmstart):
        _, _, next_state, _ = GaussianProcess.fit(
            x_train,
            y_train,
            l1_penalty=jnp.array(0.2),
            max_iterations=1,
            warmstart=warmstart,
            profile="rbf",
        )
        return next_state

    warm, cold = one_iteration(state), one_iteration(None)
    assert not np.allclose(warm.x, cold.x)
    assert not np.allclose(warm.aux[0], cold.aux[0])

    # restart throws the dual away, so even a warm first iteration starts from u = 0
    assert np.allclose(warm.u, warm.x - warm.z)
