import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feetgp import admm
from feetgp.admm import RHO_MAX, RHO_MIN, ADMMState

jax.config.update("jax_enable_x64", True)


def test_layout_round_trips_and_pools_every_output():
    rng = np.random.default_rng(0)
    o, d, group_size = 4, 3, 2
    v = jnp.asarray(rng.normal(size=(o, d * group_size)))

    grouped = admm.to_groups(v, group_size)
    assert grouped.shape == (d, o * group_size)
    assert np.allclose(admm.to_outputs(grouped, group_size), v)

    # a group is one column block across *all* outputs, not one output's block
    for group in range(d):
        block = v[:, group * group_size : (group + 1) * group_size]
        assert sorted(np.asarray(grouped[group])) == sorted(np.asarray(block).ravel())


def test_group_soft_threshold_shrinks_then_kills():
    rng = np.random.default_rng(1)
    v = jnp.asarray(rng.normal(size=(5, 3)))
    norms = np.linalg.norm(np.asarray(v), axis=-1)

    shrunk = admm.group_soft_threshold(v, jnp.array(0.5 * norms.min()))
    expected = (1 - 0.5 * norms.min() / norms)[:, None] * np.asarray(v)
    assert np.allclose(shrunk, expected)

    # past the largest norm every group dies, and stays dead rather than flipping sign
    assert np.allclose(admm.group_soft_threshold(v, jnp.array(norms.max() + 1.0)), 0.0)


def test_prox_is_the_identity_at_zero_penalty():
    """At l1 = 0 the multiplier is exactly 1, so u must stay at zero and r must vanish."""
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.normal(size=(4, 3)))
    state = ADMMState.initialize(jnp.zeros_like(x))._replace(x=x)

    state = admm.z_and_u_update(state, jnp.array(0.0))
    assert np.allclose(state.z, x)
    assert np.allclose(state.u, 0.0)
    assert admm.residuals(state, state)[0] == 0.0


def test_z_and_u_update_respects_bounds():
    x = jnp.array([[-1.0, 2.0], [0.5, 0.5]])
    bounds = jnp.stack([jnp.zeros_like(x), jnp.ones_like(x)])
    state = ADMMState.initialize(jnp.zeros_like(x))._replace(x=x)

    state = admm.z_and_u_update(state, jnp.array(0.0), bounds=bounds)
    assert np.all(np.asarray(state.z) >= 0.0)
    assert np.all(np.asarray(state.z) <= 1.0)


def test_over_relaxation_only_enters_through_x_hat():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(4, 3)))
    z = jnp.asarray(rng.normal(size=(4, 3)))
    state = ADMMState(x=x, z=z, u=jnp.zeros_like(x), rho=jnp.array(1.0))

    alpha = 1.6
    relaxed = admm.z_and_u_update(state, jnp.array(0.2), jnp.array(alpha))
    x_hat = alpha * x + (1 - alpha) * z
    assert np.allclose(relaxed.z, admm.group_soft_threshold(x_hat, jnp.array(0.2)))
    # the residual is still measured on the real x, not on x_hat
    assert np.allclose(
        admm.residuals(relaxed, state)[0], jnp.linalg.norm(x - relaxed.z)
    )


@pytest.mark.parametrize(
    "primal, dual, expected",
    [(100.0, 1.0, 2.0), (1.0, 100.0, 0.5), (1.0, 1.0, 1.0)],
)
def test_rho_adaptation_follows_residual_imbalance(primal, dual, expected):
    # z - z_prev sets the dual residual, x - z the primal one
    z = jnp.full((1, 1), dual)
    state = ADMMState(
        x=jnp.full((1, 1), primal + dual), z=z, u=jnp.ones((1, 1)), rho=jnp.array(1.0)
    )
    prev = state._replace(z=jnp.zeros((1, 1)))

    new_state, _, _ = admm.check_residuals(state, prev, jnp.array(1e-6))
    assert new_state.rho == expected
    # u carries the *applied* ratio, so the dual variable stays consistent with rho
    assert np.allclose(new_state.u, 1.0 / expected)


def test_rho_adaptation_is_clamped():
    state = ADMMState(
        x=jnp.ones((1, 1)),
        z=jnp.zeros((1, 1)),
        u=jnp.zeros((1, 1)),
        rho=jnp.array(RHO_MAX),
    )
    prev = state._replace(z=jnp.zeros((1, 1)))
    assert admm.check_residuals(state, prev, jnp.array(1e-6))[0].rho == RHO_MAX

    # a zero primal residual makes the decrease test a tautology, which is what walks
    # rho down to the floor at l1 = 0
    state = state._replace(x=jnp.zeros((1, 1)), rho=jnp.array(RHO_MIN))
    prev = state._replace(z=jnp.ones((1, 1)))
    assert admm.check_residuals(state, prev, jnp.array(1e-6))[0].rho == RHO_MIN


def test_frozen_rho_does_not_move():
    state = ADMMState(
        x=jnp.ones((1, 1)), z=jnp.zeros((1, 1)), u=jnp.ones((1, 1)), rho=jnp.array(1.0)
    )
    prev = state._replace(z=jnp.zeros((1, 1)))
    frozen, _, _ = admm.check_residuals(state, prev, jnp.array(1e-6), adapt_rho=False)
    assert frozen.rho == 1.0
    assert np.allclose(frozen.u, state.u)


def least_squares_x_update(design, targets, group_size):
    """A ridge x-update, so the fixture is a group lasso with a known KKT condition."""

    def x_update(state: ADMMState, _: int) -> tuple[ADMMState, bool]:
        target = admm.to_outputs(state.z - state.u, group_size)
        A = design.T @ design + state.rho * jnp.eye(design.shape[1])
        b = design.T @ targets + state.rho * target.T
        return (
            state._replace(x=admm.to_groups(jnp.linalg.solve(A, b).T, group_size)),
            True,
        )

    return x_update


def test_solve_reaches_the_group_lasso_optimum():
    rng = np.random.default_rng(4)
    n, d, group_size, o, l1 = 40, 4, 2, 3, 1.0
    design = jnp.asarray(rng.normal(size=(n, d * group_size)))
    targets = jnp.asarray(rng.normal(size=(n, o)))

    state, info = admm.solve(
        least_squares_x_update(design, targets, group_size),
        ADMMState.initialize(jnp.zeros((d, o * group_size))),
        jnp.array(l1),
        max_iterations=3000,
        tol=jnp.array(1e-10),
    )
    assert info["converged"], info

    theta = admm.to_outputs(state.z, group_size)
    grad = admm.to_groups((-design.T @ (targets - design @ theta.T)).T, group_size)
    for group, (grad_group, theta_group) in enumerate(zip(grad, state.z)):
        norm = np.linalg.norm(theta_group)
        if norm < 1e-6:
            assert np.linalg.norm(grad_group) <= l1 + 1e-3, group
        else:
            assert np.linalg.norm(grad_group + l1 * theta_group / norm) < 1e-3, group


def test_inexact_x_update_cannot_report_convergence():
    """A subproblem solved under budget must not let a small residual end the loop."""
    x = jnp.zeros((2, 2))

    def x_update(state: ADMMState, _: int) -> tuple[ADMMState, bool]:
        return state._replace(x=x), False

    _, info = admm.solve(
        x_update, ADMMState.initialize(x), jnp.array(0.0), max_iterations=5
    )
    assert not info["converged"]
    assert info["iterations"] == 5
    assert info["primal_residual"] == 0.0


def test_aux_passes_through_untouched():
    """aux belongs to the x-update: it must never enter z, u or either residual."""
    x = jnp.ones((2, 2))
    aux = jnp.array([3.0, 4.0])

    def x_update(state: ADMMState, iteration: int) -> tuple[ADMMState, bool]:
        assert state.aux is not None
        return state._replace(aux=state.aux + iteration), True

    state, _ = admm.solve(
        x_update, ADMMState.initialize(x, aux=aux), jnp.array(0.5), max_iterations=3
    )
    assert np.allclose(state.aux, aux + 3)
