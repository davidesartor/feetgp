import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange

from feetgp import glasso_admm
from feetgp.glasso_admm import RHO_MAX, RHO_MIN, ADMMState


def state_at(x: jnp.ndarray) -> ADMMState:
    return ADMMState(x=x, z=x, u=jnp.zeros_like(x))


def residual_state(x, z, u, prev_z, rho=jnp.array(1.0)) -> ADMMState:
    state = ADMMState(x=x, z=z, u=u, rho=rho)
    return glasso_admm.update_residuals(state, state._replace(z=prev_z))


def test_z_update_shrinks_then_kills():
    rng = np.random.default_rng(1)
    v = jnp.asarray(rng.normal(size=(5, 3)))
    norms = np.linalg.norm(np.asarray(v), axis=0)

    shrunk = glasso_admm.update_z_and_u(state_at(v), jnp.array(0.5 * norms.min())).z
    expected = (1 - 0.5 * norms.min() / norms)[None, :] * np.asarray(v)
    assert np.allclose(shrunk, expected)

    killed = glasso_admm.update_z_and_u(state_at(v), jnp.array(norms.max() + 1.0)).z
    assert np.allclose(killed, 0.0)


def test_prox_is_the_identity_at_zero_penalty():
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.normal(size=(4, 3)))

    state = glasso_admm.update_z_and_u(state_at(x), jnp.array(0.0))
    assert np.allclose(state.z, x)
    assert np.allclose(state.u, 0.0)


def test_prox_leaves_an_already_dead_group_at_zero():
    x = jnp.ones((4, 3)).at[:, 1].set(0.0)

    # a zero group divides by its own zero norm, which used to poison z with nan
    for l1_penalty in (0.0, 0.5):
        state = glasso_admm.update_z_and_u(state_at(x), jnp.array(l1_penalty))
        assert np.isfinite(state.z).all(), l1_penalty
        assert np.allclose(state.z[:, 1], 0.0)


def test_z_update_is_prox_of_x_plus_u():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(4, 3)))
    z = jnp.asarray(rng.normal(size=(4, 3)))
    u = jnp.asarray(rng.normal(size=(4, 3)))
    state = ADMMState(x=x, z=z, u=u)

    updated = glasso_admm.update_z_and_u(state, jnp.array(0.2))
    v = np.asarray(x + u)
    norms = np.linalg.norm(v, axis=0, keepdims=True)
    expected = np.maximum(0.0, 1 - 0.2 / norms) * v
    assert np.allclose(updated.z, expected)
    assert np.allclose(updated.u, u + x - updated.z)


def test_residuals_are_relative_and_count_the_iteration():
    x = jnp.array([[3.0]])
    state = residual_state(x=x, z=jnp.array([[1.0]]), u=x, prev_z=jnp.zeros((1, 1)))

    assert state.iteration == 1
    assert np.allclose(state.primal_residual, 2.0 / 3.0)
    assert np.allclose(state.dual_residual, 1.0 / 3.0)
    assert state.converged(jnp.array(1.0))
    assert not state.converged(jnp.array(1e-3))


def test_residuals_stay_finite_with_a_vanished_target():
    zero = jnp.zeros((1, 1))
    state = residual_state(x=zero, z=zero, u=zero, prev_z=zero)
    assert np.isfinite([state.primal_residual, state.dual_residual]).all()


@pytest.mark.parametrize(
    "z, prev_z, expected",
    [(0.0, 0.0, 2.0), (1.0, 0.0, 0.5), (0.0, 1.0, 1.0)],
)
def test_rho_adaptation_follows_residual_imbalance(z, prev_z, expected):
    state = residual_state(
        x=jnp.ones((1, 1)),
        z=jnp.full((1, 1), z),
        u=jnp.ones((1, 1)),
        prev_z=jnp.full((1, 1), prev_z),
    )

    adapted = glasso_admm.update_rho(state, jnp.array(True))
    assert adapted.rho == expected
    assert np.allclose(adapted.u, 1.0 / expected)


def test_rho_adaptation_is_clamped():
    state = residual_state(
        x=jnp.ones((1, 1)),
        z=jnp.zeros((1, 1)),
        u=jnp.zeros((1, 1)),
        prev_z=jnp.zeros((1, 1)),
        rho=jnp.array(RHO_MAX),
    )
    assert glasso_admm.update_rho(state, jnp.array(True)).rho == RHO_MAX

    state = residual_state(
        x=jnp.zeros((1, 1)),
        z=jnp.zeros((1, 1)),
        u=jnp.ones((1, 1)),
        prev_z=jnp.ones((1, 1)),
        rho=jnp.array(RHO_MIN),
    )
    assert glasso_admm.update_rho(state, jnp.array(False)).rho == RHO_MIN


def test_frozen_rho_does_not_move():
    state = residual_state(
        x=jnp.ones((1, 1)),
        z=jnp.zeros((1, 1)),
        u=jnp.ones((1, 1)),
        prev_z=jnp.zeros((1, 1)),
    )
    frozen = glasso_admm.update_rho(state, jnp.array(False))
    assert frozen.rho == 1.0
    assert np.allclose(frozen.u, state.u)


def least_squares_x_update(design, targets):
    """Ridge-regularized least squares, the closed form the linear model uses."""

    def x_update(state: ADMMState) -> ADMMState:
        _, _, g = state.x.shape
        target = rearrange(state.z - state.u, "o d g -> (d g) o")
        A = design.T @ design + state.rho * jnp.eye(design.shape[1])
        b = design.T @ targets + state.rho * target
        x = rearrange(jnp.linalg.solve(A, b), "(d g) o -> o d g", g=g)
        return state._replace(x=x)

    return x_update


def test_solve_reaches_the_group_lasso_optimum():
    rng = np.random.default_rng(4)
    n, d, g, o, l1 = 40, 4, 2, 3, 1.0
    design = jnp.asarray(rng.normal(size=(n, d * g)))
    targets = jnp.asarray(rng.normal(size=(n, o)))

    zeros = jnp.zeros((o, d, g))
    state = glasso_admm.solve(
        least_squares_x_update(design, targets),
        ADMMState(x=zeros, z=zeros, u=zeros),
        l1_penalty=jnp.array(l1),
        max_iterations=3000,
        tol=jnp.array(1e-6),
    )
    assert state.converged(jnp.array(1e-6))

    theta = rearrange(state.z, "o d g -> o (d g)")
    grad = rearrange(-design.T @ (targets - design @ theta.T), "(d g) o -> o d g", g=g)
    for group in range(g):
        grad_group = np.asarray(grad[..., group])
        theta_group = np.asarray(state.z[..., group])
        norm = np.linalg.norm(theta_group)
        if norm < 1e-6:
            assert np.linalg.norm(grad_group) <= l1 + 1e-3, group
        else:
            assert np.linalg.norm(grad_group + l1 * theta_group / norm) < 1e-3, group


def test_splitting_the_iterate_into_leaves_changes_nothing():
    rng = np.random.default_rng(6)
    n, d, g, o, split = 40, 4, 2, 3, 1
    design = jnp.asarray(rng.normal(size=(n, d * g)))
    targets = jnp.asarray(rng.normal(size=(n, o)))
    dense_x_update = least_squares_x_update(design, targets)

    # same update, but the outputs live in two leaves of unequal size
    def tree_x_update(state: ADMMState) -> ADMMState:
        merged = state._replace(
            **{f: jnp.concatenate(getattr(state, f)) for f in ("x", "z", "u")}
        )
        x = dense_x_update(merged).x
        return state._replace(x=(x[:split], x[split:]))

    zeros = jnp.zeros((o, d, g))
    solve = lambda x_update, x: glasso_admm.solve(
        x_update,
        ADMMState(x=x, z=x, u=x),
        l1_penalty=jnp.array(1.0),
        max_iterations=3000,
        tol=jnp.array(1e-6),
    )
    dense = solve(dense_x_update, zeros)
    tree = solve(tree_x_update, (zeros[:split], zeros[split:]))

    assert tree.iteration == dense.iteration
    assert np.allclose(jnp.concatenate(tree.z), dense.z)
    assert np.allclose(jnp.concatenate(tree.u), dense.u)
    assert tree.rho == dense.rho


def test_aux_rides_along_untouched():
    x = jnp.ones((2, 2))
    side = jnp.array([3.0, 4.0])

    def x_update(state: ADMMState) -> ADMMState:
        return state._replace(aux=(state.aux[0] + 1.0,))

    state = glasso_admm.solve(
        x_update,
        ADMMState(x=x, z=jnp.zeros_like(x), u=jnp.zeros_like(x), aux=(side,)),
        l1_penalty=jnp.array(0.5),
        max_iterations=3,
        tol=jnp.array(0.0),
    )

    # only the x update writes aux, the prox and the residuals never see it
    assert np.allclose(state.aux[0], side + state.iteration)


def test_restart_keeps_the_primal_iterate_and_drops_the_rest():
    rng = np.random.default_rng(5)
    x, z, u = (jnp.asarray(rng.normal(size=(2, 3))) for _ in range(3))
    state = ADMMState(
        x=x,
        z=z,
        u=u,
        rho=jnp.array(8.0),
        iteration=jnp.array(17),
        primal_residual=jnp.array(1e-5),
        dual_residual=jnp.array(1e-5),
    )

    restarted = state.restart()
    assert np.allclose(restarted.x, x)
    assert np.allclose(restarted.z, 0.0) and np.allclose(restarted.u, 0.0)
    assert restarted.rho == 1.0 and restarted.iteration == 0
    assert not restarted.converged(jnp.array(1e30))
