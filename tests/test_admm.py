import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange

from feetgp import glasso_admm
from feetgp.glasso_admm import RHO_MAX, RHO_MIN, ADMMState

jax.config.update("jax_enable_x64", True)


def state_at(x: jnp.ndarray) -> ADMMState:
    return ADMMState(x=x, z=x, u=jnp.zeros_like(x), rho=jnp.array(1.0))


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


def test_z_update_is_prox_of_x_plus_u():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(4, 3)))
    z = jnp.asarray(rng.normal(size=(4, 3)))
    u = jnp.asarray(rng.normal(size=(4, 3)))
    state = ADMMState(x=x, z=z, u=u, rho=jnp.array(1.0))

    updated = glasso_admm.update_z_and_u(state, jnp.array(0.2))
    v = np.asarray(x + u)
    norms = np.linalg.norm(v, axis=0, keepdims=True)
    expected = np.maximum(0.0, 1 - 0.2 / norms) * v
    assert np.allclose(updated.z, expected)
    assert np.allclose(updated.u, u + x - updated.z)


@pytest.mark.parametrize(
    "primal, dual, expected",
    [(100.0, 1.0, 2.0), (1.0, 100.0, 0.5), (1.0, 1.0, 1.0)],
)
def test_rho_adaptation_follows_residual_imbalance(primal, dual, expected):
    z = jnp.full((1, 1), dual)
    state = ADMMState(
        x=jnp.full((1, 1), primal + dual), z=z, u=jnp.ones((1, 1)), rho=jnp.array(1.0)
    )
    prev = state._replace(z=jnp.zeros((1, 1)))

    new_state, _, _ = glasso_admm.check_residuals(
        state, prev, jnp.array(1e-6), update_rho=jnp.array(True)
    )
    assert new_state.rho == expected
    assert np.allclose(new_state.u, 1.0 / expected)


def test_rho_adaptation_is_clamped():
    state = ADMMState(
        x=jnp.ones((1, 1)),
        z=jnp.zeros((1, 1)),
        u=jnp.zeros((1, 1)),
        rho=jnp.array(RHO_MAX),
    )
    prev = state._replace(z=jnp.zeros((1, 1)))
    clamped, _, _ = glasso_admm.check_residuals(
        state, prev, jnp.array(1e-6), update_rho=jnp.array(True)
    )
    assert clamped.rho == RHO_MAX

    state = state._replace(x=jnp.zeros((1, 1)), rho=jnp.array(RHO_MIN))
    prev = state._replace(z=jnp.ones((1, 1)))
    clamped, _, _ = glasso_admm.check_residuals(
        state, prev, jnp.array(1e-6), update_rho=jnp.array(True)
    )
    assert clamped.rho == RHO_MIN


def test_frozen_rho_does_not_move():
    state = ADMMState(
        x=jnp.ones((1, 1)), z=jnp.zeros((1, 1)), u=jnp.ones((1, 1)), rho=jnp.array(1.0)
    )
    prev = state._replace(z=jnp.zeros((1, 1)))
    frozen, _, _ = glasso_admm.check_residuals(
        state, prev, jnp.array(1e-6), update_rho=jnp.array(False)
    )
    assert frozen.rho == 1.0
    assert np.allclose(frozen.u, state.u)


def least_squares_x_update(design, targets, group_size):
    def x_update(state: ADMMState) -> ADMMState:
        target = rearrange(state.z - state.u, "(o g) d -> (d g) o", g=group_size)
        A = design.T @ design + state.rho * jnp.eye(design.shape[1])
        b = design.T @ targets + state.rho * target
        x = rearrange(jnp.linalg.solve(A, b), "(d g) o -> (o g) d", g=group_size)
        return state._replace(x=x)

    return x_update


def test_solve_reaches_the_group_lasso_optimum():
    rng = np.random.default_rng(4)
    n, d, group_size, o, l1 = 40, 4, 2, 3, 1.0
    design = jnp.asarray(rng.normal(size=(n, d * group_size)))
    targets = jnp.asarray(rng.normal(size=(n, o)))

    state, converged, _ = glasso_admm.solve(
        least_squares_x_update(design, targets, group_size),
        jnp.zeros((o * group_size, d)),
        l1_penalty=jnp.array(l1),
        max_iterations=3000,
        tol=1e-10,
    )
    assert converged

    theta = rearrange(state.z, "(o g) d -> o (d g)", g=group_size)
    grad = rearrange(
        -design.T @ (targets - design @ theta.T), "(d g) o -> (o g) d", g=group_size
    )
    for group, (grad_group, theta_group) in enumerate(zip(grad.T, state.z.T)):
        norm = np.linalg.norm(theta_group)
        if norm < 1e-6:
            assert np.linalg.norm(grad_group) <= l1 + 1e-3, group
        else:
            assert np.linalg.norm(grad_group + l1 * theta_group / norm) < 1e-3, group


def test_aux_passes_through_untouched():
    x = jnp.ones((2, 2))
    aux = jnp.array([3.0, 4.0])

    def x_update(state: ADMMState) -> ADMMState:
        return state._replace(aux=state.aux + 1.0)

    state, _, iterations = glasso_admm.solve(
        x_update, x, aux, jnp.array(0.5), max_iterations=3
    )
    assert np.allclose(state.aux, aux + iterations)
