from typing import Any, Callable, NamedTuple, Optional, Self
from jaxtyping import Array, Bool, Int, Float

import jax
import jax.numpy as jnp
import equinox as eqx
from einops import reduce

RHO_MIN, RHO_MAX = 1e-6, 1e6


class ADMMState(NamedTuple):
    x: Float[Array, "... g"]
    z: Float[Array, "... g"]
    u: Float[Array, "... g"]
    rho: Float[Array, ""]
    aux: Any = None


UpdateX = Callable[[ADMMState], ADMMState]


@eqx.filter_jit
def update_z_and_u(state: ADMMState, l1_penalty: Float[Array, ""]) -> ADMMState:
    # compute the group lasso proximal operator
    v = state.x + state.u
    threshold = l1_penalty / state.rho
    norm = jnp.sqrt(reduce(v**2, "... g -> g", "sum"))
    norm = jnp.maximum(norm, 0.5 * threshold)  # avoid division by zero
    z = jnp.maximum(0.0, 1 - threshold / norm) * v
    return state._replace(z=z, u=state.u + state.x - z)


@eqx.filter_jit
def check_residuals(
    state: ADMMState,
    prev: ADMMState,
    tol: Float[Array, ""],
    update_rho: Bool[Array, ""],
) -> tuple[ADMMState, Bool[Array, ""], Bool[Array, ""]]:
    # check primal residual: ||x - z||_2 < eps_abs + tol * max(||x||_2, ||z||_2)
    primal_residual = jnp.linalg.norm(state.x - state.z)
    primal_target = jnp.maximum(jnp.linalg.norm(prev.x), jnp.linalg.norm(prev.z))
    primal_ok = primal_residual <= tol * primal_target

    # check dual residual: ||rho * (z - z_prev)||_2 < eps_abs + tol * ||rho * u||_2
    dual_residual = state.rho * jnp.linalg.norm(state.z - prev.z)
    dual_target = state.rho * jnp.linalg.norm(prev.u)
    dual_ok = dual_residual <= tol * dual_target

    # adapt rho if the primal and dual residuals are very different
    increase = (primal_residual > 10 * dual_residual) & update_rho
    decrease = (dual_residual > 10 * primal_residual) & update_rho
    scale = jnp.select([increase, decrease], [2.0, 0.5], default=1.0)
    new_rho = jnp.clip(state.rho * scale, RHO_MIN, RHO_MAX)
    state = state._replace(rho=new_rho, u=state.u * (state.rho / new_rho))
    return state, primal_ok, dual_ok


@eqx.filter_jit
def solve(
    x_update: UpdateX,
    x0: Float[Array, "... g"],
    aux: Any = None,
    l1_penalty: Float[Array, ""] = jnp.array(0.0),
    *,
    z0: Optional[Float[Array, "... g"]] = None,
    u0: Optional[Float[Array, "... g"]] = None,
    rho0: Float[Array, ""] = jnp.array(1.0),
    max_iterations: int | Int[Array, ""] = 300,
    tol: float | Float[Array, ""] = 1e-3,
    adapt_rho: bool | Bool[Array, ""] = True,
) -> tuple[ADMMState, Bool[Array, ""], Int[Array, ""]]:
    # initialize while loop state
    z0 = z0 if z0 is not None else x0
    u0 = u0 if u0 is not None else jnp.zeros_like(x0)
    state = ADMMState(x=x0, z=z0, u=u0, rho=rho0, aux=aux)
    converged = jnp.asarray(False)
    iter = jnp.asarray(0)

    # define admm loop condition and body
    def while_condition(carry):
        state, converged, iter = carry
        return (iter < max_iterations) & (~converged)

    def while_body(
        carry: tuple[ADMMState, Bool[Array, ""], Int[Array, ""]],
    ) -> tuple[ADMMState, Bool[Array, ""], Int[Array, ""]]:
        state, converged, iter = carry
        new_state = x_update(state)
        new_state = update_z_and_u(new_state, l1_penalty)
        update_rho = jnp.asarray(adapt_rho & (iter < max_iterations // 2))
        new_state, primal_ok, dual_ok = check_residuals(
            new_state, state, jnp.asarray(tol), update_rho=update_rho
        )

        converged = primal_ok.all() & dual_ok.all()
        return new_state, converged, iter + 1

    # run the admm optimization loop
    state, converged, iter = jax.lax.while_loop(
        while_condition, while_body, (state, converged, iter)
    )
    return state, converged, iter
