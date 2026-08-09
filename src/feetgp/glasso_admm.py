from typing import Any, Callable, NamedTuple, Optional, Self
from jaxtyping import Array, Bool, Int, Float

import jax
import jax.numpy as jnp
import equinox as eqx
from einops import reduce

RHO_MIN, RHO_MAX = 1e-4, 1e4


class ADMMState(NamedTuple):
    x: Float[Array, "... g"]
    z: Float[Array, "... g"]
    u: Float[Array, "... g"]
    rho: Float[Array, ""] = jnp.array(1.0)
    iteration: Int[Array, ""] = jnp.array(0)
    primal_residual: Float[Array, ""] = jnp.array(jnp.inf)
    dual_residual: Float[Array, ""] = jnp.array(jnp.inf)
    aux: tuple[Any, ...] = ()

    def converged(self, tol: Float[Array, ""]) -> Bool[Array, ""]:
        """Both residuals are relative, so one tolerance covers them."""
        return (self.primal_residual <= tol) & (self.dual_residual <= tol)

    def restart(self) -> Self:
        """Warmstart from the primal iterate alone, dual and rho start over."""
        return self._replace(
            z=jnp.zeros_like(self.z),
            u=jnp.zeros_like(self.u),
            rho=jnp.array(1.0),
            iteration=jnp.array(0),
            primal_residual=jnp.array(jnp.inf),
            dual_residual=jnp.array(jnp.inf),
        )


UpdateX = Callable[[ADMMState], ADMMState]


@eqx.filter_jit
def update_z_and_u(state: ADMMState, l1_penalty: Float[Array, ""]) -> ADMMState:
    # compute the group lasso proximal operator
    v = state.x + state.u
    norm = jnp.sqrt(reduce(v**2, "... g -> g", "sum"))
    shrink = jnp.where(norm > 0.0, l1_penalty / state.rho / norm, jnp.inf)
    z = v * (1 - shrink).clip(min=0.0, max=1.0)
    return state._replace(z=z, u=state.u + state.x - z)


@eqx.filter_jit
def update_residuals(state: ADMMState, prev: ADMMState) -> ADMMState:
    # primal residual ||x - z||_2, relative to max(||x||_2, ||z||_2)
    primal = jnp.linalg.norm(state.x - state.z)
    primal_target = jnp.maximum(jnp.linalg.norm(state.x), jnp.linalg.norm(state.z))

    # dual residual ||rho (z - z_prev)||_2, relative to ||rho u||_2
    dual = state.rho * jnp.linalg.norm(state.z - prev.z)
    dual_target = state.rho * jnp.linalg.norm(state.u)

    # a zero target leaves the ratio at inf, or at zero when the residual vanishes
    return state._replace(
        iteration=state.iteration + 1,
        primal_residual=jnp.nan_to_num(primal / primal_target),
        dual_residual=jnp.nan_to_num(dual / dual_target),
    )


@eqx.filter_jit
def update_rho(state: ADMMState, adapt: Bool[Array, ""]) -> ADMMState:
    # adapt rho if the primal and dual residuals are very different
    increase = (state.primal_residual > 10 * state.dual_residual) & adapt
    decrease = (state.dual_residual > 10 * state.primal_residual) & adapt
    scale = jnp.select([increase, decrease], [2.0, 0.5], default=1.0)
    new_rho = jnp.clip(state.rho * scale, RHO_MIN, RHO_MAX)
    return state._replace(rho=new_rho, u=state.u * (state.rho / new_rho))


@eqx.filter_jit
def solve(
    x_update: UpdateX,
    state: ADMMState,
    l1_penalty: Float[Array, ""],
    *,
    max_iterations: int | Int[Array, ""] = jnp.array(300),
    tol: float | Float[Array, ""] = jnp.array(1e-5),
    adapt_rho: bool | Bool[Array, ""] = jnp.array(True),
) -> ADMMState:
    # define admm loop condition and body
    def while_condition(state: ADMMState) -> Bool[Array, ""]:
        return (state.iteration < max_iterations) & ~state.converged(jnp.asarray(tol))

    def while_body(state: ADMMState) -> ADMMState:
        new_state = x_update(state)
        new_state = update_z_and_u(new_state, l1_penalty)
        new_state = update_residuals(new_state, state)
        adapt = jnp.asarray(adapt_rho & (state.iteration < max_iterations // 2))
        return update_rho(new_state, adapt)

    # run the admm optimization loop
    return jax.lax.while_loop(while_condition, while_body, state)


@eqx.filter_jit
def kkt_certificate(
    x: Float[Array, "... g"],
    grad: Float[Array, "... g"],
    l1_penalty: Float[Array, ""],
    lower_bound: Optional[Float[Array, "... g"]] = None,
    upper_bound: Optional[Float[Array, "... g"]] = None,
) -> Float[Array, "g"]:
    # kkt of live groups: gradient plus the penalty subgradient
    norms = jnp.sqrt(reduce(x**2, "... g -> g", "sum"))
    kkt = grad + l1_penalty * x / jnp.where(norms > 0.0, norms, 1.0)

    # project out components blocked by the box constraints
    if lower_bound is not None:
        kkt = jnp.where(x <= lower_bound, kkt.clip(max=0.0), kkt)
    if upper_bound is not None:
        kkt = jnp.where(x >= upper_bound, kkt.clip(min=0.0), kkt)

    # dead groups only need their gradient inside the penalty ball
    kkt = jnp.sqrt(reduce(kkt**2, "... g -> g", "sum"))
    dead_slack = l1_penalty - jnp.sqrt(reduce(grad**2, "... g -> g", "sum"))
    return jnp.where(norms > 0.0, kkt, -dead_slack)
