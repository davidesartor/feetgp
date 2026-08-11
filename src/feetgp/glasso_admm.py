from typing import Any, Callable, NamedTuple, Optional, Self
from jaxtyping import Array, Bool, Int, Float, PyTree

import jax
import jax.numpy as jnp
import equinox as eqx
from einops import reduce

RHO_MIN, RHO_MAX = 1e-4, 1e4

# every leaf carries the group axis last, the penalty pools across all of them
Blocks = PyTree[Float[Array, "... g"]]


def group_norm(tree: Blocks) -> Float[Array, "g"]:
    """Euclidean norm per group, pooled across every leaf."""
    leaves = jax.tree.leaves(tree)
    return jnp.sqrt(sum(reduce(leaf**2, "... g -> g", "sum") for leaf in leaves))


class ADMMState(NamedTuple):
    x: Blocks
    z: Blocks
    u: Blocks
    rho: Float[Array, ""] = jnp.array(1.0)
    iteration: Int[Array, ""] = jnp.array(0)
    primal_residual: Float[Array, ""] = jnp.array(jnp.inf)
    dual_residual: Float[Array, ""] = jnp.array(jnp.inf)

    # side state the x update carries along, never penalized nor split
    aux: Any = ()

    def converged(self, tol: Float[Array, ""]) -> Bool[Array, ""]:
        """Both residuals are relative, so one tolerance covers them."""
        return (self.primal_residual <= tol) & (self.dual_residual <= tol)

    def restart(self) -> Self:
        """Warmstart from the primal iterate alone, dual and rho start over."""
        return self._replace(
            z=jax.tree.map(jnp.zeros_like, self.z),
            u=jax.tree.map(jnp.zeros_like, self.u),
            rho=jnp.array(1.0),
            iteration=jnp.array(0),
            primal_residual=jnp.array(jnp.inf),
            dual_residual=jnp.array(jnp.inf),
        )


UpdateX = Callable[[ADMMState], ADMMState]


@eqx.filter_jit
def update_z_and_u(state: ADMMState, l1_penalty: Float[Array, ""]) -> ADMMState:
    # compute the group lasso proximal operator, one shared factor per group
    v = jax.tree.map(jnp.add, state.x, state.u)
    norm = group_norm(v)
    shrink = jnp.where(norm > 0.0, l1_penalty / state.rho / norm, jnp.inf)
    keep = (1 - shrink).clip(min=0.0, max=1.0)

    z = jax.tree.map(lambda leaf: leaf * keep, v)
    u = jax.tree.map(lambda u, x, z: u + x - z, state.u, state.x, z)
    return state._replace(z=z, u=u)


@eqx.filter_jit
def update_residuals(state: ADMMState, prev: ADMMState) -> ADMMState:
    # primal residual ||x - z||_2, relative to the larger of the two blocks
    norm = lambda tree: jnp.linalg.norm(group_norm(tree))
    primal = norm(jax.tree.map(jnp.subtract, state.x, state.z))
    primal_target = jnp.maximum(norm(state.x), norm(state.z))

    # dual residual ||rho (z - z_prev)||_2, relative to ||rho u||_2
    dual = state.rho * norm(jax.tree.map(jnp.subtract, state.z, prev.z))
    dual_target = state.rho * norm(state.u)

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
    u = jax.tree.map(lambda leaf: leaf * (state.rho / new_rho), state.u)
    return state._replace(rho=new_rho, u=u)


@eqx.filter_jit
def solve(
    x_update: UpdateX,
    state: ADMMState,
    l1_penalty: Float[Array, ""],
    *,
    max_iterations: int | Int[Array, ""] = jnp.array(300),
    tol: Optional[float | Float[Array, ""]] = None,
    adapt_rho: bool | Bool[Array, ""] = jnp.array(True),
) -> ADMMState:
    # the scalar defaults are built at import time, so only x carries the true precision
    dtype = jax.tree.leaves(state.x)[0].dtype
    state = state._replace(
        rho=jnp.asarray(state.rho, dtype),
        primal_residual=jnp.asarray(state.primal_residual, dtype),
        dual_residual=jnp.asarray(state.dual_residual, dtype),
    )

    # default tol depends on the machine precision
    eps = jnp.finfo(dtype).eps
    tol = jnp.asarray(100 * eps, dtype) if tol is None else jnp.asarray(tol, dtype)

    # define admm loop condition and body
    def while_condition(state: ADMMState) -> Bool[Array, ""]:
        return (state.iteration < max_iterations) & ~state.converged(tol)

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
    x: Blocks,
    grad: Blocks,
    l1_penalty: Float[Array, ""],
    lower_bound: Optional[Blocks] = None,
    upper_bound: Optional[Blocks] = None,
) -> Float[Array, "g"]:
    # kkt of live groups: gradient plus the penalty subgradient
    norms = group_norm(x)
    safe_norms = jnp.where(norms > 0.0, norms, 1.0)
    kkt = jax.tree.map(lambda x, g: g + l1_penalty * x / safe_norms, x, grad)

    # project out components blocked by the box constraints
    if lower_bound is not None:
        clip_low = lambda k, x, lo: jnp.where(x <= lo, k.clip(max=0.0), k)
        kkt = jax.tree.map(clip_low, kkt, x, lower_bound)
    if upper_bound is not None:
        clip_high = lambda k, x, up: jnp.where(x >= up, k.clip(min=0.0), k)
        kkt = jax.tree.map(clip_high, kkt, x, upper_bound)

    # dead groups only need their gradient inside the penalty ball
    dead_slack = l1_penalty - group_norm(grad)
    return jnp.where(norms > 0.0, group_norm(kkt), -dead_slack)
