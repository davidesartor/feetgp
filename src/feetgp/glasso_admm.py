
from typing import Any, Callable, NamedTuple, Optional, Self
from jaxtyping import Array, Bool, Float, Scalar

import jax
import jax.numpy as jnp
import equinox as eqx

from einops import rearrange
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))
RHO_MIN, RHO_MAX = 1e-6, 1e6


class ADMMState(NamedTuple):

    x: Float[Array, "... g"]
    z: Float[Array, "... g"]
    u: Float[Array, "... g"]
    rho: Scalar
    aux: Any = None

    @classmethod
    def initialize(
        cls,
        x0: Float[Array, "... g"],
        rho: Scalar = jnp.array(1.0),
        aux: Any = None,
    ) -> Self:
        return cls(x=x0, z=x0, u=jnp.zeros_like(x0), rho=rho, aux=aux)


XUpdate = Callable[[ADMMState, int], tuple[ADMMState, bool]]


def to_groups(v: Float[Array, "o d*g"], group_size: int) -> Float[Array, "d o*g"]:
    return rearrange(v, "o (d g) -> d (o g)", g=group_size)


def to_outputs(v: Float[Array, "d o*g"], group_size: int) -> Float[Array, "o d*g"]:
    return rearrange(v, "d (o g) -> o (d g)", g=group_size)


@jax.jit
def group_soft_threshold(
    v: Float[Array, "... g"], threshold: Scalar
) -> Float[Array, "... g"]:
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return jnp.maximum(0.0, 1 - threshold / norm.clip(min=EPS)) * v


@eqx.filter_jit
def z_and_u_update(
    state: ADMMState,
    l1: Scalar,
    bounds: Optional[Float[Array, "2 ... g"]] = None,
) -> ADMMState:
    z = group_soft_threshold(state.x + state.u, l1 / state.rho)
    if bounds is not None:
        z = jnp.clip(z, *bounds)
    return state._replace(z=z, u=state.u + state.x - z)


@eqx.filter_jit
def residuals(state: ADMMState, prev: ADMMState) -> tuple[Scalar, Scalar]:
    return (
        jnp.linalg.norm(state.x - state.z),
        state.rho * jnp.linalg.norm(state.z - prev.z),
    )


@eqx.filter_jit
def check_residuals(
    state: ADMMState, prev: ADMMState, tol: Scalar, adapt_rho: bool = True
) -> tuple[ADMMState, Bool[Array, ""], Bool[Array, ""]]:
    eps_abs = jnp.sqrt(state.x.size) * EPS

    primal_residual, dual_residual = residuals(state, prev)
    primal_target = jnp.maximum(jnp.linalg.norm(prev.x), jnp.linalg.norm(prev.z))
    primal_ok = primal_residual < eps_abs + tol * primal_target

    dual_target = state.rho * jnp.linalg.norm(prev.u)
    dual_ok = dual_residual < eps_abs + tol * dual_target

    if adapt_rho:
        increase = primal_residual > 10 * dual_residual
        decrease = dual_residual > 10 * primal_residual
        scale = jnp.select([increase, decrease], [2.0, 0.5], default=1.0)
        rho = jnp.clip(state.rho * scale, RHO_MIN, RHO_MAX)
        state = state._replace(rho=rho, u=state.u * (state.rho / rho))
    return state, primal_ok, dual_ok


def solve(
    x_update: XUpdate,
    state: ADMMState,
    l1: Scalar,
    *,
    max_iterations: int = 300,
    tol: Scalar = jnp.array(1e-3),
    bounds: Optional[Float[Array, "2 ... g"]] = None,
    adapt_rho: bool = True,
    adapt_rho_iters: Optional[int] = None,
    log_every: int = 0,
) -> tuple[ADMMState, dict]:
    if adapt_rho_iters is None:
        adapt_rho_iters = max_iterations // 2

    info = dict(converged=False, iterations=max_iterations)
    for iteration in (pbar := tqdm(range(max_iterations), desc="ADMM")):
        new_state, exact = x_update(state, iteration)
        new_state = z_and_u_update(new_state, l1, bounds)
        primal, dual = (float(r) for r in residuals(new_state, state))
        state, primal_ok, dual_ok = check_residuals(
            new_state, state, tol, adapt_rho and iteration < adapt_rho_iters
        )

        pbar.set_postfix({"rho": state.rho.item(), "r": primal, "s": dual})
        info |= dict(
            iterations=iteration + 1, primal_residual=primal, dual_residual=dual
        )
        if log_every and iteration % log_every == 0:
            print(
                f"    iter {iteration + 1:>5}  r={primal:.4e}  s={dual:.4e}"
                f"  rho={state.rho.item():.4e}"
            )

        if exact and primal_ok.all() and dual_ok.all():
            info["converged"] = True
            print(f"ADMM converged in {iteration + 1} iterations.")
            break
    else:
        print("ADMM did not converge within the maximum number of iterations.")
    return state, info
