"""Group-lasso ADMM. The caller owns the x-update; z, u, rho and the loop live here."""

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
    """Iterates laid out as (... g): leading axes index groups, the last holds one group.

    That convention is what removes group_size from this module entirely -- the caller
    rearranges once so that every element sharing a penalty sits on the last axis.
    aux carries unpenalized parameters the x-update owns; nothing here touches it, so
    they never enter the consensus constraint or either residual.
    """

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


# state, iteration -> updated state (x and aux), and whether the subproblem was solved
# to full budget. An inexact x-update reports False to suppress the convergence test,
# so a cheap early iteration cannot fake convergence.
XUpdate = Callable[[ADMMState, int], tuple[ADMMState, bool]]


def to_groups(v: Float[Array, "o d*g"], group_size: int) -> Float[Array, "d o*g"]:
    """Model layout (outputs by flat coefficients) -> ADMM layout (groups by members).

    A group is one block of columns across *every* output, so the members of group d are
    all o outputs times the g coefficients in the block. Both models store parameters per
    output, so they convert on the way in and back out.
    """
    return rearrange(v, "o (d g) -> d (o g)", g=group_size)


def to_outputs(v: Float[Array, "d o*g"], group_size: int) -> Float[Array, "o d*g"]:
    return rearrange(v, "d (o g) -> o (d g)", g=group_size)


@jax.jit
def group_soft_threshold(
    v: Float[Array, "... g"], threshold: Scalar
) -> Float[Array, "... g"]:
    """Group-lasso prox: shrink each last-axis slice toward zero, killing it past threshold."""
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return jnp.maximum(0.0, 1 - threshold / norm.clip(min=EPS)) * v


@eqx.filter_jit
def z_and_u_update(
    state: ADMMState,
    l1: Scalar,
    alpha: Scalar = jnp.array(1.0),
    bounds: Optional[Float[Array, "2 ... g"]] = None,
) -> ADMMState:
    """Prox step, over-relaxed by alpha (Boyd 3.4.3; 1.0 is plain ADMM).

    alpha only enters through x_hat, so the primal residual is still measured on the
    real x. Raising it above 1.0 is safe only for an unconstrained problem: with a box,
    x_hat = alpha*x + (1-alpha)*z leaves it whenever x and z straddle, and at l1 = 0 the
    prox is the identity so r = (alpha-1)||x - z_prev|| is the x-update's own jitter
    rather than a consensus error, with a floor it cannot cross.
    """
    x_hat = alpha * state.x + (1 - alpha) * state.z
    z = group_soft_threshold(x_hat + state.u, l1 / state.rho)
    if bounds is not None:
        z = jnp.clip(z, *bounds)
    return state._replace(z=z, u=state.u + x_hat - z)


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
    """Boyd 3.3.1 stopping test, plus the residual-balancing rho update."""
    # the absolute floor is sqrt(p) * EPS, not EPS: both residuals are norms of
    # p-element vectors, and at l1 = 0 the prox is the identity, so u is exactly zero
    # and the dual criterion has no relative term left to carry it. Unscaled, that asks
    # an inexact x-update for a dual residual below its own noise floor.
    eps_abs = jnp.sqrt(state.x.size) * EPS

    primal_residual, dual_residual = residuals(state, prev)
    primal_target = jnp.maximum(jnp.linalg.norm(prev.x), jnp.linalg.norm(prev.z))
    primal_ok = primal_residual < eps_abs + tol * primal_target

    dual_target = state.rho * jnp.linalg.norm(prev.u)
    dual_ok = dual_residual < eps_abs + tol * dual_target

    # rho is bounded: a primal residual with an irreducible floor otherwise feeds the
    # x2 rule forever, and past ~1e12 the augmented term swamps the objective in float64
    # so the residual can never fall again. u is rescaled by the *applied* ratio, so the
    # clamp stays consistent.
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
    # over-relaxation; Boyd reports 1.5-1.8 as the useful band, 1.0 is plain ADMM
    alpha: float = 1.0,
    adapt_rho: bool = True,
    adapt_rho_iters: Optional[int] = None,
    # tqdm goes to stderr and only shows the latest iteration; a periodic stdout row is
    # what makes a residual that stalls distinguishable from one still falling
    log_every: int = 0,
) -> tuple[ADMMState, dict]:
    """Run ADMM to convergence or the iteration cap, returning the state and an info dict."""
    # tying the rho freeze to the budget keeps a retuned budget from silently moving it
    if adapt_rho_iters is None:
        adapt_rho_iters = max_iterations // 2

    info = dict(converged=False, iterations=max_iterations)
    for iteration in (pbar := tqdm(range(max_iterations), desc="ADMM")):
        new_state, exact = x_update(state, iteration)
        new_state = z_and_u_update(new_state, l1, jnp.array(alpha), bounds)
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
