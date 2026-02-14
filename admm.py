from typing import NamedTuple, Optional, Callable
from jaxtyping import Array, Float, Scalar
import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
import scipy
from tqdm import tqdm

EPS = jnp.sqrt(jnp.finfo(float).eps)


class OptState(NamedTuple):
    x: Float[Array, "n"]
    z: Float[Array, "n"]
    u: Float[Array, "n"]
    rho: float

    @classmethod
    def initialize(cls, x0: Float[Array, "n"], rho: float = 1.0):
        return cls(x=x0, z=x0.copy(), u=jnp.zeros_like(x0), rho=rho)


def minimize_with_l1_penalty(
    fun: Callable[[Float[Array, "n"]], Scalar],
    x0: Float[Array, "n"],
    l1_penalty: Float[Array, "n"],
    bounds: Optional[list[tuple[float, float]]] = None,
    max_iterations: int = 1000,
    tol: float = 1e-4,
):

    @jax.jit
    @jax.value_and_grad
    def x_update_loss(x, z, u, rho):
        return fun(x) + 0.5 * rho * jnp.sum((x - z + u) ** 2)

    def soft_threshold(x, threshold):
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)

    state = OptState.initialize(x0=x0, rho=max(l1_penalty) + EPS)

    for _ in tqdm(range(max_iterations), leave=False):
        # x update step using L-BFGS-B
        optimization = scipy.optimize.minimize(
            fun=x_update_loss,
            x0=state.x,
            args=(state.z, state.u, state.rho),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=10, ftol=EPS, gtol=0),
        )
        new_x = optimization["x"]

        # z update step using soft-thresholding
        new_z = soft_threshold(new_x + state.u, l1_penalty / state.rho)

        # u update step
        new_u = state.u + new_x - new_z

        # update rho to balance primal and dual residuals
        primal_residual = norm(new_x - new_z)
        dual_residual = state.rho * norm(new_z - state.z)
        if primal_residual > 10 * dual_residual:
            new_rho = 2 * state.rho
            new_u = new_u / 2
        elif dual_residual > 10 * primal_residual:
            new_rho = state.rho / 2
            new_u = new_u * 2
        else:
            new_rho = state.rho

        state = OptState(x=new_x, z=new_z, u=new_u, rho=new_rho)

        # check convergence
        primal_ok = primal_residual < EPS + tol * max(norm(state.x), norm(state.z))
        dual_ok = dual_residual < EPS + tol * state.rho * norm(state.u)
        if primal_ok and dual_ok:
            break
    else:
        print("ADMM did not converge within the maximum number of iterations.")

    return state
