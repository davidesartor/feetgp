from typing import NamedTuple, Optional
from jaxtyping import Array, Int, Float

import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange, reduce, einsum

from feetgp.glasso_admm import ADMMState, solve, kkt_certificate


def centered(
    x_train: Float[Array, "n d g"],
    y_train: Float[Array, "n o"],
    fit_intercept: bool,
) -> tuple[Float[Array, "d g"], Float[Array, "o"]]:
    """Training means, zero when the intercept is off so centering is a no-op."""
    if not fit_intercept:
        return jnp.zeros(x_train.shape[1:]), jnp.zeros(y_train.shape[1:])
    return x_train.mean(axis=0), y_train.mean(axis=0)


class Linear(NamedTuple):
    theta: Optional[Float[Array, "o d g"]] = None
    bias: Optional[Float[Array, "o"]] = None

    @eqx.filter_jit
    def predict(self, x: Float[Array, "... d g"]) -> Float[Array, "... o"]:
        return einsum(self.theta, x, "o d g, ... d g -> ... o") + self.bias

    @staticmethod
    @eqx.filter_jit
    def loss(
        theta: Float[Array, "o d g"],
        x: Float[Array, "n d g"],
        y: Float[Array, "n o"],
    ) -> Float[Array, ""]:
        """Smooth part of the objective, on centered data so the intercept drops out."""
        residual = y - einsum(theta, x, "o d g, n d g -> n o")
        return 0.5 * jnp.sum(residual**2)

    @staticmethod
    @eqx.filter_jit
    def lambda_max(
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
        *,
        fit_intercept: bool = True,
    ) -> Float[Array, ""]:
        """Smallest penalty that kills every group, from the gradient at theta = 0."""
        _, d, g = x_train.shape
        _, o = y_train.shape
        x_mean, y_mean = centered(x_train, y_train, fit_intercept)
        grad = jax.grad(Linear.loss)(
            jnp.zeros((o, d, g)), x_train - x_mean, y_train - y_mean
        )
        return jnp.max(jnp.sqrt(reduce(grad**2, "... g -> g", "sum")))

    @staticmethod
    @eqx.filter_jit
    def fit(
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
        l1_penalty: Float[Array, ""],
        *,
        fit_intercept: bool = True,
        max_iterations: int | Int[Array, ""] = jnp.array(300),
        **kwargs,
    ) -> tuple["Linear", Float[Array, ""], ADMMState, Float[Array, "g"]]:
        _, d, g = x_train.shape
        _, o = y_train.shape

        # intercept is unpenalized, so centering gives it in closed form
        x_mean, y_mean = centered(x_train, y_train, fit_intercept)

        # precompute constant matrices
        design = rearrange(x_train - x_mean, "n d g -> n (d g)")
        A0 = einsum(design, design, "n i, n j -> i j")
        b0 = einsum(design, y_train - y_mean, "n i, n o -> i o")

        # closed form of the quadratic minimization step
        # min_x 0.5 ||y - X x||_2^2 + (rho / 2) ||x - z + u||_2^2
        def x_update(state: ADMMState) -> ADMMState:
            target = rearrange(state.z - state.u, "o d g -> (d g) o")
            A = A0 + state.rho * jnp.eye(d * g)
            b = b0 + state.rho * target
            theta = rearrange(jnp.linalg.solve(A, b), "(d g) o -> o d g", g=g)
            return state._replace(x=theta)

        # run the ADMM solver loop
        theta0 = jnp.zeros((o, d, g))
        state = solve(
            x_update,
            ADMMState(x=theta0, z=theta0, u=theta0),
            l1_penalty=l1_penalty,
            max_iterations=max_iterations,
            **kwargs,
        )

        # build the final model
        theta = state.z
        bias = y_mean - einsum(theta, x_mean, "o d g, d g -> o")

        # optimality certificate, gradient of the least squares term
        loss, grad = jax.value_and_grad(Linear.loss)(
            theta, x_train - x_mean, y_train - y_mean
        )
        certificate = kkt_certificate(theta, grad, l1_penalty)
        return Linear(theta=theta, bias=bias), loss, state, certificate
