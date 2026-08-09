from typing import NamedTuple, Self
from jaxtyping import Array, Int, Float

import jax.numpy as jnp
import equinox as eqx
from einops import rearrange, reduce, einsum

from feetgp.glasso_admm import ADMMState, solve, kkt_certificate


class Linear(NamedTuple):
    theta: Float[Array, "o d g"]
    bias: Float[Array, "o"]

    @eqx.filter_jit
    def predict(self, x: Float[Array, "... d g"]) -> Float[Array, "... o"]:
        return einsum(self.theta, x, "o d g, ... d g -> ... o") + self.bias

    @classmethod
    @eqx.filter_jit
    def lambda_max(
        cls,
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
        *,
        fit_intercept: bool = True,
    ) -> Float[Array, ""]:
        """Smallest penalty that kills every group, from the gradient at theta = 0."""
        x_mean = x_train.mean(axis=0) if fit_intercept else 0.0
        y_mean = y_train.mean(axis=0) if fit_intercept else 0.0
        grad = einsum(x_train - x_mean, y_train - y_mean, "n d g, n o -> o d g")
        return jnp.max(jnp.sqrt(reduce(grad**2, "... g -> g", "sum")))

    @classmethod
    @eqx.filter_jit
    def fit(
        cls,
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
        l1_penalty: Float[Array, ""],
        *,
        fit_intercept: bool = True,
        max_iterations: int | Int[Array, ""] = jnp.array(300),
        **kwargs,
    ) -> tuple[Self, Float[Array, ""], ADMMState, Float[Array, "g"]]:
        _, d, g = x_train.shape
        _, o = y_train.shape

        # intercept is unpenalized, so centering gives it in closed form
        x_mean = x_train.mean(axis=0) if fit_intercept else jnp.zeros((d, g))
        y_mean = y_train.mean(axis=0) if fit_intercept else jnp.zeros((o,))

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
            x = rearrange(jnp.linalg.solve(A, b), "(d g) o -> o d g", g=g)
            return state._replace(x=x)

        # run the ADMM solver loop
        zeros = jnp.zeros((o, d, g))
        state = ADMMState(x=zeros, z=zeros, u=zeros)
        state = solve(
            x_update,
            state,
            l1_penalty=l1_penalty,
            max_iterations=max_iterations,
            **kwargs,
        )

        # build the final model
        theta = state.z
        bias = y_mean - einsum(theta, x_mean, "o d g, d g -> o")
        model = cls(theta=theta, bias=bias)

        # optimality certificate, gradient of the least squares term
        residual = y_train - model.predict(x_train)
        loss = 0.5 * jnp.sum(residual**2)
        grad = -einsum(x_train - x_mean, residual, "n d g, n o -> o d g")
        certificate = kkt_certificate(theta, grad, l1_penalty)
        return model, loss, state, certificate
