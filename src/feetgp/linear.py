from typing import NamedTuple, Self
from jaxtyping import Array, Bool, Int, Float

import jax.numpy as jnp
import equinox as eqx

from einops import rearrange, einsum

from feetgp.glasso_admm import ADMMState, kkt_certificate, solve


class GroupLassoLinear(NamedTuple):
    theta: Float[Array, "o d g"]
    bias: Float[Array, "o"]
    x_train: Float[Array, "n d g"]
    y_train: Float[Array, "n o"]

    @eqx.filter_jit
    def predict(self, x: Float[Array, "m d g"]) -> Float[Array, "m o"]:
        return einsum(self.theta, x, "o d g, m d g -> m o") + self.bias

    @classmethod
    @eqx.filter_jit
    def fit(
        cls,
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
        *,
        fit_intercept: bool = True,
        **kwargs,
    ) -> tuple[Self, Float[Array, ""], Bool[Array, ""], Int[Array, ""], dict]:
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
            target = rearrange(state.z - state.u, "(o g) d -> (d g) o", g=g)
            A = A0 + state.rho * jnp.eye(d * g)
            b = b0 + state.rho * target
            x = rearrange(jnp.linalg.solve(A, b), "(d g) o -> (o g) d", g=g)
            return state._replace(x=x)

        # run the ADMM solver loop
        state, converged, iterations = solve(
            x_update, x0=jnp.zeros((o * g, d)), **kwargs
        )

        # build the final model and compute the training loss
        theta = rearrange(state.z, "(o g) d -> o d g", g=g)
        bias = y_mean - einsum(theta, x_mean, "o d g, d g -> o")
        model = cls(theta=theta, bias=bias, x_train=x_train, y_train=y_train)
        loss = 0.5 * jnp.sum((y_train - model.predict(x_train)) ** 2)

        # optimality certificate from the closed form gradient
        theta_flat = rearrange(state.z, "(o g) d -> (d g) o", g=g)
        grad = rearrange(A0 @ theta_flat - b0, "(d g) o -> (o g) d", g=g)
        certificate = kkt_certificate(
            grad, state.z, kwargs.get("l1_penalty", jnp.array(0.0))
        )
        return model, loss, converged, iterations, certificate
