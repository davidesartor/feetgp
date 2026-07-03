from typing import NamedTuple, Self
from jaxtyping import Array, Float, Scalar, Bool

import jax
import jax.numpy as jnp
import equinox as eqx

from einops import rearrange
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


class GLASSOADMMState(NamedTuple):
    x: Float[Array, "o d*g"]
    z: Float[Array, "o d*g"]
    u: Float[Array, "o d*g"]
    rho: Scalar
    l1: Scalar
    group_size: int

    @eqx.filter_jit
    def x_update(
        self,
        x_train: Float[Array, "n d*g"],
        y_train: Float[Array, "n o"],
    ) -> Self:
        n, d = x_train.shape
        A = x_train.T @ x_train + self.rho * jnp.eye(d)
        b = x_train.T @ y_train + self.rho * (self.z - self.u).T
        return self._replace(x=jnp.linalg.solve(A, b).T)

    @eqx.filter_jit
    def z_and_u_update(self) -> Self:
        z = rearrange(self.x + self.u, "o (d g) -> (o g) d", g=self.group_size)
        group_norm = jnp.linalg.norm(z, axis=0)
        prox = jnp.maximum(0, 1 - (self.l1 / (self.rho * group_norm).clip(min=EPS)))
        z = rearrange(prox * z, "(o g) d -> o (d g)", g=self.group_size)
        u = self.u + self.x - z
        return self._replace(z=z, u=u)

    @eqx.filter_jit
    def check_residuals(
        self, prev: Self, tol: Scalar, adapt_rho: bool = True
    ) -> tuple[Self, Bool[Array, ""], Bool[Array, ""]]:
        # check primal
        primal_residual = jnp.linalg.norm(self.x - self.z)
        primal_target = jnp.maximum(jnp.linalg.norm(prev.x), jnp.linalg.norm(prev.z))
        primal_ok = primal_residual < EPS + tol * primal_target

        # check dual
        dual_residual = self.rho * jnp.linalg.norm(self.z - prev.z)
        dual_target = self.rho * jnp.linalg.norm(prev.u)
        dual_ok = dual_residual < EPS + tol * dual_target

        # update rho and u to balance primal and dual residuals
        if adapt_rho:
            increase = primal_residual > 10 * dual_residual
            decrease = dual_residual > 10 * primal_residual
            scale = jnp.select([increase, decrease], [2.0, 0.5], default=1.0)
            self = self._replace(rho=self.rho * scale, u=self.u / scale)
        return self, primal_ok, dual_ok


class GroupLassoLinear(NamedTuple):
    theta: Float[Array, "o d*g"]
    x_train: Float[Array, "n d*g"]
    y_train: Float[Array, "n o"]

    @jax.jit
    def predict(self, xs: Float[Array, "m d*g"]) -> Float[Array, "o m"]:
        return self.theta @ xs.T

    @classmethod
    def fit(
        cls,
        x_train: Float[Array, "n d*g"],
        y_train: Float[Array, "n o"],
        l1_penalty: Scalar,
        group_size: int,
        *,
        max_iterations: int = 1000,
        tol: Scalar = jnp.array(1e-3),
        adapt_rho: bool = True,
        **kwargs,  # ignored, used for api compatibility
    ) -> tuple[Self, Scalar]:
        n, d_times_g = x_train.shape
        n, o = y_train.shape
        assert d_times_g % group_size == 0

        x0 = jnp.zeros((o, d_times_g))

        # initialize ADMM state
        state = GLASSOADMMState(
            x=x0,
            z=x0,
            u=jnp.zeros((o, d_times_g)),
            rho=jnp.array(1.0),
            l1=l1_penalty,
            group_size=group_size,
        )

        # ADMM iterations
        for iter in (pbar := tqdm(range(max_iterations), desc="ADMM")):
            new_state = state.x_update(x_train, y_train)
            new_state = new_state.z_and_u_update()
            state, primal_ok, dual_ok = new_state.check_residuals(
                state, tol, adapt_rho and iter < max_iterations // 2
            )
            pbar.set_postfix({"rho": state.rho.item()})
            if primal_ok.all() and dual_ok.all():
                print(f"ADMM converged in {iter+1} iterations.")
                break
        else:
            print("ADMM did not converge within the maximum number of iterations.")

        model = cls(theta=state.z, x_train=x_train, y_train=y_train)
        loss = 0.5 * jnp.sum((y_train.T - model.predict(x_train)) ** 2)
        return model, loss
