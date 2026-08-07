from typing import NamedTuple, Optional, Self
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import equinox as eqx

from feetgp import glasso_admm
from feetgp.glasso_admm import ADMMState

jax.config.update("jax_enable_x64", True)


class GLASSOADMMState(NamedTuple):

    x: Float[Array, "o d*g"]
    z: Float[Array, "o d*g"]
    u: Float[Array, "o d*g"]
    rho: Scalar
    l1: Scalar
    group_size: int


def admm_state_from_legacy(legacy: GLASSOADMMState) -> ADMMState:
    x, z, u = (
        glasso_admm.to_groups(v, legacy.group_size) for v in (legacy.x, legacy.z, legacy.u)
    )
    return ADMMState(x=x, z=z, u=u, rho=legacy.rho)


def admm_state_from_pickle(results: dict) -> ADMMState:
    state = results["admm_state"]
    if isinstance(state, GLASSOADMMState):
        return admm_state_from_legacy(state)
    return state


@eqx.filter_jit
def x_update_solve(
    target: Float[Array, "o d*g"],
    x_train: Float[Array, "n d*g"],
    y_train: Float[Array, "n o"],
    rho: Scalar,
) -> Float[Array, "o d*g"]:
    _, d = x_train.shape
    A = x_train.T @ x_train + rho * jnp.eye(d)
    b = x_train.T @ y_train + rho * target.T
    return jnp.linalg.solve(A, b).T


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
        warmstart: Optional[ADMMState] = None,
        max_iterations: int = 1000,
        tol: Scalar = jnp.array(1e-3),
        adapt_rho: bool = True,
        adapt_rho_iters: Optional[int] = None,
        log_every: int = 0,
        **kwargs,
    ) -> tuple[Self, Scalar, ADMMState, dict]:
        _, d_times_g = x_train.shape
        _, o = y_train.shape
        assert d_times_g % group_size == 0

        state = warmstart or ADMMState.initialize(
            jnp.zeros((d_times_g // group_size, o * group_size))
        )

        def x_update(state: ADMMState, _: int) -> tuple[ADMMState, bool]:
            target = glasso_admm.to_outputs(state.z - state.u, group_size)
            x = x_update_solve(target, x_train, y_train, state.rho)
            return state._replace(x=glasso_admm.to_groups(x, group_size)), True

        state, info = glasso_admm.solve(
            x_update,
            state,
            l1_penalty,
            max_iterations=max_iterations,
            tol=tol,
            adapt_rho=adapt_rho,
            adapt_rho_iters=adapt_rho_iters,
            log_every=log_every,
        )

        model = cls(
            theta=glasso_admm.to_outputs(state.z, group_size),
            x_train=x_train,
            y_train=y_train,
        )
        loss = 0.5 * jnp.sum((y_train.T - model.predict(x_train)) ** 2)
        return model, loss, state, info
