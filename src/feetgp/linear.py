from typing import NamedTuple, Optional, Self
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import equinox as eqx

from feetgp import admm
from feetgp.admm import ADMMState

jax.config.update("jax_enable_x64", True)


class GLASSOADMMState(NamedTuple):
    """Format-4 state, kept only so pickles written before the ADMM port still load.

    Nothing constructs one any more; admm_state_from_legacy converts it.
    """

    x: Float[Array, "o d*g"]
    z: Float[Array, "o d*g"]
    u: Float[Array, "o d*g"]
    rho: Scalar
    l1: Scalar
    group_size: int


def admm_state_from_legacy(legacy: GLASSOADMMState) -> ADMMState:
    """Format-4 (o, d*g) iterates -> the (... g) layout. Lossless, layout only."""
    x, z, u = (
        admm.to_groups(v, legacy.group_size) for v in (legacy.x, legacy.z, legacy.u)
    )
    return ADMMState(x=x, z=z, u=u, rho=legacy.rho)


def admm_state_from_pickle(results: dict) -> ADMMState:
    """The state out of a result pickle, converting the format-4 layout if it is one."""
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
    """Ridge solve: with a quadratic loss the x-update is closed form."""
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
        # over-relaxation; Boyd reports 1.5-1.8 as the useful band, 1.0 is plain ADMM.
        # It is safe here and not in the GP because this problem has no box for x_hat
        # to leave, and its paths are measured good
        alpha: float = 1.6,
        log_every: int = 0,
        **kwargs,  # ignored, used for api compatibility
    ) -> tuple[Self, Scalar, ADMMState, dict]:
        _, d_times_g = x_train.shape
        _, o = y_train.shape
        assert d_times_g % group_size == 0

        state = warmstart or ADMMState.initialize(
            jnp.zeros((d_times_g // group_size, o * group_size))
        )

        def x_update(state: ADMMState, _: int) -> tuple[ADMMState, bool]:
            target = admm.to_outputs(state.z - state.u, group_size)
            x = x_update_solve(target, x_train, y_train, state.rho)
            return state._replace(x=admm.to_groups(x, group_size)), True

        state, info = admm.solve(
            x_update,
            state,
            l1_penalty,
            max_iterations=max_iterations,
            tol=tol,
            alpha=alpha,
            adapt_rho=adapt_rho,
            adapt_rho_iters=adapt_rho_iters,
            log_every=log_every,
        )

        model = cls(
            theta=admm.to_outputs(state.z, group_size),
            x_train=x_train,
            y_train=y_train,
        )
        loss = 0.5 * jnp.sum((y_train.T - model.predict(x_train)) ** 2)
        return model, loss, state, info
