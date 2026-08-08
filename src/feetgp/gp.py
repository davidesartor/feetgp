from typing import NamedTuple, Optional, Self
from jaxtyping import Array, Bool, Float, Int

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx

from einops import rearrange, repeat, pack, unpack
import numpy as np
from vlse.optim import minimise
from feetgp import glasso_admm
from feetgp.glasso_admm import ADMMState, solve

# fp64 is a must otherwise cholesky fails badly
jax.config.update("jax_enable_x64", True)


CERTIFICATE_TOLERANCE = 1e-2


def squared_distance(
    x1: Float[Array, "n d"], x2: Float[Array, "m d"]
) -> Float[Array, "n m"]:
    # fast pairwise squared distances expanding the square
    sqn1 = jnp.sum(x1**2, axis=-1)
    sqn2 = jnp.sum(x2**2, axis=-1)
    return sqn1[:, None] + sqn2[None, :] - 2.0 * x1 @ x2.T


def kernel(
    theta: Float[Array, "d"],
    x1: Float[Array, "n d"],
    x2: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    # squared exponential kernel with ARD lengthscales
    d2 = squared_distance(x1 * theta, x2 * theta)
    return jnp.exp(-0.5 * d2)


def hetgpy_auto_bounds(
    x: Float[Array, "n d"],
    min_cor: float = 0.01,
    max_cor: float = 0.5,
) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
    # normalize each column to [0, 1] so distances are comparable
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x_span = jnp.where(x_max > x_min, x_max - x_min, 1.0)
    x = (x - x_min) / x_span

    # pairwise squared distances between distinct points
    rows, cols = jnp.tril_indices(len(x), k=-1)
    dists = squared_distance(x, x)
    dists = jnp.maximum(dists[rows, cols], 0.0)
    dists = jnp.where(dists > 0, dists, jnp.nan)

    # lengthscale bounds from target correlations at distance quantiles
    lower = -jnp.nanquantile(dists, 0.05) / jnp.log(min_cor) * x_span**2
    upper = -jnp.nanquantile(dists, 0.95) / jnp.log(max_cor) * x_span**2
    return lower, upper


def gp_posterior(
    Kxx: Float[Array, "m m"],
    Kox: Float[Array, "n m"],
    Koo: Float[Array, "n n"],
    y_train: Float[Array, "n"],
    b: Float[Array, ""],
) -> tuple[Float[Array, "m"], Float[Array, "m m"]]:
    # condition on the observations
    chol = jsp.linalg.cho_factor(Koo)
    gain = jsp.linalg.cho_solve(chol, Kox)
    mean = b + gain.T @ (y_train - b)
    cov = Kxx - Kox.T @ gain

    # inflate the covariance for the estimated trend
    Kbx = jnp.ones((1, len(y_train))) @ gain
    Ki_1 = jsp.linalg.cho_solve(chol, jnp.ones_like(y_train))
    cov = cov + (1 - Kbx).T @ (1 - Kbx) / Ki_1.sum()
    return mean, cov


def gp_loglikelihood(
    Koo: Float[Array, "n n"],
    y_train: Float[Array, "n"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    K_sqrt, is_lower = jsp.linalg.cho_factor(Koo)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    # solve K @ x = 1 and K @ x = y in one cholesky solve
    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([jnp.ones_like(y_train), y_train], 1),
    ).T

    # closed-form trend and variance under the profiled likelihood
    b = (Ki_1 * y_train).sum() / Ki_1.sum()
    nu = jnp.dot((y_train - b) / len(y_train), (Ki_y - Ki_1 * b))

    loglik = -0.5 * (len(y_train) * jnp.log(nu) + logdetK)
    return (loglik, b, nu)


def autoregressive_mask(o: int, d: int, group_size: int) -> Float[Array, "o d"]:
    # zero out each output's own input group
    return (jnp.arange(o)[:, None] // group_size != jnp.arange(d)[None, :]).astype(
        float
    )


@eqx.filter_jit
def kkt_certificate(
    theta: Float[Array, "o d g"],
    w: Float[Array, "o"],
    l1: Float[Array, ""],
    x_train: Float[Array, "n d g"],
    y_train: Float[Array, "n o"],
    bounds: Float[Array, "2 d o*g"],
    mask: Optional[Float[Array, "o d"]] = None,
    chunk_size: int = 8,
) -> dict:
    n, _, group_size = x_train.shape
    design = rearrange(x_train, "n d g -> n (d g)")

    def negloglik_single_output(
        theta_i: Float[Array, "d*g"], w_i: Float[Array, ""], y_i: Float[Array, "n"]
    ) -> Float[Array, ""]:
        Koo = kernel(theta_i, design, design) + jnp.exp(w_i) * jnp.eye(n)
        loglik, _, _ = gp_loglikelihood(Koo, y_i)
        return -loglik

    # per-output gradients of the negative log likelihood
    grad_theta, grad_w = jax.lax.map(
        lambda inputs: jax.grad(negloglik_single_output, argnums=(0, 1))(*inputs),
        (rearrange(theta, "o d g -> o (d g)"), w, y_train.T),
        batch_size=chunk_size,
    )
    if mask is not None:
        grad_theta = grad_theta * repeat(mask, "o d -> o (d g)", g=group_size)

    # check group lasso stationarity, nugget is unpenalized so plain gradient
    certificate = glasso_admm.kkt_certificate(
        rearrange(grad_theta, "o (d g) -> o g d", g=group_size),
        rearrange(theta, "o d g -> o g d"),
        l1,
        rearrange(bounds, "b d (o g) -> b o g d", g=group_size),
    )
    return certificate | dict(nugget_grad=jnp.max(jnp.abs(grad_w)))


class GroupLassoGaussianProcess(NamedTuple):
    theta: Float[Array, "o d g"]
    nugget: Float[Array, "o"]
    b: Float[Array, "o"]
    nu: Float[Array, "o"]

    x_train: Float[Array, "n d g"]
    y_train: Float[Array, "n o"]

    @eqx.filter_jit
    def predict(
        self, x: Float[Array, "... m d g"]
    ) -> tuple[Float[Array, "... o m"], Float[Array, "... o m m"]]:

        def predict_single(
            x: Float[Array, "m d g"],
            theta: Float[Array, "d g"],
            nugget: Float[Array, ""],
            b: Float[Array, ""],
            nu: Float[Array, ""],
            x_train: Float[Array, "n d g"],
            y_train: Float[Array, "n"],
        ) -> tuple[Float[Array, "m"], Float[Array, "m m"]]:
            x_train = rearrange(self.x_train, "n d g -> n (d g)")
            Kox = nu * kernel(theta, x_train, x)
            Koo = nu * (
                kernel(theta, x_train, x_train) + nugget * jnp.eye(len(y_train))
            )
            Kxx = nu * kernel(theta, x, x)
            return gp_posterior(Kxx, Kox, Koo, y_train, b)

        # vectorize over outputs, scan over inputs to avoid OOM
        *b, m, d, g = x.shape
        x = rearrange(x, "... m d g -> b m (d g)")
        predict = jax.vmap(predict_single, in_axes=(None, 0, 0, 0, 0, None, 1))
        mean, cov = jax.lax.map(lambda x: predict(x, *self), x)
        mean = mean.reshape(*b, *mean.shape[1:])
        cov = cov.reshape(*b, *cov.shape[1:])
        return mean, cov

    @classmethod
    @eqx.filter_jit
    def unpack_parameters(
        cls,
        admm_theta: Float[Array, "o d g"],
        admm_w: Float[Array, "o"],
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
    ) -> tuple[Self, Float[Array, ""]]:
        theta = jnp.abs(admm_theta)
        nugget = jnp.exp(admm_w)
        design = rearrange(x_train, "n d g -> n (d g)")

        def unpack_single_output(
            theta_i: Float[Array, "d*g"],
            nugget_i: Float[Array, ""],
            y_i: Float[Array, "n"],
        ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
            Koo = kernel(theta_i, design, design) + nugget_i * jnp.eye(len(y_i))
            return gp_loglikelihood(Koo, y_i)

        # recover the profiled trend and variance for each output
        _, (llk, b, nu) = jax.lax.scan(
            lambda _, args: (_, unpack_single_output(*args)),
            None,
            (rearrange(theta, "o d g -> o (d g)"), nugget, y_train.T),
        )
        self = cls(theta, nugget, b, nu, x_train, y_train)
        return self, llk.sum()

    @classmethod
    def fit(cls, *args, **kwargs) -> tuple[Self, Float[Array, ""], ADMMState, dict]:
        """Fit and move the solver diagnostics off device."""
        self, llk, state, converged, iterations, certificate = cls._fit(*args, **kwargs)
        info = dict(
            converged=bool(converged),
            iterations=int(iterations),
            certificate={k: np.asarray(v) for k, v in certificate.items()},
        )
        return self, llk, state, info

    @classmethod
    @eqx.filter_jit
    def _fit(
        cls,
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
        l1_penalty: Float[Array, ""],
        autoregressive: bool = True,
        *,
        warmstart: Optional["Self | ADMMState"] = None,
        nugget_init: float = 0.1,
        max_iterations: int = 300,
        tol: Float[Array, ""] = jnp.array(1e-3),
        adapt_rho: bool = True,
        inner_maxiter: int = 5,
        inner_pgtol: float = 1e-2,
        inner_max_linesearch: int = 5,
        chunk_size: int = 8,
        history_length: int = 40,
        **_,
    ) -> tuple[
        Self, Float[Array, ""], ADMMState, Bool[Array, ""], Int[Array, ""], dict
    ]:
        _, d, group_size = x_train.shape
        _, o = y_train.shape
        design = rearrange(x_train, "n d g -> n (d g)")

        # box constraints for theta and the free nugget parameter
        lower, upper = (
            rearrange(bound, "(d g) -> d g", g=group_size)
            for bound in hetgpy_auto_bounds(design)
        )
        theta_max = repeat(jnp.sqrt(2.0 / lower), "d g -> d (o g)", o=o)
        bounds = jnp.stack([jnp.zeros_like(theta_max), theta_max], axis=0)
        admm_bounds = rearrange(bounds, "b d (o g) -> b o g d", o=o)
        w_min, w_max = (jnp.full((o, 1), jnp.log(g)) for g in (1e-4, 100.0))
        solver_lower = jnp.concat([jnp.zeros((o, d * group_size)), w_min], axis=-1)
        solver_upper = jnp.concat(
            [repeat(jnp.sqrt(2.0 / lower), "d g -> o (d g)", o=o), w_max], axis=-1
        )

        # initialize near the long-lengthscale end of the box
        theta_init = repeat(
            jnp.sqrt(2.0 / (0.9 * upper + 0.1 * lower)), "d g -> o g d", o=o
        )
        w_init = jnp.full((o,), jnp.log(nugget_init))

        # take the starting point from the warmstart when given
        start = ADMMState(
            x=theta_init,
            z=theta_init,
            u=jnp.zeros_like(theta_init),
            rho=jnp.array(1.0),
            aux=w_init,
        )
        if isinstance(warmstart, ADMMState):
            start = warmstart
        elif warmstart is not None:
            theta = rearrange(warmstart.theta, "o d g -> o g d")
            w = jnp.log(warmstart.nugget)
            start = start._replace(x=theta, z=theta, aux=w)

        # autoregressive outputs never see their own input group
        if autoregressive:
            keep = repeat(1.0 - jnp.eye(d), "k d -> k (d g)", g=group_size)
            masked_designs = design[None, :, :] * keep[:, None, :]
            mask = autoregressive_mask(o, d, group_size)
            flat_mask = jnp.concat(
                [repeat(mask, "o d -> o (d g)", g=group_size), jnp.ones((o, 1))],
                axis=-1,
            )
        else:
            masked_designs = jnp.broadcast_to(design, (d, *design.shape))
            mask = None
            flat_mask = jnp.ones((o, d * group_size + 1))

        group_of_output = jnp.arange(o) // group_size

        # solve each output's bounded likelihood subproblem with L-BFGS-B
        def x_update_loss(x: Float[Array, "d*g+1"], args: tuple) -> Float[Array, ""]:
            # negative log likelihood plus the augmented lagrangian penalty
            target_theta, rho, design, ys = args
            theta, w = x[:-1], x[-1]
            Koo = kernel(jax.nn.relu(theta), design, design)
            Koo = Koo + jnp.exp(w) * jnp.eye(len(ys))
            loglik, _, _ = gp_loglikelihood(Koo, ys)
            lagrangian = 0.5 * rho * jnp.sum((theta - target_theta) ** 2)
            return -loglik + lagrangian

        def x_update(state: ADMMState) -> ADMMState:
            def solve_one(
                x_i: Float[Array, "g d"],
                w_i: Float[Array, ""],
                target_i: Float[Array, "g d"],
                y_i: Float[Array, "n"],
                group_i: Int[Array, ""],
                mask_i: Float[Array, "d*g+1"],
                lower_i: Float[Array, "d*g+1"],
                upper_i: Float[Array, "d*g+1"],
            ) -> tuple[Float[Array, "g d"], Float[Array, ""]]:
                x0 = jnp.concat([rearrange(x_i, "g d -> (d g)"), w_i[None]]) * mask_i
                loss_args = (
                    rearrange(target_i, "g d -> (d g)"),
                    state.rho,
                    masked_designs[group_i],
                    y_i,
                )
                solution = minimise(
                    x_update_loss,
                    x0,
                    (lower_i, upper_i),
                    args=(loss_args,),
                    tol=inner_pgtol,
                    max_iterations=inner_maxiter,
                    history_length=history_length,
                    max_linesearch=inner_max_linesearch,
                )
                solution = solution.x * mask_i
                theta_i = rearrange(solution[:-1], "(d g) -> g d", g=group_size)
                return theta_i, solution[-1]

            theta, w = jax.vmap(solve_one, in_axes=(0, 0, 0, 1, 0, 0, 0, 0))(
                state.x,
                state.aux,
                state.z - state.u,
                y_train,
                group_of_output,
                flat_mask,
                solver_lower,
                solver_upper,
            )
            return state._replace(x=jnp.clip(theta, *admm_bounds), aux=w)

        # run the ADMM solver loop
        state, converged, iterations = solve(
            x_update,
            start.x,
            start.aux,
            l1_penalty,
            z0=start.z,
            u0=start.u,
            rho0=start.rho,
            max_iterations=max_iterations,
            tol=tol,
            adapt_rho=adapt_rho,
        )

        # build the final model and its optimality certificate
        theta = rearrange(state.z, "o g d -> o d g")
        theta = theta if mask is None else theta * mask[..., None]
        self, llk = cls.unpack_parameters(theta, state.aux, x_train, y_train)
        certificate = kkt_certificate(
            self.theta,
            state.aux,
            l1_penalty,
            x_train,
            y_train,
            bounds,
            mask=mask,
            chunk_size=chunk_size,
        )
        return self, llk, state, converged, iterations, certificate
