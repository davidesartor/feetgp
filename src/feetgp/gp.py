from typing import NamedTuple, Optional, Self
from jaxtyping import Array, Float

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx

from einops import rearrange, repeat
import numpy as np
from vlse.optim import minimise as lbfgsb_minimise

from feetgp import glasso_admm
from feetgp.glasso_admm import ADMMState

jax.config.update("jax_enable_x64", True)

G_RANGE = (1e-4, 100.0)

CERTIFICATE_TOLERANCE = 1e-2


@jax.jit
def kernel(
    theta: Float[Array, "d"],
    xs1: Float[Array, "n d"],
    xs2: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    scaled1, scaled2 = xs1 * theta, xs2 * theta
    sqnorm1 = jnp.sum(scaled1**2, axis=-1)
    sqnorm2 = jnp.sum(scaled2**2, axis=-1)
    sqdist = sqnorm1[:, None] + sqnorm2[None, :] - 2.0 * scaled1 @ scaled2.T
    return jnp.exp(-0.5 * jnp.maximum(sqdist, 0.0))


def hetgpy_auto_bounds(
    x: Float[Array, "n d g"],
    min_cor: float = 0.01,
    max_cor: float = 0.5,
) -> tuple[Float[Array, "d g"], Float[Array, "d g"]]:
    _, d, group_size = x.shape

    # normalize each column to [0, 1] so distances are comparable
    x = rearrange(jnp.asarray(x), "n d g -> n (d g)")
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x_span = jnp.where(x_max > x_min, x_max - x_min, 1.0)
    x = (x - x_min) / x_span

    # pairwise squared distances between distinct points
    sqnorm = jnp.sum(x**2, axis=-1)
    sqdist = sqnorm[:, None] + sqnorm[None, :] - 2.0 * x @ x.T
    rows, cols = jnp.tril_indices(len(x), k=-1)
    # coincident pairs excluded from the quantiles, as nan (shape stays static)
    dists = jnp.maximum(sqdist[rows, cols], 0.0)
    dists = jnp.where(dists > 0, dists, jnp.nan)

    # lengthscale bounds from target correlations at distance quantiles
    lower = -jnp.nanquantile(dists, 0.05) / jnp.log(min_cor) * x_span**2
    upper = -jnp.nanquantile(dists, 0.95) / jnp.log(max_cor) * x_span**2
    return (
        rearrange(lower, "(d g) -> d g", g=group_size),
        rearrange(upper, "(d g) -> d g", g=group_size),
    )


@eqx.filter_jit
def gp_posterior(
    Kxx: Float[Array, "m m"],
    Kox: Float[Array, "n m"],
    Koo: Float[Array, "n n"],
    observed_ys: Float[Array, "n"],
    b: Float[Array, ""],
) -> tuple[Float[Array, "m"], Float[Array, "m m"]]:
    # condition on the observations
    chol = jsp.linalg.cho_factor(Koo)
    gain = jsp.linalg.cho_solve(chol, Kox)
    mean = b + gain.T @ (observed_ys - b)
    cov = Kxx - Kox.T @ gain

    # inflate the covariance for the estimated trend
    Kbx = jnp.ones((1, len(observed_ys))) @ gain
    Ki_1 = jsp.linalg.cho_solve(chol, jnp.ones_like(observed_ys))
    cov = cov + (1 - Kbx).T @ (1 - Kbx) / Ki_1.sum()
    return mean, cov


@jax.jit
def loglikelihood(
    Koo: Float[Array, "n n"],
    observed_ys: Float[Array, "n"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    K_sqrt, is_lower = jsp.linalg.cho_factor(Koo)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([jnp.ones_like(observed_ys), observed_ys], 1),
    ).T

    # closed-form trend and variance under the profiled likelihood
    b = (Ki_1 * observed_ys).sum() / Ki_1.sum()
    nu = jnp.dot((observed_ys - b) / len(observed_ys), (Ki_y - Ki_1 * b))

    loglik = -0.5 * (len(observed_ys) * jnp.log(nu) + logdetK)
    return (loglik, b, nu)


def x_update_loss(x: Float[Array, "d*g+1"], args: tuple) -> Float[Array, ""]:
    # negative log likelihood plus the augmented lagrangian penalty
    target_theta, rho, design, ys, g_min, g_max = args
    theta, w = x[:-1], x[-1]
    Koo = kernel(jax.nn.relu(theta), design, design)
    Koo = Koo + nugget_from_w(w, g_min, g_max) * jnp.eye(len(ys))
    loglik, _, _ = loglikelihood(Koo, ys)
    lagrangian = 0.5 * rho * jnp.sum((theta - target_theta) ** 2)
    return -loglik + lagrangian


def nugget_from_w(
    w: Float[Array, ""], g_min: Float[Array, ""], g_max: Float[Array, ""]
) -> Float[Array, ""]:
    return g_min + (g_max - g_min) * jax.nn.sigmoid(w)


def w_from_nugget(
    g: Float[Array, ""], g_min: Float[Array, ""], g_max: Float[Array, ""]
) -> Float[Array, ""]:
    fraction = jnp.clip((g - g_min) / (g_max - g_min), 1e-12, 1 - 1e-12)
    return jsp.special.logit(fraction)


def autoregressive_mask(o: int, d: int, group_size: int) -> Float[Array, "o d"]:
    # zero out each output's own input group
    return (jnp.arange(o)[:, None] // group_size != jnp.arange(d)[None, :]).astype(
        float
    )


def theta_box(lower: Float[Array, "d g"], o: int) -> Float[Array, "2 d o*g"]:
    theta_max = repeat(jnp.sqrt(2.0 / lower), "d g -> d (o g)", o=o)
    return jnp.stack([jnp.zeros_like(theta_max), theta_max], axis=0)


@eqx.filter_jit
def penalized_objective(
    theta: Float[Array, "o d g"],
    g: Float[Array, "o"],
    l1: Float[Array, ""],
    x_train: Float[Array, "n d g"],
    y_train: Float[Array, "n o"],
) -> Float[Array, ""]:
    n = len(x_train)
    design = rearrange(x_train, "n d g -> n (d g)")

    def negloglik_single_output(
        theta_i: Float[Array, "d*g"], g_i: Float[Array, ""], y_i: Float[Array, "n"]
    ) -> Float[Array, ""]:
        Koo = kernel(theta_i, design, design) + g_i * jnp.eye(n)
        loglik, _, _ = loglikelihood(Koo, y_i)
        return -loglik

    # sum the per-output likelihoods and the group lasso penalty
    _, negloglik = jax.lax.scan(
        lambda _, inputs: (_, negloglik_single_output(*inputs)),
        None,
        (rearrange(theta, "o d g -> o (d g)"), g, y_train.T),
    )
    norms = jnp.linalg.norm(rearrange(theta, "o d g -> d (o g)"), axis=-1)
    return negloglik.sum() + l1 * norms.sum()


@eqx.filter_jit
def kkt_certificate(
    theta: Float[Array, "o d g"],
    w: Float[Array, "o"],
    l1: Float[Array, ""],
    x_train: Float[Array, "n d g"],
    y_train: Float[Array, "n o"],
    bounds: Float[Array, "2 d o*g"],
    mask: Optional[Float[Array, "o d"]] = None,
    g_min: Float[Array, ""] = jnp.array(1e-4),
    g_max: Float[Array, ""] = jnp.array(100.0),
    chunk_size: int = 8,
) -> dict:
    n, _, group_size = x_train.shape
    design = rearrange(x_train, "n d g -> n (d g)")

    def negloglik_single_output(
        theta_i: Float[Array, "d*g"], w_i: Float[Array, ""], y_i: Float[Array, "n"]
    ) -> Float[Array, ""]:
        Koo = kernel(theta_i, design, design) + nugget_from_w(
            w_i, g_min, g_max
        ) * jnp.eye(n)
        loglik, _, _ = loglikelihood(Koo, y_i)
        return -loglik

    # per-output gradients of the negative log likelihood
    grad_theta, grad_w = jax.lax.map(
        lambda inputs: jax.grad(negloglik_single_output, argnums=(0, 1))(*inputs),
        (rearrange(theta, "o d g -> o (d g)"), w, y_train.T),
        batch_size=chunk_size,
    )
    if mask is not None:
        grad_theta = grad_theta * repeat(mask, "o d -> o (d g)", g=group_size)

    # stationarity of live groups: gradient plus the penalty subgradient
    grad_groups = rearrange(grad_theta, "o (d g) -> d (o g)", g=group_size)
    theta_groups = rearrange(theta, "o d g -> d (o g)")
    norms = jnp.linalg.norm(theta_groups, axis=-1)
    live = norms > 0.0
    direction = theta_groups / jnp.where(live, norms, 1.0)[:, None]
    stationarity = grad_groups + l1 * direction

    # project out components blocked by the box constraints
    lower, upper = bounds
    stationarity = jnp.where(
        theta_groups <= lower, jnp.minimum(stationarity, 0.0), stationarity
    )
    stationarity = jnp.where(
        theta_groups >= upper, jnp.maximum(stationarity, 0.0), stationarity
    )

    # dead groups only need their gradient inside the penalty ball
    live_kkt = jnp.linalg.norm(stationarity, axis=-1)
    dead_slack = l1 - jnp.linalg.norm(grad_groups, axis=-1)
    return dict(
        max_live_kkt=jnp.max(jnp.where(live, live_kkt, 0.0)),
        live_kkt=jnp.where(live, live_kkt, jnp.nan),
        dead_slack=jnp.where(live, jnp.nan, dead_slack),
        nugget_grad=jnp.max(jnp.abs(grad_w)),
    )


class GroupLassoGaussianProcess(NamedTuple):
    theta: Float[Array, "o d g"]
    g: Float[Array, "o"]
    b: Float[Array, "o"]
    nu: Float[Array, "o"]

    x_train: Float[Array, "n d g"]
    y_train: Float[Array, "n o"]

    @eqx.filter_jit
    def predict(
        self, xs: Float[Array, "m d g"], covariance: bool = False
    ) -> Float[Array, "o m"] | tuple[Float[Array, "o m"], Float[Array, "o m m"]]:
        design = rearrange(self.x_train, "n d g -> n (d g)")
        xs = rearrange(xs, "m d g -> m (d g)")

        def predict_single_output(
            theta: Float[Array, "d*g"],
            g: Float[Array, ""],
            b: Float[Array, ""],
            nu: Float[Array, ""],
            y_train: Float[Array, "n"],
        ) -> Float[Array, "m"] | tuple[Float[Array, "m"], Float[Array, "m m"]]:
            Kox = nu * kernel(theta, design, xs)
            Koo = nu * (kernel(theta, design, design) + g * jnp.eye(len(y_train)))
            if covariance:
                Kxx = nu * kernel(theta, xs, xs)
                return gp_posterior(Kxx, Kox, Koo, y_train, b)
            gain = jsp.linalg.cho_solve(jsp.linalg.cho_factor(Koo), Kox)
            return b + gain.T @ (y_train - b)

        _, posterior = jax.lax.scan(
            lambda _, args: (_, predict_single_output(*args)),
            None,
            (
                rearrange(self.theta, "o d g -> o (d g)"),
                self.g,
                self.b,
                self.nu,
                self.y_train.T,
            ),
        )
        return posterior

    @classmethod
    @eqx.filter_jit
    def unpack_parameters(
        cls,
        admm_theta: Float[Array, "o d g"],
        admm_w: Float[Array, "o"],
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
        g_min: Float[Array, ""] = jnp.array(1e-4),
        g_max: Float[Array, ""] = jnp.array(1.0),
    ) -> tuple[Self, Float[Array, ""]]:
        theta = jnp.abs(admm_theta)
        g = nugget_from_w(admm_w, g_min, g_max)
        design = rearrange(x_train, "n d g -> n (d g)")

        def unpack_single_output(
            theta_i: Float[Array, "d*g"],
            g_i: Float[Array, ""],
            y_i: Float[Array, "n"],
        ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
            Koo = kernel(theta_i, design, design) + g_i * jnp.eye(len(y_i))
            return loglikelihood(Koo, y_i)

        # recover the profiled trend and variance for each output
        _, (llk, b, nu) = jax.lax.scan(
            lambda _, args: (_, unpack_single_output(*args)),
            None,
            (rearrange(theta, "o d g -> o (d g)"), g, y_train.T),
        )
        self = cls(theta, g, b, nu, x_train, y_train)
        return self, llk.sum()

    @classmethod
    def fit(
        cls,
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
        l1_penalty: Float[Array, ""],
        autoregressive: bool = True,
        *,
        warmstart: Optional["Self | ADMMState"] = None,
        auto_bounds: Optional[tuple[Float[Array, "d g"], Float[Array, "d g"]]] = None,
        g_range: tuple[float, float] = G_RANGE,
        g_init: float = 0.1,
        max_iterations: int = 300,
        tol: Float[Array, ""] = jnp.array(1e-3),
        adapt_rho: bool = True,
        inner_maxiter: int = 5,
        inner_pgtol: float = 1e-2,
        inner_max_linesearch: int = 5,
        chunk_size: int = 8,
        history_length: int = 40,
        **_,
    ) -> tuple[Self, Float[Array, ""], ADMMState, dict]:
        _, d, group_size = x_train.shape
        _, o = y_train.shape
        design = rearrange(x_train, "n d g -> n (d g)")

        # box constraints for theta and the free nugget parameter
        lower, upper = auto_bounds or hetgpy_auto_bounds(x_train)
        bounds = theta_box(lower, o)
        admm_bounds = rearrange(bounds, "b d (o g) -> b o g d", o=o)
        unbounded = jnp.full((o, 1), jnp.inf)
        solver_lower = jnp.concat([jnp.zeros((o, d * group_size)), -unbounded], axis=-1)
        solver_upper = jnp.concat(
            [repeat(jnp.sqrt(2.0 / lower), "d g -> o (d g)", o=o), unbounded], axis=-1
        )

        # initialize near the long-lengthscale end of the box
        theta_init = repeat(
            jnp.sqrt(2.0 / (0.9 * upper + 0.1 * lower)), "d g -> o g d", o=o
        )
        g_min, g_max = jnp.array(g_range[0]), jnp.array(g_range[1])
        w_init = jnp.full((o,), w_from_nugget(jnp.array(g_init), g_min, g_max))

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
            w = w_from_nugget(warmstart.g, g_min, g_max)
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
        def solve_outputs(
            x0: Float[Array, "o d*g+1"],
            target: Float[Array, "o d*g"],
            rho: Float[Array, ""],
        ) -> Float[Array, "o d*g+1"]:
            def solve_one(args):
                x0_i, target_i, y_i, group_i, mask_i, lower_i, upper_i = args
                solution = lbfgsb_minimise(
                    x_update_loss,
                    x0_i,
                    (lower_i, upper_i),
                    args=((target_i, rho, masked_designs[group_i], y_i, g_min, g_max),),
                    tol=inner_pgtol,
                    max_iterations=inner_maxiter,
                    history_length=history_length,
                    max_linesearch=inner_max_linesearch,
                )
                return solution.x * mask_i

            return jax.lax.map(
                solve_one,
                (
                    x0,
                    target,
                    y_train.T,
                    group_of_output,
                    flat_mask,
                    solver_lower,
                    solver_upper,
                ),
                batch_size=chunk_size,
            )

        # define the x update function for ADMM
        def x_update(state: ADMMState) -> ADMMState:
            theta = rearrange(state.x, "o g d -> o (d g)")
            x0 = jnp.concat([theta, state.aux[..., None]], axis=-1) * flat_mask
            target = rearrange(state.z - state.u, "o g d -> o (d g)")
            solution = solve_outputs(x0, target, state.rho)
            theta = rearrange(solution[:, :-1], "o (d g) -> o g d", g=group_size)
            return state._replace(x=jnp.clip(theta, *admm_bounds), aux=solution[:, -1])

        # run the ADMM solver loop
        state, converged, iterations = glasso_admm.solve(
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
        info = dict(converged=bool(converged), iterations=int(iterations))

        # build the final model and its optimality certificate
        theta = rearrange(state.z, "o g d -> o d g")
        theta = theta if mask is None else theta * mask[..., None]
        self, llk = cls.unpack_parameters(
            theta, state.aux, x_train, y_train, g_min=g_min, g_max=g_max
        )
        certificate = kkt_certificate(
            self.theta,
            state.aux,
            l1_penalty,
            x_train,
            y_train,
            bounds,
            mask=mask,
            g_min=g_min,
            g_max=g_max,
            chunk_size=chunk_size,
        )
        info["certificate"] = {k: np.asarray(v) for k, v in certificate.items()}
        return self, llk, state, info
