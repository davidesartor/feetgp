from typing import NamedTuple, Optional, Self
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx

from einops import rearrange
import numpy as np
from scipy.spatial.distance import cdist
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
    x: Float[Array, "n d"],
    min_cor: float = 0.01,
    max_cor: float = 0.5,
) -> tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d"]]:
    x = np.asarray(x)
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x_span = np.where(x_max > x_min, x_max - x_min, 1.0)
    x = (x - x_min) / x_span

    dists = cdist(x, x, metric="sqeuclidean")
    dists = dists[np.tril(dists, k=-1) > 0]

    lower = -np.quantile(dists, 0.05) / np.log(min_cor) * x_span**2
    upper = -np.quantile(dists, 0.95) / np.log(max_cor) * x_span**2
    return lower, upper


@eqx.filter_jit
def gp_posterior(
    Kox: Float[Array, "n m"],
    Koo: Float[Array, "n n"],
    observed_ys: Float[Array, "n"],
    b: Scalar,
    Kxx: Optional[Float[Array, "m m"]] = None,
) -> Float[Array, "m"] | tuple[Float[Array, "m"], Float[Array, "m m"]]:
    chol = jsp.linalg.cho_factor(Koo)
    gain = jsp.linalg.cho_solve(chol, Kox)
    mean = b + gain.T @ (observed_ys - b)
    if Kxx is None:
        return mean

    cov = Kxx - Kox.T @ gain

    Kbx = jnp.ones((1, len(observed_ys))) @ gain
    Ki_1 = jsp.linalg.cho_solve(chol, jnp.ones_like(observed_ys))
    cov = cov + (1 - Kbx).T @ (1 - Kbx) / Ki_1.sum()
    return mean, cov


@jax.jit
def loglikelihood(
    Koo: Float[Array, "n n"],
    observed_ys: Float[Array, "n"],
) -> tuple[Scalar, Scalar, Scalar]:
    K_sqrt, is_lower = jsp.linalg.cho_factor(Koo)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([jnp.ones_like(observed_ys), observed_ys], 1),
    ).T

    b = (Ki_1 * observed_ys).sum() / Ki_1.sum()
    nu = jnp.dot((observed_ys - b) / len(observed_ys), (Ki_y - Ki_1 * b))

    loglik = -0.5 * (len(observed_ys) * jnp.log(nu) + logdetK)
    return (loglik, b, nu)


def admm_x_update_loss(
    x: Float[Array, "d*g+1"],
    args: tuple,
) -> Scalar:
    target_theta, rho, x_train, y_train, g_min, g_max = args
    theta, w = x[:-1], x[-1]
    Koo = kernel(jax.nn.relu(theta), x_train, x_train)
    Koo = Koo + nugget_from_w(w, g_min, g_max) * jnp.eye(len(y_train))
    loglik, _, _ = loglikelihood(Koo, y_train)
    lagrangian = 0.5 * rho * jnp.sum((theta - target_theta) ** 2)
    return -loglik + lagrangian


def nugget_from_w(w: Scalar, g_min: Scalar, g_max: Scalar) -> Scalar:
    return g_min + (g_max - g_min) * jax.nn.sigmoid(w)


def w_from_nugget(g: Scalar, g_min: Scalar, g_max: Scalar) -> Scalar:
    fraction = jnp.clip((g - g_min) / (g_max - g_min), 1e-12, 1 - 1e-12)
    return jsp.special.logit(fraction)


def autoregressive_mask(
    o: int, d_times_g: int, group_size: int
) -> Float[Array, "o d*g+1"]:
    output = jnp.arange(o)[:, None] // group_size
    column = jnp.arange(d_times_g)[None, :] // group_size
    return jnp.concatenate(
        [(output != column).astype(float), jnp.ones((o, 1))], axis=-1
    )


def theta_box(
    lower: Float[np.ndarray, "d*g"], o: int, d_times_g: int, group_size: int
) -> Float[Array, "2 d o*g"]:
    theta_max = glasso_admm.to_groups(
        jnp.broadcast_to(jnp.sqrt(2.0 / lower), (o, d_times_g)), group_size
    )
    return jnp.stack([jnp.zeros_like(theta_max), theta_max], axis=0)


@eqx.filter_jit
def penalized_objective(
    theta: Float[Array, "o d*g"],
    g: Float[Array, "o"],
    l1: Scalar,
    x_train: Float[Array, "n d*g"],
    y_train: Float[Array, "n o"],
    group_size: int,
) -> Scalar:
    n = len(x_train)

    def negloglik_single_output(
        theta_i: Float[Array, "d*g"], g_i: Scalar, y_i: Float[Array, "n"]
    ) -> Scalar:
        Koo = kernel(theta_i, x_train, x_train) + g_i * jnp.eye(n)
        loglik, _, _ = loglikelihood(Koo, y_i)
        return -loglik

    _, negloglik = jax.lax.scan(
        lambda _, inputs: (_, negloglik_single_output(*inputs)),
        None,
        (theta, g, y_train.T),
    )
    norms = jnp.linalg.norm(glasso_admm.to_groups(theta, group_size), axis=-1)
    return negloglik.sum() + l1 * norms.sum()


@eqx.filter_jit
def kkt_certificate(
    theta: Float[Array, "o d*g"],
    w: Float[Array, "o"],
    l1: Scalar,
    x_train: Float[Array, "n d*g"],
    y_train: Float[Array, "n o"],
    group_size: int,
    bounds: Float[Array, "2 d o*g"],
    mask: Optional[Float[Array, "o d*g+1"]] = None,
    g_min: Scalar = jnp.array(1e-4),
    g_max: Scalar = jnp.array(100.0),
    chunk_size: int = 8,
) -> dict:
    n = len(x_train)

    def negloglik_single_output(
        theta_i: Float[Array, "d*g"], w_i: Scalar, y_i: Float[Array, "n"]
    ) -> Scalar:
        Koo = kernel(theta_i, x_train, x_train) + nugget_from_w(
            w_i, g_min, g_max
        ) * jnp.eye(n)
        loglik, _, _ = loglikelihood(Koo, y_i)
        return -loglik

    grad_theta, grad_w = jax.lax.map(
        lambda inputs: jax.grad(negloglik_single_output, argnums=(0, 1))(*inputs),
        (theta, w, y_train.T),
        batch_size=chunk_size,
    )
    if mask is not None:
        grad_theta = grad_theta * mask[:, :-1]

    grad_groups = glasso_admm.to_groups(grad_theta, group_size)
    theta_groups = glasso_admm.to_groups(theta, group_size)
    norms = jnp.linalg.norm(theta_groups, axis=-1)
    live = norms > 0.0

    direction = theta_groups / jnp.where(live, norms, 1.0)[:, None]
    stationarity = grad_groups + l1 * direction
    lower, upper = bounds
    stationarity = jnp.where(
        theta_groups <= lower, jnp.minimum(stationarity, 0.0), stationarity
    )
    stationarity = jnp.where(
        theta_groups >= upper, jnp.maximum(stationarity, 0.0), stationarity
    )
    live_kkt = jnp.linalg.norm(stationarity, axis=-1)
    dead_slack = l1 - jnp.linalg.norm(grad_groups, axis=-1)
    return dict(
        max_live_kkt=jnp.max(jnp.where(live, live_kkt, 0.0)),
        live_kkt=jnp.where(live, live_kkt, jnp.nan),
        dead_slack=jnp.where(live, jnp.nan, dead_slack),
        nugget_grad=jnp.max(jnp.abs(grad_w)),
    )


class GLASSOADMMState(NamedTuple):

    x: Float[Array, "o d*g+1"]
    z: Float[Array, "o d*g+1"]
    u: Float[Array, "o d*g+1"]
    rho: Scalar
    l1: Scalar
    group_size: int
    bounds: Optional[Float[Array, "2 o d*g+1"]] = None
    g_min: Scalar = jnp.array(1e-4)
    g_max: Scalar = jnp.array(1.0)


def admm_state_from_legacy(legacy: GLASSOADMMState) -> ADMMState:
    x, z, u = (
        glasso_admm.to_groups(v[:, :-1], legacy.group_size)
        for v in (legacy.x, legacy.z, legacy.u)
    )
    return ADMMState(x=x, z=z, u=u, rho=legacy.rho, aux=legacy.x[:, -1])


def admm_state_from_pickle(results: dict) -> ADMMState:
    state = results["admm_state"]
    if isinstance(state, GLASSOADMMState):
        return admm_state_from_legacy(state)
    return state


@eqx.filter_jit
def x_update_solve(
    x0: Float[Array, "o d*g+1"],
    target_theta: Float[Array, "o d*g"],
    rho: Scalar,
    masked_designs: Float[Array, "n_groups n d*g"],
    y_train: Float[Array, "n o"],
    group_size: int,
    bounds: tuple[Float[Array, "o d*g+1"], Float[Array, "o d*g+1"]],
    mask: Optional[Float[Array, "o d*g+1"]] = None,
    g_min: Scalar = jnp.array(1e-4),
    g_max: Scalar = jnp.array(1.0),
    maxiter: int = 5,
    chunk_size: int = 8,
    history_length: int = 40,
    tol: float = 1e-2,
    max_linesearch: int = 5,
) -> Float[Array, "o d*g+1"]:
    x0 = x0 if mask is None else x0 * mask
    lower, upper = bounds
    group_of_output = jnp.arange(len(x0)) // group_size

    def solve_one_output(args) -> Float[Array, "d*g+1"]:
        x0_i, target_i, y_i, group_i, mask_i, lower_i, upper_i = args
        solution = lbfgsb_minimise(
            admm_x_update_loss,
            x0_i,
            (lower_i, upper_i),
            args=((target_i, rho, masked_designs[group_i], y_i, g_min, g_max),),
            tol=tol,
            max_iterations=maxiter,
            history_length=history_length,
            max_linesearch=max_linesearch,
        )
        return solution.x * mask_i

    return jax.lax.map(
        solve_one_output,
        (
            x0,
            target_theta,
            y_train.T,
            group_of_output,
            jnp.ones_like(x0) if mask is None else mask,
            lower,
            upper,
        ),
        batch_size=chunk_size,
    )


class GroupLassoGaussianProcess(NamedTuple):
    theta: Float[Array, "o d*g"]
    g: Float[Array, "o"]
    b: Float[Array, "o"]
    nu: Float[Array, "o"]

    x_train: Float[Array, "n d*g"]
    y_train: Float[Array, "n o"]

    @eqx.filter_jit
    def predict(
        self, xs: Float[Array, "m d*g"], covariance: bool = False
    ) -> Float[Array, "o m"] | tuple[Float[Array, "o m"], Float[Array, "o m m"]]:
        def predict_single_output(
            theta: Float[Array, "d*g"],
            g: Scalar,
            b: Scalar,
            nu: Scalar,
            y_train: Float[Array, "n"],
        ) -> Float[Array, "m"] | tuple[Float[Array, "m"], Float[Array, "m m"]]:
            Kox = nu * kernel(theta, self.x_train, xs)
            Koo = nu * (
                kernel(theta, self.x_train, self.x_train)
                + g * jnp.eye(len(self.y_train))
            )
            Kxx = nu * kernel(theta, xs, xs) if covariance else None
            return gp_posterior(Kox, Koo, y_train, b, Kxx)

        _, posterior = jax.lax.scan(
            lambda _, args: (_, predict_single_output(*args)),
            None,
            (self.theta, self.g, self.b, self.nu, self.y_train.T),
        )
        return posterior

    @classmethod
    @eqx.filter_jit
    def unpack_parameters(
        cls,
        admm_theta: Float[Array, "o d*g"],
        admm_w: Float[Array, "o"],
        x_train: Float[Array, "n d*g"],
        y_train: Float[Array, "n o"],
        g_min: Scalar = jnp.array(1e-4),
        g_max: Scalar = jnp.array(1.0),
    ) -> tuple[Self, Scalar]:
        theta = jnp.abs(admm_theta)
        g = nugget_from_w(admm_w, g_min, g_max)

        def unpack_single_output(
            theta: Float[Array, "d*g"],
            g: Scalar,
            y_train: Float[Array, "n"],
        ) -> tuple[Scalar, Scalar, Scalar]:
            Koo = kernel(theta, x_train, x_train) + g * jnp.eye(len(y_train))
            return loglikelihood(Koo, y_train)

        _, (llk, b, nu) = jax.lax.scan(
            lambda _, args: (_, unpack_single_output(*args)),
            None,
            (theta, g, y_train.T),
        )
        self = cls(theta, g, b, nu, x_train, y_train)
        return self, llk.sum()

    @classmethod
    def fit(
        cls,
        x_train: Float[Array, "n d*g"],
        y_train: Float[Array, "n o"],
        l1_penalty: Scalar,
        group_size: int,
        autoregressive: bool = True,
        *,
        warmstart: Optional["Self | ADMMState"] = None,
        auto_bounds: Optional[
            tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d"]]
        ] = None,
        g_range: tuple[float, float] = G_RANGE,
        g_init: float = 0.1,
        max_iterations: int = 300,
        tol: Scalar = jnp.array(1e-3),
        adapt_rho: bool = True,
        adapt_rho_iters: Optional[int] = None,
        inner_maxiter: int = 5,
        inner_maxiter_init: int = 1,
        inner_pgtol: float = 1e-2,
        inner_max_linesearch: int = 5,
        chunk_size: int = 8,
        history_length: int = 40,
        log_every: int = 0,
    ) -> tuple[Self, Scalar, ADMMState, dict]:
        _, d_times_g = x_train.shape
        _, o = y_train.shape
        assert d_times_g % group_size == 0

        lower, upper = auto_bounds or hetgpy_auto_bounds(x_train)
        bounds = theta_box(lower, o, d_times_g, group_size)
        unbounded = jnp.full((o, 1), jnp.inf)
        solver_bounds = (
            jnp.concat([glasso_admm.to_outputs(bounds[0], group_size), -unbounded], axis=-1),
            jnp.concat([glasso_admm.to_outputs(bounds[1], group_size), unbounded], axis=-1),
        )
        theta_init = glasso_admm.to_groups(
            jnp.broadcast_to(
                jnp.sqrt(2.0 / (0.9 * upper + 0.1 * lower)), (o, d_times_g)
            ),
            group_size,
        )
        g_min, g_max = jnp.array(g_range[0]), jnp.array(g_range[1])
        w_init = jnp.full((o,), w_from_nugget(jnp.array(g_init), g_min, g_max))
        state = ADMMState.initialize(theta_init, aux=w_init)

        if isinstance(warmstart, ADMMState):
            state = warmstart
        elif warmstart is not None:
            theta = glasso_admm.to_groups(warmstart.theta, group_size)
            w = w_from_nugget(warmstart.g, g_min, g_max)
            state = state._replace(x=theta, z=theta, aux=w)

        n_groups = d_times_g // group_size
        if autoregressive:
            group_columns = jnp.arange(d_times_g)[None, :] // group_size
            keep = (jnp.arange(n_groups)[:, None] != group_columns).astype(
                x_train.dtype
            )
            masked_designs = x_train[None, :, :] * keep[:, None, :]
            mask = autoregressive_mask(o, d_times_g, group_size)
        else:
            masked_designs = jnp.broadcast_to(x_train, (n_groups, *x_train.shape))
            mask = None

        def x_update(state: ADMMState, iteration: int) -> tuple[ADMMState, bool]:
            maxiter = min(inner_maxiter, inner_maxiter_init * 2**iteration)
            theta = glasso_admm.to_outputs(state.x, group_size)
            x0 = jnp.concat([theta, state.aux[..., None]], axis=-1)
            target = glasso_admm.to_outputs(state.z - state.u, group_size)
            solution = x_update_solve(
                x0,
                target,
                state.rho,
                masked_designs,
                y_train,
                group_size,
                solver_bounds,
                mask=mask,
                g_min=g_min,
                g_max=g_max,
                maxiter=maxiter,
                chunk_size=chunk_size,
                history_length=history_length,
                tol=inner_pgtol,
                max_linesearch=inner_max_linesearch,
            )
            theta = glasso_admm.to_groups(solution[:, :-1], group_size)
            state = state._replace(x=jnp.clip(theta, *bounds), aux=solution[:, -1])
            return state, maxiter == inner_maxiter

        state, info = glasso_admm.solve(
            x_update,
            state,
            l1_penalty,
            max_iterations=max_iterations,
            tol=tol,
            bounds=bounds,
            adapt_rho=adapt_rho,
            adapt_rho_iters=adapt_rho_iters,
            log_every=log_every,
        )

        theta = glasso_admm.to_outputs(state.z, group_size)
        theta = theta if mask is None else theta * mask[:, :-1]
        self, llk = cls.unpack_parameters(
            theta, state.aux, x_train, y_train, g_min=g_min, g_max=g_max
        )

        certificate = kkt_certificate(
            self.theta,
            state.aux,
            l1_penalty,
            x_train,
            y_train,
            group_size,
            bounds,
            mask=mask,
            g_min=g_min,
            g_max=g_max,
            chunk_size=chunk_size,
        )
        info["certificate"] = {k: np.asarray(v) for k, v in certificate.items()}
        return self, llk, state, info
