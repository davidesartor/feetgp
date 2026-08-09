from typing import Literal, NamedTuple, Self
from jaxtyping import Array, Int, Float

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
from einops import rearrange, repeat, pack, unpack
from vlse.optim import minimise

from feetgp.glasso_admm import ADMMState, solve, kkt_certificate

KernelProfile = Literal["rbf", "matern52"]


def kernel(
    profile: KernelProfile,
    theta: Float[Array, "d"],
    x1: Float[Array, "n d"],
    x2: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Stationary ARD kernel, distances scaled by the precisions theta."""
    # the exponent is linear in theta, so the squared distance expands
    sqn1 = jnp.sum(theta * x1**2, axis=-1)
    sqn2 = jnp.sum(theta * x2**2, axis=-1)
    d2 = sqn1[:, None] + sqn2[None, :] - 2.0 * (theta * x1) @ x2.T
    d2 = jnp.maximum(d2, 0.0)

    if profile == "rbf":
        return jnp.exp(-0.5 * d2)
    elif profile == "matern52":
        # sqrt is not differentiable at zero, so keep the diagonal off it
        r = jnp.sqrt(5.0 * jnp.where(d2 > 0.0, d2, 1.0))
        r = jnp.where(d2 > 0.0, r, 0.0)
        return (1.0 + r + r**2 / 3.0) * jnp.exp(-r)
    else:
        raise ValueError(f"unknown kernel profile {profile!r}")


def hetgpy_auto_bounds(
    x: Float[Array, "n d"],
    min_cor: float = 0.01,
    max_cor: float = 0.5,
) -> tuple[Float[Array, "d"], Float[Array, "d"], Float[Array, "d"]]:
    # normalize each column to [0, 1] so distances are comparable
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x_span = jnp.where(x_max > x_min, x_max - x_min, 1.0)
    x = (x - x_min) / x_span

    # pairwise squared distances between distinct points
    rows, cols = jnp.tril_indices(len(x), k=-1)
    dists = jnp.sum((x[rows] - x[cols]) ** 2, axis=-1)
    dists = jnp.maximum(dists, 0.0)
    dists = jnp.where(dists > 0, dists, jnp.nan)

    # lengthscale bounds from target rbf correlations at distance quantiles
    lower = -0.5 * jnp.nanquantile(dists, 0.05) / jnp.log(min_cor) * x_span**2
    upper = -0.5 * jnp.nanquantile(dists, 0.95) / jnp.log(max_cor) * x_span**2
    init = 0.9 * upper + 0.1 * lower

    # invert into the precision parametrization
    init, lower, upper = 1.0 / init, 1.0 / upper, 1.0 / lower
    return init, lower, upper


def cosine_schedule(
    t: Float[Array, ""], min: float = 1e-4, max: float = 1e-1
) -> Float[Array, ""]:
    """Anneal from maximum to minimum over t in [0, 1], flat at both ends."""
    t = 0.5 * (1.0 + jnp.cos(jnp.pi * t.clip(0.0, 1.0)))
    return jnp.exp(jnp.log(min) + (jnp.log(max) - jnp.log(min)) * t)


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


class GaussianProcess(NamedTuple):
    profile: KernelProfile

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
            # flatten the group dimension for the kernel computations
            x = rearrange(x, "m d g -> m (d g)")
            theta = rearrange(theta, "d g -> (d g)")
            x_train = rearrange(x_train, "n d g -> n (d g)")

            # compute the kernel matrices
            Kox = nu * kernel(self.profile, theta, x_train, x)
            Koo = nu * (kernel(self.profile, theta, x_train, x_train))
            Koo = Koo + nugget * jnp.eye(len(y_train))
            Kxx = nu * kernel(self.profile, theta, x, x)
            return gp_posterior(Kxx, Kox, Koo, y_train, b)

        # vectorize over outputs, scan over batch axes to avoid OOM
        x, batch = pack([x], "* m d g")
        predict = jax.vmap(predict_single, in_axes=(None, 0, 0, 0, 0, None, 1))
        params = (self.theta, self.nugget, self.b, self.nu)
        mean, cov = jax.lax.map(
            lambda x: predict(x, *params, self.x_train, self.y_train), x
        )
        [mean] = unpack(mean, batch, "* o m")
        [cov] = unpack(cov, batch, "* o m1 m2")
        return mean, cov

    @classmethod
    @eqx.filter_jit
    def fit(
        cls,
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
        l1_penalty: Float[Array, ""],
        *,
        profile: KernelProfile,
        warmstart: ADMMState | None = None,
        nugget_range: Float[Array, "2"] = jnp.array([0.001, 100]),
        max_iterations: int | Int[Array, ""] = jnp.array(300),
        **kwargs,
    ) -> tuple[Self, Float[Array, ""], ADMMState, Float[Array, "g"]]:
        _, d, g = x_train.shape
        _, o = y_train.shape
        x_train_flat = rearrange(x_train, "n d g -> n (d g)")

        # data driven bounds and init for gp mle fit with l-bfsg-b
        theta_init, theta_lower, theta_upper = hetgpy_auto_bounds(x_train_flat)
        theta_lower = jnp.zeros_like(theta_lower)  # allow shrink to zero
        to_groups = lambda v: rearrange(v, "(d g) -> d g", g=g)
        theta_init, theta_lower, theta_upper = map(
            to_groups, (theta_init, theta_lower, theta_upper)
        )
        lower = (theta_lower, jnp.log(nugget_range[0]))
        upper = (theta_upper, jnp.log(nugget_range[1]))

        # negative loglikelihood for gp mle fit
        def negative_loglikelihood(
            theta: Float[Array, "d g"],
            nugget: Float[Array, ""],
            y: Float[Array, "n"],
        ):
            theta = rearrange(theta, "d g -> (d g)")
            Koo = kernel(profile, theta, x_train_flat, x_train_flat)
            Koo = Koo + nugget * jnp.eye(len(y))
            llk, b, nu = gp_loglikelihood(Koo, y)
            return -llk, (b, nu)

        # x update step, one bound constrained gp mle per output
        def x_update(state: ADMMState) -> ADMMState:
            def solve_single(
                theta: Float[Array, "d g"],
                log_nugget: Float[Array, ""],
                theta_target: Float[Array, "d g"],
                y: Float[Array, "n"],
            ) -> tuple[Float[Array, "d g"], Float[Array, ""]]:
                # augmented Lagrangian objective for the x update step
                def objective(
                    theta_and_log_nugget: tuple[Float[Array, "d g"], Float[Array, ""]],
                ):
                    theta, log_nugget = theta_and_log_nugget
                    loss, _ = negative_loglikelihood(theta, jnp.exp(log_nugget), y)
                    lagrangian = 0.5 * jnp.sum((theta - theta_target) ** 2)
                    return loss + state.rho * lagrangian

                # optimize for a single output with l-bfgs-b
                res = minimise(
                    objective,
                    x0=(theta, log_nugget),
                    bounds=(lower, upper),
                    tol=cosine_schedule(state.iteration / max_iterations),
                )
                theta, log_nugget = res.x
                return theta, log_nugget

            # vectorize over outputs
            theta, log_nugget = jax.vmap(solve_single)(
                state.x, state.aux[0], state.z - state.u, y_train.T
            )
            return state._replace(x=theta, aux=(log_nugget,))

        # run the ADMM solver loop, cold start unless warmstarted
        theta0 = repeat(theta_init, "d g -> o d g", o=o)
        state = (
            warmstart.restart()
            if warmstart is not None
            else ADMMState(
                x=theta0,
                z=jnp.zeros_like(theta0),
                u=jnp.zeros_like(theta0),
                aux=(jnp.zeros((o,)),),
            )
        )
        state = solve(
            x_update,
            state,
            l1_penalty=l1_penalty,
            max_iterations=max_iterations,
            **kwargs,
        )

        # build the final model
        theta = state.z
        nugget = jnp.exp(state.aux[0])
        (nll, (b, nu)), grad = jax.vmap(
            jax.value_and_grad(negative_loglikelihood, has_aux=True)
        )(theta, nugget, y_train.T)

        # check group lasso stationarity, nugget is unpenalized so plain gradient
        certificate = kkt_certificate(theta, grad, l1_penalty, theta_lower, theta_upper)

        model = cls(profile, theta, nugget, b, nu, x_train, y_train)
        return model, nll.sum(), state, certificate
