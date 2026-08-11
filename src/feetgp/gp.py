from typing import Literal, NamedTuple, Optional
from jaxtyping import Array, Int, Float

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
from einops import rearrange, reduce, repeat, pack, unpack
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
    sqn1 = jnp.sum(theta * x1**2, axis=-1)
    sqn2 = jnp.sum(theta * x2**2, axis=-1)
    d2 = sqn1[:, None] + sqn2[None, :] - 2.0 * (theta * x1) @ x2.T
    d2 = jnp.maximum(d2, 0.0)

    if profile == "rbf":
        return jnp.exp(-0.5 * d2)
    elif profile == "matern52":
        r = jnp.sqrt(5.0 * jnp.where(d2 > 0.0, d2, 1.0))
        matern = (1.0 + r + r**2 / 3.0) * jnp.exp(-r)
        return jnp.where(d2 > 0.0, matern, 1.0 - 5.0 / 6.0 * d2)
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
    theta: Optional[Float[Array, "o d g"]] = None
    nugget: Optional[Float[Array, "o"]] = None

    # profiled out of the likelihood, so the solver never sees these two
    b: Optional[Float[Array, "o"]] = None
    nu: Optional[Float[Array, "o"]] = None

    @eqx.filter_jit
    def predict(
        self,
        x: Float[Array, "... m d g"],
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
    ) -> tuple[Float[Array, "... o m"], Float[Array, "... o m m"]]:
        profile = self.profile

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
            Kox = nu * kernel(profile, theta, x_train, x)
            Koo = nu * (kernel(profile, theta, x_train, x_train))
            Koo = Koo + nugget * jnp.eye(len(y_train))
            Kxx = nu * kernel(profile, theta, x, x)
            return gp_posterior(Kxx, Kox, Koo, y_train, b)

        # vectorize over outputs, scan over batch axes to avoid OOM
        x, batch = pack([x], "* m d g")
        predict = jax.vmap(predict_single, in_axes=(None, 0, 0, 0, 0, None, 1))
        params = (self.theta, self.nugget, self.b, self.nu)
        mean, cov = jax.lax.map(lambda x: predict(x, *params, x_train, y_train), x)
        [mean] = unpack(mean, batch, "* o m")
        [cov] = unpack(cov, batch, "* o m1 m2")
        return mean, cov

    @staticmethod
    def loss(
        theta: Float[Array, "d g"],
        nugget: Float[Array, ""],
        x_train_flat: Float[Array, "n dg"],
        y: Float[Array, "n"],
        *,
        profile: KernelProfile,
    ) -> tuple[Float[Array, ""], tuple[Float[Array, ""], Float[Array, ""]]]:
        """Profiled gp mle objective, with the closed-form trend and variance."""
        theta = rearrange(theta, "d g -> (d g)")
        Koo = kernel(profile, theta, x_train_flat, x_train_flat)
        Koo = Koo + nugget * jnp.eye(len(y))
        llk, b, nu = gp_loglikelihood(Koo, y)
        return -llk, (b, nu)

    @eqx.filter_jit
    def lambda_max(
        self,
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
        *,
        nugget_range: tuple[float, float] = (0.001, 100),
    ) -> Float[Array, ""]:
        """Smallest penalty that kills every group, from the gradient at theta = 0."""
        profile = self.profile
        _, d, g = x_train.shape
        x_train_flat = rearrange(x_train, "n d g -> n (d g)")
        theta_zero = jnp.zeros((d, g), dtype=x_train.dtype)
        log_nugget_range = jnp.log(jnp.asarray(nugget_range, dtype=x_train.dtype))

        def loss(
            theta: Float[Array, "d g"],
            log_nugget: Float[Array, ""],
            y: Float[Array, "n"],
        ) -> Float[Array, ""]:
            nll, _ = GaussianProcess.loss(
                theta, jnp.exp(log_nugget), x_train_flat, y, profile=profile
            )
            return nll

        def gradient_at_zero(y: Float[Array, "n"]) -> Float[Array, "d g"]:
            # the nugget is the only free parameter left when every group is dead
            log_nugget = minimise(
                lambda log_nugget: loss(theta_zero, log_nugget, y),
                x0=jnp.mean(log_nugget_range),
                bounds=(log_nugget_range[0], log_nugget_range[1]),
            ).x
            return jax.grad(loss)(theta_zero, log_nugget, y)

        # theta sits on its lower bound, so only inward components can revive a group
        grad = jnp.minimum(jax.vmap(gradient_at_zero)(y_train.T), 0.0)
        return jnp.max(jnp.sqrt(reduce(grad**2, "o d g -> g", "sum")))

    @eqx.filter_jit
    def fit(
        self,
        x_train: Float[Array, "n d g"],
        y_train: Float[Array, "n o"],
        l1_penalty: Float[Array, ""],
        *,
        warmstart: ADMMState | None = None,
        nugget_range: tuple[float, float] = (0.001, 100),
        max_iterations: int | Int[Array, ""] = jnp.array(300),
        **kwargs,
    ) -> tuple["GaussianProcess", Float[Array, ""], ADMMState, Float[Array, "g"]]:
        profile = self.profile
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
        log_nugget_range = jnp.log(jnp.asarray(nugget_range, dtype=x_train.dtype))
        lower = (theta_lower, log_nugget_range[0])
        upper = (theta_upper, log_nugget_range[1])

        loss = lambda theta, nugget, y: GaussianProcess.loss(
            theta, nugget, x_train_flat, y, profile=profile
        )

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
                    nll, _ = loss(theta, jnp.exp(log_nugget), y)
                    lagrangian = 0.5 * jnp.sum((theta - theta_target) ** 2)
                    return nll + state.rho * lagrangian

                # optimize for a single output with l-bfgs-b
                res = minimise(
                    objective,
                    x0=(theta, log_nugget),
                    bounds=(lower, upper),
                    tol=cosine_schedule(state.iteration / max_iterations),
                )
                return res.x

            # vectorize over outputs, the nugget riding along outside the iterate
            (log_nugget,) = state.aux
            theta, log_nugget = jax.vmap(solve_single)(
                state.x, log_nugget, state.z - state.u, y_train.T
            )
            return state._replace(x=theta, aux=(log_nugget,))

        # run the ADMM solver loop, cold start unless warmstarted
        theta0 = repeat(theta_init, "d g -> o d g", o=o)
        zeros = jnp.zeros_like(theta0)
        state = (
            warmstart.restart()
            if warmstart is not None
            else ADMMState(x=theta0, z=zeros, u=zeros, aux=(jnp.zeros((o,)),))
        )
        state = solve(
            x_update,
            state,
            l1_penalty=l1_penalty,
            max_iterations=max_iterations,
            **kwargs,
        )

        # z is the shrunk iterate, the nugget comes from the aux block
        theta = state.z
        (log_nugget,) = state.aux
        (nll, (b, nu)), grad = jax.vmap(jax.value_and_grad(loss, has_aux=True))(
            theta, jnp.exp(log_nugget), y_train.T
        )

        # check group lasso stationarity of the penalized block alone
        certificate = kkt_certificate(
            theta, grad, l1_penalty, theta_lower, theta_upper
        )
        model = self._replace(theta=theta, nugget=jnp.exp(log_nugget), b=b, nu=nu)
        return model, nll.sum(), state, certificate
