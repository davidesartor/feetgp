from typing import NamedTuple, Optional, Self
from jaxtyping import Array, Float, Scalar, Bool

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx

from einops import rearrange
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


@jax.jit
def kernel(
    theta: Float[Array, "d"],
    xs1: Float[Array, "n d"],
    xs2: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    def k(x1, x2):
        return jnp.exp(-0.5 * jnp.sum(theta**2 * (x1 - x2) ** 2))

    return jax.vmap(jax.vmap(k, (None, 0)), (0, None))(xs1, xs2)


def hetgpy_auto_bounds(
    x: Float[Array, "n d"],
    min_cor: float = 0.01,
    max_cor: float = 0.5,
) -> tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d"]]:
    # rescale each input dimension to [0, 1] (constant columns are left untouched)
    x = np.asarray(x) # type: ignore
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x_span = np.where(x_max > x_min, x_max - x_min, 1.0)
    x = (x - x_min) / x_span

    # pairwise squared distances, keeping only the strictly-lower nonzero pairs
    dists = cdist(x, x, metric="sqeuclidean")
    dists = dists[np.tril(dists, k=-1) > 0]

    # hetgpy heuristic: pick lengthscales so the given distance quantiles map to
    # the target correlations min_cor (short) and max_cor (long)
    lower = -np.quantile(dists, 0.05) / np.log(min_cor) * x_span**2
    upper = -np.quantile(dists, 0.95) / np.log(max_cor) * x_span**2
    return lower, upper


@jax.jit
def gp_posterior(
    Kxx: Float[Array, "m m"],
    Kox: Float[Array, "n m"],
    Koo: Float[Array, "n n"],
    observed_ys: Float[Array, "n"],
    b: Scalar,
) -> tuple[Float[Array, "m"], Float[Array, "m m"]]:
    # posterior mean and covariance
    gain = jnp.linalg.solve(Koo, Kox).T
    mean = b + gain @ (observed_ys - b)
    cov = Kxx - gain @ Kox

    # Add correction based on the trend estimation correlation
    Kbx = jnp.ones((1, len(observed_ys))) @ gain.T
    cov = cov + (1 - Kbx).T @ (1 - Kbx) / jnp.linalg.inv(Koo).sum()
    return mean, cov


@jax.jit
def loglikelihood(
    Koo: Float[Array, "n n"],
    observed_ys: Float[Array, "n"],
) -> tuple[Scalar, Scalar, Scalar]:
    # cholesky of K and compute logdet
    K_sqrt, is_lower = jsp.linalg.cho_factor(Koo)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    # compute Ki_1=(K^-1 @ 1) and Ki_y=(K^-1 @ y)
    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([jnp.ones_like(observed_ys), observed_ys], 1),
    ).T

    # compute optimal trend b and scale nu
    b = (Ki_1 * observed_ys).sum() / Ki_1.sum()
    nu = jnp.dot((observed_ys - b) / len(observed_ys), (Ki_y - Ki_1 * b))

    # likelihood when marginalizing over trend and variance
    loglik = -0.5 * (len(observed_ys) * jnp.log(nu) + logdetK)
    return (loglik, b, nu)


@eqx.filter_jit
@jax.value_and_grad
def admm_theta_update_loss(
    theta: Float[Array, "d*g"],
    g: Scalar,
    z: Float[Array, "d*g+1"],
    u: Float[Array, "d*g+1"],
    rho: Scalar,
    x_train: Float[Array, "n d*g"],
    y_train: Float[Array, "n"],
):
    # theta-only step of the x-update: optimize the lengthscales with the nugget g
    # held fixed. Gradient (via value_and_grad) is w.r.t. theta.
    Koo = kernel(theta, x_train, x_train)
    Koo = Koo + g * jnp.eye(len(y_train))
    loglik, _, _ = loglikelihood(Koo, y_train)
    target_theta = (z - u)[:-1]
    lagrangian = 0.5 * rho * jnp.sum((theta - target_theta) ** 2)
    return -loglik + lagrangian


@eqx.filter_jit
def admm_g_update_loss(
    g: Scalar,
    theta: Float[Array, "d*g"],
    x_train: Float[Array, "n d*g"],
    y_train: Float[Array, "n"],
) -> Scalar:
    # g-only step of the x-update: optimize the nugget with the lengthscales theta
    # held fixed. g carries no group-lasso penalty, so this is just -loglik and is a
    # 1D bounded problem. Used for the grid probe (value only).
    Koo = kernel(theta, x_train, x_train)
    Koo = Koo + g * jnp.eye(len(y_train))
    loglik, _, _ = loglikelihood(Koo, y_train)
    return -loglik


@eqx.filter_jit
@jax.value_and_grad
def admm_g_update_loss_and_grad(
    g: Float[Array, "1"],
    theta: Float[Array, "d*g"],
    x_train: Float[Array, "n d*g"],
    y_train: Float[Array, "n"],
) -> Scalar:
    # same objective as admm_g_update_loss, but returns (value, grad) for the
    # gradient-based local polish started from the best grid point.
    Koo = kernel(theta, x_train, x_train)
    Koo = Koo + g[0] * jnp.eye(len(y_train))
    loglik, _, _ = loglikelihood(Koo, y_train)
    return -loglik


class GLASSOADMMState(NamedTuple):
    x: Float[Array, "o d*g+1"]
    z: Float[Array, "o d*g+1"]
    u: Float[Array, "o d*g+1"]
    rho: Scalar
    l1: Scalar
    group_size: int
    bounds: Float[Array, "2 o d*g+1"]

    @classmethod
    def initialize(
        cls,
        bounds: Float[Array, "2 o d*g+1"],
        x0: Float[Array, "o d*g+1"],
        penalty: Scalar,
        group_size: int,
    ) -> Self:
        return cls(
            x=x0,
            z=x0,
            u=jnp.zeros_like(x0),
            rho=jnp.array(1.0),
            l1=penalty,
            group_size=group_size,
            bounds=bounds,
        )

    def x_update(
        self,
        x_train: Float[Array, "n d*g"],
        y_train: Float[Array, "n o"],
        autoregressive: bool,
        maxiter: int = 1000,
        n_jobs: int = -1,
        g_grid_size: int = 15,
    ) -> Self:
        def solve_ith_output(i):
            x_train_i = np.array(x_train)
            if autoregressive:
                # zero out the group corresponding to the i-th output in the training data
                start = (i // self.group_size) * self.group_size
                x_train_i[:, start : start + self.group_size] = 0.0

            y_train_i = y_train[:, i]
            theta0 = self.x[i][:-1]
            g0 = self.x[i][-1]

            # ---- Step 1: optimize theta (lengthscales), g fixed -------------------
            theta_bounds = list(
                zip(self.bounds[0, i, :-1], self.bounds[1, i, :-1])
            )
            new_theta = sp.optimize.minimize(
                fun=admm_theta_update_loss,
                x0=theta0,
                args=(g0, self.z[i], self.u[i], self.rho, x_train_i, y_train_i),
                jac=True,
                method="L-BFGS-B",
                bounds=theta_bounds,
                options=dict(maxiter=maxiter, ftol=EPS, gtol=0.0),
            ).x
            new_theta = jnp.asarray(new_theta)

            # ---- Step 2: optimize g (nugget), theta fixed -------------------------
            # grid-probe the (log-spaced) range for the best g, then run a
            # gradient-based local polish over the full range from that point.
            g_lo = float(self.bounds[0, i, -1])
            g_hi = float(self.bounds[1, i, -1])
            g_grid = jnp.geomspace(g_lo, g_hi, g_grid_size)
            grid_losses = jax.vmap(
                lambda g: admm_g_update_loss(
                    g, new_theta, jnp.asarray(x_train_i), y_train_i
                )
            )(g_grid)
            g_best = float(g_grid[int(jnp.argmin(grid_losses))])
            new_g = sp.optimize.minimize(
                fun=admm_g_update_loss_and_grad,
                x0=np.array([g_best]),
                args=(new_theta, x_train_i, y_train_i),
                jac=True,
                method="L-BFGS-B",
                bounds=[(g_lo, g_hi)],
                options=dict(maxiter=maxiter, ftol=EPS, gtol=0.0),
            ).x[0]

            return np.concatenate([np.asarray(new_theta), [new_g]])

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(solve_ith_output)(i) for i in range(len(self.x))
        )
        new_x = np.stack([np.array(r) for r in results])
        if autoregressive:
            for i in range(len(self.x)):
                start = (i // self.group_size) * self.group_size
                new_x[i, start : start + self.group_size] = 0.0
        return self._replace(x=jnp.array(new_x))

    @eqx.filter_jit
    def z_and_u_update(self) -> Self:
        # split theta and g
        theta = (self.x + self.u)[:, :-1]
        g = (self.x + self.u)[:, -1:]

        # apply group lasso proximal operator to theta
        theta = rearrange(theta, "o (d g) -> (o g) d", g=self.group_size)
        group_norm = jnp.linalg.norm(theta, axis=0)
        prox = jnp.maximum(0, 1 - (self.l1 / (self.rho * group_norm).clip(min=EPS)))
        theta = prox * theta
        theta = rearrange(theta, "(o g) d -> o (d g)", g=self.group_size)

        # recompose z and clip to bounds (no penalty for g)
        z = jnp.concatenate([theta, g], axis=-1)
        z = jnp.clip(z, *self.bounds)  # in case bounds do not include 0

        u = self.u + self.x - z
        return self._replace(z=z, u=u)

    @eqx.filter_jit
    def check_residuals(
        self, prev: Self, tol: Scalar, adapt_rho: bool = False
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
        if not covariance:
            mean, cov = jax.vmap(lambda x: self.predict(x, covariance=True))(
                xs[:, None, :]
            )
            return mean.squeeze(-1).T

        def predict_single_output(
            theta: Float[Array, "d*g"],
            g: Scalar,
            b: Scalar,
            nu: Scalar,
            y_train: Float[Array, "n"],
        ) -> tuple[Float[Array, "m"], Float[Array, "m m"]]:
            Kxx = nu * kernel(theta, xs, xs)
            Kox = nu * kernel(theta, self.x_train, xs)
            Koo = nu * (
                kernel(theta, self.x_train, self.x_train)
                + g * jnp.eye(len(self.y_train))
            )
            mean, cov = gp_posterior(Kxx, Kox, Koo, y_train, b)
            return mean, cov

        # use scan instead of vmap to avoid OOM
        _, (mean, cov) = jax.lax.scan(
            lambda _, args: (_, predict_single_output(*args)),
            None,
            (self.theta, self.g, self.b, self.nu, self.y_train.T),
        )
        return mean, cov

    @classmethod
    @eqx.filter_jit
    def unpack_parameters(
        cls,
        admm_x: Float[Array, "o d*g+1"],
        x_train: Float[Array, "n d*g"],
        y_train: Float[Array, "n o"],
    ) -> tuple[Self, Scalar]:
        # extract the optimal parameters and infer the rest
        theta, g = admm_x[..., :-1], admm_x[..., -1]

        # use scan instead of vmap over theta to avoid OOM
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
        warmstart: Optional[GLASSOADMMState] = None,
        g_range: tuple[float, float] = (EPS, 1.0),
        g_init: float = 0.1,
        max_iterations: int = 100,
        tol: Scalar = jnp.array(1e-4),
        adapt_rho: bool = True,
        n_jobs: int = -1,
    ) -> tuple[Self, GLASSOADMMState, Scalar]:
        _, d_times_g = x_train.shape
        _, o = y_train.shape
        assert d_times_g % group_size == 0

        # initialize ADMM state
        if warmstart is not None:
            state = warmstart._replace(
                rho=jnp.ones_like(warmstart.rho),
                l1=l1_penalty,
                u=jnp.zeros_like(warmstart.u),
            )
        else:
            # data-driven (hetgpy) lengthscale bounds; hetgpy's Gaussian kernel is
            # exp(-d^2 / l), ours is exp(-0.5 * theta^2 * d^2), so theta = sqrt(2 / l).
            # short lengthscale (lower) -> largest theta; the group lasso can drive
            # theta to 0, so its lower bound is 0 everywhere.
            lower, upper = hetgpy_auto_bounds(x_train)
            theta_max = jnp.broadcast_to(jnp.sqrt(2.0 / lower), (o, d_times_g))
            theta_min = jnp.zeros((o, d_times_g))
            theta_bounds = jnp.stack([theta_min, theta_max], axis=0)
            theta_init = jnp.broadcast_to(
                jnp.sqrt(2.0 / (0.9 * upper + 0.1 * lower)), (o, d_times_g)
            )

            # g is optimized within its bounds, initialized at g_init
            g_min = jnp.ones((o,)) * g_range[0]
            g_max = jnp.ones((o,)) * g_range[1]
            g_bounds = jnp.stack([g_min, g_max], axis=0)
            g_init_vec = jnp.ones((o,)) * g_init

            bounds = jnp.concat([theta_bounds, g_bounds[..., None]], axis=-1)
            x0 = jnp.concat([theta_init, g_init_vec[..., None]], axis=-1)
            state = GLASSOADMMState.initialize(bounds, x0, l1_penalty, group_size)

        for iter in (pbar := tqdm(range(max_iterations), desc="ADMM")):
            new_state = state.x_update(x_train, y_train, autoregressive, n_jobs=n_jobs)
            new_state = new_state.z_and_u_update()
            state, primal_ok, dual_ok = new_state.check_residuals(state, tol, adapt_rho)
            pbar.set_postfix({"rho": state.rho.item()})
            if primal_ok.all() and dual_ok.all():
                print(f"ADMM converged in {iter + 1} iterations.")
                break
        else:
            print("ADMM did not converge within the maximum number of iterations.")

        self, llk = cls.unpack_parameters(state.x, x_train, y_train)
        return self, state, llk
