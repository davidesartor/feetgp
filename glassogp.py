from typing import NamedTuple, Optional, Self
from cycler import K
from jaxtyping import Array, Float, Scalar, Bool

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx

from einops import rearrange
import scipy
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


@jax.jit
def kernel(
    theta: Float[Array, "d"],
    xs1: Float[Array, "n d"],
    xs2: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    d2 = jnp.sum(theta * (xs1[:, None, :] - xs2[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-0.5 * d2)


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


class GLASSOADMMState(NamedTuple):
    x: Float[Array, "o d*g+1"]
    z: Float[Array, "o d*g+1"]
    u: Float[Array, "o d*g+1"]
    rho: Float[Array, "d"]
    l1: Scalar
    group_size: int
    bounds: Float[Array, "2 o d*g+1"]

    @classmethod
    def initialize(
        cls,
        bounds: Float[Array, "2 o d*g+1"],
        penalty: Scalar,
        group_size: int,
    ) -> Self:
        d = (bounds.shape[-1] - 1) // group_size
        vmin, vmax = bounds
        x0 = 0.1 * vmax + 0.9 * vmin
        return cls(
            x=x0,
            z=x0,
            u=jnp.zeros_like(x0),
            rho=penalty * jnp.ones((d,)),
            l1=penalty,
            group_size=group_size,
            bounds=bounds,
        )

    @staticmethod
    @jax.jit
    @jax.value_and_grad
    def admm_x_update_loss(
        x: Float[Array, "d*g+1"],
        z: Float[Array, "d*g+1"],
        u: Float[Array, "d*g+1"],
        rho: Float[Array, "d"],
        x_train: Float[Array, "n d*g"],
        y_train: Float[Array, "n"],
        mask: Float[Array, "d*g"],
    ):
        theta, g = x[:-1], x[-1]

        # used a masked version of lenghtscales for likelihood
        Koo = kernel(theta * mask, x_train, x_train)
        Koo = Koo + g * jnp.eye(len(y_train))
        loglik, _, _ = loglikelihood(Koo, y_train)

        # use different rho for each group, no penalty for g
        target_theta = (z - u)[:-1]
        squared_error = rearrange(
            (theta - target_theta) ** 2, "(d g) -> g d", d=len(rho)
        )
        lagrangian = jnp.sum(0.5 * rho * squared_error)

        return -loglik + lagrangian

    def x_update(
        self,
        x_train: Float[Array, "n d*g"],
        y_train: Float[Array, "n o"],
        autoregressive: bool,
        maxiter: int = 10,
    ) -> Self:
        new_x = []
        for i, (x, z, u, y, bmin, bmax) in enumerate(
            zip(self.x, self.z, self.u, y_train.T, *self.bounds)
        ):
            # zero out self referential entries if autoregressive
            theta_mask = jnp.ones(x_train.shape[-1])
            if autoregressive:
                start = (i // self.group_size) * self.group_size
                theta_mask = theta_mask.at[start : start + self.group_size].set(0.0)

            # optimize with L-BFGS-B
            res = scipy.optimize.minimize(
                fun=self.admm_x_update_loss,
                x0=x,
                args=(z, u, self.rho, x_train, y, theta_mask),
                jac=True,
                method="L-BFGS-B",
                bounds=[(a, b) for a, b in zip(bmin, bmax)],
                options=dict(maxiter=maxiter, ftol=EPS, gtol=0),
            )

            new_x.append(res.x)
        new_x = jnp.stack(new_x, axis=0)
        return self._replace(x=new_x)

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
        self, prev: Self, tol: Scalar
    ) -> tuple[Self, Bool[Array, "d"], Bool[Array, "d"]]:
        def grouped_norm(x: Float[Array, "o d*g+1"]) -> Float[Array, "d"]:
            x = rearrange(x[..., :-1], "o (d g) -> d (o g)", g=self.group_size)
            return jnp.linalg.norm(x, axis=-1)

        # check primal
        primal_residual = grouped_norm(self.x - self.z)
        primal_target = jnp.maximum(grouped_norm(prev.x), grouped_norm(prev.z))
        primal_ok = primal_residual < EPS + tol * primal_target

        # check dual
        dual_residual = self.rho * grouped_norm(self.z - prev.z)
        dual_target = self.rho * grouped_norm(prev.u)
        dual_ok = dual_residual < EPS + tol * dual_target

        # update rho and u to balance primal and dual residuals
        increase = primal_residual > 10 * dual_residual
        decrease = dual_residual > 10 * primal_residual
        scale = jnp.select([increase, decrease], [2.0, 0.5], default=1.0)

        self = self._replace(
            rho=self.rho * scale,
            u=self.u.at[:, :-1].mul(jnp.repeat(1 / scale, self.group_size)),
        )
        return self, primal_ok, dual_ok


class GroupLassoGaussianProcess(NamedTuple):
    theta: Float[Array, "o d*g"]
    g: Float[Array, "o"]
    b: Float[Array, "o"]
    nu: Float[Array, "o"]

    x_train: Float[Array, "n d*g"]
    y_train: Float[Array, "n o"]
    Koo: Float[Array, "o n n"]

    @jax.jit
    def predict(
        self, xs: Float[Array, "m d*g"]
    ) -> tuple[Float[Array, "o m"], Float[Array, "o m m"]]:
        # use scan instead of vmap to avoid OOM
        def scan_kernel(theta, xs1, xs2):
            body = lambda _, t: (_, kernel(t, xs1, xs2))
            _, K = jax.lax.scan(body, None, theta)
            return K

        nu = self.nu[:, None, None]
        Kxx = nu * scan_kernel(self.theta, xs, xs)
        Kox = nu * scan_kernel(self.theta, self.x_train, xs)
        Koo = nu * self.Koo
        return jax.vmap(gp_posterior)(Kxx, Kox, Koo, self.y_train.T, self.b)

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
        _, Koo = jax.lax.scan(
            lambda _, t: (_, kernel(t, x_train, x_train)), None, theta
        )
        Koo = Koo + g[..., None, None] * jnp.eye(len(y_train))
        # again, use scan instead of vmap to avoid OOM
        _, (llk, b, nu) = jax.lax.scan(
            lambda _, K_y: (_, loglikelihood(*K_y)), None, (Koo, y_train.T)
        )
        self = cls(theta, g, b, nu, x_train, y_train, Koo)
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
        theta_range: tuple[float, float] = (0.0, 100.0),
        g_range: tuple[float, float] = (EPS, 10.0),
        max_iterations: int = 100,
        tol: Scalar = jnp.array(1e-4),
    ) -> tuple[Self, GLASSOADMMState, Scalar]:
        n, d_times_g = x_train.shape
        n, o = y_train.shape
        assert d_times_g % group_size == 0

        # get the bounds for the optimization
        theta_min = jnp.ones((o, d_times_g)) * theta_range[0]
        theta_max = jnp.ones((o, d_times_g)) * theta_range[1]
        theta_bounds = jnp.stack([theta_min, theta_max], axis=0)
        g_min = jnp.ones((o,)) * g_range[0]
        g_max = jnp.ones((o,)) * g_range[1]
        g_bounds = jnp.stack([g_min, g_max], axis=0)
        bounds = jnp.concat([theta_bounds, g_bounds[..., None]], axis=-1)

        # initialize ADMM state
        state = (
            GLASSOADMMState.initialize(bounds, l1_penalty, group_size)
            if warmstart is None
            else warmstart._replace(
                rho=l1_penalty * jnp.ones_like(warmstart.rho),
                l1=l1_penalty,
            )
        )

        for iter in (pbar := tqdm(range(max_iterations), desc="ADMM")):
            new_state = state.x_update(x_train, y_train, autoregressive)
            new_state = new_state.z_and_u_update()
            state, primal_ok, dual_ok = new_state.check_residuals(state, tol)
            pbar.set_postfix({"rho": (state.rho.min().item(), state.rho.max().item())})
            if primal_ok.all() and dual_ok.all():
                print(f"ADMM converged in {iter+1} iterations.")
                break
        else:
            print("ADMM did not converge within the maximum number of iterations.")

        self, llk = cls.unpack_parameters(state.x, x_train, y_train)
        return self, state, llk
