from typing import NamedTuple, Optional
from jaxtyping import Array, Float, Key
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.numpy.linalg import norm
from einops import rearrange, einsum
import scipy
from tqdm import tqdm

EPS = float(jnp.sqrt(jnp.finfo(float).eps))

FIXED_G = jnp.array(0.001)
ADAPT_RHO = False
BFGS_ITERS = 1000


class Gaussian(NamedTuple):
    mean: Float[Array, "n"]
    cov: Float[Array, "n n"]


class Parameters(NamedTuple):
    theta: Float[Array, "o d d"]
    g: Float[Array, "o"]
    b: Float[Array, "o"]
    nu: Float[Array, "o"]


class ADMMState(NamedTuple):
    x: Float[Array, "o d*d+1"]
    z: Float[Array, "o d*d+1"]
    u: Float[Array, "o d*d+1"]
    rho: float


def kernel(
    x1: Float[Array, "n d"],
    x2: Float[Array, "m d"],
    theta: Float[Array, "d d"],
) -> Float[Array, "n m"]:
    def dist(a, b):
        theta_ab = theta * (a - b)
        return jnp.sum(theta_ab**2)

    dist = jax.vmap(jax.vmap(dist, (None, 0)), (0, None))
    return jnp.exp(-dist(x1, x2))


def likelihood(
    theta: Float[Array, "d d"],
    g: Float[Array, ""],
    x: Float[Array, "n d"],
    y: Float[Array, "n"],
):
    n, d = x.shape

    # foward of kernel
    K = kernel(x, x, theta)
    K = K + jnp.eye(n) * (EPS + g)

    # cholesky of K and compute logdet
    K_sqrt, is_lower = jsp.linalg.cho_factor(K)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    # compute Ki_1=(K^-1 @ 1) and Ki_y=(K^-1 @ y)
    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([jnp.ones_like(y), y], 1),
    ).T

    # compute optimal trend b
    b = (Ki_1 * y).sum() / Ki_1.sum()
    nu = jnp.dot((y - b) / n, (Ki_y - Ki_1 * b))

    # # likelihood when marginalizing over trend and variance
    loglik = -0.5 * (n * jnp.log(nu) + logdetK)
    return loglik, b, nu


@jax.jit
@jax.value_and_grad
def admm_x_update_loss(
    admm_x: Float[Array, "d*d+1"],
    admm_z: Float[Array, "d*d+1"],
    admm_u: Float[Array, "d*d+1"],
    rho: float,
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n"],
):
    n, d = x_train.shape
    # theta, g = admm_x[:-1], admm_x[-1]
    theta, _ = admm_x[:-1], admm_x[-1]
    theta = theta.reshape(d, d)
    g = FIXED_G
    loglik, _, _ = likelihood(theta, g, x_train, y_train)
    lagrangian = 0.5 * rho * jnp.sum((admm_x - admm_z + admm_u) ** 2)
    return -loglik + lagrangian


def admm_x_update(
    admm_x: Float[Array, "o d*d+1"],
    admm_z: Float[Array, "o d*d+1"],
    admm_u: Float[Array, "o d*d+1"],
    rho: float,
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n o"],
) -> Float[Array, "o d*d+1"]:
    new_x = []
    for i, (x, z, u, y) in enumerate(zip(admm_x, admm_z, admm_u, y_train.T)):
        ret = scipy.optimize.minimize(
            fun=admm_x_update_loss,
            x0=x,
            args=(z, u, rho, x_train, y),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=BFGS_ITERS, ftol=EPS, gtol=0),
        ).x
        new_x.append(ret)
    new_x = jnp.stack(new_x, axis=0)
    return new_x


@jax.jit
def admm_z_update(
    admm_x: Float[Array, "o d*d+1"],
    admm_z: Float[Array, "o d*d+1"],
    admm_u: Float[Array, "o d*d+1"],
    rho: float,
    l1_penalty: float,
) -> Float[Array, "o d*d+1"]:
    theta = (admm_x + admm_u)[:, :-1]
    g = (admm_x + admm_u)[:, -1]

    o, d = theta.shape
    d = int(d**0.5)

    Z = rearrange(theta, "o (d1 d2) -> (o d1) d2", d1=d, d2=d)
    U, S, V = jnp.linalg.svd(Z, full_matrices=False)
    S = jnp.maximum(0, S - l1_penalty / rho)
    theta = U @ jnp.diag(S) @ V
    theta = rearrange(theta, "(o d1) d2 -> o (d1 d2)", o=o)

    new_z = admm_z.at[:, :-1].set(theta)
    new_z = new_z.at[:, -1].set(g)  # no regularization on g
    return new_z


def admm(
    warmstart: ADMMState,
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n o"],
    l1_penalty: float,
    max_iterations: int,
    tollerance: float,
):
    n, d = x_train.shape
    n, o = y_train.shape

    admm_x = warmstart.x
    admm_z = warmstart.z
    admm_u = warmstart.u
    rho = warmstart.rho

    if not ADAPT_RHO or ADAPT_RHO:
        rho = l1_penalty

    trajectory = [ADMMState(admm_x, admm_z, admm_u, rho)]
    for iter in (pbar := tqdm(range(max_iterations), desc="ADMM")):
        new_admm_x = admm_x_update(admm_x, admm_z, admm_u, rho, x_train, y_train)
        new_admm_z = admm_z_update(new_admm_x, admm_z, admm_u, rho, l1_penalty)
        new_admm_u = admm_u + new_admm_x - new_admm_z

        # check convergence
        primal_residual = norm(new_admm_x - new_admm_z, axis=-1)
        primal_target = jnp.maximum(norm(admm_x, axis=-1), norm(admm_z, axis=-1))
        primal_ok = primal_residual < EPS + tollerance * primal_target
        dual_residual = rho * norm(new_admm_z - admm_z, axis=-1)
        dual_target = rho * norm(admm_u, axis=-1)
        dual_ok = dual_residual < EPS + tollerance * dual_target

        # update rho to balance primal and dual residuals
        if ADAPT_RHO:
            if (
                jnp.square(primal_residual).sum()
                > 100 * jnp.square(dual_residual).sum()
            ):
                rho = 2 * rho
                new_admm_u = new_admm_u / 2
            elif (
                jnp.square(dual_residual).sum()
                > 100 * jnp.square(primal_residual).sum()
            ):
                rho = rho / 2
                new_admm_u = new_admm_u * 2

        pbar.set_postfix(
            {
                "primal:": f"{(primal_residual / (primal_target + EPS)).max():.5f}",
                "dual:": f"{(dual_residual / (dual_target + EPS)).max():.5f}",
                "rho": rho,
            }
        )

        # update state and possibly early stop
        admm_x, admm_z, admm_u = new_admm_x, new_admm_z, new_admm_u
        trajectory.append(ADMMState(admm_x, admm_z, admm_u, rho))
        if primal_ok.all() and dual_ok.all():
            break
    else:
        print("ADMM did not converge within the maximum number of iterations.")
    return trajectory


@dataclass
class GaussianProcessRegressor:
    x_train: Float[Array, "n d"]
    y_train: Float[Array, "n o"]
    n_groups: int
    normalize: bool = True
    max_iterations: int = 1000
    tollerance: float = 1e-4
    seed: int = 42
    verbose: bool = False

    # params to fit after training
    parameters: Parameters = field(init=False)
    x: Float[Array, "n d"] = field(init=False)
    y: Float[Array, "n o"] = field(init=False)
    bounds: Float[Array, "2 o d"] = field(init=False)
    trajectory: list[ADMMState] = field(init=False)

    def __post_init__(self):
        n, d = self.x_train.shape
        n, o = self.y_train.shape

        # initialize ADMM state
        init_theta = jnp.stack([jnp.eye(d).flatten() for _ in range(o)])
        init_g = 0.1 * jnp.ones((o, 1))
        admm_state = ADMMState(
            x=jnp.concat([init_theta, init_g], axis=-1),
            z=jnp.concat([init_theta, init_g], axis=-1),
            u=jnp.zeros((o, d + 1)),
            rho=1.0,
        )
        self.trajectory = [admm_state]

    def fit(self, l1_penalty: float):
        n, d = self.x_train.shape
        n, o = self.y_train.shape

        # run admm
        trajectory = admm(
            warmstart=self.trajectory[-1],
            x_train=self.x_train,
            y_train=self.y_train,
            l1_penalty=l1_penalty,
            max_iterations=self.max_iterations,
            tollerance=self.tollerance,
        )
        self.trajectory.extend(trajectory[1:])  # skip the warmstart state

        # extract the optimal parameters and infer the rest
        admm_x, admm_z, admm_u, rho = self.trajectory[-1]
        theta = admm_x[..., :-1].reshape(o, d, d)
        g = admm_x[..., -1]

        # iterate to avoid OOM with vmap
        llk_b_nu = [
            likelihood(theta_i, g_i, self.x_train, y_i)
            for theta_i, g_i, y_i in zip(theta, g, self.y_train.T)
        ]
        llk, b, nu = jnp.array(list(zip(*llk_b_nu)))

        self.parameters = Parameters(theta, g, b, nu)
        if self.verbose:
            print(f"Optimal theta: {theta}")
            print(f"Optimal g: {g}")
            print(f"Optimal b: {b}")
            print(f"Optimal nu: {nu}")
            print(f"Log-likelihood at optimum: {llk.sum()}")
            print()
        return self

    def predict(self, x: Float[Array, "n d"]) -> Gaussian:
        n, d = self.x_train.shape
        n, o = self.y_train.shape

        def predict_single(theta, g, b, nu, y):
            g = FIXED_G
            Koo = nu * (
                kernel(self.x_train, self.x_train, theta) + jnp.eye(n) * (EPS + g)
            )
            Kxo = nu * kernel(x, self.x_train, theta)
            Kxx = nu * kernel(x, x, theta)

            # posterior mean and covariance
            mean = b + Kxo @ jnp.linalg.solve(Koo, y - b)
            cov = Kxx - Kxo @ jnp.linalg.solve(Koo, Kxo.T)

            # Add correction based on the trend estimation correlation (?)
            Kbx = jnp.ones((1, n)) @ jnp.linalg.solve(Koo, Kxo.T)
            cov = cov + (1 - Kbx).T @ (1 - Kbx) / jnp.linalg.inv(Koo).sum()
            return Gaussian(mean=mean, cov=cov)

        preds = [
            predict_single(theta_i, g_i, b_i, nu_i, y_i)
            for theta_i, g_i, b_i, nu_i, y_i in zip(*self.parameters, self.y_train.T)
        ]
        return Gaussian(
            mean=jnp.stack([pred.mean for pred in preds]),
            cov=jnp.stack([pred.cov for pred in preds]),
        )
