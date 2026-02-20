from typing import NamedTuple, Optional
from jaxtyping import Array, Float, Key
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.numpy.linalg import norm
import scipy
from tqdm import tqdm

EPS = float(jnp.sqrt(jnp.finfo(float).eps))


def squared_distance(
    x1: Float[Array, "n d"],
    x2: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    def dist(a, b):
        return jnp.sum((a - b) ** 2)

    dist = jax.vmap(jax.vmap(dist, (None, 0)), (0, None))
    return dist(x1, x2)


def kernel(
    x1: Float[Array, "n d"],
    x2: Float[Array, "m d"],
    theta: Float[Array, "d"],
) -> Float[Array, "n m"]:
    return jnp.exp(-squared_distance(x1 * theta, x2 * theta))


def likelihood(
    theta: Float[Array, "d"],
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
    admm_x: Float[Array, "d+1"],
    admm_z: Float[Array, "d+1"],
    admm_u: Float[Array, "d+1"],
    rho: float,
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n"],
):
    theta, g = admm_x[:-1], admm_x[-1]
    loglik, _, _ = likelihood(theta, g, x_train, y_train)
    lagrangian = 0.5 * rho * jnp.sum((admm_x - admm_z + admm_u) ** 2)
    return -loglik + lagrangian


def admm_x_update(
    admm_x: Float[Array, "o d+1"],
    admm_z: Float[Array, "o d+1"],
    admm_u: Float[Array, "o d+1"],
    rho: float,
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n o"],
    bounds: Float[Array, "2 o d+1"],
) -> Float[Array, "o d+1"]:
    new_x = [
        scipy.optimize.minimize(
            fun=admm_x_update_loss,
            x0=x,
            args=(z, u, rho, x_train, y),
            jac=True,
            method="L-BFGS-B",
            bounds=[(a, b) for a, b in zip(bmin, bmax)],
            options=dict(maxiter=10, ftol=EPS, gtol=0),
        ).x
        for x, z, u, y, bmin, bmax in zip(admm_x, admm_z, admm_u, y_train.T, *bounds)
    ]
    new_x = jnp.stack(new_x, axis=0)
    return new_x


@jax.jit
def admm_z_update(
    admm_x: Float[Array, "o d+1"],
    admm_z: Float[Array, "o d+1"],
    admm_u: Float[Array, "o d+1"],
    rho: float,
    l1_penalty: float,
    bounds: Float[Array, "2 o d+1"],
) -> Float[Array, "o d+1"]:
    theta = (admm_x + admm_u)[:, :-1]
    g = (admm_x + admm_u)[:, -1]
    prox = jnp.maximum(0, 1 - (l1_penalty / rho / norm(theta, axis=0)))
    theta = prox * theta
    new_z = admm_z.at[:, :-1].set(theta)
    new_z = new_z.at[:, -1].set(g)  # no regularization on g
    return new_z.clip(*bounds)  # in case bounds do not include 0


def admm(
    x0: Float[Array, "o d+1"],
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n o"],
    bounds: Float[Array, "2 o d+1"],
    l1_penalty: float,
    max_iterations: int,
    tollerance: float,
):
    admm_x = x0
    admm_z = x0
    admm_u = jnp.zeros_like(x0)
    rho = 1.0

    trajectory = [(admm_x, admm_z, admm_u, rho)]
    for iter in (pbar := tqdm(range(max_iterations), desc="ADMM")):
        new_admm_x = admm_x_update(
            admm_x, admm_z, admm_u, rho, x_train, y_train, bounds
        )
        new_admm_z = admm_z_update(new_admm_x, admm_z, admm_u, rho, l1_penalty, bounds)
        new_admm_u = admm_u + new_admm_x - new_admm_z

        # check convergence
        primal_residual = norm(new_admm_x - new_admm_z, axis=-1)
        primal_target = jnp.maximum(norm(admm_x, axis=-1), norm(admm_z, axis=-1))
        primal_ok = primal_residual < EPS + tollerance * primal_target
        dual_residual = rho * norm(new_admm_z - admm_z, axis=-1)
        dual_target = rho * norm(admm_u, axis=-1)
        dual_ok = dual_residual < EPS + tollerance * dual_target
        pbar.set_postfix(
            {
                "primal:": (primal_residual / (primal_target + EPS)).max(),
                "dual:": (dual_residual / (dual_target + EPS)).max(),
            }
        )

        # update rho to balance primal and dual residuals
        if primal_residual.sum() > 10 * dual_residual.sum():
            rho = 2 * rho
            new_admm_u = new_admm_u / 2
        elif dual_residual.sum() > 10 * primal_residual.sum():
            rho = rho / 2
            new_admm_u = new_admm_u * 2

        # update state and possibly early stop
        admm_x, admm_z, admm_u = new_admm_x, new_admm_z, new_admm_u
        trajectory.append((admm_x, admm_z, admm_u, rho))
        if primal_ok.all() and dual_ok.all():
            break
    else:
        print("ADMM did not converge within the maximum number of iterations.")
    return trajectory


def hetgpy_auto_bounds(x, min_cor=0.01, max_cor=0.5):
    # rescale X to [0,1]^d
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x = (x - x_min) @ jnp.diag(1 / (x_max - x_min))
    # compute pairwise distances only for proper pair
    dists = squared_distance(x, x)
    dists = dists[jnp.tril(dists, k=-1) > 0]
    # magic initialization using inverse of kernel
    lower = -jnp.quantile(dists, q=0.05) / jnp.log(min_cor) * (x_max - x_min) ** 2
    upper = -jnp.quantile(dists, q=0.95) / jnp.log(max_cor) * (x_max - x_min) ** 2
    return lower, upper


class Gaussian(NamedTuple):
    mean: Float[Array, "n"]
    cov: Float[Array, "n n"]


class Parameters(NamedTuple):
    theta: Float[Array, "o d"]
    g: Float[Array, "o"]
    b: Float[Array, "o"]
    nu: Float[Array, "o"]


@dataclass
class GaussianProcessRegressor:
    l1_penalty: float
    max_iterations: int = 1000
    tollerance: float = 1e-4
    multi_start: int = 1
    seed: int = 42
    verbose: bool = False
    init_theta: Optional[Float[Array, "o d"]] = None
    init_g: Optional[Float[Array, "o"]] = None

    # params to fit after training
    parameters: Parameters = field(init=False)
    x: Float[Array, "n d"] = field(init=False)
    y: Float[Array, "n o"] = field(init=False)
    trajectory: list = field(init=False)
    bounds: Float[Array, "2 o d"] = field(init=False)

    def fit(self, x: Float[Array, "n d"], y: Float[Array, "n 1"]):
        n, d = x.shape
        n, o = y.shape

        # initialization
        # hetgpy uses a different parametrization for theta
        # i.e. out_theta = 1 / sqrt(hetgpy_theta)
        lower, upper = hetgpy_auto_bounds(x)
        init_theta = jnp.tile(1 / (0.9 * upper + 0.1 * lower) ** 0.5, (o, 1))
        init_g = jnp.array([0.1] * o)
        if self.init_theta is not None:
            init_theta = self.init_theta
        if self.init_g is not None:
            init_g = self.init_g
        if self.verbose:
            print(f"Initial theta: {init_theta}")
            print(f"Initial g: {init_g}")
            print()

        # bounds for optimization
        # also here we need to transform and invert the theta range w.r.t. hetgpy
        theta_bounds = jnp.array([1 / upper**0.5, 1 / lower**0.5])
        g_bounds = jnp.array([EPS, 1e2])
        bounds = jnp.concat([theta_bounds, g_bounds[:, None]], axis=-1)
        bounds = jnp.tile(bounds[:, None, :], (1, o, 1))
        bounds = bounds.at[0, ..., :-1].set(0.0)
        self.bounds = bounds
        if self.verbose:
            print(f"Bounds for theta:")
            print(f"Min: {self.bounds[0, ..., :-1]}")
            print(f"Max: {self.bounds[1, ..., :-1]}")
            print(f"Bounds for g:")
            print(f"Min: {self.bounds[0, ..., -1]}")
            print(f"Max: {self.bounds[1, ..., -1]}")
            print()

        results = []
        for rng_key in jr.split(jr.key(self.seed), self.multi_start):
            # get a random init point
            rng_a, rng_b = jr.split(rng_key)
            a = jr.uniform(rng_a)
            init_theta = a * init_theta + (1 - a) * jr.uniform(
                rng_b,
                init_theta.shape,
                minval=bounds[0, ..., :-1],
                maxval=bounds[1, ..., :-1],
            )
            x0 = jnp.concatenate([init_theta, init_g[:, None]], axis=-1)

            # run admm
            trajectory = admm(
                x0=x0,
                x_train=x,
                y_train=y,
                bounds=self.bounds,
                l1_penalty=self.l1_penalty,
                max_iterations=self.max_iterations,
                tollerance=self.tollerance,
            )

            # extract the optimal parameters and infer the rest
            admm_x, admm_z, admm_u, rho = trajectory[-1]
            theta = admm_x[:, :-1]
            g = admm_x[:, -1]
            llk, b, nu = jax.vmap(likelihood, in_axes=(0, 0, None, 1))(theta, g, x, y)
            loss = -llk.sum() + self.l1_penalty * jnp.sum(norm(theta, axis=0))
            results.append((loss, trajectory, theta, g, b, nu))
            if self.verbose:
                print(f"Optimal theta: {theta}")
                print(f"Optimal g: {g}")
                print(f"Optimal b: {b}")
                print(f"Optimal nu: {nu}")
                print(f"Log-likelihood at optimum: {llk}")
                print()

        # best result (minimum loss) across the multiple random restarts
        trajectory, theta, g, b, nu = min(results, key=lambda x: x[0])[1:]

        # save stuff
        self.trajectory = trajectory
        self.parameters = Parameters(theta, g, b, nu)
        self.x = x
        self.y = y
        return self

    def predict(self, x: Float[Array, "n d"]) -> Gaussian:
        n, d = self.x.shape
        n, o = self.y.shape

        def predict_single(params, y):
            theta, g, b, nu = params
            Koo = nu * (kernel(self.x, self.x, theta) + jnp.eye(n) * (EPS + g))
            Kxo = nu * kernel(x, self.x, theta)
            Kxx = nu * kernel(x, x, theta)

            # posterior mean and covariance
            mean = b + Kxo @ jnp.linalg.solve(Koo, y - b)
            cov = Kxx - Kxo @ jnp.linalg.solve(Koo, Kxo.T)

            # Add correction based on the trend estimation correlation (?)
            Kbx = jnp.ones((1, n)) @ jnp.linalg.solve(Koo, Kxo.T)
            cov = cov + (1 - Kbx).T @ (1 - Kbx) / jnp.linalg.inv(Koo).sum()
            return Gaussian(mean=mean, cov=cov)

        return jax.vmap(predict_single)(self.parameters, self.y.T)
