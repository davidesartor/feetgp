from typing import NamedTuple, Optional
from jaxtyping import Array, Float, Key
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.numpy.linalg import norm
from einops import rearrange
import scipy
from tqdm import tqdm

EPS = float(jnp.sqrt(jnp.finfo(float).eps))


class ADMMState(NamedTuple):
    x: Float[Array, "o d"]
    z: Float[Array, "o d"]
    u: Float[Array, "o d"]
    rho: float
    l: float


@jax.jit
def admm_x_update(
    admm_x: Float[Array, "o d"],
    admm_z: Float[Array, "o d"],
    admm_u: Float[Array, "o d"],
    rho: float,
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n o"],
) -> Float[Array, "o d"]:
    n, d = x_train.shape
    A = x_train.T @ x_train + rho * jnp.eye(d)
    b = x_train.T @ y_train + rho * (admm_z - admm_u).T
    print(A.shape, b.shape)
    new_x = jnp.linalg.solve(A, b).T
    return new_x


@partial(jax.jit, static_argnames=["n_groups"])
def admm_z_update(
    admm_x: Float[Array, "o d"],
    admm_z: Float[Array, "o d"],
    admm_u: Float[Array, "o d"],
    rho: float,
    n_groups: int,
    l1_penalty: float,
) -> Float[Array, "o d"]:
    new_z = rearrange(admm_x + admm_u, "o (d k) -> (o k) d", k=n_groups)
    prox = jnp.maximum(0, 1 - (l1_penalty / rho / norm(new_z, axis=0)))
    new_z = prox * new_z
    new_z = rearrange(new_z, "(o k) d -> o (d k)", k=n_groups)
    return new_z


def admm(
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n o"],
    l1_penalty: float,
    n_groups: int,
    max_iterations: int,
    tollerance: float,
):
    n, d = x_train.shape
    n, o = y_train.shape

    admm_x = jnp.zeros((o, d))
    admm_z = jnp.zeros((o, d))
    admm_u = jnp.zeros((o, d))
    rho = 1.0

    trajectory = [ADMMState(admm_x, admm_z, admm_u, rho, l1_penalty)]
    for iter in (pbar := tqdm(range(max_iterations), desc="ADMM")):
        new_admm_x = admm_x_update(
            admm_x, admm_z, admm_u, rho, x_train, y_train
        )
        new_admm_z = admm_z_update(
            new_admm_x, admm_z, admm_u, rho, n_groups, l1_penalty
        )
        new_admm_u = admm_u + new_admm_x - new_admm_z

        # check convergence
        primal_residual = norm(new_admm_x - new_admm_z, axis=-1)
        primal_target = jnp.maximum(norm(admm_x, axis=-1), norm(admm_z, axis=-1))
        primal_ok = primal_residual < EPS + tollerance * primal_target
        dual_residual = rho * norm(new_admm_z - admm_z, axis=-1)
        dual_target = rho * norm(admm_u, axis=-1)
        dual_ok = dual_residual < EPS + tollerance * dual_target

        # update rho to balance primal and dual residuals
        if jnp.square(primal_residual).sum() > 100 * jnp.square(dual_residual).sum():
            rho = 2 * rho
            new_admm_u = new_admm_u / 2
        elif jnp.square(dual_residual).sum() > 100 * jnp.square(primal_residual).sum():
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
        trajectory.append(ADMMState(admm_x, admm_z, admm_u, rho, l1_penalty))
        if primal_ok.all() and dual_ok.all():
            break
    else:
        print("ADMM did not converge within the maximum number of iterations.")
    return trajectory


@dataclass
class LinearRegressor:
    x_train: Float[Array, "n d"]
    y_train: Float[Array, "n o"]
    n_groups: int
    max_iterations: int = 1000
    tollerance: float = 1e-4
    seed: int = 42
    verbose: bool = False

    # params to fit after training
    parameters: Float[Array, "o d"] = field(init=False)
    x: Float[Array, "n d"] = field(init=False)
    y: Float[Array, "n o"] = field(init=False)
    trajectory: list[ADMMState] = field(init=False)

    def __post_init__(self):
        n, d = self.x_train.shape
        n, o = self.y_train.shape
        self.trajectory = []

    def fit(self, l1_penalty: float):
        # run admm
        trajectory = admm(
            x_train=self.x_train,
            y_train=self.y_train,
            l1_penalty=l1_penalty,
            n_groups=self.n_groups,
            max_iterations=self.max_iterations,
            tollerance=self.tollerance,
        )
        self.trajectory.extend(trajectory)  

        # extract the optimal parameters and infer the rest
        admm_x, admm_z, admm_u, rho, l = self.trajectory[-1]
        self.parameters = admm_x
        if self.verbose:
            print(f"Optimal parameters: {self.parameters}")
            print()
        return self

    def predict(self, x: Float[Array, "n d"]) -> Float[Array, "n o"]:
        return self.parameters @ x.T