from typing import NamedTuple, Optional
from jaxtyping import Array, Float, Key
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from einops import rearrange, einsum
from tqdm import tqdm

EPS = float(jnp.sqrt(jnp.finfo(float).eps))


class ADMMState(NamedTuple):
    x: Float[Array, "g i o"]
    z: Float[Array, "g i o"]
    u: Float[Array, "g i o"]
    rho: float
    l: float


@jax.jit
def admm_x_update(
    admm_x: Float[Array, "g i o"],
    admm_z: Float[Array, "g i o"],
    admm_u: Float[Array, "g i o"],
    rho: float,
    x_train: Float[Array, "n i"],
    y_train: Float[Array, "n g o"],
) -> Float[Array, "g i o"]:
    # x = min f(x) = 1/2 ||X @ x - y||^2 + rho/2 ||x - z + u||^2
    # df/dx = X.T @ (X @ x - y) + rho * (x - z + u) = 0
    # x = (X.T @ X + rho I)^-1 (rho (z - u) + X.T y)
    n, i = x_train.shape
    n, g, o = y_train.shape
    A = einsum(x_train, x_train, "n i1, n i2 -> i1 i2") + rho * jnp.eye(i)
    b1 = einsum(x_train, y_train, "n i, n g o -> i g o")
    b1 = rearrange(b1, "i g o -> i (g o)")
    b2 = rho * (admm_z - admm_u)
    b2 = rearrange(b2, "g i o -> i (g o)")
    new_x = rearrange(jnp.linalg.solve(A, b1 + b2), "i (g o) -> g i o", g=g)
    return new_x


@jax.jit
def admm_z_update(
    admm_x: Float[Array, "g i o"],
    admm_z: Float[Array, "g i o"],
    admm_u: Float[Array, "g i o"],
    rho: float,
    l1_penalty: float,
) -> Float[Array, "g i o"]:
    # define Z := [z1, z2, ..., zg] and X := [x1, x2, ..., xg] and U := [u1, u2, ..., ug]
    # Z* = min lambda ||Z||_* + rho/2 ||Z-X+U||_F^2
    # obs: the term ||Z||_* is invariant under unitary transformations
    # obs: the term ||Z-X+U||_F^2 is minimized when Z has the same singluar vectors as X+U
    # i.e. the SVD are: Z* = U S V*, X+U = U S' V*
    # Z* = min lambda sum(diag(S)) + rho/2 sum(diag((S-S')^2))
    # solve in terms of the singular values of Z*:
    # S = max(0, S' - lambda/rho)
    g, i, o = admm_x.shape
    Z = rearrange(admm_x + admm_u, "g i o -> (g i) o")
    U, S, V = jnp.linalg.svd(Z, full_matrices=False)
    S = jnp.maximum(0, S - l1_penalty / rho)
    new_z = U @ jnp.diag(S) @ V
    new_z = rearrange(new_z, "(g i) o -> g i o", g=g)
    return new_z


def admm(
    x_train: Float[Array, "n i"],
    y_train: Float[Array, "n g o"],
    l1_penalty: float,
    max_iterations: int,
    tollerance: float,
):
    n, d = x_train.shape
    n, g, o = y_train.shape

    admm_x = jnp.zeros((g, d, o))
    admm_z = jnp.zeros((g, d, o))
    admm_u = jnp.zeros((g, d, o))
    rho = 1.0

    trajectory = [ADMMState(admm_x, admm_z, admm_u, rho, l1_penalty)]
    for iter in (pbar := tqdm(range(max_iterations), desc="ADMM")):
        new_admm_x = admm_x_update(admm_x, admm_z, admm_u, rho, x_train, y_train)
        new_admm_z = admm_z_update(new_admm_x, admm_z, admm_u, rho, l1_penalty)
        new_admm_u = admm_u + new_admm_x - new_admm_z

        # check convergence
        primal_residual = norm(new_admm_x - new_admm_z)
        primal_target = jnp.maximum(norm(admm_x), norm(admm_z))
        primal_ok = primal_residual < EPS + tollerance * primal_target
        dual_residual = rho * norm(new_admm_z - admm_z)
        dual_target = rho * norm(admm_u)
        dual_ok = dual_residual < EPS + tollerance * dual_target

        # update rho to balance primal and dual residuals
        if primal_residual > 10 * dual_residual:
            rho = 2 * rho
            new_admm_u = new_admm_u / 2
        elif dual_residual > 10 * primal_residual:
            rho = rho / 2
            new_admm_u = new_admm_u * 2

        pbar.set_postfix(
            {
                "primal:": f"{(primal_residual / (primal_target + EPS)):.5f}",
                "dual:": f"{(dual_residual / (dual_target + EPS)):.5f}",
                "rho": rho,
            }
        )

        # update state and possibly early stop
        admm_x, admm_z, admm_u = new_admm_x, new_admm_z, new_admm_u
        trajectory.append(ADMMState(admm_x, admm_z, admm_u, rho, l1_penalty))
        if primal_ok and dual_ok:
            break
    else:
        print("ADMM did not converge within the maximum number of iterations.")
    return trajectory


@dataclass
class LinearRegressor:
    x_train: Float[Array, "n i"]
    y_train: Float[Array, "n g o"]
    max_iterations: int = 1000
    tollerance: float = 1e-4
    seed: int = 42
    verbose: bool = False
    # params to fit after training
    parameters: Float[Array, "g i o"] = field(init=False)
    x: Float[Array, "n i"] = field(init=False)
    y: Float[Array, "n g o"] = field(init=False)
    trajectory: list[ADMMState] = field(init=False, default_factory=list)

    def fit(self, l1_penalty: float):
        # run admm
        trajectory = admm(
            x_train=self.x_train,
            y_train=self.y_train,
            l1_penalty=l1_penalty,
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

    def predict(self, x: Float[Array, "n i"]) -> Float[Array, "n g o"]:
        return einsum(self.parameters, x, "g i o, n i -> n g o")
