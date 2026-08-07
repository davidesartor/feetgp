
import argparse
import functools

print = functools.partial(print, flush=True)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from feetgp import glasso_admm
from feetgp.glasso_admm import ADMMState
from feetgp.gp import (
    autoregressive_mask,
    hetgpy_auto_bounds,
    w_from_nugget,
    x_update_solve,
)
from feetgp.inclinerunning import InclineRunning

parser = argparse.ArgumentParser()
parser.add_argument("--subsample", type=int, default=20)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--l1_penalty", type=float, default=30.0)
parser.add_argument("--iterations", type=int, default=120)
parser.add_argument("--chunk_size", type=int, default=39)
parser.add_argument("--inner_maxiter", type=int, default=5)
parser.add_argument("--inner_tol", type=float, default=1e-2)
parser.add_argument("--adapt_rho_iters", type=int, default=60)
parser.add_argument("--log_every", type=int, default=5)
args = parser.parse_args()

print("JAX devices:", jax.devices())

group_size = 6 if args.feet == "both" else 3
data = InclineRunning(
    subsample=args.subsample, feet=args.feet, target=args.target, inclines="all"
)
x_train = jnp.asarray(data.x_train)
y_train = jnp.asarray(data.y_train)
n, d = x_train.shape
_, o = y_train.shape
n_groups = d // group_size
autoregressive = args.target == "markers"
print(f"n={n} d={d} o={o} groups={n_groups} lambda={args.l1_penalty:g}")

lower, upper = hetgpy_auto_bounds(x_train)
theta_max = glasso_admm.to_groups(jnp.broadcast_to(jnp.sqrt(2.0 / lower), (o, d)), group_size)
theta_init = glasso_admm.to_groups(
    jnp.broadcast_to(jnp.sqrt(2.0 / (0.9 * upper + 0.1 * lower)), (o, d)), group_size
)
bounds = jnp.stack([jnp.zeros_like(theta_max), theta_max])
unbounded = jnp.full((o, 1), jnp.inf)
solver_bounds = (
    jnp.concat([glasso_admm.to_outputs(bounds[0], group_size), -unbounded], axis=-1),
    jnp.concat([glasso_admm.to_outputs(bounds[1], group_size), unbounded], axis=-1),
)
g_min, g_max = jnp.array(1e-4), jnp.array(100.0)
w_init = jnp.full((o,), w_from_nugget(jnp.array(0.1), g_min, g_max))
state = ADMMState.initialize(theta_init, aux=w_init)
l1 = jnp.array(args.l1_penalty)

if autoregressive:
    group_columns = jnp.arange(d)[None, :] // group_size
    keep = (jnp.arange(n_groups)[:, None] != group_columns).astype(x_train.dtype)
    masked_designs = x_train[None, :, :] * keep[:, None, :]
    mask = autoregressive_mask(o, d, group_size)
else:
    masked_designs = jnp.broadcast_to(x_train, (n_groups, *x_train.shape))
    mask = None


def group_norms(a):
    return jnp.linalg.norm(a, axis=-1)


def group_signs(a):
    return jnp.sign(a.sum(-1))


previous_signs = group_signs(state.x)
flips = jnp.zeros(n_groups)
print()
print(f"{'iter':>5} {'r':>10} {'s':>10} {'rho':>9} {'act':>4}  group norms of x")
for iter in range(args.iterations):
    maxiter = min(args.inner_maxiter, 2**iter)
    solution = x_update_solve(
        jnp.concat(
            [glasso_admm.to_outputs(state.x, group_size), state.aux[..., None]], axis=-1
        ),
        glasso_admm.to_outputs(state.z - state.u, group_size),
        state.rho,
        masked_designs,
        y_train,
        group_size,
        solver_bounds,
        mask=mask,
        g_min=g_min,
        g_max=g_max,
        maxiter=maxiter,
        chunk_size=args.chunk_size,
        tol=args.inner_tol,
    )
    theta = glasso_admm.to_groups(solution[:, :-1], group_size)
    new_state = state._replace(x=jnp.clip(theta, *bounds), aux=solution[:, -1])
    new_state = glasso_admm.z_and_u_update(new_state, l1, bounds)
    primal, dual = (float(r) for r in glasso_admm.residuals(new_state, state))

    signs = group_signs(new_state.x)
    flips = flips + (signs != previous_signs)
    previous_signs = signs

    state, primal_ok, dual_ok = glasso_admm.check_residuals(
        new_state, state, jnp.array(1e-3), iter < args.adapt_rho_iters
    )
    if iter % args.log_every == 0:
        xn, zn = group_norms(state.x), group_norms(state.z)
        active = int(jnp.sum(zn > 1e-8))
        print(
            f"{iter + 1:>5} {primal:>10.3e} {dual:>10.3e}"
            f" {state.rho.item():>9.2e} {active:>4}  "
            + " ".join(f"{v:6.2f}" for v in xn)
        )
        print(
            f"{'':>5} {'':>10} {'':>10} {'':>9} {'z':>4}  "
            + " ".join(f"{v:6.2f}" for v in zn)
        )
    if primal_ok.all() and dual_ok.all():
        print(f"converged at {iter + 1}")
        break

print()
print("sign flips per group over the run:")
print(" ".join(f"{int(v):6d}" for v in flips))
print(f"||u|| per group: ", " ".join(f"{v:6.2f}" for v in group_norms(state.u)))
