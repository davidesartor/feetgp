"""What does one unit of inner budget buy, and what does it cost?

optimistix counts one function evaluation per step; L-BFGS-B counts a whole line search
per iteration, so the two `maxiter` arguments are not the same currency. This measures
both in the currency that matters -- wall time and augmented-Lagrangian value reached --
on a single real x-update.
"""

import argparse
import functools
import glob
import pickle
import re
import time

print = functools.partial(print, flush=True)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from vlse.lbfgsb import minimise

from feetgp import admm
from feetgp.glassogp import (
    admm_state_from_pickle,
    admm_x_update_loss,
    autoregressive_mask,
    hetgpy_auto_bounds,
)
from feetgp.inclinerunning import InclineRunning


def nearest_cached(run_dir: str, l1: float) -> tuple[str, float]:
    paths = glob.glob(f"{run_dir}/lambda=*.pkl")
    parsed = [
        (float(m.group(1)), p)
        for p in paths
        if (m := re.search(r"lambda=([0-9.e+-]+)\.pkl", p))
    ]
    value, path = min(parsed, key=lambda vp: abs(vp[0] - l1))
    return path, value


parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--warm_lambda", type=float, required=True)
parser.add_argument("--subsample", type=int, default=20)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--chunk_size", type=int, default=39)
parser.add_argument("--history_length", type=int, default=40)
parser.add_argument("--tol", type=float, default=1e-2)
parser.add_argument("--maxiters", type=int, nargs="+", default=[5, 10, 20, 50])
parser.add_argument("--max_linesearches", type=int, nargs="+", default=[5, 30])
args = parser.parse_args()

data = InclineRunning(
    subsample=args.subsample, feet=args.feet, target=args.target, inclines="all"
)
x_train = jnp.asarray(data.x_train)
y_train = jnp.asarray(data.y_train)
group_size = 6 if args.feet == "both" else 3
n, d_times_g = x_train.shape
o = y_train.shape[1]

warm_path, warm_value = nearest_cached(args.run_dir, args.warm_lambda)
with open(warm_path, "rb") as f:
    warm = admm_state_from_pickle(pickle.load(f))
print(f"warmstart {warm_path} (lambda={warm_value:.6g}, rho={float(warm.rho):.4g})")

# bounds are rederived from the data, exactly as fit does
lower_auto, _ = hetgpy_auto_bounds(x_train)
theta_max = jnp.broadcast_to(jnp.sqrt(2.0 / lower_auto), (o, d_times_g))
lower = jnp.concatenate([jnp.zeros((o, d_times_g)), jnp.full((o, 1), -jnp.inf)], -1)
upper = jnp.concatenate([theta_max, jnp.full((o, 1), jnp.inf)], -1)

group_columns = jnp.arange(d_times_g)[None, :] // group_size
n_groups = d_times_g // group_size
keep = (jnp.arange(n_groups)[:, None] != group_columns).astype(x_train.dtype)
masked_designs = x_train[None, :, :] * keep[:, None, :]
mask = autoregressive_mask(o, d_times_g, group_size)
group_of_output = jnp.arange(o) // group_size

# the same [theta, w] the x-update starts from, and the same target it aims at
x0 = (
    jnp.concat([admm.to_outputs(warm.x, group_size), warm.aux[..., None]], axis=-1)
    * mask
)
target_theta = admm.to_outputs(warm.z - warm.u, group_size)


def x_update(maxiter: int, max_linesearch: int):
    """One full x-update, returning the final objective and evaluation count per output."""

    def solve_one(args_i):
        x0_i, target_i, y_i, group_i, mask_i, lower_i, upper_i = args_i
        loss_args = (
            target_i,
            warm.rho,
            masked_designs[group_i],
            y_i,
            jnp.array(1e-4),
            jnp.array(100.0),
        )
        state = minimise(
            admm_x_update_loss,
            x0_i,
            bounds=(lower_i, upper_i),
            args=(loss_args,),
            tol=args.tol,
            max_iterations=maxiter,
            history_length=args.history_length,
            max_linesearch=max_linesearch,
        )
        return state.x * mask_i, state.f, state.n_fun_eval, state.iteration, state.error

    return jax.lax.map(
        solve_one,
        (x0, target_theta, y_train.T, group_of_output, mask, lower, upper),
        batch_size=args.chunk_size,
    )


for max_linesearch in args.max_linesearches:
    for maxiter in args.maxiters:
        run = jax.jit(functools.partial(x_update, maxiter, max_linesearch))
        x, f, n_eval, iterations, error = jax.block_until_ready(run())
        t0 = time.perf_counter()
        x, f, n_eval, iterations, error = jax.block_until_ready(run())
        dt = time.perf_counter() - t0
        print(
            f"maxiter={maxiter:3d} max_linesearch={max_linesearch:3d}: {dt:6.2f}s  "
            f"objective={float(f.sum()):.8g}  "
            f"evals/output mean={float(n_eval.mean()):6.1f} max={int(n_eval.max()):4d}  "
            f"iters mean={float(iterations.mean()):5.1f} at_cap={int((iterations >= maxiter).sum())}/{o}  "
            f"pgrad median={float(jnp.median(error)):.3e}"
        )
