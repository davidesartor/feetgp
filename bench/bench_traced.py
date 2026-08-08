"""Static vs traced L-BFGS-B iteration bound: is the static win an unrolling artifact of tiny budgets?"""

import argparse
import functools
import time

print = functools.partial(print, flush=True)

import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange, repeat
from vlse.optim import minimise as lbfgsb_minimise

jax.config.update("jax_enable_x64", True)

from feetgp.gp import (
    G_RANGE,
    autoregressive_mask,
    hetgpy_auto_bounds,
    w_from_nugget,
    x_update_loss,
)
from feetgp.inclinerunning import InclineRunning

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/Incline Running")
parser.add_argument("--subsample", type=int, default=400)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--budgets", type=int, nargs="+", default=[1, 5, 10, 30])
parser.add_argument("--modes", type=str, nargs="+", default=["static", "traced"])
parser.add_argument("--chunk_size", type=int, default=78)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--inner_tol", type=float, default=1e-2)
parser.add_argument("--inner_max_linesearch", type=int, default=5)
parser.add_argument("--history_length", type=int, default=40)
args = parser.parse_args()

print("JAX devices:", jax.devices())

data = InclineRunning(
    path=args.data_dir,
    subsample=args.subsample,
    feet=args.feet,
    target="markers",
    inclines="all",
)
x_train = jnp.asarray(data.x_train)
y_train = jnp.asarray(data.y_train)
n, d, group_size = x_train.shape
_, o = y_train.shape
chunk_size = min(args.chunk_size, o)
print(f"n={n} d={d} g={group_size} o={o} chunk={chunk_size}")

# the same problem fit() hands the inner solver on its first x-update
design = rearrange(x_train, "n d g -> n (d g)")
lower, upper = hetgpy_auto_bounds(x_train)
unbounded = jnp.full((o, 1), jnp.inf)
solver_lower = jnp.concat([jnp.zeros((o, d * group_size)), -unbounded], axis=-1)
solver_upper = jnp.concat(
    [repeat(jnp.sqrt(2.0 / lower), "d g -> o (d g)", o=o), unbounded], axis=-1
)
theta_init = repeat(jnp.sqrt(2.0 / (0.9 * upper + 0.1 * lower)), "d g -> o (d g)", o=o)
g_min, g_max = jnp.array(G_RANGE[0]), jnp.array(G_RANGE[1])
w_init = jnp.full((o, 1), w_from_nugget(jnp.array(0.1), g_min, g_max))

keep = repeat(1.0 - jnp.eye(d), "k d -> k (d g)", g=group_size)
masked_designs = design[None, :, :] * keep[:, None, :]
mask = autoregressive_mask(o, d, group_size)
flat_mask = jnp.concat(
    [repeat(mask, "o d -> o (d g)", g=group_size), jnp.ones((o, 1))], axis=-1
)
group_of_output = jnp.arange(o) // group_size

x0 = jnp.concat([theta_init, w_init], axis=-1) * flat_mask
target = theta_init
rho = jnp.array(1.0)


def build(traced: bool):
    def solve_outputs(x0, target, rho, maxiter):
        def solve_one(carried):
            x0_i, target_i, y_i, group_i, mask_i, lower_i, upper_i = carried
            solution = lbfgsb_minimise(
                x_update_loss,
                x0_i,
                (lower_i, upper_i),
                args=((target_i, rho, masked_designs[group_i], y_i, g_min, g_max),),
                tol=args.inner_tol,
                max_iterations=maxiter,
                history_length=args.history_length,
                max_linesearch=args.inner_max_linesearch,
            )
            return solution.x * mask_i, solution.iteration

        return jax.lax.map(
            solve_one,
            (
                x0,
                target,
                y_train.T,
                group_of_output,
                flat_mask,
                solver_lower,
                solver_upper,
            ),
            batch_size=chunk_size,
        )

    return eqx.filter_jit(solve_outputs)


print(
    f"\n{'budget':>7} {'mode':>7} {'compile':>9} {'best':>9} {'median':>9}"
    f" {'iters mean':>11} {'iters max':>10}"
)

for budget in args.budgets:
    for mode in args.modes:
        traced = mode == "traced"
        solve_outputs = build(traced)
        bound = jnp.array(budget) if traced else budget

        start = time.perf_counter()
        lowered = solve_outputs.lower(x0, target, rho, bound).compile()
        compile_time = time.perf_counter() - start

        runs = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            solution, iterations = jax.block_until_ready(
                solve_outputs(x0, target, rho, bound)
            )
            runs.append(time.perf_counter() - start)
        runs.sort()

        print(
            f"{budget:>7} {mode:>7}"
            f" {compile_time:>8.2f}s {runs[0]:>8.2f}s {runs[len(runs) // 2]:>8.2f}s"
            f" {float(iterations.mean()):>11.2f} {int(iterations.max()):>10}"
        )
