"""How fast is one batched x-update on this device, and how far does chunk_size pay?

Scratch diagnostic, not part of the pipeline. Run it on the same devices the real
jobs use; the x-update is essentially the whole cost of an ADMM iteration.
"""

import argparse
import functools
import time

# a job that hits its walltime should still show the rows it did finish
print = functools.partial(print, flush=True)

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from feetgp import admm
from feetgp.admm import ADMMState
from feetgp.glassogp import (
    admm_x_update_loss,
    autoregressive_mask,
    hetgpy_auto_bounds,
    w_from_nugget,
    x_update_solve,
)
from feetgp.inclinerunning import InclineRunning

parser = argparse.ArgumentParser()
parser.add_argument("--subsample", type=int, default=20)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--ungroup_feet", action="store_true", default=False)
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--l1_penalty", type=float, default=30.0)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
parser.add_argument("--history", type=int, nargs="+", default=[40])
args = parser.parse_args()

print("JAX devices:", jax.devices())

group_size = 6 if (args.feet == "both" and not args.ungroup_feet) else 3
data = InclineRunning(
    subsample=args.subsample, feet=args.feet, target=args.target, inclines="all"
)
x_train = jnp.asarray(data.x_train)
y_train = jnp.asarray(data.y_train)
n, d = x_train.shape
_, o = y_train.shape
autoregressive = args.target == "markers"
print(f"n={n} d={d} o={o} group_size={group_size} maxiter={args.maxiter}")


def best_of(fn, reps: int = 3) -> float:
    fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


############################################################
# Build the state fit() would, at the [theta, w] layout
############################################################
g_min, g_max = jnp.array(1e-4), jnp.array(100.0)
lower, upper = hetgpy_auto_bounds(x_train)
theta_max = admm.to_groups(jnp.broadcast_to(jnp.sqrt(2.0 / lower), (o, d)), group_size)
theta_init = admm.to_groups(
    jnp.broadcast_to(jnp.sqrt(2.0 / (0.9 * upper + 0.1 * lower)), (o, d)), group_size
)
# w is not in the consensus at all: it rides in aux, and the nugget saturates into
# [g_min, g_max], so only theta needs a box; see glassogp.fit
bounds = jnp.stack([jnp.zeros_like(theta_max), theta_max])
w_init = jnp.full((o,), w_from_nugget(jnp.array(0.1), g_min, g_max))
state = ADMMState.initialize(theta_init, aux=w_init)
l1 = jnp.array(args.l1_penalty)

n_groups = d // group_size
if autoregressive:
    group_columns = jnp.arange(d)[None, :] // group_size
    keep = (jnp.arange(n_groups)[:, None] != group_columns).astype(x_train.dtype)
    masked_designs = x_train[None, :, :] * keep[:, None, :]
    mask = autoregressive_mask(o, d, group_size)
else:
    masked_designs = jnp.broadcast_to(x_train, (n_groups, *x_train.shape))
    mask = None


############################################################
# One objective evaluation: the floor no solver step can beat
############################################################
def as_outputs(state: ADMMState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """(x with the w column appended, the x-update target), both per output."""
    x = jnp.concat(
        [admm.to_outputs(state.x, group_size), state.aux[..., None]], axis=-1
    )
    return x, admm.to_outputs(state.z - state.u, group_size)


x0, target = as_outputs(state)
eval_args = (target[0], state.rho, x_train, y_train[:, 0], g_min, g_max)
one_eval = best_of(
    lambda: jax.block_until_ready(admm_x_update_loss(x0[0], eval_args)), reps=10
)
print()
print(f"one objective eval  {one_eval * 1000:9.2f} ms")

############################################################
# A whole x-update at each chunk size
############################################################
# one ADMM cycle up front, so every timed row starts from the same realistic z and u
solution = x_update_solve(
    x0,
    target,
    state.rho,
    masked_designs,
    y_train,
    group_size,
    mask=mask,
    g_min=g_min,
    g_max=g_max,
    maxiter=args.maxiter,
    chunk_size=1,
)
theta = jnp.clip(admm.to_groups(solution[:, :-1], group_size), *bounds)
warm = admm.z_and_u_update(
    state._replace(x=theta, aux=solution[:, -1]), l1, bounds=bounds
)
warm_x0, warm_target = as_outputs(warm)

for history in args.history:
    print()
    print(f"history_length={history}")
    print(f"{'chunk':>7} {'x-update':>11} {'per output':>12} {'speedup':>9}")
    baseline = None
    for chunk in args.chunks:
        if chunk > o:
            continue
        try:
            wall = best_of(
                lambda: jax.block_until_ready(
                    x_update_solve(
                        warm_x0,
                        warm_target,
                        warm.rho,
                        masked_designs,
                        y_train,
                        group_size,
                        mask=mask,
                        g_min=g_min,
                        g_max=g_max,
                        maxiter=args.maxiter,
                        chunk_size=chunk,
                        history_length=history,
                    )
                ),
                reps=3,
            )
        except Exception as err:  # out of memory at this chunk size
            print(f"{chunk:>7}   {type(err).__name__}")
            break
        baseline = wall if baseline is None else baseline
        print(
            f"{chunk:>7} {wall:>10.2f}s {wall / o * 1000:>11.1f}ms"
            f" {baseline / wall:>8.2f}x"
        )
