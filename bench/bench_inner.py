"""Where does the inner L-BFGS budget actually go?

Scratch diagnostic, not part of the pipeline. The real fit caps at inner_maxiter=1000
every ADMM iteration from iter 7 on, which either means the outputs genuinely need 1000
steps or means the termination criterion never fires. This tells them apart, per output,
and prices a looser inner tolerance.
"""

import argparse
import functools
import time

print = functools.partial(print, flush=True)

import jax
import jax.numpy as jnp
import optimistix as optx

jax.config.update("jax_enable_x64", True)

from feetgp import admm
from feetgp.admm import EPS, ADMMState
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
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--l1_penalty", type=float, default=0.0)
parser.add_argument("--warm_iters", type=int, nargs="+", default=[0, 3, 8])
parser.add_argument("--tols", type=float, nargs="+", default=[EPS, 1e-6, 1e-4, 1e-3])
parser.add_argument("--inner_maxiter", type=int, default=1000)
parser.add_argument("--chunk_size", type=int, default=32)
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
autoregressive = args.target == "markers"
print(f"n={n} d={d} o={o} group_size={group_size} lambda={args.l1_penalty:g}")

auto_bounds = hetgpy_auto_bounds(x_train)
n_groups = d // group_size
if autoregressive:
    group_columns = jnp.arange(d)[None, :] // group_size
    keep = (jnp.arange(n_groups)[:, None] != group_columns).astype(x_train.dtype)
    masked_designs = x_train[None, :, :] * keep[:, None, :]
    mask = autoregressive_mask(o, d, group_size)
else:
    masked_designs = jnp.broadcast_to(x_train, (n_groups, *x_train.shape))
    mask = jnp.ones((o, d + 1))


g_min, g_max = jnp.array(1e-4), jnp.array(100.0)
lower, upper = auto_bounds
theta_max = admm.to_groups(jnp.broadcast_to(jnp.sqrt(2.0 / lower), (o, d)), group_size)
bounds = jnp.stack([jnp.zeros_like(theta_max), theta_max])
l1 = jnp.array(args.l1_penalty)


def instrumented_x_update(state: ADMMState, rtol: float, maxiter: int):
    """Same solve as x_update_solve, but keeps num_steps and the loss reached."""
    solver = optx.LBFGS(rtol=rtol, atol=rtol, history_length=40)
    x0 = jnp.concat(
        [admm.to_outputs(state.x, group_size), state.aux[..., None]], axis=-1
    )
    target = admm.to_outputs(state.z - state.u, group_size)
    group_of_output = jnp.arange(o) // group_size

    def solve_one_output(args_):
        x0_i, target_i, y_i, group_i, mask_i = args_
        loss_args = (target_i, state.rho, masked_designs[group_i], y_i, g_min, g_max)
        solution = optx.minimise(
            admm_x_update_loss,
            solver,
            x0_i * mask_i,
            args=loss_args,
            max_steps=maxiter,
            throw=False,
        )
        loss = admm_x_update_loss(solution.value, loss_args)
        return solution.value * mask_i, solution.stats["num_steps"], loss

    return jax.lax.map(
        solve_one_output,
        (x0, target, y_train.T, group_of_output, mask),
        batch_size=args.chunk_size,
    )


def fresh_state() -> ADMMState:
    theta_init = admm.to_groups(
        jnp.broadcast_to(jnp.sqrt(2.0 / (0.9 * upper + 0.1 * lower)), (o, d)),
        group_size,
    )
    w_init = jnp.full((o,), w_from_nugget(jnp.array(0.1), g_min, g_max))
    return ADMMState.initialize(theta_init, aux=w_init)


# walk the real ADMM recursion so the later probes see a realistic z/u/rho, not the init
state = fresh_state()
for warm in range(max(args.warm_iters) + 1):
    if warm in args.warm_iters:
        print()
        print(f"--- after {warm} ADMM iterations (rho={float(state.rho):g}) ---")
        print(
            f"{'inner rtol':>11} {'wall':>8} {'steps mean/med/max':>20} {'hit cap':>8} {'loss sum':>16}"
        )
        for rtol in args.tols:
            start = time.perf_counter()
            _, steps, loss = instrumented_x_update(state, rtol, args.inner_maxiter)
            jax.block_until_ready(steps)
            wall = time.perf_counter() - start
            steps = jnp.asarray(steps)
            at_cap = int(jnp.sum(steps >= args.inner_maxiter))
            print(
                f"{rtol:>11.2e} {wall:>7.1f}s"
                f" {float(jnp.mean(steps)):>6.0f}/{float(jnp.median(steps)):>5.0f}"
                f"/{int(jnp.max(steps)):>5d} {at_cap:>4d}/{o} {float(jnp.sum(loss)):>16.4f}"
            )
        # the chunk pays the max over its members, not the mean: quantify the tax
        _, steps, _ = instrumented_x_update(state, EPS, args.inner_maxiter)
        steps = jnp.asarray(steps).reshape(-1)
        chunks = [steps[i : i + args.chunk_size] for i in range(0, o, args.chunk_size)]
        paid = sum(int(jnp.max(c)) * len(c) for c in chunks)
        print(
            f"straggler tax at chunk_size={args.chunk_size}:"
            f" paid {paid} output-steps vs {int(jnp.sum(steps))} needed"
            f" ({paid / max(int(jnp.sum(steps)), 1):.2f}x)"
        )

    solution = x_update_solve(
        jnp.concat(
            [admm.to_outputs(state.x, group_size), state.aux[..., None]], axis=-1
        ),
        admm.to_outputs(state.z - state.u, group_size),
        state.rho,
        masked_designs,
        y_train,
        group_size,
        mask=mask,
        g_min=g_min,
        g_max=g_max,
        maxiter=min(args.inner_maxiter, 20 * 2**warm),
        chunk_size=args.chunk_size,
    )
    theta = admm.to_groups(solution[:, :-1], group_size)
    new_state = state._replace(x=jnp.clip(theta, *bounds), aux=solution[:, -1])
    new_state = admm.z_and_u_update(new_state, l1, bounds=bounds)
    state, _, _ = admm.check_residuals(new_state, state, jnp.array(1e-3))
