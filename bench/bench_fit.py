
import argparse
import functools
import time

print = functools.partial(print, flush=True)

import jax
import jax.numpy as jnp
from einops import rearrange

jax.config.update("jax_enable_x64", True)

from feetgp.gp import GroupLassoGaussianProcess, hetgpy_auto_bounds
from feetgp.inclinerunning import InclineRunning

parser = argparse.ArgumentParser()
parser.add_argument("--subsample", type=int, default=20)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--ungroup_feet", action="store_true", default=False)
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--l1_penalty", type=float, default=0.0)
parser.add_argument("--maxiters", type=int, nargs="+", default=[5, 10, 20, 40, 80])
parser.add_argument("--tol", type=float, default=1e-3)
parser.add_argument("--chunks", type=int, nargs="+", default=[8, 32])
parser.add_argument("--inner_maxiter", type=int, default=5)
parser.add_argument("--inner_tol", type=float, default=1e-2)
parser.add_argument("--history_length", type=int, default=40)
parser.add_argument("--theta_scale", type=float, default=1.0)
parser.add_argument("--adapt_rho_frac", type=float, default=0.5)
parser.add_argument("--log_every", type=int, default=0)
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
print(
    f"n={n} d={d} o={o} group_size={group_size}"
    f" lambda={args.l1_penalty:g} tol={args.tol:g}"
)

auto_bounds = hetgpy_auto_bounds(x_train)
auto_bounds = (auto_bounds[0] / args.theta_scale**2, auto_bounds[1])

total_iterations = max(args.maxiters)
for chunk in args.chunks:
    if chunk > o:
        continue
    print()
    print(
        f"chunk_size={chunk} inner_maxiter={args.inner_maxiter} "
        f"inner_tol={args.inner_tol}"
        f" theta_scale={args.theta_scale}"
    )
    print(
        f"{'iters':>7} {'cumulative':>11} {'s/iter':>8} {'loglik':>13}"
        f" {'primal':>10} {'dual':>10} {'active':>8} {'r_box':>10} {'n_over':>6}"
    )
    warmstart, cumulative, previous_iters = None, 0.0, 0
    for maxiter in args.maxiters:
        start = time.perf_counter()
        model, llk, state, info = GroupLassoGaussianProcess.fit(
            x_train=x_train,
            y_train=y_train,
            l1_penalty=jnp.array(args.l1_penalty),
            group_size=group_size,
            autoregressive=args.target == "markers",
            auto_bounds=auto_bounds,
            warmstart=warmstart,
            max_iterations=maxiter - previous_iters,
            adapt_rho_iters=max(
                int(total_iterations * args.adapt_rho_frac) - previous_iters, 0
            ),
            tol=jnp.array(args.tol),
            chunk_size=chunk,
            inner_maxiter=args.inner_maxiter,
            inner_pgtol=args.inner_tol,
            history_length=args.history_length,
            log_every=args.log_every,
        )
        jax.block_until_ready(state.z)
        cumulative += time.perf_counter() - start
        groups = rearrange(model.theta, "o (d g) -> d (o g)", g=group_size)
        n_active = int(jnp.sum(jnp.linalg.norm(groups, axis=-1) > 1e-8))
        theta_max = jnp.sqrt(2.0 / auto_bounds[0])
        over = jnp.abs(state.x[:, :-1]) > theta_max
        r_box = float(
            jnp.linalg.norm(jnp.where(over, (state.x - state.z)[:, :-1], 0.0))
        )
        print(
            f"{maxiter:>7} {cumulative:>10.1f}s {cumulative / maxiter:>7.2f}s"
            f" {float(llk):>13.3f} {info['primal_residual']:>10.2e}"
            f" {info['dual_residual']:>10.2e} {n_active:>4}/{d // group_size}"
            f" {r_box:>10.2e} {int(over.sum()):>6}"
        )
        warmstart, previous_iters = state, maxiter
        if info["converged"]:
            print(f"converged at {maxiter} iterations")
            break
