"""How long is one ADMM fit at a single lambda, and what does chunk_size buy?

Scratch diagnostic, not part of the pipeline. Run it on the devices the real jobs use.
The inner L-BFGS budget ramps, so per-iteration cost is not flat; the total is what
multiplies out over a lambda sweep.
"""

import argparse
import functools
import time

# a job that hits its walltime should still show the rows it did finish
print = functools.partial(print, flush=True)

import jax
import jax.numpy as jnp
from einops import rearrange

jax.config.update("jax_enable_x64", True)

from feetgp.glassogp import GroupLassoGaussianProcess, hetgpy_auto_bounds
from feetgp.inclinerunning import InclineRunning

parser = argparse.ArgumentParser()
parser.add_argument("--subsample", type=int, default=20)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--ungroup_feet", action="store_true", default=False)
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--l1_penalty", type=float, default=0.0)
# one row per budget, so a job that hits its walltime still shows the cost curve
parser.add_argument("--maxiters", type=int, nargs="+", default=[5, 10, 20, 40, 80])
parser.add_argument("--tol", type=float, default=1e-3)
parser.add_argument("--chunks", type=int, nargs="+", default=[8, 32])
parser.add_argument("--inner_maxiter", type=int, default=50)
parser.add_argument("--inner_tol", type=float, default=1e-4)
parser.add_argument("--history_length", type=int, default=40)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--theta_scale", type=float, default=1.0)
# fraction of the run after which rho stops adapting; 0.5 is what fit() defaults to
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

# shared across every row, so no row pays the n*n cdist
auto_bounds = hetgpy_auto_bounds(x_train)
# widen the theta box by theta_scale: theta_max = sqrt(2 / lower)
auto_bounds = (auto_bounds[0] / args.theta_scale**2, auto_bounds[1])

# each row continues the previous row's ADMM state rather than refitting, so the rows
# are a convergence curve of one fit and the last row's wall is the real cost of it
total_iterations = max(args.maxiters)
for chunk in args.chunks:
    if chunk > o:
        continue
    print()
    print(
        f"chunk_size={chunk} inner_maxiter={args.inner_maxiter} "
        f"inner_tol={args.inner_tol} alpha={args.alpha}"
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
            # keep the rho freeze at the same absolute iteration the whole run would use
            adapt_rho_iters=max(
                int(total_iterations * args.adapt_rho_frac) - previous_iters, 0
            ),
            tol=jnp.array(args.tol),
            chunk_size=chunk,
            inner_maxiter=args.inner_maxiter,
            alpha=args.alpha,
            inner_rtol=args.inner_tol,
            inner_atol=args.inner_tol,
            history_length=args.history_length,
            log_every=args.log_every,
        )
        jax.block_until_ready(state.z)
        cumulative += time.perf_counter() - start
        groups = rearrange(model.theta, "o (d g) -> d (o g)", g=group_size)
        n_active = int(jnp.sum(jnp.linalg.norm(groups, axis=-1) > 1e-8))
        # how much of the primal residual is x sitting outside the theta box
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
