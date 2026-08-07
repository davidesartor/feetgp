
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

from feetgp.gp import (
    GroupLassoGaussianProcess,
    admm_state_from_pickle,
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
parser.add_argument("--warm_lambda", type=float, default=None)
parser.add_argument("--knot_lambda", type=float, required=True)
parser.add_argument("--subsample", type=int, default=20)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--ungroup_feet", action="store_true", default=False)
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--maxiter", type=int, default=60)
parser.add_argument("--chunk_size", type=int, default=39)
parser.add_argument("--inner_maxiters", type=int, nargs="+", default=[5])
parser.add_argument("--inner_max_linesearch", type=int, default=5)
parser.add_argument("--inner_tols", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
args = parser.parse_args()

data = InclineRunning(
    subsample=args.subsample,
    feet=args.feet,
    target=args.target,
    inclines="all",
)
x_train = jnp.asarray(data.x_train)
y_train = jnp.asarray(data.y_train)
group_size = 6 if (args.feet == "both" and not args.ungroup_feet) else 3
autoregressive = args.target == "markers"
auto_bounds = hetgpy_auto_bounds(x_train)

warm = None
if args.warm_lambda is not None:
    warm_path, warm_value = nearest_cached(args.run_dir, args.warm_lambda)
    with open(warm_path, "rb") as f:
        warm = admm_state_from_pickle(pickle.load(f))
    print(f"warmstart {warm_path} (lambda={warm_value:.6g}, rho={float(warm.rho):.4g})")
else:
    print("cold start")


def run(inner_tol: float, inner_maxiter: int) -> None:
    label = f"lbfgsb inner_tol={inner_tol:g} inner_maxiter={inner_maxiter}"
    print(f"\n=== {label}")
    t0 = time.perf_counter()
    model, llk, state, info = GroupLassoGaussianProcess.fit(
        x_train=x_train,
        y_train=y_train,
        l1_penalty=jnp.array(args.knot_lambda),
        group_size=group_size,
        autoregressive=autoregressive,
        warmstart=warm,
        auto_bounds=auto_bounds,
        max_iterations=args.maxiter,
        tol=jnp.array(1e-3),
        chunk_size=args.chunk_size,
        inner_maxiter=inner_maxiter,
        inner_pgtol=inner_tol,
        inner_max_linesearch=args.inner_max_linesearch,
        log_every=10,
    )
    dt = time.perf_counter() - t0
    theta = model.theta.reshape(model.theta.shape[0], -1, group_size)
    norms = jnp.linalg.norm(theta, axis=(0, 2))
    n_active = int((norms > 0).sum())
    iterations = max(int(info["iterations"]), 1)
    print(
        f"{label}: primal={float(info['primal_residual']):.4e} "
        f"dual={float(info['dual_residual']):.4e} "
        f"converged={bool(info['converged'])} in {iterations} iters, "
        f"{dt:.1f}s ({dt / iterations:.2f}s/iter), "
        f"active={n_active}/{norms.shape[0]} loglik={float(llk):.6g}"
    )


for inner_tol in args.inner_tols:
    for inner_maxiter in args.inner_maxiters:
        run(inner_tol, inner_maxiter)
