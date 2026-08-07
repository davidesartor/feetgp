
import argparse
import glob
import re
import functools
import pickle
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
parser.add_argument("--warm_lambda", type=float, required=True)
parser.add_argument("--knot_lambda", type=float, required=True)
parser.add_argument("--subsample", type=int, default=20)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--ungroup_feet", action="store_true", default=False)
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--maxiter", type=int, default=60)
parser.add_argument("--chunk_size", type=int, default=39)
parser.add_argument("--inner_maxiters", type=int, nargs="*", default=[50, 200])
parser.add_argument("--rhos", type=float, nargs="+", default=[])
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

warm_path, warm_value = nearest_cached(args.run_dir, args.warm_lambda)
print(f"warmstart file {warm_path} (lambda={warm_value:.6g})")
with open(warm_path, "rb") as f:
    warm = admm_state_from_pickle(pickle.load(f))
print(f"warmstart from lambda={args.warm_lambda:.6g}, rho={float(warm.rho):.4g}")


def run(inner_maxiter: int, rho: float | None) -> None:
    state = warm if rho is None else warm._replace(rho=jnp.array(rho))
    label = f"inner_maxiter={inner_maxiter} rho={'warm' if rho is None else rho}"
    print(f"\n=== {label}")
    t0 = time.perf_counter()
    _, _, _, info = GroupLassoGaussianProcess.fit(
        x_train=x_train,
        y_train=y_train,
        l1_penalty=jnp.array(args.knot_lambda),
        group_size=group_size,
        autoregressive=autoregressive,
        warmstart=state,
        auto_bounds=auto_bounds,
        max_iterations=args.maxiter,
        tol=jnp.array(1e-3),
        chunk_size=args.chunk_size,
        inner_maxiter=inner_maxiter,
        inner_rtol=1e-4,
        inner_atol=1e-4,
        log_every=10,
    )
    dt = time.perf_counter() - t0
    print(
        f"{label}: primal={float(info['primal_residual']):.4e} "
        f"dual={float(info['dual_residual']):.4e} "
        f"converged={bool(info['converged'])} in {int(info['iterations'])} iters, "
        f"{dt:.1f}s ({dt / max(int(info['iterations']), 1):.2f}s/iter)"
    )


for inner_maxiter in args.inner_maxiters:
    run(inner_maxiter, None)
for rho in args.rhos:
    run(min(args.inner_maxiters, default=50), rho)
