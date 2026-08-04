"""Does widening the x-update batch to a full vmap over outputs actually pay?

chunk_size == the output count is a full vmap; anything less is a sequential chain of
that many batches. Times a fixed number of ADMM iterations at each width, warmstarted
from a cached state so every setting solves the same problem from the same point.
"""

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

from feetgp.glassogp import (
    GroupLassoGaussianProcess,
    admm_state_from_pickle,
    hetgpy_auto_bounds,
)
from feetgp.inclinerunning import InclineRunning

def nearest_cached(run_dir: str, l1: float) -> tuple[str, float]:
    """Closest cached lambda to the one asked for, so a bench never dies on a typo."""
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
parser.add_argument("--l1", type=float, required=True)
parser.add_argument("--chunk_sizes", type=int, nargs="+", default=[26, 39, 78])
parser.add_argument("--subsample", type=int, default=20)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--ungroup_feet", action="store_true", default=False)
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--maxiter", type=int, default=10)
parser.add_argument("--inner_maxiter", type=int, default=50)
args = parser.parse_args()

data = InclineRunning(
    subsample=args.subsample, feet=args.feet, target=args.target, inclines="all"
)
x_train = jnp.asarray(data.x_train)
y_train = jnp.asarray(data.y_train)
group_size = 6 if (args.feet == "both" and not args.ungroup_feet) else 3
auto_bounds = hetgpy_auto_bounds(x_train)

warm_path, warm_value = nearest_cached(args.run_dir, args.warm_lambda)
print(f"warmstart file {warm_path} (lambda={warm_value:.6g})")
with open(warm_path, "rb") as f:
    warm = admm_state_from_pickle(pickle.load(f))
print(f"{y_train.shape[1]} outputs, n={x_train.shape[0]}, d={x_train.shape[1]}")

for chunk_size in args.chunk_sizes:
    t0 = time.perf_counter()
    _, _, _, info = GroupLassoGaussianProcess.fit(
        x_train=x_train,
        y_train=y_train,
        l1_penalty=jnp.array(args.l1),
        group_size=group_size,
        autoregressive=args.target == "markers",
        warmstart=warm,
        auto_bounds=auto_bounds,
        max_iterations=args.maxiter,
        tol=jnp.array(0.0),
        chunk_size=chunk_size,
        inner_maxiter=args.inner_maxiter,
        inner_rtol=1e-4,
        inner_atol=1e-4,
        log_every=0,
    )
    dt = time.perf_counter() - t0
    iterations = int(info["iterations"])
    print(
        f"chunk_size={chunk_size:3d}  {dt:7.1f}s total  "
        f"{dt / iterations:6.2f}s/iter over {iterations} iters  "
        f"primal={float(info['primal_residual']):.4e}"
    )
