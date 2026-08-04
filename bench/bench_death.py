"""Can a group reach exactly zero at high lambda, or does the path only shrink?

Chains fits upward from a cached state exactly as the sweep does, and reports the
active-group count and the two smallest group norms at each step.
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
import numpy as np

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
parser.add_argument("--lambdas", type=float, nargs="+", required=True)
parser.add_argument("--subsample", type=int, default=20)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--ungroup_feet", action="store_true", default=False)
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--maxiter", type=int, default=120)
parser.add_argument("--chunk_size", type=int, default=39)
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
    state = admm_state_from_pickle(pickle.load(f))
print(f"warmstart from lambda={args.warm_lambda:.6g}, rho={float(state.rho):.4g}")


def group_norms(z) -> np.ndarray:
    return np.asarray(jnp.linalg.norm(z, axis=-1))


for l1 in args.lambdas:
    t0 = time.perf_counter()
    _, _, state, info = GroupLassoGaussianProcess.fit(
        x_train=x_train,
        y_train=y_train,
        l1_penalty=jnp.array(l1),
        group_size=group_size,
        autoregressive=args.target == "markers",
        warmstart=state,
        auto_bounds=auto_bounds,
        max_iterations=args.maxiter,
        tol=jnp.array(1e-3),
        chunk_size=args.chunk_size,
        inner_maxiter=args.inner_maxiter,
        inner_rtol=1e-4,
        inner_atol=1e-4,
        log_every=0,
    )
    norms = np.sort(group_norms(state.z))
    dt = time.perf_counter() - t0
    print(
        f"lambda={l1:9.3f}  active={int((norms > 0).sum())}/{len(norms)}  "
        f"smallest={norms[0]:.4f},{norms[1]:.4f}  max={norms[-1]:.3f}  "
        f"rho={float(state.rho):.4g}  l1/rho={l1 / float(state.rho):.3f}  "
        f"conv={bool(info['converged'])} in {int(info['iterations'])} iters, {dt:.0f}s"
    )
