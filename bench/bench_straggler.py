"""Per-output inner-solver iteration counts for one ADMM iteration: is the vmap ragged?"""

import argparse
import functools
import time

print = functools.partial(print, flush=True)

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from feetgp.gp import GroupLassoGaussianProcess, hetgpy_auto_bounds
from feetgp.inclinerunning import InclineRunning

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/Incline Running")
parser.add_argument("--subsample", type=int, default=20)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--l1_penalty", type=float, default=0.0)
parser.add_argument("--chunks", type=int, nargs="+", default=[78, 39, 13, 1])
parser.add_argument("--inner_maxiter", type=int, default=5)
parser.add_argument("--inner_maxiter_init", type=int, default=5)
parser.add_argument("--inner_tol", type=float, default=1e-2)
parser.add_argument("--inner_max_linesearch", type=int, default=5)
args = parser.parse_args()

print("JAX devices:", jax.devices())

data = InclineRunning(
    path=args.data_dir,
    subsample=args.subsample,
    feet=args.feet,
    target=args.target,
    inclines="all",
)
x_train = jnp.asarray(data.x_train)
y_train = jnp.asarray(data.y_train)
n, d, group_size = x_train.shape
_, o = y_train.shape
print(f"n={n} d={d} g={group_size} o={o} lambda={args.l1_penalty:g}")

auto_bounds = hetgpy_auto_bounds(x_train)


def one_admm_iteration(chunk_size: int) -> None:
    start = time.perf_counter()
    _, _, state, info = GroupLassoGaussianProcess.fit(
        x_train=x_train,
        y_train=y_train,
        l1_penalty=jnp.array(args.l1_penalty),
        autoregressive=args.target == "markers",
        auto_bounds=auto_bounds,
        max_iterations=1,
        chunk_size=chunk_size,
        inner_maxiter=args.inner_maxiter,
        inner_maxiter_init=args.inner_maxiter_init,
        inner_pgtol=args.inner_tol,
        inner_max_linesearch=args.inner_max_linesearch,
        collect_inner_stats=True,
    )
    jax.block_until_ready(state.z)
    wall = time.perf_counter() - start

    stats = info["inner_stats"][0]
    iterations = stats["iterations"]
    n_fun_eval = stats["n_fun_eval"]
    n_chunks = int(np.ceil(o / chunk_size))
    padded = np.pad(
        iterations.astype(float),
        (0, n_chunks * chunk_size - o),
        constant_values=np.nan,
    ).reshape(n_chunks, chunk_size)
    chunk_maxima = np.nanmax(padded, axis=1)

    # what the ragged exit costs: sum of chunk maxima against the work actually wanted
    served = float(chunk_maxima.sum() * chunk_size)
    useful = float(iterations.sum())

    print(f"\n=== chunk_size={chunk_size} ({n_chunks} chunks) wall={wall:.1f}s")
    print(f"inner iterations per output: {np.bincount(iterations, minlength=1)}")
    print(
        f"  min={iterations.min()} median={np.median(iterations):.1f}"
        f" max={iterations.max()} mean={iterations.mean():.2f}"
    )
    print(
        f"  n_fun_eval min={n_fun_eval.min()} median={np.median(n_fun_eval):.1f}"
        f" max={n_fun_eval.max()} mean={n_fun_eval.mean():.2f}"
    )
    print(f"  failed_linesearch: {int(stats['failed_linesearch'].sum())}/{o}")
    print(f"  chunk maxima: {chunk_maxima.astype(int)}")
    print(
        f"  straggler waste: {served:.0f} lane-iterations served for {useful:.0f}"
        f" useful ({served / max(useful, 1):.2f}x)"
    )


for chunk_size in args.chunks:
    if chunk_size <= o:
        one_admm_iteration(chunk_size)
