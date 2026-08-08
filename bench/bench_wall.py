"""Wall clock of one ADMM fit under the current parametrization, no cached pickles needed."""

import argparse
import functools
import time

print = functools.partial(print, flush=True)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from feetgp.gp import GroupLassoGaussianProcess, hetgpy_auto_bounds
from feetgp.inclinerunning import InclineRunning

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/Incline Running")
parser.add_argument("--subsample", type=int, default=20)
parser.add_argument("--feet", type=str, default="both")
parser.add_argument("--ungroup_feet", action="store_true", default=False)
parser.add_argument("--target", type=str, default="markers")
parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0])
parser.add_argument("--maxiter", type=int, default=300)
parser.add_argument("--tol", type=float, default=1e-3)
parser.add_argument("--chunk_size", type=int, default=39)
parser.add_argument("--inner_maxiter", type=int, default=5)
parser.add_argument("--inner_tol", type=float, default=1e-2)
parser.add_argument("--inner_max_linesearch", type=int, default=5)
parser.add_argument("--history_length", type=int, default=40)
parser.add_argument("--log_every", type=int, default=25)
args = parser.parse_args()

print("JAX devices:", jax.devices())

data = InclineRunning(
    path=args.data_dir,
    subsample=args.subsample,
    feet=args.feet,
    target=args.target,
    inclines="all",
    ungroup_feet=args.ungroup_feet,
)
x_train = jnp.asarray(data.x_train)
y_train = jnp.asarray(data.y_train)
n, d, group_size = x_train.shape
_, o = y_train.shape
print(f"n={n} d={d} g={group_size} o={o}")

auto_bounds = hetgpy_auto_bounds(x_train)
chunk_size = min(args.chunk_size, o)

print(
    f"{'lambda':>12} {'start':>8} {'iters':>6} {'wall':>9} {'s/iter':>8}"
    f" {'primal':>10} {'dual':>10} {'conv':>5} {'maxkkt':>10} {'active':>8}"
)


def timed(l1_penalty: float, warmstart, label: str):
    start = time.perf_counter()
    model, _, state, info = GroupLassoGaussianProcess.fit(
        x_train=x_train,
        y_train=y_train,
        l1_penalty=jnp.array(l1_penalty),
        autoregressive=args.target == "markers",
        warmstart=warmstart,
        auto_bounds=auto_bounds,
        max_iterations=args.maxiter,
        tol=jnp.array(args.tol),
        chunk_size=chunk_size,
        inner_maxiter=args.inner_maxiter,
        inner_pgtol=args.inner_tol,
        inner_max_linesearch=args.inner_max_linesearch,
        history_length=args.history_length,
        log_every=args.log_every,
    )
    jax.block_until_ready(state.z)
    wall = time.perf_counter() - start
    iterations = max(int(info["iterations"]), 1)
    certificate = info.get("certificate", {})
    max_kkt = float(certificate.get("max_live_kkt", jnp.nan))
    n_active = int(jnp.sum(jnp.linalg.norm(state.z, axis=-1) > 0))
    print(
        f"{l1_penalty:>12.4g} {label:>8} {iterations:>6} {wall:>8.1f}s"
        f" {wall / iterations:>7.2f}s {float(info['primal_residual']):>10.2e}"
        f" {float(info['dual_residual']):>10.2e} {str(bool(info['converged'])):>5}"
        f" {max_kkt:>10.2e} {n_active:>8}"
    )
    return state


# what a cold single-start fit costs at each lambda
for l1_penalty in args.lambdas:
    timed(l1_penalty, None, "cold")

# what a path actually pays per knot: chained, with run.py's rho reset off the lambda=0 handoff
warm = None
for l1_penalty in args.lambdas:
    label = "cold" if warm is None else "chained"
    warm = timed(l1_penalty, warm, label)
    if l1_penalty == 0.0:
        warm = warm._replace(rho=jnp.array(1.0))
