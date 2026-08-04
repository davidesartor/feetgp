"""Wall-clock of vlse's lbfgsb.minimise against scipy's Fortran L-BFGS-B on the vlse battery."""

import argparse
import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
import vlse
from scipy.optimize import Bounds, minimize as scipy_minimize

from vlse.lbfgsb import minimise

jax.config.update("jax_enable_x64", True)

TOL = 1e-9
MAX_ITERATIONS = 1000

# one instance per vlse function
BATTERY = [
    vlse.Ackley(d=5),
    vlse.Beale(),
    vlse.Bohachevsky(variant=1),
    vlse.Booth(),
    vlse.Branin(),
    vlse.Bukin6(),
    vlse.Camel3(),
    vlse.Camel6(),
    vlse.Colville(),
    vlse.CrossInTray(),
    vlse.DeJong5(),
    vlse.DixonPrice(d=5),
    vlse.DropWave(),
    vlse.Easom(),
    vlse.EggHolder(),
    vlse.Forrester(),
    vlse.ForresterLowFidelity(),
    vlse.GoldsteinPrice(),
    vlse.GramacyLee(),
    vlse.Griewank(d=4),
    vlse.Hartmann3(),
    vlse.Hartmann4(),
    vlse.Hartmann6(),
    vlse.HolderTable(),
    vlse.Langermann(),
    vlse.Levy(d=6),
    vlse.Levy13(),
    vlse.Matyas(),
    vlse.McCormick(),
    vlse.Michalewicz(d=5),
    vlse.Perm(d=4),
    vlse.Perm0(d=4),
    vlse.Powell(d=8),
    vlse.PowerSum(d=4),
    vlse.Rastrigin(d=4),
    vlse.Rosenbrock(d=6),
    vlse.RotatedHyperEllipsoid(d=5),
    vlse.Schaffer2(),
    vlse.Schaffer4(),
    vlse.Schwefel(d=3),
    vlse.Shekel(),
    vlse.Shubert(),
    vlse.Sphere(d=8),
    vlse.StyblinskiTang(d=5),
    vlse.SumPowers(d=4),
    vlse.SumSquares(d=6),
    vlse.Trid(d=6),
    vlse.Zakharov(d=5),
]


def box_of(f):
    return tuple(
        jnp.broadcast_to(jnp.asarray(bound, dtype=jnp.float64), (f.d,))
        for bound in f.domain
    )


def timed(call, repeats):
    """Median wall time of a call that has already been warmed up."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def scipy_solve(value_and_grad, x0, lower, upper):
    counter = [0]

    def numpy_value_and_grad(x):
        counter[0] += 1
        f, grad = value_and_grad(x)
        return float(f), np.asarray(grad, dtype=np.float64)

    result = scipy_minimize(
        numpy_value_and_grad,
        np.asarray(x0, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=Bounds(np.asarray(lower), np.asarray(upper)),
        options=dict(maxiter=MAX_ITERATIONS, maxcor=10, ftol=0.0, gtol=TOL),
    )
    return result.nit, counter[0]


def benchmark(f, n_starts, batch_size, repeats):
    lower, upper = box_of(f)
    rng = np.random.default_rng(0)
    starts = jnp.asarray(
        rng.uniform(np.asarray(lower), np.asarray(upper), size=(n_starts, f.d))
    )

    solve = jax.jit(
        lambda x0: minimise(
            f, x0, (lower, upper), tol=TOL, max_iterations=MAX_ITERATIONS
        )
    )
    compile_start = time.perf_counter()
    jax.block_until_ready(solve(starts[0]))
    compile_seconds = time.perf_counter() - compile_start

    # one solve at a time, which is what scipy is restricted to
    sequential = (
        timed(lambda: [jax.block_until_ready(solve(x0)) for x0 in starts], repeats)
        / n_starts
    )

    batched_starts = jnp.asarray(
        rng.uniform(np.asarray(lower), np.asarray(upper), size=(batch_size, f.d))
    )
    solve_batch = jax.jit(jax.vmap(solve))
    jax.block_until_ready(solve_batch(batched_starts))
    batched = (
        timed(lambda: jax.block_until_ready(solve_batch(batched_starts)), repeats)
        / batch_size
    )

    value_and_grad = jax.jit(jax.value_and_grad(f))
    jax.block_until_ready(value_and_grad(starts[0]))
    dispatch = timed(lambda: jax.block_until_ready(value_and_grad(starts[0])), 50)

    # scipy's own gradient stays on the host: charging it a GPU round trip per evaluation
    # would measure our dispatch overhead, not its solver
    host_value_and_grad = jax.jit(jax.value_and_grad(f), backend="cpu")
    jax.block_until_ready(host_value_and_grad(np.asarray(starts[0])))

    scipy_iterations, scipy_evals = [], []

    def run_scipy():
        scipy_iterations.clear()
        scipy_evals.clear()
        for x0 in starts:
            nit, nev = scipy_solve(host_value_and_grad, x0, lower, upper)
            scipy_iterations.append(nit)
            scipy_evals.append(nev)

    scipy_seconds = timed(run_scipy, repeats) / n_starts

    iterations = [int(solve(x0).iteration) for x0 in starts]
    return dict(
        name=type(f).__name__,
        d=int(f.d),
        compile=compile_seconds,
        sequential=sequential,
        batched=batched,
        scipy=scipy_seconds,
        dispatch=dispatch,
        iterations=float(np.mean(iterations)),
        scipy_iterations=float(np.mean(scipy_iterations)),
        scipy_evals=float(np.mean(scipy_evals)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="cpu")
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = jax.devices()[0]
    print(
        f"# tag={args.tag} device={device.device_kind} ({device.platform}) host={platform.node()}"
    )
    print(f"# starts={args.starts} batch={args.batch} repeats={args.repeats}")
    header = (
        f"{'function':22s} {'d':>2s} {'ours_ms':>8s} {'scipy_ms':>9s} {'speedup':>8s} "
        f"{'batch_us':>9s} {'b_speedup':>10s} {'disp_us':>8s} {'it':>6s} {'sp_it':>6s} {'sp_ev':>6s} {'jit_s':>6s}"
    )
    print(header)

    rows = []
    for f in BATTERY:
        row = benchmark(f, args.starts, args.batch, args.repeats)
        rows.append(row)
        print(
            f"{row['name']:22s} {row['d']:2d} {row['sequential'] * 1e3:8.3f} {row['scipy'] * 1e3:9.3f} "
            f"{row['scipy'] / row['sequential']:8.2f} {row['batched'] * 1e6:9.2f} "
            f"{row['scipy'] / row['batched']:10.1f} {row['dispatch'] * 1e6:8.1f} "
            f"{row['iterations']:6.1f} {row['scipy_iterations']:6.1f} {row['scipy_evals']:6.1f} {row['compile']:6.2f}",
            flush=True,
        )

    ours = np.array([r["sequential"] for r in rows])
    theirs = np.array([r["scipy"] for r in rows])
    batch = np.array([r["batched"] for r in rows])
    print(
        f"\n# totals: ours {ours.sum() * 1e3:.1f}ms  scipy {theirs.sum() * 1e3:.1f}ms  "
        f"batched {batch.sum() * 1e3:.3f}ms per solve summed over {len(rows)} functions"
    )
    print(
        f"# geomean speedup: sequential {np.exp(np.mean(np.log(theirs / ours))):.2f}x  "
        f"batched {np.exp(np.mean(np.log(theirs / batch))):.1f}x"
    )
    print(
        f"# mean iterations: ours {np.mean([r['iterations'] for r in rows]):.1f}  scipy {np.mean([r['scipy_iterations'] for r in rows]):.1f}"
    )
    print(f"# total compile: {sum(r['compile'] for r in rows):.1f}s")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                dict(tag=args.tag, device=device.device_kind, rows=rows),
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
