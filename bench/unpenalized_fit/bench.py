"""Wall time of an unpenalized GP fit as the training set grows."""

import argparse
import os
import time
import jax
import jax.numpy as jnp
from einops import reduce

from bench.common import chip_name, eps_multiple, time_call, write_row
from feetgp.gp import GaussianProcess
from feetgp.inclinerunning import InclineRunning

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/Incline Running")
    parser.add_argument(
        "--train_sizes", type=int, nargs="+", default=[128, 256, 512, 1024]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float64"]
    )
    parser.add_argument(
        "--profile", type=str, default="rbf", choices=["rbf", "matern52"]
    )
    parser.add_argument(
        "--feet", type=str, default="both", choices=["both", "left_only", "right_only"]
    )
    parser.add_argument(
        "--target", type=str, default="markers", choices=["markers", "forces"]
    )
    parser.add_argument(
        "--inclines", type=str, default="inc0", choices=["all", "inc0", "inc5", "inc10"]
    )
    parser.add_argument("--ungroup_feet", action="store_true", default=False)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--max_repeats", type=int, default=20)
    parser.add_argument("--time_budget_s", type=float, default=0.0)
    return parser.parse_args()


def sanity_checks(
    model, state, certificate, n_groups: int, max_iterations: int
) -> dict:
    """At lambda = 0 nothing may be shrunk away and the fit must be stationary."""
    group_norms = jnp.sqrt(reduce(model.theta**2, "o d g -> g", "sum"))
    n_active = int(jnp.sum(group_norms > 1e-8))
    dtype = model.theta.dtype
    return dict(
        n_active=n_active,
        max_kkt=float(jnp.abs(certificate).max()),
        all_groups_alive=n_active == n_groups,
        finite=bool(jnp.isfinite(model.theta).all() & jnp.isfinite(model.nu).all()),
        converged=int(state.iteration) < max_iterations,
        primal_residual_eps=eps_multiple(state.primal_residual, dtype),
        dual_residual_eps=eps_multiple(state.dual_residual, dtype),
    )


if __name__ == "__main__":
    args = parse_args()
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    print("JAX devices:", jax.devices(), flush=True)
    chip = chip_name(jax.devices()[0])
    deadline = time.perf_counter() + args.time_budget_s if args.time_budget_s else 0.0

    for train_size in args.train_sizes:
        data = InclineRunning(
            path=args.data_dir,
            train_size=train_size,
            seed=args.seed,
            feet=args.feet,
            target=args.target,
            inclines=args.inclines,
            ungroup_feet=args.ungroup_feet,
        )
        x_train = jnp.asarray(data.x_train, dtype=args.dtype)
        n, d, g = x_train.shape

        y_train = jnp.asarray(data.y_train, dtype=args.dtype)
        o = y_train.shape[1]

        row = dict(
            chip=chip, dtype=args.dtype, profile=args.profile,
            n=n, d=d, g=g, o=o,
        )  # fmt: skip

        # full fit at lambda = 0, default iteration cap and tolerance
        def fit():
            return GaussianProcess.fit(
                x_train,
                y_train,
                jnp.zeros((), dtype=args.dtype),
                profile=args.profile,
                warmstart=None,
                max_iterations=args.maxiter,
            )

        # one row per repeat, so a job killed mid-sweep still leaves its measurements
        for repeat in range(args.max_repeats):
            try:
                seconds, (model, nll, state, certificate) = time_call(fit)
            except Exception as error:  # out of memory is the expected wall here
                print(f"n={n} o={o} FAILED: {type(error).__name__}")
                print(f"    {str(error).splitlines()[0]}", flush=True)
                write_row(
                    RESULTS_PATH, row | dict(repeat=repeat, status=type(error).__name__)
                )
                break

            checks = sanity_checks(model, state, certificate, g, args.maxiter)
            print(
                f"n={n} d={d} g={g} o={o} {args.profile} repeat={repeat}"
                f" fit={seconds:.2f}s"
                f" admm_iters={int(state.iteration)}"
                f" nll={float(nll):.4f} max_kkt={checks['max_kkt']:.4f}"
                f" active={checks['n_active']}/{g}"
                f" residuals={checks['primal_residual_eps']:.3g}"
                f"/{checks['dual_residual_eps']:.3g} eps",
                flush=True,
            )
            write_row(
                RESULTS_PATH,
                row
                | dict(
                    repeat=repeat,
                    fit_s=round(seconds, 3),
                    admm_iters=int(state.iteration),
                    nll=float(nll),
                    status="ok",
                )
                | checks,
            )

            # stop once another fit of the same cost would run past the walltime cap
            if deadline and time.perf_counter() + seconds > deadline:
                print(f"n={n} budget spent after {repeat + 1} repeats", flush=True)
                break

        del data, x_train, y_train
