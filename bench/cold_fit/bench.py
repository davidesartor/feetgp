"""Wall time of a cold unregularized GP fit as the training set grows."""

import argparse
import csv
import fcntl
import os
import platform
import time
import jax
import jax.numpy as jnp

from feetgp.gp import AutoregressiveGaussianProcess, GaussianProcess
from feetgp.inclinerunning import InclineRunning

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")

FIELDS = [
    "chip",
    "dtype",
    "profile",
    "n",
    "d",
    "g",
    "o",
    "maxiter",
    "repeat",
    "fit_s",
    "per_admm_iter_s",
    "nll",
    "max_kkt",
    "status",
]


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
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--max_repeats", type=int, default=20)
    parser.add_argument("--time_budget_s", type=float, default=0.0)
    return parser.parse_args()


def time_fit(fit) -> tuple[float, tuple]:
    """Wall time of one blocking fit, repeat 0 also carrying the compile."""
    start = time.perf_counter()
    result = jax.block_until_ready(fit())
    return time.perf_counter() - start, result


def chip_name(device) -> str:
    """Device kind, except on cpu where every host reports the useless "cpu"."""
    if device.platform != "cpu":
        return device.device_kind
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "cpu"


def write_row(row: dict) -> None:
    """Append under an exclusive lock, so parallel array tasks share one csv."""
    with open(RESULTS_PATH, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    args = parse_args()
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    print("JAX devices:", jax.devices(), flush=True)
    chip = chip_name(jax.devices()[0])
    deadline = time.perf_counter() + args.time_budget_s if args.time_budget_s else 0.0

    autoregressive = args.target == "markers"
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

        if autoregressive:
            targets = ()
            o = d * g
        else:
            targets = (jnp.asarray(data.y_train, dtype=args.dtype),)
            o = targets[0].shape[1]

        model_cls = AutoregressiveGaussianProcess if autoregressive else GaussianProcess
        row = dict(
            chip=chip, dtype=args.dtype, profile=args.profile,
            n=n, d=d, g=g, o=o, maxiter=args.maxiter,
        )  # fmt: skip

        # lambda = 0 never trips the dual tolerance, so every fit runs the full budget
        def fit():
            return model_cls.fit(
                x_train,
                *targets,
                jnp.zeros((), dtype=args.dtype),
                profile=args.profile,
                warmstart=None,
                max_iterations=args.maxiter,
                tol=jnp.zeros((), dtype=args.dtype),
            )

        # one row per repeat, so a job killed mid-sweep still leaves its measurements
        for repeat in range(args.max_repeats):
            try:
                seconds, (_, nll, _, certificate) = time_fit(fit)
            except Exception as error:  # out of memory is the expected wall here
                print(f"n={n} o={o} FAILED: {type(error).__name__}")
                print(f"    {str(error).splitlines()[0]}", flush=True)
                write_row(row | dict(repeat=repeat, status=type(error).__name__))
                break

            print(
                f"n={n} d={d} g={g} o={o} {args.profile} repeat={repeat}"
                f" fit={seconds:.2f}s"
                f" per_admm_iter={seconds / args.maxiter:.3f}s"
                f" per_iter_per_output={1e3 * seconds / args.maxiter / o:.2f}ms"
                f" nll={float(nll):.4f} max_kkt={float(jnp.abs(certificate).max()):.4f}",
                flush=True,
            )
            write_row(
                row
                | dict(
                    repeat=repeat,
                    fit_s=round(seconds, 3),
                    per_admm_iter_s=round(seconds / args.maxiter, 4),
                    nll=float(nll),
                    max_kkt=float(jnp.abs(certificate).max()),
                    status="ok",
                )
            )

            # stop once another fit of the same cost would run past the walltime cap
            if deadline and time.perf_counter() + seconds > deadline:
                print(f"n={n} budget spent after {repeat + 1} repeats", flush=True)
                break

        del data, x_train, targets
