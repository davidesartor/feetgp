"""Does lambda_max actually kill every group, and what does that fit cost?"""

import argparse
import os
import time
import jax
import jax.numpy as jnp
from einops import reduce

from bench.common import chip_name, eps_multiple, time_call, write_row
from feetgp.glasso_admm import ADMMState
from feetgp.gp import GaussianProcess
from feetgp.linear import Linear
from feetgp.inclinerunning import InclineRunning

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")
DTYPE = "float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/Incline Running")
    parser.add_argument(
        "--train_sizes", type=int, nargs="+", default=[64, 128, 256, 512]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile", type=str, default="rbf", choices=["rbf", "matern52"]
    )
    parser.add_argument("--linear_model", action="store_true", default=False)
    parser.add_argument(
        "--feet", type=str, default="both", choices=["both", "left_only", "right_only"]
    )
    parser.add_argument(
        "--target", type=str, default="forces", choices=["markers", "forces"]
    )
    parser.add_argument(
        "--inclines", type=str, default="inc0", choices=["all", "inc0", "inc5", "inc10"]
    )
    parser.add_argument("--ungroup_feet", action="store_true", default=False)
    parser.add_argument("--maxiter", type=int, default=300)
    # just above lambda_max theta=0 must be stationary, just below it must be escapable
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[2.0, 1.05, 0.95, 0.5]
    )
    # gradient evaluated at epsilon * theta_init; sweep checks lambda_max stability
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--max_repeats", type=int, default=1)
    parser.add_argument("--time_budget_s", type=float, default=0.0)
    return parser.parse_args()


def n_active_groups(model) -> int:
    norms = jnp.sqrt(reduce(model.theta**2, "... g -> g", "sum"))
    return int(jnp.sum(norms > 1e-8))


if __name__ == "__main__":
    args = parse_args()
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
        x_train = jnp.asarray(data.x_train, dtype=DTYPE)
        y_train = jnp.asarray(data.y_train, dtype=DTYPE)
        n, d, g = x_train.shape
        o = y_train.shape[1]

        model = Linear if args.linear_model else GaussianProcess
        model_kwargs = {} if args.linear_model else dict(profile=args.profile)
        row = dict(
            chip=chip, dtype=DTYPE, model="linear" if args.linear_model else "gp",
            profile=args.profile, target=args.target, n=n, d=d, g=g, o=o,
        )  # fmt: skip

        # linear lambda_max has no degenerate zero, so no epsilon to sweep
        epsilons = [0.0] if args.linear_model else args.epsilons
        for epsilon in epsilons:
            lambda_kwargs = model_kwargs if args.linear_model else model_kwargs | dict(epsilon=epsilon)  # fmt: skip
            try:
                lambda_seconds, lambda_max = time_call(
                    lambda: model.lambda_max(x_train, y_train, **lambda_kwargs)
                )
            except Exception as error:
                message = str(error).splitlines()[0]
                print(
                    f"n={n} eps={epsilon} lambda_max FAILED:"
                    f" {type(error).__name__}: {message}",
                    flush=True,
                )
                write_row(
                    RESULTS_PATH,
                    row | dict(epsilon=epsilon, status=type(error).__name__, error=message),  # fmt: skip
                )
                continue

            lambda_max = float(lambda_max)
            print(
                f"n={n} d={d} g={g} o={o} eps={epsilon}"
                f" lambda_max={lambda_max:.6g} in {lambda_seconds:.2f}s",
                flush=True,
            )

            # lambda_max only certifies theta=0 as stationary, so above it start
            # there; below it a cold start must be able to escape
            dead = ADMMState(
                x=jnp.zeros((o, d, g), dtype=DTYPE),
                z=jnp.zeros((o, d, g), dtype=DTYPE),
                u=jnp.zeros((o, d, g), dtype=DTYPE),
                aux=(jnp.zeros((o,), dtype=DTYPE),),
            )

            for ratio in args.ratios:
                l1_penalty = jnp.asarray(ratio * lambda_max, dtype=DTYPE)

                def fit():
                    return model.fit(
                        x_train,
                        y_train,
                        l1_penalty,
                        max_iterations=args.maxiter,
                        warmstart=dead if ratio >= 1.0 else None,
                        **model_kwargs,
                    )

                for repeat in range(args.max_repeats):
                    try:
                        seconds, (fitted, nll, state, certificate) = time_call(fit)
                    except Exception as error:  # out of memory is the expected wall
                        print(f"n={n} ratio={ratio} FAILED: {type(error).__name__}")
                        print(f"    {str(error).splitlines()[0]}", flush=True)
                        write_row(
                            RESULTS_PATH,
                            row | dict(epsilon=epsilon, ratio=ratio, repeat=repeat, status=type(error).__name__),  # fmt: skip
                        )
                        break

                    # above lambda_max theta=0 must be a fixed point, below it not
                    n_active = n_active_groups(fitted)
                    passed = (n_active == 0) if ratio >= 1.0 else (n_active >= 1)
                    stationary = float(jnp.max(certificate)) <= 0.0
                    passed = passed and (stationary if ratio >= 1.0 else True)
                    converged = int(state.iteration) < args.maxiter
                    primal_eps = eps_multiple(state.primal_residual, DTYPE)
                    dual_eps = eps_multiple(state.dual_residual, DTYPE)
                    print(
                        f"n={n} eps={epsilon} ratio={ratio}"
                        f" lambda={float(l1_penalty):.6g} repeat={repeat}"
                        f" fit={seconds:.2f}s admm_iters={int(state.iteration)}"
                        f" active={n_active}/{g}"
                        f" (expected {'0' if ratio >= 1.0 else '>=1'})"
                        f" converged={converged}"
                        f" residuals={primal_eps:.3g}/{dual_eps:.3g} eps"
                        f" max_kkt={float(jnp.max(certificate)):.4g}"
                        f" {'PASS' if passed else 'FAIL'}",
                        flush=True,
                    )
                    write_row(
                        RESULTS_PATH,
                        row
                        | dict(
                            epsilon=epsilon,
                            ratio=ratio,
                            l1_penalty=float(l1_penalty),
                            lambda_max=lambda_max,
                            lambda_max_s=round(lambda_seconds, 3),
                            repeat=repeat,
                            fit_s=round(seconds, 3),
                            admm_iters=int(state.iteration),
                            converged=converged,
                            primal_residual_eps=primal_eps,
                            dual_residual_eps=dual_eps,
                            max_kkt=float(jnp.max(certificate)),
                            nll=float(nll),
                            n_active=n_active,
                            stationary=stationary,
                            passed=passed,
                            status="ok",
                        ),
                    )

                    if deadline and time.perf_counter() + seconds > deadline:
                        print(f"n={n} budget spent", flush=True)
                        break

        del data, x_train, y_train
