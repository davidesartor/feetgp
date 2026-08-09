from typing import Callable
from jaxtyping import Array, Float, Scalar
from numpy.typing import NDArray

import os
import json
import argparse
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
from einops import rearrange, reduce
from sklearn.metrics import r2_score

from feetgp.glasso_admm import ADMMState
from feetgp.gp import GaussianProcess
from feetgp.linear import Linear
from feetgp.inclinerunning import InclineRunning
from feetgp.store import RunStore, model_to_arrays, state_to_arrays

print("JAX devices:", jax.devices())


Model = GaussianProcess | Linear


def git_revision() -> tuple[str | None, bool]:
    """Commit hash and whether the tree is dirty, both None/False outside git."""
    git_dir = os.path.dirname(os.path.abspath(__file__))

    def run(*command: str) -> str:
        result = subprocess.run(
            command, cwd=git_dir, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    try:
        return run("git", "rev-parse", "HEAD"), bool(
            run("git", "status", "--porcelain")
        )
    except (subprocess.CalledProcessError, OSError):
        return None, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data/Incline Running")
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--feet", type=str, default="both", choices=["both", "left_only", "right_only"]
    )
    parser.add_argument(
        "--target", type=str, default="markers", choices=["markers", "forces"]
    )
    parser.add_argument(
        "--inclines", type=str, default="all", choices=["all", "inc0", "inc5", "inc10"]
    )

    parser.add_argument("--linear_model", action="store_true", default=False)
    parser.add_argument(
        "--profile", type=str, default="rbf", choices=["rbf", "matern52"]
    )
    parser.add_argument("--ungroup_feet", action="store_true", default=False)
    parser.add_argument(
        "--relative", type=str, nargs="?", default=None, const="midpoint"
    )

    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--lambda_budget", type=int, default=40)
    parser.add_argument("--lambda_ratio", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def linear_model(
    args: argparse.Namespace,
    x_train: Float[Array, "n d g"],
    y_train: Float[Array, "n o"],
) -> tuple[Callable, Callable, Callable]:
    """Closed-form x update, so there is nothing to warmstart."""

    def fit(l1_penalty: float, warmstart: ADMMState | None = None):
        return Linear.fit(
            x_train,
            y_train,
            jnp.array(l1_penalty),
            max_iterations=args.maxiter,
            tol=jnp.array(args.tol),
        )

    def predict(model: Linear, x: Float[Array, "m d g"]) -> Float[NDArray, "m o"]:
        return np.asarray(model.predict(x))

    def lambda_max() -> float:
        return float(Linear.lambda_max(x_train, y_train))

    return fit, predict, lambda_max


def gp_model(
    args: argparse.Namespace,
    x_train: Float[Array, "n d g"],
    y_train: Float[Array, "n o"],
) -> tuple[Callable, Callable, Callable]:
    def fit(l1_penalty: float, warmstart: ADMMState | None = None):
        return GaussianProcess.fit(
            x_train,
            y_train,
            jnp.array(l1_penalty),
            profile=args.profile,
            warmstart=warmstart,
            max_iterations=args.maxiter,
            tol=jnp.array(args.tol),
        )

    def predict(
        model: GaussianProcess, x: Float[Array, "m d g"]
    ) -> Float[NDArray, "m o"]:
        # mock query dim: per-point 1x1 covariance instead of full m x m
        mean, _ = model.predict(x[:, None])
        return np.asarray(rearrange(mean, "m o 1 -> m o"))

    def lambda_max() -> float:
        return float(GaussianProcess.lambda_max(x_train, y_train, profile=args.profile))

    return fit, predict, lambda_max


if __name__ == "__main__":
    args = parse_args()

    run_name = (
        f"model={'linear' if args.linear_model else 'gp'}"
        f"/target={args.target}"
        f"/feet={args.feet}{'_ungrouped' if args.ungroup_feet else ''}"
        f"/inclines={args.inclines}_n={args.train_size or 'half'}_seed={args.seed}"
        f"{f'_relative={args.relative}' if args.relative else ''}"
    )
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to {save_dir}")

    data = InclineRunning(
        path=args.data_dir,
        train_size=args.train_size,
        seed=args.seed,
        feet=args.feet,
        target=args.target,
        inclines=args.inclines,
        relative=args.relative,
        ungroup_feet=args.ungroup_feet,
    )

    x_train = jnp.asarray(data.x_train)
    x_test = jnp.asarray(data.x_test)
    n, d, g = x_train.shape

    y_train = jnp.asarray(data.y_train)
    y_test = jnp.asarray(data.y_test)

    revision, dirty = git_revision()
    if dirty:
        print(
            "=" * 72 + "\nWARNING: git tree is DIRTY — meta.json will record"
            f" {revision} + dirty=true.\nCommit before any run whose results"
            " you intend to keep.\n" + "=" * 72
        )

    meta = dict(
        args=vars(args),
        group_size=d,
        n_groups=g,
        # markers are their own targets, so that run is the autoregressive one
        autoregressive=args.target == "markers",
        run_name=run_name,
        git_revision=revision,
        git_dirty=dirty,
        x_columns=data.x_columns,
        y_columns=data.y_columns,
        group_labels=data.group_labels,
    )
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    store = RunStore(save_dir)

    build = linear_model if args.linear_model else gp_model
    fit, predict, lambda_max = build(args, x_train, y_train)

    def group_norms(model: Model) -> Float[NDArray, "g"]:
        norms = jnp.sqrt(reduce(model.theta**2, "... g -> g", "sum"))
        return np.asarray(norms)

    def r2_scores(
        model: Model, x: Float[Array, "m d g"], y: Float[Array, "m o"]
    ) -> Float[NDArray, "o"]:
        return r2_score(np.asarray(y), predict(model, x), multioutput="raw_values")

    def record(
        l1_penalty: float,
        model: Model,
        loss: Scalar,
        state: ADMMState,
        certificate: Float[Array, "g"],
    ) -> dict:
        gn = group_norms(model)
        r2_test = r2_scores(model, x_test, y_test)
        r2_train = r2_scores(model, x_train, y_train)
        n_active = int(np.sum(gn > 1e-8))
        print(f"lambda = {l1_penalty:.4g}")
        print(
            f"    converged = {bool(state.converged(args.tol))}"
            f" in {int(state.iteration)} iterations"
            f" (r={float(state.primal_residual):.3e},"
            f" s={float(state.dual_residual):.3e})"
        )
        print(f"    active groups = {n_active}/{len(gn)}")
        print(f"    max gnorm = {gn.max():.4f}")
        print(f"    max kkt violation = {float(jnp.max(certificate)):.3e}")
        print(f"    r2 (test)  = [{r2_test.min():.3f}, {r2_test.max():.3f}]")
        print(f"    r2 (train) = [{r2_train.min():.3f}, {r2_train.max():.3f}]")

        row = dict(
            l1_penalty=l1_penalty,
            n_active=n_active,
            n_groups=len(gn),
            converged=bool(state.converged(args.tol)),
            iterations=int(state.iteration),
            primal_residual=float(state.primal_residual),
            dual_residual=float(state.dual_residual),
            max_kkt=float(jnp.max(certificate)),
            loss=float(loss),
            group_norms=gn.tolist(),
            r2_test=r2_test.tolist(),
            r2_train=r2_train.tolist(),
        )
        arrays = (
            model_to_arrays(model)
            | state_to_arrays(state)
            | dict(certificate=np.asarray(certificate))
        )
        return store.append(row, arrays)

    def fit_or_load(
        l1_penalty: float, warmstart: ADMMState | None = None
    ) -> tuple[dict, ADMMState | None]:
        cached = None if args.overwrite else store.find(l1_penalty)
        if cached is not None:
            print(f"lambda = {l1_penalty:.4g} (cached, resuming)")
            print(f"    active groups = {cached['n_active']}/{cached['n_groups']}")
            return cached, store.load_state(cached["index"])
        row = record(l1_penalty, *fit(l1_penalty, warmstart=warmstart))
        return row, store.load_state(row["index"])

    path: dict[float, int] = {}

    def fit_and_track(
        l1_penalty: float, warmstart: ADMMState | None = None
    ) -> tuple[dict, ADMMState | None]:
        l1_penalty = float(l1_penalty)
        row, state = fit_or_load(l1_penalty, warmstart=warmstart)
        path[l1_penalty] = row["n_active"]
        return row, state

    # the zero model anchors the path, everything above this penalty is all dead
    start = lambda_max()
    print(f"lambda_max = {start:.6g}")

    # relax geometrically from the zero model, each fit warmstarted from the last
    warmstart, n_groups = None, None
    for step in range(args.lambda_budget):
        row, warmstart = fit_and_track(start * args.lambda_ratio**step, warmstart)
        n_groups = row["n_groups"]
        if row["n_active"] == n_groups:
            break
    else:
        print(f"Support still incomplete after {args.lambda_budget} lambdas.")

    # the unpenalized fit is the accuracy ceiling, and the cheapest one to reach here
    fit_and_track(0.0, warmstart)

    print("Final path (lambda, active groups):")
    for l1 in sorted(path):
        print(f"    {l1:.6g}  {path[l1]}/{n_groups}")
