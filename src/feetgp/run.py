from jaxtyping import Array, Float, Scalar
from numpy.typing import NDArray

import os
import glob
import json
import pickle
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

STATE_FORMAT = 9

LAMBDA_START_FRACTION = 0.02

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
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--lambda_budget", type=int, default=100)
    parser.add_argument("--lambda_step", type=float, default=1.3)
    parser.add_argument("--lambda_refine", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # markers are their own targets, so that run is the autoregressive one
    autoregressive = args.target == "markers"

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
    o = y_train.shape[1]

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
        autoregressive=autoregressive,
        run_name=run_name,
        git_revision=revision,
        git_dirty=dirty,
        x_columns=data.x_columns,
        y_columns=data.y_columns,
        group_labels=data.group_labels,
    )
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    def cache_path(l1_penalty: float) -> str:
        return os.path.join(save_dir, f"lambda={float(l1_penalty):.9e}.pkl")

    def find_cached(l1_penalty: float) -> str | None:
        for path in glob.glob(os.path.join(save_dir, "lambda=*.pkl")):
            name = os.path.basename(path).removeprefix("lambda=").removesuffix(".pkl")
            try:
                cached_penalty = float(name)
            except ValueError:
                continue
            if np.isclose(cached_penalty, l1_penalty, rtol=1e-6, atol=0.0):
                return path
        return None

    def fit(l1_penalty: float, warmstart: ADMMState | None = None):
        if args.linear_model:
            # the linear x update is closed form, so there is nothing to warmstart
            return Linear.fit(
                x_train,
                y_train,
                jnp.array(l1_penalty),
                max_iterations=args.maxiter,
                tol=jnp.array(args.tol),
            )

        return GaussianProcess.fit(
            x_train,
            y_train,
            jnp.array(l1_penalty),
            profile=args.profile,
            warmstart=warmstart,
            max_iterations=args.maxiter,
            tol=jnp.array(args.tol),
        )

    def group_norms(model: Model) -> Float[NDArray, "g"]:
        norms = jnp.sqrt(reduce(model.theta**2, "... g -> g", "sum"))
        return np.asarray(norms)

    def predict(model: Model, x: Float[Array, "m d g"]) -> Float[NDArray, "m o"]:
        if args.linear_model:
            prediction = model.predict(x)
        else:
            # mock query dim: per-point 1x1 covariance instead of full m x m
            prediction, _ = model.predict(x[:, None])
            prediction = rearrange(prediction, "m ... 1 -> m ...")
        return np.asarray(rearrange(prediction, "m ... -> m (...)"))

    def r2_scores(
        model: Model, x: Float[Array, "m d g"], y: Float[Array, "m o"]
    ) -> Float[NDArray, "o"]:
        return r2_score(np.asarray(y), predict(model, x), multioutput="raw_values")

    def without_training_data(model: Model) -> Model:
        fields = {f: None for f in ("x_train", "y_train") if f in model._fields}
        return model._replace(**fields)

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

        results = dict(
            l1_penalty=l1_penalty,
            model=without_training_data(model),
            admm_state=state,
            certificate=np.asarray(certificate),
            group_norms=gn,
            r2_test=r2_test,
            r2_train=r2_train,
            loss=loss,
            n_active=n_active,
            state_format=STATE_FORMAT,
        )
        with open(cache_path(l1_penalty), "wb") as f:
            pickle.dump(results, f)
        return results

    def fit_or_load(
        l1_penalty: float, warmstart: ADMMState | None = None
    ) -> tuple[dict, ADMMState | None]:
        cached = find_cached(l1_penalty)
        if not args.overwrite and cached is not None:
            with open(cached, "rb") as f:
                results = pickle.load(f)
            n_active = results["n_active"]
            n_groups = len(results["group_norms"])
            print(f"lambda = {l1_penalty:.4g} (cached, resuming)")
            print(f"    active groups = {n_active}/{n_groups}")
            if results.get("state_format") != STATE_FORMAT:
                print("    stale pickle (older state format), warmstarting cold")
                return results, None
            return results, results.get("admm_state")
        results = record(l1_penalty, *fit(l1_penalty, warmstart=warmstart))
        return results, results["admm_state"]

    path: dict[float, int] = {}
    states: dict[float, ADMMState | None] = {}

    def fit_and_track(l1_penalty: float, warmstart: ADMMState | None = None) -> dict:
        l1_penalty = float(l1_penalty)
        results, state = fit_or_load(l1_penalty, warmstart=warmstart)
        path[l1_penalty] = results["n_active"]
        states[l1_penalty] = state
        return results

    # the unpenalized fit anchors the path and warmstarts every descent from dense
    results = fit_and_track(0.0)
    dense_warmstart = states[0.0]
    gn = results["group_norms"]
    n_groups = len(gn)

    # back off until a penalty keeps the full support alive
    l1_penalty = LAMBDA_START_FRACTION * float(gn.max())
    for _ in range(args.lambda_budget):
        results = fit_and_track(l1_penalty, warmstart=dense_warmstart)
        if results["n_active"] >= path[0.0]:
            break
        l1_penalty /= args.lambda_step
    else:
        print(f"No lambda holds the full support within {args.lambda_budget} steps.")

    # climb until every group is dead, chaining warmstarts along the path
    warmstart = states[l1_penalty]
    for _ in range(args.lambda_budget):
        l1_penalty *= args.lambda_step
        results = fit_and_track(l1_penalty, warmstart=warmstart)
        warmstart = states[l1_penalty]
        if results["n_active"] == 0:
            break
    else:
        print(f"Failed to kill every group within {args.lambda_budget} lambdas.")

    # bisect wherever more than one group dies at once
    for _ in range(args.lambda_refine):
        grid = sorted(path)
        min_width = np.log(1.001)
        gaps = [
            (path[lo] - path[hi], np.log(hi / lo), lo, hi)
            for lo, hi in zip(grid, grid[1:])
            if lo > 0 and path[lo] - path[hi] > 1 and np.log(hi / lo) > min_width
        ]
        if not gaps:
            print("No lambda interval left where more than one group dies at once.")
            break
        lost, _, lo, hi = max(gaps)
        midpoint = float(np.sqrt(lo * hi))
        print(f"Refining [{lo:.4g}, {hi:.4g}], {lost} groups die there.")
        fit_and_track(midpoint, warmstart=states[lo])
    else:
        print(f"Refinement budget of {args.lambda_refine} lambdas spent.")

    print("Final path (lambda, active groups):")
    for l1 in sorted(path):
        print(f"    {l1:.6g}  {path[l1]}/{n_groups}")
