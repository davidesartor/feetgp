from jaxtyping import Float, Scalar
from numpy.typing import NDArray
from jaxtyping import Array

import os
import glob
import json
import pickle
import argparse
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
from einops import rearrange
from sklearn.metrics import r2_score

from feetgp import glasso_admm
from feetgp import gp
from feetgp import linear
from feetgp.gp import GroupLassoGaussianProcess, hetgpy_auto_bounds
from feetgp.linear import GroupLassoLinear
from feetgp.inclinerunning import InclineRunning

jax.config.update("jax_enable_x64", True)

STATE_FORMAT = 5

LAMBDA_START_FRACTION = 0.02

print("JAX devices:", jax.devices())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data/Incline Running")
    parser.add_argument("--subsample", type=int, default=1)
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
    parser.add_argument("--ungroup_feet", action="store_true", default=False)
    parser.add_argument(
        "--relative", type=str, nargs="?", default=None, const="midpoint"
    )

    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--inner_maxiter", type=int, default=5)
    parser.add_argument("--inner_tol", type=float, default=1e-2)
    parser.add_argument("--inner_max_linesearch", type=int, default=5)
    parser.add_argument("--history_length", type=int, default=40)
    parser.add_argument("--adapt_rho_iters", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--lambda_budget", type=int, default=100)
    parser.add_argument("--lambda_step", type=float, default=1.3)
    parser.add_argument("--lambda_refine", type=int, default=25)
    parser.add_argument("--n_starts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    group_size = 6 if (args.feet == "both" and not args.ungroup_feet) else 3
    autoregressive = args.target == "markers"

    run_name = (
        f"model={'linear' if args.linear_model else 'gp'}"
        f"/target={args.target}"
        f"/feet={args.feet}{'_ungrouped' if args.ungroup_feet else ''}"
        f"/inclines={args.inclines}_sub={args.subsample}"
        f"{f'_relative={args.relative}' if args.relative else ''}"
    )
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to {save_dir}")

    data = InclineRunning(
        path=args.data_dir,
        subsample=args.subsample,
        feet=args.feet,
        target=args.target,
        inclines=args.inclines,
        relative=args.relative,
    )

    x_train = jnp.asarray(data.x_train)
    y_train = jnp.asarray(data.y_train)
    x_test = jnp.asarray(data.x_test)
    y_test = jnp.asarray(data.y_test)

    n, d = x_train.shape
    _, o = y_train.shape

    def git_revision() -> tuple[str | None, bool]:
        try:
            git_dir = os.path.dirname(os.path.abspath(__file__))
            rev = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=git_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=git_dir,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, OSError):
            return None, False
        return rev.stdout.strip(), bool(status.stdout.strip())

    def group_label(columns: list[str]) -> str:
        names = sorted({c.rsplit(" ", 1)[0] for c in columns})
        if len(names) == 1:
            return f"{names[0][0]} {names[0][1:]}"
        return names[0][1:]

    revision, dirty = git_revision()
    if dirty:
        print(
            "=" * 72 + "\nWARNING: git tree is DIRTY — meta.json will record"
            f" {revision} + dirty=true.\nCommit before any run whose results"
            " you intend to keep.\n" + "=" * 72
        )

    meta = dict(
        args=vars(args),
        group_size=group_size,
        autoregressive=autoregressive,
        run_name=run_name,
        git_revision=revision,
        git_dirty=dirty,
        x_columns=data.x_columns,
        y_columns=data.y_columns,
        group_labels=[
            group_label(data.x_columns[i : i + group_size])
            for i in range(0, d, group_size)
        ],
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

    auto_bounds = None if args.linear_model else hetgpy_auto_bounds(x_train)

    chunk_size = min(args.chunk_size, y_train.shape[1])

    admm_state_from_legacy = (
        linear.admm_state_from_legacy
        if args.linear_model
        else gp.admm_state_from_legacy
    )

    def fit(l1_penalty: float, warmstart=None):
        model_cls = GroupLassoLinear if args.linear_model else GroupLassoGaussianProcess
        return model_cls.fit(
            x_train=x_train,
            y_train=y_train,
            l1_penalty=jnp.array(l1_penalty),
            group_size=group_size,
            autoregressive=autoregressive,
            warmstart=warmstart,
            auto_bounds=auto_bounds,
            max_iterations=args.maxiter,
            tol=jnp.array(args.tol),
            adapt_rho_iters=args.adapt_rho_iters,
            chunk_size=chunk_size,
            inner_maxiter=args.inner_maxiter,
            inner_pgtol=args.inner_tol,
            inner_max_linesearch=args.inner_max_linesearch,
            history_length=args.history_length,
            log_every=args.log_every,
        )

    def random_start(l1_penalty: float, k: int) -> glasso_admm.ADMMState:
        rng = np.random.default_rng([k, int(np.float64(l1_penalty).view(np.uint64))])
        lower, upper = auto_bounds
        low, high = np.sqrt(2.0 / upper), np.sqrt(2.0 / lower)
        theta = np.exp(rng.uniform(np.log(low), np.log(high), size=(o, d)))
        g_min, g_max = (jnp.array(g) for g in gp.G_RANGE)
        w = gp.w_from_nugget(jnp.array(0.1), g_min, g_max)
        return glasso_admm.ADMMState.initialize(
            glasso_admm.to_groups(jnp.asarray(theta), group_size),
            aux=jnp.full((o,), float(w)),
        )

    def fit_multistart(l1_penalty: float, warmstart=None):
        if args.linear_model or args.n_starts == 1:
            return fit(l1_penalty, warmstart=warmstart)

        starts = [("chained" if warmstart is not None else "default", warmstart)]
        chained_is_dense = (
            warmstart is not None
            and states.get(0.0) is not None
            and warmstart.x is states[0.0].x
        )
        if l1_penalty > 0 and states.get(0.0) is not None and not chained_is_dense:
            starts.append(("dense", states[0.0]._replace(rho=jnp.array(1.0))))
        while len(starts) < args.n_starts:
            starts.append(
                (f"random_{len(starts)}", random_start(l1_penalty, len(starts)))
            )
        starts = starts[: args.n_starts]

        print(f"lambda = {l1_penalty:.4g}, {len(starts)} starts")
        fits = {}
        for label, start in starts:
            model, llk, state, info = fit(l1_penalty, warmstart=start)
            objective = float(
                gp.penalized_objective(
                    model.theta,
                    model.g,
                    jnp.asarray(l1_penalty),
                    x_train,
                    y_train,
                    group_size,
                )
            )
            fits[label] = (objective, model, llk, state, info)
            print(
                f"    start {label}: objective = {objective:.6f},"
                f" converged = {info['converged']} in {info['iterations']} iterations"
            )

        winner = min(fits, key=lambda label: fits[label][0])
        objective, model, llk, state, info = fits[winner]
        info["winner"] = winner
        info["start_objectives"] = {label: fits[label][0] for label in fits}
        info["starts"] = {
            label: dict(
                converged=bool(fits[label][4]["converged"]),
                iterations=int(fits[label][4]["iterations"]),
            )
            for label in fits
        }
        print(f"    winner = {winner}")
        return model, llk, state, info

    def group_norms(
        model: GroupLassoGaussianProcess | GroupLassoLinear,
    ) -> Float[NDArray, "d"]:
        groups = rearrange(model.theta, "o (d g) -> d (o g)", g=group_size)
        norms = np.linalg.norm(np.asarray(groups), axis=-1)
        return norms

    def r2_scores(
        model: GroupLassoGaussianProcess | GroupLassoLinear,
        x: Float[Array, "m d"],
        y: Float[Array, "m o"],
    ) -> Float[Array, "o"]:
        y_pred = np.array(model.predict(x))
        r2 = jnp.array([r2_score(y[:, j], y_pred[j, :]) for j in range(o)])
        return r2

    def record(
        l1_penalty: float,
        model: GroupLassoGaussianProcess | GroupLassoLinear,
        llk: Scalar,
        admm_state,
        info: dict,
    ) -> dict:
        gn = group_norms(model)
        r2_test = r2_scores(model, x_test, y_test)
        r2_train = r2_scores(model, x_train, y_train)
        n_active = int(np.sum(gn > 1e-8))
        print(f"lambda = {l1_penalty:.4g}")
        print(
            f"    converged = {info['converged']} in {info['iterations']} iterations"
            f" (r={info.get('primal_residual', float('nan')):.3e},"
            f" s={info.get('dual_residual', float('nan')):.3e})"
        )
        print(f"    active groups = {n_active}/{len(gn)}")
        print(f"    max gnorm = {gn.max():.4f}")
        print(f"    r2 (test)  = [{r2_test.min():.3f}, {r2_test.max():.3f}]")
        print(f"    r2 (train) = [{r2_train.min():.3f}, {r2_train.max():.3f}]")
        certificate = info.get("certificate")
        if certificate is not None:
            print(
                f"    max live KKT = {float(certificate['max_live_kkt']):.3e}"
                f" (nugget grad {float(certificate['nugget_grad']):.3e})"
            )

        results = dict(
            l1_penalty=l1_penalty,
            model=model._replace(x_train=None, y_train=None),
            admm_state=admm_state,
            group_norms=gn,
            r2_test=r2_test,
            r2_train=r2_train,
            llk=llk,
            n_active=n_active,
            info=info,
            state_format=STATE_FORMAT,
        )
        with open(cache_path(l1_penalty), "wb") as f:
            pickle.dump(results, f)
        return results

    def fit_or_load(l1_penalty: float, warmstart=None) -> tuple[dict, object]:
        cached = find_cached(l1_penalty)
        if not args.overwrite and cached is not None:
            with open(cached, "rb") as f:
                results = pickle.load(f)
            n_active = results["n_active"]
            n_groups = len(results["group_norms"])
            print(f"lambda = {l1_penalty:.4g} (cached, resuming)")
            print(f"    active groups = {n_active}/{n_groups}")
            state_format = results.get("state_format")
            state = results.get("admm_state")
            if state_format == 4 and state is not None:
                state = admm_state_from_legacy(state)
            elif state_format != STATE_FORMAT:
                print(
                    "    stale pickle (older nugget parametrization), warmstarting cold"
                )
                return results, None
            return results, state
        model, llk, state, info = fit_multistart(l1_penalty, warmstart=warmstart)
        results = record(l1_penalty, model, llk, state, info)
        return results, state

    path: dict[float, int] = {}
    states: dict[float, object] = {}

    def fit_and_track(l1_penalty: float, warmstart=None) -> dict:
        l1_penalty = float(l1_penalty)
        results, state = fit_or_load(l1_penalty, warmstart=warmstart)
        path[l1_penalty] = results["n_active"]
        states[l1_penalty] = state
        return results

    results = fit_and_track(0.0, warmstart=None)
    unpenalized_warmstart = states[0.0]._replace(rho=jnp.array(1.0))
    gn = results["group_norms"]
    n_groups = len(gn)

    l1_penalty = LAMBDA_START_FRACTION * float(gn.max())
    for _ in range(args.lambda_budget):
        results = fit_and_track(l1_penalty, warmstart=unpenalized_warmstart)
        if results["n_active"] >= path[0.0]:
            break
        l1_penalty /= args.lambda_step
    else:
        print(f"No lambda holds the full support within {args.lambda_budget} steps.")

    warmstart = states[l1_penalty]
    for _ in range(args.lambda_budget):
        l1_penalty *= args.lambda_step
        results = fit_and_track(l1_penalty, warmstart=warmstart)
        warmstart = states[l1_penalty]
        if results["n_active"] == 0:
            break
    else:
        print(f"Failed to kill every group within {args.lambda_budget} lambdas.")

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
