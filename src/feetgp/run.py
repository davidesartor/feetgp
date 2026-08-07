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

from feetgp import admm
from feetgp import glassogp
from feetgp import linear
from feetgp.glassogp import GroupLassoGaussianProcess, hetgpy_auto_bounds
from feetgp.linear import GroupLassoLinear
from feetgp.inclinerunning import InclineRunning

jax.config.update("jax_enable_x64", True)

# what admm_state.x means. 1: a linear g in the last column. 2: w = log(g - g_min).
# 3: w = logit((g - g_min) / (g_max - g_min)), the saturating nugget. Reading one
# format's w as another's is a wrong-nugget bug that leaves no trace, so old pickles
# are kept as results but refused as warmstarts. 4: same w, but theta is back in the
# nonnegative orthant, so a format-3 state's negative thetas would clip to zero.
# 5 (current): the shared ADMM machinery. Same parametrization as 4, but the iterates
# are laid out (groups, group members) and the nugget moved out of x into state.aux --
# a pure relabelling, so format 4 is converted rather than refused.
STATE_FORMAT = 5

# where the sweep starts, as a fraction of the largest group norm at lambda=0. The
# useful lambda range sits on the scale of those norms -- a group dies once the prox
# threshold l1 / rho reaches its own norm -- and they span well under a decade, so
# starting a little under the band and stepping up covers the whole path.
LAMBDA_START_FRACTION = 0.02

print("JAX devices:", jax.devices())


if __name__ == "__main__":
    ##############################################################
    # Parse arguments
    ##############################################################
    parser = argparse.ArgumentParser()
    # DATASET ARGS
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

    # MODELLING ABLATIONS
    parser.add_argument("--linear_model", action="store_true", default=False)
    parser.add_argument("--ungroup_feet", action="store_true", default=False)
    # pass --relative alone for the LMAL/MMAL midpoint, or --relative MARKER for a specific marker
    parser.add_argument(
        "--relative", type=str, nargs="?", default=None, const="midpoint"
    )

    # OPTIMIZATION ARGS
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-3)
    # outputs solved per batched x-update; bigger keeps the device busier, at n*n each
    parser.add_argument("--chunk_size", type=int, default=8)
    # "optimistix" is unconstrained L-BFGS plus a projection onto the box, "lbfgsb" is
    # vlse's bounded solver. Measured at one knot from a single warmstart, lbfgsb took
    # 67.4s and 16 ADMM iterations against optimistix's 488.7s and 35, at equal-or-better
    # loglik -- but that is a single knot, so optimistix stays the default
    parser.add_argument(
        "--solver", choices=("optimistix", "lbfgsb"), default="optimistix"
    )
    # inner L-BFGS budget per ADMM iteration. A cheap inexact x-update run many times
    # beats an exact one run a few times: the cost is iterations * outputs * inner_steps.
    # counts whole line searches under lbfgsb and single steps under optimistix, hence
    # the two defaults, resolved below
    parser.add_argument("--inner_maxiter", type=int, default=None)
    # optimistix: step tolerance, not gradient -- 1e-4 matches 1.49e-8 to 4e-6 relative in
    # half the steps. lbfgsb: projected-gradient tolerance, scipy's pgtol
    parser.add_argument("--inner_tol", type=float, default=None)
    # lbfgsb only, the straggler's leash: at inner_maxiter=12, cutting 30 -> 5 took one
    # x-update from 28.24s to 9.45s for the same objective to eight digits
    parser.add_argument("--inner_max_linesearch", type=int, default=5)
    parser.add_argument("--history_length", type=int, default=40)
    # iteration after which rho is frozen so the problem stops moving; defaults to
    # max_iterations // 2, which is what it was implicitly tied to
    parser.add_argument("--adapt_rho_iters", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--lambda_budget", type=int, default=100)
    # a fine grid is what makes groups die one at a time instead of in clumps
    parser.add_argument("--lambda_step", type=float, default=1.3)
    # extra fits spent bisecting the intervals where more than one group died at once.
    # a geometric grid spreads points evenly in log lambda, but the deaths are not
    # spread evenly, so uniform refinement is mostly wasted on the flat ends
    parser.add_argument("--lambda_refine", type=int, default=25)
    # GP fits per lambda: chained warmstart, dense (lambda=0) start, then randoms. The
    # objective is even in theta, so death is absorbing and a single chained start bakes
    # continuation bias into the path; with the dense start in the race at every lambda
    # the winner is chosen by the true penalized objective instead. 1 restores the old
    # single-start behavior; the linear model is convex and always runs one start
    parser.add_argument("--n_starts", type=int, default=3)
    # by default an existing lambda pickle is reused (resume); --overwrite refits
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    # the two inner solvers stop on different criteria and count different things, so an
    # unset budget or tolerance means whatever is right for the solver in use
    if args.inner_maxiter is None:
        args.inner_maxiter = 5 if args.solver == "lbfgsb" else 50
    if args.inner_tol is None:
        args.inner_tol = 1e-2 if args.solver == "lbfgsb" else 1e-4

    group_size = 6 if (args.feet == "both" and not args.ungroup_feet) else 3
    autoregressive = args.target == "markers"

    ##############################################################
    # Build run directory from parameters
    ##############################################################
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

    ############################################################
    # Load and prepare data
    ############################################################
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

    ############################################################
    # Record what produced this run, so plots.py does not have to
    # reverse-engineer it from the directory name
    ############################################################
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
        """Name a penalty group from its columns: LCAL1+RCAL1 -> CAL1, LCAL1 -> L CAL1."""
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
        """Locate a saved result by the lambda value in its filename, not by its format.

        Old zero-padded names parse the same way, so they are still found; ones whose
        rounding lost too many digits simply miss and get refit under the new name.
        """
        for path in glob.glob(os.path.join(save_dir, "lambda=*.pkl")):
            name = os.path.basename(path).removeprefix("lambda=").removesuffix(".pkl")
            try:
                cached_penalty = float(name)
            except ValueError:
                continue
            if np.isclose(cached_penalty, l1_penalty, rtol=1e-6, atol=0.0):
                return path
        return None

    # the lengthscale bounds only depend on x_train, but cost an n*n cdist; the whole
    # sweep shares one set instead of refitting them per lambda
    auto_bounds = None if args.linear_model else hetgpy_auto_bounds(x_train)

    # a chunk wider than the number of outputs just pads; --target forces has 3
    chunk_size = min(args.chunk_size, y_train.shape[1])

    admm_state_from_legacy = (
        linear.admm_state_from_legacy
        if args.linear_model
        else glassogp.admm_state_from_legacy
    )

    def fit(l1_penalty: float, warmstart=None):
        model_cls = GroupLassoLinear if args.linear_model else GroupLassoGaussianProcess
        return model_cls.fit(
            x_train=x_train,
            y_train=y_train,
            l1_penalty=jnp.array(l1_penalty),
            group_size=group_size,
            autoregressive=autoregressive,  # type: ignore only used for GP model
            warmstart=warmstart,  # type: ignore only used for GP model
            auto_bounds=auto_bounds,  # type: ignore only used for GP model
            max_iterations=args.maxiter,
            tol=jnp.array(args.tol),
            adapt_rho_iters=args.adapt_rho_iters,
            chunk_size=chunk_size,  # type: ignore only used for GP model
            solver=args.solver,  # type: ignore only used for GP model
            inner_maxiter=args.inner_maxiter,  # type: ignore only used for GP model
            inner_rtol=args.inner_tol,  # type: ignore only used for GP model
            inner_atol=args.inner_tol,  # type: ignore only used for GP model
            inner_pgtol=args.inner_tol,  # type: ignore only used for GP model
            inner_max_linesearch=args.inner_max_linesearch,  # type: ignore GP only
            history_length=args.history_length,  # type: ignore only used for GP model
            log_every=args.log_every,
        )

    def random_start(l1_penalty: float, k: int) -> admm.ADMMState:
        """Fresh ADMM state, theta log-uniform inside the hetgpy lengthscale band."""
        # seeded from (k, lambda bits) so a resumed run redraws the same starts
        rng = np.random.default_rng([k, int(np.float64(l1_penalty).view(np.uint64))])
        lower, upper = auto_bounds
        low, high = np.sqrt(2.0 / upper), np.sqrt(2.0 / lower)
        theta = np.exp(rng.uniform(np.log(low), np.log(high), size=(o, d)))
        g_min, g_max = (jnp.array(g) for g in glassogp.G_RANGE)
        w = glassogp.w_from_nugget(jnp.array(0.1), g_min, g_max)
        return admm.ADMMState.initialize(
            admm.to_groups(jnp.asarray(theta), group_size),
            aux=jnp.full((o,), float(w)),
        )

    def fit_multistart(l1_penalty: float, warmstart=None):
        """Fit from several starts, winner by the true penalized objective.

        The chained warmstart carries continuation bias -- death is absorbing, so it can
        only lose groups -- and the dense lambda=0 start is what can win them back.
        """
        if args.linear_model or args.n_starts == 1:
            return fit(l1_penalty, warmstart=warmstart)

        starts = [("chained" if warmstart is not None else "default", warmstart)]
        # at the first lambda above 0 the chained warmstart is itself the dense start
        # (states[0.0] with rho reset), so racing both would fit the same start twice
        chained_is_dense = (
            warmstart is not None
            and states.get(0.0) is not None
            and warmstart.x is states[0.0].x
        )
        if l1_penalty > 0 and states.get(0.0) is not None and not chained_is_dense:
            # rho reset as for the chained handoff: lambda=0 walks rho to RHO_MIN, where
            # the prox threshold l1 / (rho * norm) kills every group on contact
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
                glassogp.penalized_objective(
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

        # remove the training data from the model before saving, to reduce pickle size
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
        """Fit at l1_penalty, or reuse a previously saved result to resume a run.

        Returns the results dict and the ADMM state (or, for legacy pickles that
        predate it, the model) used to warmstart the next fit.
        """
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
            # format 4 differs from 5 only in layout, so it converts instead of being
            # thrown away; anything older parametrizes the nugget differently and
            # reading one format's w as another's is a silent wrong-nugget bug
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

    # every lambda the sweep touches, cached or freshly fit, so the refinement pass
    # can see the whole path and warmstart from a neighbour it already has in memory
    path: dict[float, int] = {}
    states: dict[float, object] = {}

    def fit_and_track(l1_penalty: float, warmstart=None) -> dict:
        # lambdas derived from jax scalars cannot key a dict until they are python floats
        l1_penalty = float(l1_penalty)
        results, state = fit_or_load(l1_penalty, warmstart=warmstart)
        path[l1_penalty] = results["n_active"]
        states[l1_penalty] = state
        return results

    ############################################################
    # Unpenalized fit: sets the lambda scale, and is the only warmstart with full support
    ############################################################
    results = fit_and_track(0.0, warmstart=None)
    # rho is not inherited from lambda=0. There the prox is the identity, so u is
    # identically zero and the augmented term is vacuous -- rho decaying to RHO_MIN is
    # the adaptation correctly annihilating it, and rho -> 0 is what makes the x-update
    # the exact MLE this lambda wants. But it is meaningless to the next lambda, where a
    # prox threshold of l1 / (rho * norm) at rho=2e-6 kills every group on contact.
    unpenalized_warmstart = states[0.0]._replace(rho=jnp.array(1.0))
    gn = results["group_norms"]
    n_groups = len(gn)

    ############################################################
    # Walk lambda up from under the group-norm band until every group is dead.
    #
    # Strictly upward, and every fit warmstarted from the lambda below it. Death is
    # absorbing here: the GP objective is exactly even in theta, so grad f(0) = 0, and
    # once a group has x = z = 0 its u freezes, the x-update target z - u = -u is
    # nonpositive and clips straight back to 0. Nothing can resurrect it. So a fit must
    # never be handed a warmstart sparser than its own solution -- which is what walking
    # lambda downward did, and it dragged whole paths to 0 active groups.
    ############################################################
    # LAMBDA_START_FRACTION only guesses the scale, so drop until some lambda still holds
    # the full support. These probes warmstart from lambda=0, never from the probe above:
    # a dense warmstart has no dead group to inherit, which is what makes descending safe
    # here and unsafe in the walk below.
    l1_penalty = LAMBDA_START_FRACTION * float(gn.max())
    for _ in range(args.lambda_budget):
        results = fit_and_track(l1_penalty, warmstart=unpenalized_warmstart)
        # against the unpenalized support, not n_groups: some groups come out of lambda=0
        # at exactly zero norm already, and no lambda can then bring them back
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

    ############################################################
    # Refine: bisect the intervals where the support dropped by more than one group
    ############################################################
    for _ in range(args.lambda_refine):
        grid = sorted(path)
        # (groups lost, log-width) over adjacent pairs; a pair that lost one group is
        # already resolved, and the flat ends lose none, so neither is worth a fit
        # groups that genuinely die at the same lambda would otherwise be bisected until
        # the budget runs out, so intervals narrower than min_width are left alone
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
