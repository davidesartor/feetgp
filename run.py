from jaxtyping import Float, Scalar
from numpy.typing import NDArray
from jaxtyping import Array

import os
import pickle
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from einops import rearrange
from sklearn.metrics import r2_score

from glassogp import GroupLassoGaussianProcess, kernel, loglikelihood
from linear import GroupLassoLinear
from inclinerunning import InclineRunning

jax.config.update("jax_enable_x64", True)
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
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--lambda_budget", type=int, default=100)
    parser.add_argument("--lambda_step", type=float, default=2.0)
    # by default an existing lambda pickle is reused (resume); --overwrite refits
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

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

    def fit(l1_penalty: float, warmstart=None):
        model_cls = GroupLassoLinear if args.linear_model else GroupLassoGaussianProcess
        return model_cls.fit(
            x_train=x_train,
            y_train=y_train,
            l1_penalty=jnp.array(l1_penalty),
            group_size=group_size,
            autoregressive=autoregressive,  # type: ignore only used for GP model
            warmstart=warmstart,  # type: ignore only used for GP model
            max_iterations=args.maxiter,
            tol=jnp.array(args.tol),
            n_jobs=args.n_jobs,  # type: ignore only used for GP model
        )

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
    ) -> dict:
        gn = group_norms(model)
        r2_test = r2_scores(model, x_test, y_test)
        r2_train = r2_scores(model, x_train, y_train)
        n_active = int(np.sum(gn > 1e-8))
        print(f"lambda = {l1_penalty:.4g}")
        print(f"    active groups = {n_active}/{len(gn)}")
        print(f"    max gnorm = {gn.max():.4f}")
        print(f"    r2 (test)  = [{r2_test.min():.3f}, {r2_test.max():.3f}]")
        print(f"    r2 (train) = [{r2_train.min():.3f}, {r2_train.max():.3f}]")

        # remove the training data from the model before saving, to reduce pickle size
        results = dict(
            l1_penalty=l1_penalty,
            model=model._replace(x_train=None, y_train=None),
            group_norms=gn,
            r2_test=r2_test,
            r2_train=r2_train,
            llk=llk,
            n_active=n_active,
        )
        path = os.path.join(save_dir, f"lambda={float(l1_penalty):013.6f}.pkl")
        with open(path, "wb") as f:
            pickle.dump(results, f)
        return results

    def fit_or_load(l1_penalty: float, warmstart=None) -> tuple[dict, object]:
        """Fit at l1_penalty, or reuse a previously saved result to resume a run.

        Returns the results dict and the previous solution's model instance,
        used to warmstart the next fit.
        """
        path = os.path.join(save_dir, f"lambda={float(l1_penalty):013.6f}.pkl")
        if not args.overwrite and os.path.exists(path):
            with open(path, "rb") as f:
                results = pickle.load(f)
            n_active = results["n_active"]
            n_groups = len(results["group_norms"])
            print(f"lambda = {l1_penalty:.4g} (cached, resuming)")
            print(f"    active groups = {n_active}/{n_groups}")
            return results, results["model"]
        model, llk = fit(l1_penalty, warmstart=warmstart)
        results = record(l1_penalty, model, llk)
        return results, model

    ############################################################
    # Probe lambda_max: unpenalized fit gives upper bound on group norms
    ############################################################
    results, unpenalized_warmstart = fit_or_load(0.0, warmstart=None)
    gn = results["group_norms"]

    ############################################################
    # Pivot: lambda where penalty term balances the log-likelihood gained
    # by fitting over the null (theta=0, i.e. linear: A=0) model
    ############################################################
    fitted_model = results["model"]
    if args.linear_model:
        null_loss = 0.5 * jnp.sum(y_train**2)
        delta = results["llk"] - null_loss
    else:
        # same nugget (g) per output as the fitted model, or the null-model
        # kernel (all ones, since theta=0) is singular and cho_factor -> nan
        K0 = kernel(jnp.zeros(d), x_train, x_train)
        null_ll = lambda g, y: loglikelihood(K0 + g * jnp.eye(n), y)[0]
        llks = jax.vmap(null_ll)(fitted_model.g, y_train.T)
        delta = results["llk"] - jnp.sum(llks)
    lambda_pivot = 0.1 * abs(delta) / gn.sum()
    results, pivot_warmstart = fit_or_load(
        lambda_pivot, warmstart=unpenalized_warmstart
    )

    ############################################################
    # Search lambda_max: double from pivot until all groups die
    ############################################################
    warmstart = pivot_warmstart
    lambda_max = lambda_pivot
    for _ in range(args.lambda_budget // 2):
        lambda_max *= args.lambda_step
        results, warmstart = fit_or_load(lambda_max, warmstart=warmstart)
        if results["n_active"] == 0:
            break
    else:
        print(f"Failed to find lambda_max with {args.lambda_budget} steps.")

    ############################################################
    # Search lambda_min: halve from pivot until full support
    ############################################################
    warmstart = pivot_warmstart
    lambda_min = lambda_pivot
    for _ in range(args.lambda_budget // 2):
        lambda_min /= args.lambda_step
        results, warmstart = fit_or_load(lambda_min, warmstart=warmstart)
        if results["n_active"] == len(results["group_norms"]):
            break
    else:
        print(f"Failed to find lambda_min with {args.lambda_budget} steps.")
