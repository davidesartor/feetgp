from jaxtyping import Float, Scalar
from numpy.typing import NDArray

import os
import pickle
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from einops import rearrange
from sklearn.metrics import r2_score

from glassogp import GroupLassoGaussianProcess
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
    parser.add_argument("--relative", type=str, default=None)

    # OPTIMIZATION ARGS
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--lambda_budget", type=int, default=100)
    parser.add_argument("--lambda_step", type=float, default=2.0)
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
        if args.linear_model:
            model = GroupLassoLinear.fit(
                x_train=x_train,
                y_train=y_train,
                l1_penalty=jnp.array(l1_penalty),
                group_size=group_size,
                max_iterations=args.maxiter,
                tol=jnp.array(args.tol),
            )
            return model, None, None
        else:
            return GroupLassoGaussianProcess.fit(
                x_train=x_train,
                y_train=y_train,
                l1_penalty=jnp.array(l1_penalty),
                group_size=group_size,
                autoregressive=autoregressive,
                warmstart=warmstart,
                max_iterations=args.maxiter,
                tol=jnp.array(args.tol),
            )

    def group_norms(
        model: GroupLassoGaussianProcess | GroupLassoLinear,
    ) -> Float[NDArray, "d"]:
        params = (
            model.theta if isinstance(model, GroupLassoGaussianProcess) else model.A
        )
        groups = rearrange(params, "o (d g) -> d (o g)", g=group_size)
        norms = np.linalg.norm(np.asarray(groups), axis=-1)
        return norms

    def r2_scores(
        model: GroupLassoGaussianProcess | GroupLassoLinear,
    ) -> Float[NDArray, "o"]:
        y_pred = np.array(model.predict(x_test))
        r2 = np.array([r2_score(y_test[:, j], y_pred[j, :]) for j in range(o)])
        return r2

    def record(
        l1_penalty: float,
        model: GroupLassoGaussianProcess | GroupLassoLinear,
        llk: Scalar | None,
    ):
        gn = group_norms(model)
        r2 = r2_scores(model)
        n_active = int(np.sum(gn > 1e-8))
        print(f"lambda = {l1_penalty:.4g}")
        print(f"    active groups = {n_active}/{len(gn)}")
        print(f"    max gnorm = {gn.max():.4f}")
        print(f"    r2 = [{r2.min():.3f}, {r2.max():.3f}]")

        fname = f"lambda={l1_penalty:.6e}.pkl"
        with open(os.path.join(save_dir, fname), "wb") as f:
            results = dict(
                l1_penalty=float(l1_penalty),
                model=(
                    model
                    if isinstance(model, GroupLassoLinear)
                    else model._replace(x_train=None, y_train=None)
                ),
                group_norms=gn,
                r2=r2,
                llk=float(llk) if llk is not None else None,
                n_active=n_active,
            )
            pickle.dump(results, f)
        return gn, r2, n_active

    ############################################################
    # Probe lambda_max: unpenalized fit gives upper bound on group norms
    ############################################################
    unpenalized_model, unpenalized_admm, llk = fit(l1_penalty=0.0, warmstart=None)
    gn, r2, max_active_groups = record(0.0, unpenalized_model, llk)

    ############################################################
    # Pivot: lambda where penalty term balances the negative log-likelihood
    ############################################################
    lambda_pivot = 1 / gn.sum()
    pivot_model, pivot_admm_state, llk = fit(
        l1_penalty=lambda_pivot, warmstart=unpenalized_admm
    )
    gn, r2, active_groups = record(lambda_pivot, pivot_model, llk)

    ############################################################
    # Search lambda_max: double from pivot until all groups die
    ############################################################
    warmstart = pivot_admm_state
    lambda_max = lambda_pivot
    for _ in range(args.lambda_budget // 2):
        lambda_max *= args.lambda_step
        model, warmstart, llk = fit(l1_penalty=lambda_max, warmstart=warmstart)
        gn, r2, n_active = record(lambda_max, model, llk)
        if n_active == 0:
            break
    else:
        print(f"Failed to find lambda_max with {args.lambda_budget} steps.")

    ############################################################
    # Search lambda_min: halve from pivot until full support
    ############################################################
    warmstart = pivot_admm_state
    lambda_min = lambda_pivot
    for _ in range(args.lambda_budget // 2):
        lambda_min /= args.lambda_step
        model, warmstart, llk = fit(l1_penalty=lambda_min, warmstart=warmstart)
        gn, r2, n_active = record(lambda_min, model, llk)
        if n_active == len(gn):
            break
    else:
        print(f"Failed to find lambda_min with {args.lambda_budget} steps.")
