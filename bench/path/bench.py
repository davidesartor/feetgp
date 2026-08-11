"""Multi-start backward regularization path: K noised paths from lambda_max down."""

import argparse
import os

import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat

from bench.common import chip_name, eps_multiple, time_call, write_row
from feetgp.glasso_admm import group_norm
from feetgp.gp import GaussianProcess, hetgpy_auto_bounds

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")
DTYPE = "float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/Incline Running")
    parser.add_argument("--train_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile", type=str, default="rbf", choices=["rbf", "matern52"]
    )
    parser.add_argument(
        "--feet", type=str, default="both", choices=["both", "left_only", "right_only"]
    )
    parser.add_argument(
        "--target", type=str, default="forces", choices=["markers", "forces"]
    )
    parser.add_argument(
        "--inclines", type=str, default="inc0", choices=["all", "inc0", "inc5", "inc10"]
    )
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--n_paths", type=int, default=8)
    parser.add_argument("--n_lambdas", type=int, default=20)
    parser.add_argument("--lambda_min_ratio", type=float, default=0.01)
    parser.add_argument("--p_entry", type=float, default=0.7)
    parser.add_argument("--kick_range", type=float, nargs=2, default=[0.03, 0.3])
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--cold_reference", action="store_true", default=True)
    parser.add_argument("--no_cold_reference", dest="cold_reference", action="store_false")  # fmt: skip
    return parser.parse_args()


def group_grad_norms(theta, log_nugget, x_flat, y_train, profile):
    """Inward gradient norm per group, pooled over outputs, at the fitted point."""

    def nll(theta, log_nugget, y):
        loss, _ = GaussianProcess.loss(
            theta, jnp.exp(log_nugget), x_flat, y, profile=profile
        )
        return loss

    grad = jax.vmap(jax.grad(nll))(theta, log_nugget, y_train.T)
    return jnp.sqrt(reduce(jnp.minimum(grad, 0.0) ** 2, "o d g -> g", "sum"))


def kick(state, grad_norms, l1_penalty, key, theta_init, args):
    """Open eligible dead groups with noised magnitudes, jitter the live ones."""
    o, d, g = state.x.shape
    active = group_norm(state.z) > 0.0
    eligible = ~active & (grad_norms > l1_penalty)

    key_open, key_size, key_jitter = jax.random.split(key, 3)
    opened = eligible & (jax.random.uniform(key_open, (g,)) < args.p_entry)

    # kick size: log-uniform multiple of theta_init, same for every output
    low, high = jnp.log(jnp.asarray(args.kick_range))
    scale = jnp.exp(jax.random.uniform(key_size, (g,), minval=low, maxval=high))
    kick_value = repeat(theta_init * scale, "d g -> o d g", o=o)

    # small relative jitter on live coordinates keeps same-mask paths distinct
    noise = 1.0 + args.jitter * jax.random.normal(key_jitter, state.x.shape)
    x = jnp.where(active, state.x * noise, jnp.where(opened, kick_value, state.x))

    # opened groups anchor at x via z - u, but with a residual that cannot read zero
    z = jnp.where(opened, 0.0, state.z)
    u = jnp.where(opened, -x, state.u)
    return state._replace(x=x, z=z, u=u), opened


def active_groups(state) -> str:
    return "".join("1" if a else "0" for a in group_norm(state.z) > 0.0)


if __name__ == "__main__":
    from feetgp.inclinerunning import InclineRunning

    args = parse_args()
    print("JAX devices:", jax.devices(), flush=True)
    chip = chip_name(jax.devices()[0])

    data = InclineRunning(
        path=args.data_dir,
        train_size=args.train_size,
        seed=args.seed,
        feet=args.feet,
        target=args.target,
        inclines=args.inclines,
    )
    x_train = jnp.asarray(data.x_train, dtype=DTYPE)
    y_train = jnp.asarray(data.y_train, dtype=DTYPE)
    n, d, g = x_train.shape
    o = y_train.shape[1]
    x_flat = rearrange(x_train, "n d g -> n (d g)")
    theta_init = rearrange(hetgpy_auto_bounds(x_flat)[0], "(d g) -> d g", g=g)

    row = dict(
        chip=chip, dtype=DTYPE, profile=args.profile, target=args.target,
        n=n, d=d, g=g, o=o, seed=args.seed, groups=data.group_labels,
    )  # fmt: skip

    lambda_max = float(
        GaussianProcess.lambda_max(
            x_train, y_train, profile=args.profile, epsilon=args.epsilon
        )
    )
    lambdas = jnp.geomspace(
        1.05 * lambda_max, args.lambda_min_ratio * lambda_max, args.n_lambdas
    )
    print(f"n={n} d={d} g={g} o={o} lambda_max={lambda_max:.6g}", flush=True)

    def fit(l1_penalty, warmstart, restart):
        return GaussianProcess.fit(
            x_train,
            y_train,
            jnp.asarray(l1_penalty, dtype=DTYPE),
            profile=args.profile,
            warmstart=warmstart,
            restart=restart,
            max_iterations=args.maxiter,
        )

    # shared root fit at the top of the path, cold from theta_init, expected to die
    seconds, (model, nll, root, certificate) = time_call(
        lambda: fit(lambdas[0], None, True)
    )
    print(
        f"root lambda={float(lambdas[0]):.6g} fit={seconds:.2f}s"
        f" active={active_groups(root)} nll={float(nll):.6g}"
        f" rho={float(root.rho):.4g}",
        flush=True,
    )

    grad_norms = group_grad_norms(root.z, root.aux[0], x_flat, y_train, args.profile)
    keys = jax.random.split(jax.random.key(args.seed), args.n_paths)
    paths = [dict(state=root, grad_norms=grad_norms, key=key) for key in keys]

    for step, l1_penalty in enumerate(lambdas[1:], start=1):
        for path_id, path in enumerate(paths):
            path["key"], key_step = jax.random.split(path["key"])
            state, opened = kick(
                path["state"],
                path["grad_norms"],
                l1_penalty,
                key_step,
                theta_init,
                args,
            )
            seconds, (model, nll, state, certificate) = time_call(
                lambda: fit(l1_penalty, state, False)
            )
            path["state"] = state
            path["grad_norms"] = group_grad_norms(
                state.z, state.aux[0], x_flat, y_train, args.profile
            )

            active = active_groups(state)
            print(
                f"step={step} lambda={float(l1_penalty):.4g} path={path_id}"
                f" opened={int(opened.sum())} active={active}"
                f" fit={seconds:.2f}s iters={int(state.iteration)}"
                f" nll={float(nll):.6g} max_kkt={float(jnp.max(certificate)):.4g}",
                flush=True,
            )
            write_row(
                RESULTS_PATH,
                row
                | dict(
                    kind="path",
                    step=step,
                    l1_penalty=float(l1_penalty),
                    lambda_max=lambda_max,
                    path=path_id,
                    opened=int(opened.sum()),
                    active=active,
                    n_active=active.count("1"),
                    norms=[float(v) for v in group_norm(state.z)],
                    rho=float(state.rho),
                    fit_s=round(seconds, 3),
                    admm_iters=int(state.iteration),
                    primal_residual_eps=eps_multiple(state.primal_residual, DTYPE),
                    dual_residual_eps=eps_multiple(state.dual_residual, DTYPE),
                    max_kkt=float(jnp.max(certificate)),
                    nll=float(nll),
                ),
            )

        # cold baseline at the same lambda: what the warm starts must beat
        if args.cold_reference:
            seconds, (model, nll, state, certificate) = time_call(
                lambda: fit(l1_penalty, None, True)
            )
            print(
                f"step={step} lambda={float(l1_penalty):.4g} path=cold"
                f" active={active_groups(state)}"
                f" fit={seconds:.2f}s iters={int(state.iteration)}"
                f" nll={float(nll):.6g}",
                flush=True,
            )
            write_row(
                RESULTS_PATH,
                row
                | dict(
                    kind="cold",
                    step=step,
                    l1_penalty=float(l1_penalty),
                    lambda_max=lambda_max,
                    active=active_groups(state),
                    n_active=active_groups(state).count("1"),
                    norms=[float(v) for v in group_norm(state.z)],
                    rho=float(state.rho),
                    fit_s=round(seconds, 3),
                    admm_iters=int(state.iteration),
                    max_kkt=float(jnp.max(certificate)),
                    nll=float(nll),
                ),
            )
