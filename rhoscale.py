"""Scales that bound rho: group norms, nll curvature, and u growth at lambda_max."""

import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat

from feetgp.glasso_admm import ADMMState, group_norm
from feetgp.gp import GaussianProcess, hetgpy_auto_bounds
from feetgp.inclinerunning import InclineRunning

EPSILON = 1e-2
data = InclineRunning(
    path="data/Incline Running",
    train_size=128,
    seed=0,
    feet="both",
    target="forces",
    inclines="inc0",
)
x_train = jnp.asarray(data.x_train, dtype="float32")
y_train = jnp.asarray(data.y_train, dtype="float32")
n, d, g = x_train.shape
o = y_train.shape[1]
x_flat = rearrange(x_train, "n d g -> n (d g)")
theta_init = rearrange(hetgpy_auto_bounds(x_flat)[0], "(d g) -> d g", g=g)

lambda_max = float(
    GaussianProcess.lambda_max(x_train, y_train, profile="rbf", epsilon=EPSILON)
)
theta_eps = repeat(EPSILON * theta_init, "d g -> o d g", o=o)
theta_full = repeat(theta_init, "d g -> o d g", o=o)
zeros = jnp.zeros_like(theta_full)


def grad_norms(theta, log_nugget):
    def nll(theta, log_nugget, y):
        loss, _ = GaussianProcess.loss(
            theta, jnp.exp(log_nugget), x_flat, y, profile="rbf"
        )
        return loss

    grad = jax.vmap(jax.grad(nll))(theta, log_nugget, y_train.T)
    return jnp.sqrt(reduce(grad**2, "o d g -> g", "sum"))


print(f"lambda_max={lambda_max:.6g}", flush=True)
for tag, theta in [("eps", theta_eps), ("init", theta_full)]:
    norms = group_norm(theta)
    gn = grad_norms(theta, jnp.zeros((o,)))
    print(
        f"{tag}: group_norm min={float(norms.min()):.4g} max={float(norms.max()):.4g}"
        f" | grad_norm min={float(gn.min()):.4g} max={float(gn.max()):.4g}"
        f" | rho_upper=lambda_max/max_norm={lambda_max / float(norms.max()):.4g}",
        flush=True,
    )

# unpenalized MLE scale: what x runs to when rho is too small to hold the anchor
model, nll, state, cert = GaussianProcess.fit(
    x_train,
    y_train,
    jnp.asarray(0.0, dtype="float32"),
    profile="rbf",
    max_iterations=200,
)
mle_norms = group_norm(state.z)
print(
    f"mle: group_norm min={float(mle_norms.min()):.4g} max={float(mle_norms.max()):.4g}"
    f" rho_upper={lambda_max / float(mle_norms.max()):.4g}",
    flush=True,
)

# trace u growth at lambda_max from a zero z, sweeping fixed rho
for rho in [0.1, 1.0, 10.0, 100.0, lambda_max / float(mle_norms.max())]:
    warm = ADMMState(
        x=theta_eps,
        z=zeros,
        u=zeros,
        rho=jnp.asarray(rho, dtype="float32"),
        aux=(jnp.zeros((o,)),),
    )
    model, nll, state, cert = GaussianProcess.fit(
        x_train,
        y_train,
        jnp.asarray(lambda_max, dtype="float32"),
        profile="rbf",
        warmstart=warm,
        restart=False,
        max_iterations=300,
    )
    active = "".join("1" if a else "0" for a in group_norm(state.z) > 0.0)
    print(
        f"rho0={rho:.4g} active={active} iters={int(state.iteration)}"
        f" u_norm={float(jnp.linalg.norm(state.u)):.4g}"
        f" x_norm={float(jnp.linalg.norm(state.x)):.4g}"
        f" rho_final={float(state.rho):.4g} nll={float(nll):.6g}",
        flush=True,
    )
