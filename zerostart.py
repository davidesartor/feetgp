"""Does a fit started near zero stay at zero just below lambda_max?"""

import jax.numpy as jnp
from einops import rearrange, repeat

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
print(f"lambda_max={lambda_max:.6g}", flush=True)

theta_eps = repeat(EPSILON * theta_init, "d g -> o d g", o=o)
zeros = jnp.zeros_like(theta_eps)

for mult, x0, tag in [
    (1.05, theta_eps, "eps"),
    (0.99, theta_eps, "eps"),
    (0.90, theta_eps, "eps"),
    (1.05, repeat(theta_init, "d g -> o d g", o=o), "init"),
]:
    l1 = jnp.asarray(mult * lambda_max, dtype="float32")
    warm = ADMMState(x=x0, z=x0, u=zeros, aux=(jnp.zeros((o,)),))
    model, nll, state, cert = GaussianProcess.fit(
        x_train,
        y_train,
        l1,
        profile="rbf",
        warmstart=warm,
        restart=False,
        max_iterations=300,
    )
    active = "".join("1" if a else "0" for a in group_norm(state.z) > 0.0)
    print(
        f"mult={mult} start={tag} active={active} iters={int(state.iteration)}"
        f" nll={float(nll):.6g} max_kkt={float(jnp.max(cert)):.4g}"
        f" rho={float(state.rho):.4g}",
        flush=True,
    )
