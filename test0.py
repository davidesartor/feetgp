import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import plotly.graph_objects as go
import gp
import os

os.makedirs("test0", exist_ok=True)
jax.config.update("jax_enable_x64", True)


############################################################
# Define a (simple) mock problem for testing
############################################################
def f(x):
    x1, x2 = x[:, 0], x[:, 1]
    y = jnp.cos(jnp.pi * x1) + 0.1 * jnp.sin(3 * jnp.pi * x2)
    return y.reshape(-1, 1)


N_GRID = 30
N_TRAIN = 100

x_test = jnp.meshgrid(jnp.linspace(-1.0, 1.0, N_GRID), jnp.linspace(-1.0, 1.0, N_GRID))
x_test = jnp.stack([x_test[0].ravel(), x_test[1].ravel()], axis=-1)
y_test = f(x_test)

x_train = jr.uniform(jr.key(42), (N_TRAIN, 2), minval=-1.0, maxval=1.0)
y_train = f(x_train)

print("Train points shape:", x_train.shape)  # (N_TRAIN, 2)
print("Test points shape:", x_test.shape)  # (N_GRID*N_GRID, 2)
print()


############################################################
# 3d plot of the function test/train points used
############################################################
print("Creating 3D scatter plot of test and train points...")
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=x_test[:, 0],
            y=x_test[:, 1],
            z=y_test[:, 0],
            mode="markers",
            name="Test Points",
        ),
        go.Scatter3d(
            x=x_train[:, 0],
            y=x_train[:, 1],
            z=y_train[:, 0],
            mode="markers",
            name="Train Points",
        ),
    ]
)
fig.update_traces(marker=dict(size=3))
fig.write_html("test0/function_surface.html")
print("Done!")
print()

############################################################
# Fit unregularized model
############################################################
print("Fitting and saving the model...")
model = gp.GaussianProcessRegressor(
    l1_penalty=0.0,
    max_admm_iterations=1000,
    max_bfgs_iterations=100,
    tollerance=1e-4,
    verbose=True,
)
model.fit(x=x_train, y=y_train)

xs = jnp.array([x for x, z, u, rho in model.trajectory])
zs = jnp.array([z for x, z, u, rho in model.trajectory])
us = jnp.array([u for x, z, u, rho in model.trajectory])
rhos = jnp.array([rho for x, z, u, rho in model.trajectory])
np.savez(
    "test0/model.npz",
    theta=model.parameters.theta,
    g=model.parameters.g,
    b=model.parameters.b,
    nu=model.parameters.nu,
    admm_x_trajectory=xs,
    admm_z_trajectory=zs,
    admm_u_trajectory=us,
    admm_rho_trajectory=rhos,
)
print("Done!")
print()


############################################################
# Interactive plot of loss surface
############################################################
print("Computing log-likelihood contours...")
theta_max = model.bounds[1, 0, :-1]
t1 = jnp.linspace(0, theta_max[0], 50)
t2 = jnp.linspace(0, theta_max[1], 50)
thetas = jnp.stack([t.ravel() for t in jnp.meshgrid(t1, t2)], axis=1)
gs = 10 ** jnp.linspace(-8, 2, 30)

loglikelihood_fn = jax.vmap(gp.likelihood, in_axes=(0, None, None, None))
ll = [loglikelihood_fn(thetas, g, x_train, y_train[:, 0])[0] for g in gs]
ll = jnp.stack(ll)

print("Creating interactive contour plot of log-likelihood...")
fig = go.Figure(
    data=[
        go.Contour(
            z=z,
            x=t1,
            y=t2,
            zmin=float(ll.min()),
            zmax=float(ll.max()),
            colorscale="Viridis",
            visible=i == 0,
        )
        for i, z in enumerate(ll.reshape(-1, len(t1), len(t2)))
    ],
)
fig.update_layout(
    sliders=[
        dict(
            currentvalue={"prefix": "g: "},
            pad={"t": 50},
            steps=[
                dict(
                    method="update",
                    args=[{"visible": [i == j for j in range(len(gs))]}],
                    label=f"{g:.1e}",
                )
                for i, g in enumerate(gs)
            ],
        )
    ]
)
fig.write_html("test0/likelihood_contours.html")
print("Done!")
print()


############################################################
# Plot the loss surface and ADMM trajectory
############################################################
print("Plotting the ADMM trajectory...")
xs = jnp.array([x[0] for x, z, u, rho in model.trajectory])
zs = jnp.array([z[0] for x, z, u, rho in model.trajectory])


def loss_fn(theta, g, admm_z, admm_u, rho):
    admm_x = jnp.concatenate([theta, g], axis=-1)
    return gp.admm_x_update_loss(admm_x, admm_z, admm_u, rho, x_train, y_train[:, 0])


loss_fn = jax.vmap(loss_fn, in_axes=(0, None, None, None, None))

traces = []
for i in range(len(model.trajectory) - 1):
    x, z, u, rho = model.trajectory[i]
    x_next, z_next, u_next, rho_next = model.trajectory[i + 1]
    g_next = x_next[:, -1]
    loss, _ = loss_fn(thetas, g_next, z, u, rho)
    true_argmin = thetas[jnp.argmin(loss)]
    traces += [
        go.Contour(
            z=loss.reshape(len(t1), len(t2)),
            x=t1,
            y=t2,
            zmin=float(loss.min()),
            zmax=float(loss.max()),
            colorscale="Viridis",
            showscale=False,
            visible=i == 0,
        ),
        go.Scatter(
            x=xs[: i + 1, 0],
            y=xs[: i + 1, 1],
            mode="markers+lines",
            marker=dict(color="red", size=10),
            line=dict(color="red", width=2),
            name=f"x",
            visible=i == 0,
        ),
        go.Scatter(
            x=zs[: i + 1, 0],
            y=zs[: i + 1, 1],
            mode="markers+lines",
            marker=dict(color="green", size=10, symbol="x"),
            line=dict(color="green", width=1),
            name=f"z",
            visible=i == 0,
        ),
        go.Scatter(
            x=[true_argmin[0]],
            y=[true_argmin[1]],
            mode="markers",
            marker=dict(
                color="white", size=10, symbol="star", line=dict(color="black", width=1)
            ),
            name=f"true argmin",
            visible=i == 0,
        ),
    ]

fig = go.Figure(data=traces)
fig.update_layout(
    sliders=[
        dict(
            currentvalue={"prefix": "step: "},
            pad={"t": 50},
            steps=[
                dict(
                    method="update",
                    args=[{"visible": [j // 4 == i for j in range(len(traces))]}],
                    label=f"{i}",
                )
                for i in range(len(traces) // 4)
            ],
        )
    ]
)
fig.write_html("test0/optimization_trajectory.html")
print("Done!")
print()
