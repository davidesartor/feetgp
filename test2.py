import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gp
import os

os.makedirs("test2", exist_ok=True)
jax.config.update("jax_enable_x64", True)

############################################################
# Variants
############################################################
multi_start = 5
jitter = True
log_scale_plots = False


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
fig.write_html("test2/function_surface.html")
print("Done!")
print()


############################################################
# Fit models using various regularizations
############################################################
print("Fitting and saving the models...")
models = []
lambdas = 10 ** jnp.linspace(0, 4, 10)
for l in lambdas:
    model = gp.GaussianProcessRegressor(
        l1_penalty=l,
        max_iterations=1000,
        tollerance=1e-4,
        multi_start=multi_start,
        jitter=jitter,
        verbose=True,
    )
    model.fit(x=x_train, y=y_train)
    models.append(model)

    xs = jnp.array([x for x, z, u, rho in model.trajectory])
    zs = jnp.array([z for x, z, u, rho in model.trajectory])
    us = jnp.array([u for x, z, u, rho in model.trajectory])
    rhos = jnp.array([rho for x, z, u, rho in model.trajectory])
    np.savez(
        f"test2/model_{len(models)}_l1={l:.1e}.npz",
        l1_penalty=l,
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
    # Plot the loss surface and ADMM trajectory
    ############################################################
    print("Plotting the ADMM trajectory...")
    theta_max = model.bounds[1, 0, :-1]
    t1 = jnp.exp(jnp.linspace(jnp.log(1e-10), jnp.log(theta_max[0]), 50))
    t2 = jnp.exp(jnp.linspace(jnp.log(1e-10), jnp.log(theta_max[1]), 50))
    thetas = jnp.stack([t.ravel() for t in jnp.meshgrid(t1, t2)], axis=1)
    xs = jnp.array([x[0] for x, z, u, rho in model.trajectory])
    zs = jnp.array([z[0] for x, z, u, rho in model.trajectory])

    def loss_fn(theta, g, admm_z, admm_u, rho):
        admm_x = jnp.concatenate([theta, g], axis=-1)
        return gp.admm_x_update_loss(
            admm_x, admm_z, admm_u, rho, x_train, y_train[:, 0]
        )

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
                x=xs[: i + 1, 0].clip(t1.min(), t1.max()),  # avoid log(0) in the plot
                y=xs[: i + 1, 1].clip(t2.min(), t2.max()),  # avoid log(0) in the plot
                mode="markers+lines",
                marker=dict(color="red", size=10),
                line=dict(color="red", width=2),
                name=f"x",
                visible=i == 0,
            ),
            go.Scatter(
                x=zs[: i + 1, 0].clip(t1.min(), t1.max()),  # avoid log(0) in the plot
                y=zs[: i + 1, 1].clip(t2.min(), t2.max()),  # avoid log(0) in the plot
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
                    color="white",
                    size=10,
                    symbol="star",
                    line=dict(color="black", width=1),
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
    if log_scale_plots:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    fig.write_html(f"test2/optimization_trajectory_{len(models)}_l1={l:.8f}.html")
    print("Done!")
    print()


############################################################
# Plot the effect of the regularization on the fit
############################################################
print("Plotting the effect of the regularization on the fit...")

y_preds = [model.predict(x_test) for model in models]
r2_scores = [r2_score(y_test[:, 0], y_pred.mean[0, :]) for y_pred in y_preds]
thetas = jnp.array([model.parameters.theta[0] for model in models])
gs = jnp.array([model.parameters.g[0] for model in models])

fig = make_subplots(rows=2, cols=1)
fig.add_trace(
    go.Scatter(x=lambdas, y=r2_scores, mode="markers+lines", name="R2 Score"),
    row=1,
    col=1,
)
for d in range(2):
    fig.add_trace(
        go.Scatter(
            x=lambdas,
            y=thetas[:, d],
            mode="markers+lines",
            name=f"theta_{d}",
        ),
        row=2,
        col=1,
    )
fig.add_trace(
    go.Scatter(
        x=lambdas,
        y=gs,
        mode="markers+lines",
        name="g",
    ),
    row=2,
    col=1,
)
fig.update_xaxes(type="log", row=1, col=1)
fig.update_xaxes(type="log", row=2, col=1)
fig.write_html(f"test2/effect_of_regularization.html")
print("Done!")
print()
