import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gp
import os
import pandas as pd

os.makedirs("test3", exist_ok=True)
jax.config.update("jax_enable_x64", True)

############################################################
# Variants
############################################################
multi_start = 10
subsample_factor = 100
inputs = 10
outputs = 10

############################################################
# Define a (simple) mock problem for testing
############################################################
path = "OneDrive_1_03-02-2026/Left_bf_10kmh_Filtered.txt"
df = pd.read_csv(path, sep="\t", skiprows=[0, 2, 3, 4], index_col=1)
df = df.drop(columns=["Unnamed: 0"])
df.reset_index(inplace=True)
print(df.shape)
x = df.iloc[:, 1:].values
x = x[~np.isnan(x).any(axis=1)]

# subsample and make a mock regression problem
x = jnp.array(x[::subsample_factor, :])
x_train, x_test = x[::2, :inputs], x[1::2, :inputs]
y_train, y_test = x[::2, :outputs], x[1::2, :outputs]
print("train:", x_train.shape, y_train.shape)
print("test:", x_test.shape, y_test.shape)
n, d = x_train.shape
n, o = y_train.shape

print("Train points shape:", x_train.shape)  # (N_TRAIN, 2)
print("Test points shape:", x_test.shape)  # (N_GRID*N_GRID, 2)
print()


############################################################
# Fit models using various regularizations
############################################################
print("Fitting and saving the models...")
models = []
lambdas = 10 ** jnp.linspace(0, 4, 10)[::-1]
for l in lambdas:
    model = gp.GaussianProcessRegressor(
        l1_penalty=l,
        max_iterations=1000,
        tollerance=1e-4,
        multi_start=multi_start,
        init_theta=models[-1].parameters.theta if models else None,
        verbose=True,
    )
    model.fit(x=x_train, y=y_train)
    models.append(model)

    xs = jnp.array([x for x, z, u, rho in model.trajectory])
    zs = jnp.array([z for x, z, u, rho in model.trajectory])
    us = jnp.array([u for x, z, u, rho in model.trajectory])
    rhos = jnp.array([rho for x, z, u, rho in model.trajectory])
    np.savez(
        f"test3/model_{len(models)}_l1={l:.1e}.npz",
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
# Plot the effect of the regularization on the fit
############################################################
print("Plotting the effect of the regularization on the fit...")

y_preds = [model.predict(x_test) for model in models]
r2 = [
    [r2_score(y_test[:, i], y_pred.mean[i, :]) for y_pred in y_preds] for i in range(o)
]
r2 = jnp.array(r2)
thetas = jnp.array([model.parameters.theta.sum(-2) for model in models])
gs = jnp.array([model.parameters.g.sum() for model in models])

fig = make_subplots(rows=2, cols=1)
for i in range(o):
    fig.add_trace(
        go.Scatter(x=lambdas, y=r2[i], mode="markers+lines", name=f"R2_{i} Score"),
        row=1,
        col=1,
    )
for j in range(d):
    fig.add_trace(
        go.Scatter(
            x=lambdas,
            y=thetas[:, j],
            mode="markers+lines",
            name=f"theta_{j}",
        ),
        row=2,
        col=1,
    )
fig.add_trace(
    go.Scatter(
        x=lambdas,
        y=gs,
        mode="markers+lines",
        name=f"g",
    ),
    row=2,
    col=1,
)
fig.update_xaxes(type="log", row=1, col=1)
fig.update_xaxes(type="log", row=2, col=1)
fig.write_html(f"test3/effect_of_regularization.html")
print("Done!")
print()
