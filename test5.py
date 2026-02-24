import jax
import jax.numpy as jnp
import numpy as np

from sklearn.metrics import r2_score
import gp
import os
import pandas as pd
from einops import rearrange
import matplotlib.pyplot as plt

os.makedirs("test5", exist_ok=True)
jax.config.update("jax_enable_x64", True)

############################################################
# Parameters
############################################################

NORMALIZE = True
ADMM_MAX_ITER = 500
ADMM_TOL = 1e-3
LAMBDAS = 10 ** jnp.linspace(1, 6, 100)


MARKERS = [
    "CAL1",
    "CUB",
    "LCAL",
    "LMAL",
    "MCAL",
    "MMAL",
    "MT1B",
    "MT1H",
    "MT2H",
    "MT5B",
    "MT5H",
    "NAV",
    "TOE",
]
MARKERS = [f"L{m}" for m in MARKERS]
GROUPS = 3  # x, y, z
SUBSAMPLE_FACTOR = 20  # 10Hz


############################################################
# Load and prepare data
############################################################

filenames = [
    "inc0_10kmh.tsv",
    "inc0_12kmh.tsv",
    "inc0_14kmh.tsv",
    "inc5_10kmh.tsv",
    "inc5_12kmh.tsv",
    "inc5_14kmh.tsv",
    "inc10_10kmh.tsv",
    "inc10_12kmh.tsv",
    "inc10_14kmh.tsv",
]

dfs = []
for f in filenames:
    # ignore first 10 lines of metadata
    df = pd.read_csv(f"Incline Running/{f}", sep="\t", skiprows=10)
    df = df.iloc[:, :-1]  # drops last column which is empty
    # only keep columns that start with the marker names
    df = df[df.columns[df.columns.str.contains("|".join(MARKERS))]]
    # sort columns by marker names
    df = df[sorted(df.columns, key=lambda x: MARKERS.index(x[:-2]))]
    print("Read file:", f, " data with shape:", df.shape)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

# convert to numpy and drop rows with nans
x = df.values
x = x[~np.isnan(x).any(axis=1)]
x = jnp.array(x[1::SUBSAMPLE_FACTOR])
x_train, x_test = x[::2, :], x[1::2, :]
y_train, y_test = x[::2, :], x[1::2, :]
print("train:", x_train.shape, y_train.shape)
print("test:", x_test.shape, y_test.shape)
n, d = x_train.shape
n, o = y_train.shape


if NORMALIZE:  # normalize inputs and standardize outputs
    y_mean = jnp.mean(y_train, axis=0, keepdims=True)
    y_std = jnp.std(y_train, axis=0, keepdims=True)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    x_min = jnp.min(x_train, axis=0, keepdims=True)
    x_max = jnp.max(x_train, axis=0, keepdims=True)
    x_train = (x_train - x_min) / (x_max - x_min)
    x_test = (x_test - x_min) / (x_max - x_min)

print("Train points shape:", x_train.shape)
print("Test points shape:", x_test.shape)
print()

#############################################################
## Single yolo run
#############################################################


model = gp.GaussianProcessRegressor(
    x_train=x_train,
    y_train=y_train,
    n_groups=GROUPS,
    max_iterations=ADMM_MAX_ITER,
    tollerance=ADMM_TOL,
    verbose=False,
)


thetas = np.nan * np.zeros([len(LAMBDAS), o, d])
r2s = np.nan * np.zeros([len(LAMBDAS), o])

for i, l in enumerate(LAMBDAS):
    # keep going with the optimization using new lambda
    model.fit(l1_penalty=l)

    # compute R2 score for each output dimension
    y_pred = model.predict(x_test).mean
    r2s[i, :] = jnp.array([r2_score(y_test[:, i], y_pred[i, :]) for i in range(o)])
    # compute glasso penalty for each output dimension
    thetas[i, :, :] = model.parameters.theta

    # aggregate each marker
    r2_avgs = rearrange(r2s, "... (m k) -> ... m k", k=GROUPS).mean(-1)
    glassos = jnp.linalg.norm(
        rearrange(thetas, "... o (m k) -> ... (o k) m", k=GROUPS), axis=-2
    )

    # plot progress
    plt.figure(figsize=[10, 5])
    colors = plt.cm.tab20(np.linspace(0, 1, len(MARKERS)))
    plt.subplot(1, 2, 1)
    for j in range(len(MARKERS)):
        plt.plot(LAMBDAS, glassos[:, j], color=colors[j], label=MARKERS[j][1:])
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    for j in range(len(MARKERS)):
        plt.plot(LAMBDAS, r2_avgs[:, j], color=colors[j], label=MARKERS[j][1:])
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.grid()
    plt.legend()

    plt.savefig(f"test5/regularization_effect.pdf")
    plt.close()

    # save intermediate model parameters
    np.savez(
        f"test5/model_{i}.npz",
        l=l,
        theta=model.parameters.theta,
        g=model.parameters.g,
        b=model.parameters.b,
        nu=model.parameters.nu,
    )

# save aggregate results
np.savez(
    f"test5/aggregate_results.npz",
    l=LAMBDAS,
    theta=thetas,
    r2=r2s,
    r2_avg=rearrange(r2s, "... (m k) -> ... m k", k=GROUPS).mean(-1),
    glasso=jnp.linalg.norm(
        rearrange(thetas, "... o (m k) -> ... (o k) m", k=GROUPS), axis=-2
    ),
)
