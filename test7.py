import jax
import jax.numpy as jnp
import numpy as np

from sklearn.metrics import r2_score
import gp
import os
import pandas as pd
from einops import rearrange
import matplotlib.pyplot as plt

os.makedirs("test7", exist_ok=True)
jax.config.update("jax_enable_x64", True)

############################################################
# Parameters
############################################################

NORMALIZE = True
ADMM_MAX_ITER = 1000
ADMM_TOL = 1e-4
LAMBDAS = 10 ** jnp.linspace(-2, 4, 100)


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
MARKERS = [p + x for x in MARKERS for p in ("L", "R")]
GROUPS = 6  # LX, LY, LZ, RX, RY, RZ
SUBSAMPLE_FACTOR = 20  # 10Hz


############################################################
# Load and prepare data
############################################################

filenames = [
    "inc0_10kmh",
    "inc0_12kmh",
    "inc0_14kmh",
    "inc5_10kmh",
    "inc5_12kmh",
    "inc5_14kmh",
    "inc10_10kmh",
    "inc10_12kmh",
    "inc10_14kmh",
]

df_markers = []
for f in filenames:
    # ignore first 10 lines of metadata
    df = pd.read_csv(f"Incline Running/{f}.tsv", sep="\t", skiprows=10)
    df = df.iloc[:, :-1]  # drops last column which is empty
    # only keep columns that start with the marker names
    df = df[df.columns[df.columns.str.contains("|".join(MARKERS))]]
    # sort columns by marker names
    df = df[sorted(df.columns, key=lambda x: MARKERS.index(x[:-2]))]
    print(f"Read file: {f}, data with shape: {df.shape}")
    df_markers.append(df)
df_markers = pd.concat(df_markers, ignore_index=True)

# read tsv file, ignore first 10 lines of metadata
df_forces = []
for f in filenames:
    df = pd.read_excel(f"Incline Running/{f}_f_1.xlsx")
    df = df.iloc[:, 1:]  # drops first column which is the idx
    df = df.groupby(df.index // 5).mean()  # match the marker sampling rate
    df_forces.append(df)
    print(f"Read file: {f} with shape {df.shape}")
df_forces = pd.concat(df_forces, ignore_index=True)
df_forces


# convert to numpy and drop rows with nans
x = df_markers.values
y = np.cbrt(df_forces.values)  # cube root to make the distribution more normal
x = jnp.array(x[1::SUBSAMPLE_FACTOR])
y = jnp.array(y[1::SUBSAMPLE_FACTOR])
x, y = x[~np.isnan(x).any(axis=1)], y[~np.isnan(y).any(axis=1)]
x_train, x_test = x[::2, :], x[1::2, :]
y_train, y_test = y[::2, :], y[1::2, :]
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
    n_groups=1,
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
    glassos = jnp.linalg.norm(
        rearrange(thetas, "... o (m k) -> ... (o k) m", k=GROUPS), axis=-2
    )

    # plot progress
    plt.figure(figsize=[10, 5])
    colors = plt.cm.tab20(np.linspace(0, 1, len(MARKERS) // 2))
    plt.subplot(1, 2, 1)
    for j in range(len(MARKERS) // 2):
        plt.plot(LAMBDAS, glassos[:, j], color=colors[j], label=MARKERS[2*j][1:])
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(LAMBDAS, r2s[:, 0], label="X")
    plt.plot(LAMBDAS, r2s[:, 1], label="Y")
    plt.plot(LAMBDAS, r2s[:, 2], label="Z")
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.grid()
    plt.legend()

    plt.savefig(f"test7/regularization_effect.pdf")
    plt.close()

    # save intermediate model parameters
    np.savez(
        f"test7/model_{i}.npz",
        l=l,
        theta=model.parameters.theta,
        g=model.parameters.g,
        b=model.parameters.b,
        nu=model.parameters.nu,
    )

# save aggregate results
np.savez(
    f"test7/aggregate_results.npz",
    l=LAMBDAS,
    theta=thetas,
    r2=r2s,
    glasso=jnp.linalg.norm(
        rearrange(thetas, "... o (m k) -> ... (o k) m", k=GROUPS), axis=-2
    ),
)
