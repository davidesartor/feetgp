import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import gp


jax.config.update("jax_enable_x64", True)
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)


# load and preprocess data
path = "OneDrive_1_03-02-2026/Left_bf_10kmh_Filtered.txt"
df = pd.read_csv(path, sep="\t", skiprows=[0, 2, 3, 4], index_col=1)
df = df.drop(columns=["Unnamed: 0"])
df.reset_index(inplace=True)
print(df.shape)
x = df.iloc[:, 1:].values
# filter nans
x = x[~np.isnan(x).any(axis=1)]

# subsample and make a mock regression problem
x = jnp.array(x[::10, :])
assert x.shape[1] % 3 == 0, "Number of features should be a multiple of 3"

# split into train and test
# x_train, x_test = x[: len(x) // 2], x[len(x) // 2 :]
# y_train, y_test = x[: len(x) // 2], x[len(x) // 2 :]
x_train, x_test = x[::2], x[1::2]
y_train, y_test = x[::2], x[1::2]
print("train:", x_train.shape, y_train.shape)
print("test:", x_test.shape, y_test.shape)
n, d = x_train.shape
n, o = y_train.shape


# train models with different regularization strengths
models = []
penalties = list(10 ** jnp.linspace(0, 6, 10))[::-1]
for i, l in enumerate(penalties):
    model = gp.GaussianProcessRegressor(
        l1_penalty=l,
        max_iterations=100,
        tollerance=1e-3,
        multi_init=20,
        #jitter=True,
        #theta_min=models[-1].parameters.theta if models else None,
    )
    model.fit(x=x_train, y=y_train)
    models.append(model)

    # evaluate models and plot global results
    preds = [model.predict(x=x_test) for model in models]
    r2 = [
        [r2_score(y_true=y_test[:, i], y_pred=pred.mean[i]) for i in range(d)]
        for pred in preds
    ]
    thetas = np.array([model.parameters.theta for model in models])
    gs = np.array([model.parameters.g for model in models])

    # reshape stuff
    r2 = np.array(r2).clip(min=1e-4)  # avoid log(0) in plot

    thetas = thetas.reshape(-1, o, d // 3, 3)
    thetas_avg = thetas.mean(-1).mean(-2)  # average over same marker
    thetas_max = thetas.max(-1).max(-2)  # max over same marker
    thetas_min = thetas.min(-1).min(-2)  # min over same marker
    r2 = r2.reshape(-1, d // 3, 3).mean(-1)  # average over same marker

    print()
    print("L1 penalty:", l)
    print("R2 values:", r2[-1])
    print("Learned max-lengthscales:", thetas_max[-1])
    print("Learned min-lengthscales:", thetas_min[-1])
    print()

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    for j in range(d // 3):
        plt.plot(penalties[: i + 1], (1 - r2[:, j]), marker="o", label=f"marker_{j}")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel("Regularization lambda")
    plt.ylabel("(1-R2) on test set")
    plt.title("Effect of regularization on GP performance")
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)

    for j in range(d // 3):
        plt.plot(penalties[: i + 1], thetas_avg[:, j], marker="o", label=f"marker_{j}")
        plt.fill_between(
            penalties[: i + 1], thetas_min[:, j], thetas_max[:, j], alpha=0.1
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Regularization lambda")
    plt.ylabel("Learned lengthscale")
    plt.title("Effect of regularization on learned lengthscale")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("figures/test1_results.pdf")
    plt.close()


# save model trajectory for visualization
for i, model in enumerate(models):
    x, z, u, rho = zip(*model.trajectory)
    np.savez(
        f"models/trajectory_{i}.npz",
        x=np.array(x),
        z=np.array(z),
        u=np.array(u),
        rho=np.array(rho),
        penalty=model.l1_penalty,
        x_train=np.array(x_train),
        y_train=np.array(y_train),
    )
