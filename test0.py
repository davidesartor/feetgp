import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import r2_score
import gp


jax.config.update("jax_enable_x64", True)


# define the problem
def f(x):
    x1, x2 = x[:, 0], x[:, 1]
    y = jnp.cos(jnp.pi * x1) + 0.1 * jnp.sin(3 * jnp.pi * x2)
    return y.reshape(-1, 1)


N = 30
x_test = jnp.meshgrid(jnp.linspace(-1.0, 1.0, N), jnp.linspace(-1.0, 1.0, N))
x_test = jnp.stack([x_test[0].ravel(), x_test[1].ravel()], axis=-1)
y_test = f(x_test)

x_train = jr.uniform(jr.key(0), (100, 2), minval=-1.0, maxval=1.0)
y_train = f(x_train)


# train models with different regularization strengths
models = []
regularizations = list(10 ** jnp.linspace(0, 4, 20))[::-1]
for i, l in enumerate(regularizations):
    model = gp.GaussianProcessRegressor(
        l1_penalty=l,
        max_iterations=1000,
        tollerance=1e-4,
        jitter=True,
        warm_start_theta=models[-1].parameters.theta if models else None,
    )
    model.fit(x=x_train, y=y_train)
    models.append(model)

    # visualize the loss surface and ADMM trajectory
    xs = jnp.array([x for x, z, u, rho in model.trajectory])
    zs = jnp.array([z for x, z, u, rho in model.trajectory])
    us = jnp.array([u for x, z, u, rho in model.trajectory])
    lims = (min(xs.min(), zs.min() - 0.01), max(xs.max(), zs.max()) + 0.01)
    t1 = jnp.linspace(*lims, 50)
    t2 = jnp.linspace(*lims, 50)
    theta = jnp.stack([t.ravel() for t in jnp.meshgrid(t1, t2)], axis=1)
    loglikelihood = jax.jit(jax.vmap(gp.likelihood, in_axes=(0, None, None, None)))

    plt.figure(figsize=(8, 6))
    # plot the loss surface
    ll, _, _ = loglikelihood(theta, model.parameters.g[0], x_train, y_train[:, 0])
    z = -ll + model.l1_penalty * jnp.sum(jnp.abs(theta), axis=-1)
    plt.contourf(t1, t2, z.reshape(50, 50), levels=50, cmap="viridis")
    plt.xlabel("Lengthscale 0")
    plt.ylabel("Lengthscale 1")
    plt.grid()

    # plot the ADMM trajectory
    plt.plot(*xs[:, 0, :2].T, marker="*", color="green", label="x path", alpha=0.5)
    plt.plot(*zs[:, 0, :2].T, marker="*", color="blue", label="z path", alpha=0.5)

    # plot the minimum and learned parameters
    plt.plot(*theta[jnp.argmin(z)], marker="o", color="white", label="minimum")
    plt.plot(
        *model.parameters.theta[0], marker="x", color="red", label="Learned parameters"
    )
    plt.legend()
    plt.title(f"Lambda = {model.l1_penalty}")
    plt.savefig(f"figures/test_{i}.pdf")
    plt.close()


# evaluate models and plot global results
preds = [model.predict(x=x_test) for model in models]
r2 = np.array([r2_score(y_true=y_test[:, 0], y_pred=pred.mean[0]) for pred in preds])
thetas = np.array([model.parameters.theta[0] for model in models])
gs = np.array([model.parameters.g[0] for model in models])

regularizations = [r for r in regularizations]

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(regularizations, r2, marker="o", label="y")
plt.xscale("log")
plt.xlabel("Regularization lambda")
plt.ylabel("R2 on test set")
plt.title("Effect of regularization on GP performance")
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(regularizations, thetas, marker="o", label="lengthscale")
# plt.plot(regularizations, gs, marker="o", label="nugget", color="black", alpha=0.1)
plt.xscale("log")
plt.xlabel("Regularization lambda")
plt.ylabel("Learned lengthscale")
plt.title("Effect of regularization on learned lengthscale")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/test0_results.pdf")
plt.close()


# save model trajectory for visualization
import numpy as np

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
