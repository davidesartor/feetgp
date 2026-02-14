import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import r2_score
import gp


jax.config.update("jax_enable_x64", True)


def f(x):
    x1, x2 = x[:, 0], x[:, 1]
    y = jnp.cos(jnp.pi * x1) + 0.1 * jnp.sin(3 * jnp.pi * x2)
    return y.reshape(-1, 1)


N = 30
x_test = jnp.meshgrid(jnp.linspace(-1.0, 1.0, N), jnp.linspace(-1.0, 1.0, N))
x_test = jnp.stack([x_test[0].ravel(), x_test[1].ravel()], axis=-1)
y_test = f(x_test)

x_train = jr.uniform(jr.key(0), (300, 2), minval=-1.0, maxval=1.0)
y_train = f(x_train)


models = []
regularizations = list(10 ** jnp.linspace(0, 6, 50))[::-1]
for l in regularizations:
    model = gp.GaussianProcessRegressor(
        l1_penalty=l,
        warm_start=models[-1].parameters if models else None,
        # max_iterations=16,
    )
    debug = model.fit(x=x_train, y=y_train)
    models.append(model)

preds = [model.predict(x=x_test) for model in models]
r2 = [r2_score(y_true=y_test[:, 0], y_pred=pred.mean[0]) for pred in preds]
thetas = [model.parameters.theta[0] for model in models]
gs = [model.parameters.g[0] for model in models]

regularizations = [r for r in regularizations]

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.semilogx(regularizations, r2, marker="o", label="y")
plt.ylim(-0.1, 1.1)
plt.xlabel("Regularization lambda")
plt.ylabel("R2 on test set")
plt.title("Effect of regularization on GP performance")
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.semilogx(regularizations, thetas, marker="o", label="lengthscale")
# plt.semilogx(regularizations, gs, marker="o", label="signal variance")
plt.xlabel("Regularization lambda")
plt.ylabel("Learned lengthscale")
plt.title("Effect of regularization on learned lengthscale")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("test_results.pdf")
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


t1 = 10 ** jnp.linspace(-6, 1, 30)
t2 = 10 ** jnp.linspace(-6, 1, 30)
theta = jnp.stack([t.ravel() for t in jnp.meshgrid(t1, t2)], axis=1)
loglikelihood = jax.jit(jax.vmap(gp.likelihood, in_axes=(0, None, None, None)))

for i, model in enumerate(gp.tqdm(models, desc="Plotting loss surfaces")):
    ll, _, _ = loglikelihood(theta, model.parameters.g[0], x_train, y_train[:, 0])
    z = -ll + model.l1_penalty * jnp.sum(jnp.abs(theta), -1)

    # loss surface plot
    plt.figure(figsize=(8, 6))
    plt.contourf(t1, t2, z.reshape(30, 30), levels=50, cmap="viridis")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Lengthscale 0")
    plt.ylabel("Lengthscale 1")
    plt.grid()
    plt.plot(
        *theta[jnp.argmin(z)],
        marker="o",
        color="white",
        label="Minimum of loss surface",
    )
    plt.plot(
        *model.parameters.theta[0], marker="x", color="red", label="Learned parameters"
    )

    xs = jnp.array([x for x, z, u, rho in model.trajectory])
    zs = jnp.array([z for x, z, u, rho in model.trajectory])
    us = jnp.array([u for x, z, u, rho in model.trajectory])
    plt.plot(
        *xs[:, 0, :2].clip(min=theta.min()+1e-8).T,
        marker="*",
        color="green",
        label="x path",
        alpha=0.5,
    )
    plt.plot(
        *zs[:, 0, :2].clip(min=theta.min()+1e-8).T,
        marker="*",
        color="blue",
        label="z path",
        alpha=0.5,
    )
    plt.plot(
        *((zs - us)[:, 0, :2].clip(min=theta.min()+1e-8).T),
        marker="*",
        color="black",
        label="z-u path",
        alpha=0.5,
    )

    plt.legend()
    plt.title(f"Lambda = {model.l1_penalty}")
    plt.savefig(f"test_{i}.pdf")
    plt.close()