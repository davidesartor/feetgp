from scipy.optimize import minimize as scipy_optimize_minimize
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

EPS = jnp.sqrt(jnp.finfo(float).eps)


def squared_distance(x1, x2, order: int = 2):
    def dist(a, b):
        # IMPORTANT: use lax.cond to avoid propagating NaNs in the gradient
        return jax.lax.cond(
            jnp.all(a == b),
            lambda: 0.0,
            lambda: jnp.linalg.norm(a - b, ord=order),
        )

    dist = jax.vmap(jax.vmap(dist, (0, None)), (None, 0))
    return dist(x1, x2) ** 2


def cov_gen(X1, X2, theta):
    B = X1 / jnp.sqrt(theta)
    A = X2 / jnp.sqrt(theta)
    d = squared_distance(A, B)
    return jnp.exp(-d)


def auto_bounds(X, min_cor=0.01, max_cor=0.5):
    # rescale X to [0,1]^d
    x_min, x_max = X.min(axis=0), X.max(axis=0)
    X = (X - x_min) @ jnp.diag(1 / (x_max - x_min))

    # compute pairwise distances only for proper pairs
    dists = squared_distance(X, X)
    dists = dists[jnp.tril(dists, k=-1) > 0]

    # magic initialization using inverse of kernel
    lower = -jnp.quantile(dists, q=0.05) / jnp.log(min_cor) * (x_max - x_min) ** 2
    upper = -jnp.quantile(dists, q=0.95) / jnp.log(max_cor) * (x_max - x_min) ** 2
    return lower, upper


class homGP:
    def mleHomGP(self, X0, Z0):
        N, D = X0.shape

        # Determine bounds
        lower, upper = auto_bounds(X0)
        bounds = [(l, u) for l, u in zip(lower, upper)] + [(EPS, 1e2)]

        # Define initial parameters
        init_theta = 0.9 * lower + 0.1 * upper
        init_g = jnp.array([0.1])
        par_init = jnp.hstack((init_theta, init_g))

        # Definition of mle loss
        @jax.jit
        @jax.value_and_grad
        def fn(params, X0, Z0):
            N, D = X0.shape
            theta, g = params[:-1], params[-1]

            # foward of kernel
            C = cov_gen(X0, X0, theta)
            K = C + jnp.eye(N) * (EPS + g)

            # computes the trend with kriging interpolator
            Ki = jnp.linalg.inv(K)
            b = Ki.sum(axis=1) @ Z0 / Ki.sum()

            # likelihood when marginalizing over trend and variance
            psi = (Z0 - b).T @ jnp.linalg.solve(K, (Z0 - b))
            loglik = -0.5 * (N * jnp.log(1 / N * psi) + jnp.linalg.slogdet(K).logabsdet)
            return -loglik

        # The actual optimization call
        out = scipy_optimize_minimize(
            fun=fn,
            args=(X0, Z0),
            x0=par_init,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=100, ftol=EPS, gtol=0),
        )

        # extract the results and save stuff
        self.theta, self.g = out["x"][:-1], out["x"][-1]
        self.X0 = X0
        self.Z0 = Z0

        # compute optimal trend and variance
        K = cov_gen(X0, X0, self.theta) + jnp.eye(N) * (EPS + self.g)
        Ki = jnp.linalg.inv(K)
        self.b = Ki.sum(axis=1) @ self.Z0 / Ki.sum()
        self.nu = 1 / N * (self.Z0 - self.b).T @ Ki @ (self.Z0 - self.b)
        return self

    def predict(self, x):
        # here i'm recomputing for clarity, ideally this would be cached
        N, D = self.X0.shape
        Koo = self.nu * (
            cov_gen(self.X0, self.X0, self.theta) + jnp.eye(N) * (EPS + self.g)
        )
        Kxo = self.nu * cov_gen(X1=x, X2=self.X0, theta=self.theta)
        Kxx = self.nu * cov_gen(X1=x, X2=x, theta=self.theta)

        # posterior mean and covariance
        mean = self.b + Kxo @ jnp.linalg.solve(Koo, self.Z0 - self.b)
        cov = Kxx - Kxo @ jnp.linalg.solve(Koo, Kxo.T)

        # Add correction based on the trend estimation correlation
        Kbx = jnp.ones((1, N)) @ jnp.linalg.solve(Koo, Kxo.T)
        cov = cov + (1 - Kbx).T @ (1 - Kbx) / jnp.linalg.inv(Koo).sum()

        preds = dict(mean=mean, cov=cov)
        return preds
