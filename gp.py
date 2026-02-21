from typing import NamedTuple, Optional
from jaxtyping import Array, Float, Key
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.numpy.linalg import norm
from einops import rearrange
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt

EPS = float(jnp.sqrt(jnp.finfo(float).eps))

ADAPT_RHO = False
#ADAPT_RHO = True

SELF_ZERO = True
#SELF_ZERO = False

def squared_distance(
    x1: Float[Array, "n d"],
    x2: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    def dist(a, b):
        return jnp.sum((a - b) ** 2)

    dist = jax.vmap(jax.vmap(dist, (None, 0)), (0, None))
    return dist(x1, x2)

def kernel(
    x1: Float[Array, "n d"],
    x2: Float[Array, "m d"],
    theta: Float[Array, "d"],
) -> Float[Array, "n m"]:
    return jnp.exp(-squared_distance(x1 * theta, x2 * theta))


def likelihood(
    theta: Float[Array, "d"],
    g: Float[Array, ""],
    x: Float[Array, "n d"],
    y: Float[Array, "n"],
):
    n, d = x.shape

    # foward of kernel
    K = kernel(x, x, theta)
    K = K + jnp.eye(n) * (EPS + g)

    # cholesky of K and compute logdet
    K_sqrt, is_lower = jsp.linalg.cho_factor(K)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    # compute Ki_1=(K^-1 @ 1) and Ki_y=(K^-1 @ y)
    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([jnp.ones_like(y), y], 1),
    ).T

    # compute optimal trend b
    b = (Ki_1 * y).sum() / Ki_1.sum()
    nu = jnp.dot((y - b) / n, (Ki_y - Ki_1 * b))

    # # likelihood when marginalizing over trend and variance
    loglik = -0.5 * (n * jnp.log(nu) + logdetK)
    return loglik, b, nu


@jax.jit
@jax.value_and_grad
def admm_x_update_loss(
    admm_x: Float[Array, "d+1"],
    admm_z: Float[Array, "d+1"],
    admm_u: Float[Array, "d+1"],
    rho: float,
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n"],
):
    #theta, g = admm_x[:-1], admm_x[-1]
    theta, _ = admm_x[:-1], admm_x[-1]
    #g = 0.1
    #g = 0.01
    g = 0.001
    print("ignoring g arg.")
    loglik, _, _ = likelihood(theta, g, x_train, y_train)
    lagrangian = 0.5 * rho * jnp.sum((admm_x - admm_z + admm_u) ** 2)
    return -loglik + lagrangian



def admm_x_update(
    admm_x: Float[Array, "o d+1"],
    admm_z: Float[Array, "o d+1"],
    admm_u: Float[Array, "o d+1"],
    rho: float,
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n o"],
    bounds: Float[Array, "2 o d+1"],
) -> Float[Array, "o d+1"]:
    new_x = []
    for i, x, z, u, y, bmin, bmax in zip(range(admm_x.shape[0]), admm_x, admm_z, admm_u, y_train.T, *bounds):
        #print(i)
        x_train_i = jnp.copy(x_train)
        if SELF_ZERO:
            # Stores indices of columns pertaining to current marker. 
            cm_dims = jnp.arange(x_train.shape[0]).reshape([-1,3])[i//3,:] 
            x_train_i = x_train_i.at[:,cm_dims].set(0.)

        ret = scipy.optimize.minimize(
            fun=admm_x_update_loss,
            x0=x,
            args=(z, u, rho, x_train_i, y),
            jac=True,
            method="L-BFGS-B",
            bounds=[(a, b) for a, b in zip(bmin, bmax)],
            #options=dict(maxiter=10, ftol=EPS, gtol=0),
            options=dict(maxiter=1000, ftol=EPS, gtol=0),
        ).x
        new_x.append(ret)
    #
    new_x = jnp.stack(new_x, axis=0)
    return new_x


@jax.jit
def admm_z_update(
    admm_x: Float[Array, "o d+1"],
    admm_z: Float[Array, "o d+1"],
    admm_u: Float[Array, "o d+1"],
    rho: float,
    l1_penalty: float,
    bounds: Float[Array, "2 o d+1"],
) -> Float[Array, "o d+1"]:
    theta = (admm_x + admm_u)[:, :-1]
    g = (admm_x + admm_u)[:, -1]
    theta = rearrange(theta, "o (d k) -> (o k) d", k=3)
    prox = jnp.maximum(0, 1 - (l1_penalty / rho / norm(theta, axis=0)))
    theta = prox * theta
    theta = rearrange(theta, "(o k) d -> o (d k)", k=3)
    new_z = admm_z.at[:, :-1].set(theta)
    new_z = new_z.at[:, -1].set(g)  # no regularization on g
    return new_z.clip(*bounds)  # in case bounds do not include 0


def admm(
    x_train: Float[Array, "n d"],
    y_train: Float[Array, "n o"],
    bounds: Float[Array, "2 o d+1"],
    l1_penalty: float,
    x0: Float[Array, "o d+1"] | None = None,
    z0: Float[Array, "o d+1"] | None = None,
    u0: Float[Array, "o d+1"] | None = None,
    rho : float | None = None,
    max_iterations: int = 100,
    tollerance: float = 1e-4,
):
    if x0 is None:
        admm_x = jnp.zeros([y_train.shape[1], x_train.shape[1]+1])
    else:
        admm_x = x0
    if z0 is None:
        admm_z = x0
    else:
        admm_z = z0
    if u0 is None:
        admm_u = jnp.zeros_like(x0)
    else:
        admm_u = u0
    #rho = 1.0
    #print("rho is diff!")
    #rho = 1e5
    print("rho is lam!")
    rho = l1_penalty

    trajectory = [(admm_x, admm_z, admm_u, rho)]
    for iter in (pbar := tqdm(range(max_iterations), desc="ADMM")):
        new_admm_x = admm_x_update(
            admm_x, admm_z, admm_u, rho, x_train, y_train, bounds
        )
        #print(f"New x: {admm_x}")
        new_admm_z = admm_z_update(new_admm_x, admm_z, admm_u, rho, l1_penalty, bounds)
        #print(f"New z: {admm_z}")
        new_admm_u = admm_u + new_admm_x - new_admm_z

        # check convergence
        primal_residual = norm(new_admm_x - new_admm_z, axis=-1)
        primal_target = jnp.maximum(norm(admm_x, axis=-1), norm(admm_z, axis=-1))
        primal_ok = primal_residual < EPS + tollerance * primal_target
        dual_residual = rho * norm(new_admm_z - admm_z, axis=-1)
        dual_target = rho * norm(admm_u, axis=-1)
        dual_ok = dual_residual < EPS + tollerance * dual_target

        # update rho to balance primal and dual residuals
        if ADAPT_RHO:
            if primal_residual.sum() > 10 * dual_residual.sum():
                rho = 2 * rho
                new_admm_u = new_admm_u / 2
            elif dual_residual.sum() > 10 * primal_residual.sum():
                rho = rho / 2
                new_admm_u = new_admm_u * 2

        pbar.set_postfix(
            {
                "primal:": f"{(primal_residual / (primal_target + EPS)).max():.5f}",
                "dual:": f"{(dual_residual / (dual_target + EPS)).max():.5f}",
                "rho": rho,
            }
        )

        # update state and possibly early stop
        admm_x, admm_z, admm_u = new_admm_x, new_admm_z, new_admm_u
        trajectory.append((admm_x, admm_z, admm_u, rho))
        if primal_ok.all() and dual_ok.all():
            break
    else:
        print("ADMM did not converge within the maximum number of iterations.")
    return trajectory


def hetgpy_auto_bounds(x, min_cor=0.01, max_cor=0.5):
    # rescale X to [0,1]^d
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x = (x - x_min) @ jnp.diag(1 / (x_max - x_min))
    # compute pairwise distances only for proper pair
    dists = squared_distance(x, x)
    dists = dists[jnp.tril(dists, k=-1) > 0]
    # magic initialization using inverse of kernel
    lower = -jnp.quantile(dists, q=0.05) / jnp.log(min_cor) * (x_max - x_min) ** 2
    upper = -jnp.quantile(dists, q=0.95) / jnp.log(max_cor) * (x_max - x_min) ** 2
    return lower, upper


class Gaussian(NamedTuple):
    mean: Float[Array, "n"]
    cov: Float[Array, "n n"]


class Parameters(NamedTuple):
    theta: Float[Array, "o d"]
    g: Float[Array, "o"]
    b: Float[Array, "o"]
    nu: Float[Array, "o"]


@dataclass
class GaussianProcessRegressor:
    l1_penalty: float
    max_iterations: int = 1000
    tollerance: float = 1e-4
    seed: int = 42
    verbose: bool = False
    init_theta: Optional[Float[Array, "o d"]] = None
    init_g: Optional[Float[Array, "o"]] = None

    # params to fit after training
    parameters: Parameters = field(init=False)
    x: Float[Array, "n d"] = field(init=False)
    y: Float[Array, "n o"] = field(init=False)
    trajectory: list = field(init=False)
    bounds: Float[Array, "2 o d"] = field(init=False)

    def fit(self, x: Float[Array, "n d"], y: Float[Array, "n 1"], method ='admm'):
        n, d = x.shape
        n, o = y.shape

        # initialization
        # hetgpy uses a different parametrization for theta
        # i.e. out_theta = 1 / sqrt(hetgpy_theta)
        lower, upper = hetgpy_auto_bounds(x)
        init_theta = jnp.tile(1 / (0.9 * upper + 0.1 * lower) ** 0.5, (o, 1))
        init_g = jnp.array([0.1] * o)
        if self.init_theta is not None:
            init_theta = self.init_theta
        if self.init_g is not None:
            init_g = self.init_g
        if self.verbose:
            print(f"Initial theta: {init_theta}")
            print(f"Initial g: {init_g}")
            print()

        # bounds for optimization
        # also here we need to transform and invert the theta range w.r.t. hetgpy
        theta_bounds = jnp.array([1 / upper**0.5, 1 / lower**0.5])
        g_bounds = jnp.array([EPS, 1e2])
        bounds = jnp.concat([theta_bounds, g_bounds[:, None]], axis=-1)
        bounds = jnp.tile(bounds[:, None, :], (1, o, 1))
        bounds = bounds.at[0, ..., :-1].set(0.0)
        self.bounds = bounds
        if self.verbose:
            print(f"Bounds for theta:")
            print(f"Min: {self.bounds[0, ..., :-1]}")
            print(f"Max: {self.bounds[1, ..., :-1]}")
            print(f"Bounds for g:")
            print(f"Min: {self.bounds[0, ..., -1]}")
            print(f"Max: {self.bounds[1, ..., -1]}")
            print()

        results = []
        x0 = jnp.concatenate([init_theta, init_g[:, None]], axis=-1)

        # run admm
        if method=='admm':
            trajectory = admm(
                x_train=x,
                y_train=y,
                bounds=self.bounds,
                l1_penalty=self.l1_penalty,
                x0=x0,
                max_iterations=self.max_iterations,
                tollerance=self.tollerance,
            )
        elif method=='pgd':
            import IPython; IPython.embed()

            def prox_l1(theta, lam):
                theta = rearrange(theta, "o (d k) -> (o k) d", k=3)
                prox = jnp.maximum(0, 1 - (lam / norm(theta, axis=0)))
                theta = prox * theta
                theta = rearrange(theta, "(o k) d -> o (d k)", k=3)
                return theta

            x_use = jnp.stack([x for _ in range(x.shape[1]) ])
            if SELF_ZERO:
                for i in range(x.shape[1]):
                    cm_dims = jnp.arange(x.shape[0]).reshape([-1,3])[i//3,:] 
                    x_use = x_use.at[i,:,cm_dims].set(0.)
            
            def _total_nll(theta, g, x_use, y):
                print("assuming that we are not deleting self!")
                #likelihood(theta[1,:], g[1,:], x, y[:,1])
                lls, _, _ = jax.vmap(likelihood, [0,0,0,1])(theta, g, x_use, y)
                return(-jnp.sum(lls))
            total_nll = jax.jit(jax.value_and_grad(_total_nll, argnums=(0,1)))

            theta = init_theta
            g = init_g[:,None]
            #g = jnp.ones_like(init_g[:,None])

            #lr_theta = 1e-3
            lr_theta = 1e-4
            iters = 1000    

            learn_g = False
            #learn_g = True
            chill_g = True

            #g_cands = jnp.logspace(jnp.log10(g_bounds[0]), jnp.log10(g_bounds[1]),num=10)

            traj_theta = jnp.nan*jnp.zeros((iters,)+theta.shape)
            traj_g = jnp.nan*jnp.zeros((iters,)+g.shape)
            costs = jnp.nan*jnp.zeros(iters)
            costs_f = jnp.nan*jnp.zeros(iters)
            costs_g = jnp.nan*jnp.zeros(iters)
            for i in tqdm(range(iters)):

                if i == 658:
                    break

                # Assuming common g for all problems! 
                if learn_g:
                    if chill_g:
                        lb = jnp.power(10, (i/iters) * -4 + (1-i/iters)*-1)
                        g_cands = jnp.logspace(lb, jnp.log10(g_bounds[1]),num=10)
                    else:
                        g_cands = jnp.logspace(jnp.log10(g_bounds[0]), jnp.log10(g_bounds[1]),num=10)
                    g_grid = jnp.stack([g_cands for _ in range(g.shape[0])],axis=1)[:,:,None]

                    g_cost, _ = jax.vmap(total_nll, [None,0,None,None])(theta, g_grid, x_use, y)
                    g_opt = g_cands[jnp.argmin(g_cost)]
                    g = jnp.ones_like(g)*g_opt

                # GD step.
                cost, (theta_grad, g_grad) = total_nll(theta, g, x_use, y)
                theta = theta - lr_theta*theta_grad
                theta = prox_l1(theta, self.l1_penalty * lr_theta)
                #g = g - lr_g*g_grad

                # Projection
                print("this should be before prox")
                theta = theta.clip(*bounds[:,:,:-1])
                #g = g.clip(*bounds[:,:,-1:])

                pen = self.l1_penalty*jnp.sum(jnp.sqrt(jnp.sum(jnp.square(theta),axis=0)))
                costs_f = costs_f.at[i].set(cost)
                costs_g = costs_g.at[i].set(pen)
                costs = costs.at[i].set(cost+pen)
                traj_theta = traj_theta.at[i,:,:].set(theta)
                traj_g = traj_g.at[i].set(g)

                if i > 0:
                    d = costs[i] - costs[i-1]
                    if d > 0:
                        print(i)
                            

            fig = plt.figure(figsize=[15,5])
            plt.subplot(1,3,1)
            plt.plot(costs, label = 'f+g')
            plt.plot(costs_f, label = 'nll')
            plt.plot(costs_g, label = 'pen')
            plt.legend()
            ax = plt.gca().twinx()
            ax.scatter(jnp.arange(iters-1),jnp.diff(costs))
            plt.axhline(0, alpha=0.5, color='gray')
            plt.title('Cost')
            plt.subplot(1,3,2)
            for i in range(theta.shape[0]):
                for j in range(theta.shape[0]):
                    plt.plot(traj_theta[:,i,j])
            plt.title(r"$\theta$")
            plt.subplot(1,3,3)
            plt.plot(traj_g[:,:,0])
            plt.title("g")
            plt.savefig('pgd.pdf')
            plt.close()


        else:
            raise Exception('Unknown method in fit.')

        # extract the optimal parameters and infer the rest
        admm_x, admm_z, admm_u, rho = trajectory[-1]
        theta = admm_x[:, :-1]
        g = admm_x[:, -1]
        llk, b, nu = jax.vmap(likelihood, in_axes=(0, 0, None, 1))(theta, g, x, y)
        glasso = jnp.sum(norm(rearrange(theta, "o (d k) -> (o k) d", k=3), axis=0))
        loss = -llk.sum() + self.l1_penalty * glasso
        results.append((loss, trajectory, theta, g, b, nu))
        if self.verbose:
            print(f"Optimal theta: {theta}")
            print(f"Optimal g: {g}")
            print(f"Optimal b: {b}")
            print(f"Optimal nu: {nu}")
            print(f"Log-likelihood at optimum: {llk}")
            print()

        #import IPython; IPython.embed()

        # best result (minimum loss) across the multiple random restarts
        trajectory, theta, g, b, nu = min(results, key=lambda x: x[0])[1:]

        # save stuff
        self.trajectory = trajectory
        self.parameters = Parameters(theta, g, b, nu)
        self.x = x
        self.y = y
        return self

    def predict(self, x: Float[Array, "n d"]) -> Gaussian:
        n, d = self.x.shape
        n, o = self.y.shape

        def predict_single(params, y):
            theta, g, b, nu = params
            Koo = nu * (kernel(self.x, self.x, theta) + jnp.eye(n) * (EPS + g))
            Kxo = nu * kernel(x, self.x, theta)
            Kxx = nu * kernel(x, x, theta)

            # posterior mean and covariance
            mean = b + Kxo @ jnp.linalg.solve(Koo, y - b)
            cov = Kxx - Kxo @ jnp.linalg.solve(Koo, Kxo.T)

            # Add correction based on the trend estimation correlation (?)
            Kbx = jnp.ones((1, n)) @ jnp.linalg.solve(Koo, Kxo.T)
            cov = cov + (1 - Kbx).T @ (1 - Kbx) / jnp.linalg.inv(Koo).sum()
            return Gaussian(mean=mean, cov=cov)

        return jax.vmap(predict_single)(self.parameters, self.y.T)
