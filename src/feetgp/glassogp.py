from typing import NamedTuple, Optional, Self
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import optimistix as optx

from einops import rearrange
import numpy as np
from scipy.spatial.distance import cdist

from feetgp import admm
from feetgp.admm import ADMMState

# the bounded x-update is opt-in (--solver lbfgsb); optimistix stays the default
try:
    from vlse.optim import minimise as lbfgsb_minimise
except ImportError:
    lbfgsb_minimise = None

jax.config.update("jax_enable_x64", True)

# the nugget's reachable range. The floor stops the likelihood running away to
# interpolation as g -> 0; the ceiling leaves room for outputs noisier than their signal
G_RANGE = (1e-4, 100.0)

# provisional pass/fail line on max_live_kkt for reporting; recalibrate once a real
# path has been measured. Certificates are stored raw in the pickles, so changing this
# reinterprets old runs without refitting them
CERTIFICATE_TOLERANCE = 1e-2


@jax.jit
def kernel(
    theta: Float[Array, "d"],
    xs1: Float[Array, "n d"],
    xs2: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Squared-exponential kernel, expanded as |z1|^2 + |z2|^2 - 2 z1.z2.

    The expansion makes the pairwise work one matmul instead of an n*m*d broadcast,
    which also keeps the gradient off that tensor.
    """
    scaled1, scaled2 = xs1 * theta, xs2 * theta
    sqnorm1 = jnp.sum(scaled1**2, axis=-1)
    sqnorm2 = jnp.sum(scaled2**2, axis=-1)
    sqdist = sqnorm1[:, None] + sqnorm2[None, :] - 2.0 * scaled1 @ scaled2.T
    return jnp.exp(-0.5 * jnp.maximum(sqdist, 0.0))


def hetgpy_auto_bounds(
    x: Float[Array, "n d"],
    min_cor: float = 0.01,
    max_cor: float = 0.5,
) -> tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d"]]:
    # rescale each input dimension to [0, 1] (constant columns are left untouched)
    x = np.asarray(x)  # type: ignore
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    x_span = np.where(x_max > x_min, x_max - x_min, 1.0)
    x = (x - x_min) / x_span

    # pairwise squared distances, keeping only the strictly-lower nonzero pairs
    dists = cdist(x, x, metric="sqeuclidean")
    dists = dists[np.tril(dists, k=-1) > 0]

    # hetgpy heuristic: pick lengthscales so the given distance quantiles map to
    # the target correlations min_cor (short) and max_cor (long)
    lower = -np.quantile(dists, 0.05) / np.log(min_cor) * x_span**2
    upper = -np.quantile(dists, 0.95) / np.log(max_cor) * x_span**2
    return lower, upper


@eqx.filter_jit
def gp_posterior(
    Kox: Float[Array, "n m"],
    Koo: Float[Array, "n n"],
    observed_ys: Float[Array, "n"],
    b: Scalar,
    Kxx: Optional[Float[Array, "m m"]] = None,
) -> Float[Array, "m"] | tuple[Float[Array, "m"], Float[Array, "m m"]]:
    """Posterior mean, plus covariance if Kxx is given. Kxx=None skips the m*m work."""
    chol = jsp.linalg.cho_factor(Koo)
    gain = jsp.linalg.cho_solve(chol, Kox)
    mean = b + gain.T @ (observed_ys - b)
    if Kxx is None:
        return mean

    cov = Kxx - Kox.T @ gain

    # Add correction based on the trend estimation correlation
    Kbx = jnp.ones((1, len(observed_ys))) @ gain
    Ki_1 = jsp.linalg.cho_solve(chol, jnp.ones_like(observed_ys))
    cov = cov + (1 - Kbx).T @ (1 - Kbx) / Ki_1.sum()
    return mean, cov


@jax.jit
def loglikelihood(
    Koo: Float[Array, "n n"],
    observed_ys: Float[Array, "n"],
) -> tuple[Scalar, Scalar, Scalar]:
    # cholesky of K and compute logdet
    K_sqrt, is_lower = jsp.linalg.cho_factor(Koo)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    # compute Ki_1=(K^-1 @ 1) and Ki_y=(K^-1 @ y)
    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([jnp.ones_like(observed_ys), observed_ys], 1),
    ).T

    # compute optimal trend b and scale nu
    b = (Ki_1 * observed_ys).sum() / Ki_1.sum()
    nu = jnp.dot((observed_ys - b) / len(observed_ys), (Ki_y - Ki_1 * b))

    # likelihood when marginalizing over trend and variance
    loglik = -0.5 * (len(observed_ys) * jnp.log(nu) + logdetK)
    return (loglik, b, nu)


def admm_x_update_loss(
    x: Float[Array, "d*g+1"],
    args: tuple,
) -> Scalar:
    """x-update objective for one output. x is [theta, w], unconstrained on R.

    The kernel sees relu(theta), not theta, which is what makes the negative orthant
    flat in the likelihood and kills the mirror well: without it the objective is
    exactly even, so a target z - u below zero -- which is exactly what a group being
    killed looks like -- makes the reflected well the deeper one and L-BFGS falls into
    it. The relu costs no smoothness, because evenness also forces dloglik/dtheta_d = 0
    at theta_d = 0, so loglik(relu(theta)) is C1 across the boundary.

    The augmented term keeps the *raw* theta so that a coordinate started on the wrong
    side of zero is still pulled back to its target; if it saw relu(theta) too, the whole
    negative orthant would be flat and such a coordinate could never escape.

    w carries no augmented term at all: the nugget is an ADMM aux variable, not part of
    the consensus, so it is fit by the likelihood alone.
    """
    target_theta, rho, x_train, y_train, g_min, g_max = args
    theta, w = x[:-1], x[-1]
    Koo = kernel(jax.nn.relu(theta), x_train, x_train)
    Koo = Koo + nugget_from_w(w, g_min, g_max) * jnp.eye(len(y_train))
    loglik, _, _ = loglikelihood(Koo, y_train)
    lagrangian = 0.5 * rho * jnp.sum((theta - target_theta) ** 2)
    return -loglik + lagrangian


def nugget_from_w(w: Scalar, g_min: Scalar, g_max: Scalar) -> Scalar:
    """g in (g_min, g_max) by construction, so the x-update needs no box on w."""
    return g_min + (g_max - g_min) * jax.nn.sigmoid(w)


def w_from_nugget(g: Scalar, g_min: Scalar, g_max: Scalar) -> Scalar:
    fraction = jnp.clip((g - g_min) / (g_max - g_min), 1e-12, 1 - 1e-12)
    return jsp.special.logit(fraction)


def autoregressive_mask(
    o: int, d_times_g: int, group_size: int
) -> Float[Array, "o d*g+1"]:
    """0 over each output's own marker group, 1 elsewhere (log g always kept)."""
    output = jnp.arange(o)[:, None] // group_size
    column = jnp.arange(d_times_g)[None, :] // group_size
    return jnp.concatenate(
        [(output != column).astype(float), jnp.ones((o, 1))], axis=-1
    )


def theta_box(
    lower: Float[np.ndarray, "d*g"], o: int, d_times_g: int, group_size: int
) -> Float[Array, "2 d o*g"]:
    """The [0, sqrt(2/lower)] box in group layout, shared by fit and the certificate."""
    theta_max = admm.to_groups(
        jnp.broadcast_to(jnp.sqrt(2.0 / lower), (o, d_times_g)), group_size
    )
    return jnp.stack([jnp.zeros_like(theta_max), theta_max], axis=0)


@eqx.filter_jit
def penalized_objective(
    theta: Float[Array, "o d*g"],
    g: Float[Array, "o"],
    l1: Scalar,
    x_train: Float[Array, "n d*g"],
    y_train: Float[Array, "n o"],
    group_size: int,
) -> Scalar:
    """True nonsmooth objective sum_o -loglik_o + l1 sum_g ||theta_g||; ranks starts."""
    n = len(x_train)

    def negloglik_single_output(
        theta_i: Float[Array, "d*g"], g_i: Scalar, y_i: Float[Array, "n"]
    ) -> Scalar:
        Koo = kernel(theta_i, x_train, x_train) + g_i * jnp.eye(n)
        loglik, _, _ = loglikelihood(Koo, y_i)
        return -loglik

    # scan over outputs, one n*n kernel live at a time
    _, negloglik = jax.lax.scan(
        lambda _, inputs: (_, negloglik_single_output(*inputs)),
        None,
        (theta, g, y_train.T),
    )
    norms = jnp.linalg.norm(admm.to_groups(theta, group_size), axis=-1)
    return negloglik.sum() + l1 * norms.sum()


@eqx.filter_jit
def kkt_certificate(
    theta: Float[Array, "o d*g"],
    w: Float[Array, "o"],
    l1: Scalar,
    x_train: Float[Array, "n d*g"],
    y_train: Float[Array, "n o"],
    group_size: int,
    bounds: Float[Array, "2 d o*g"],
    mask: Optional[Float[Array, "o d*g+1"]] = None,
    g_min: Scalar = jnp.array(1e-4),
    g_max: Scalar = jnp.array(100.0),
    chunk_size: int = 8,
) -> dict:
    """First-order optimality of (theta, w), independent of the ADMM residual test.

    Live groups get the box-projected stationarity residual of f + l1||theta_g||; dead
    groups get the subgradient slack l1 - ||grad_g f||, which the even objective makes
    trivially l1 -- recorded anyway, to document that only local optimality is certified
    (multi-start is what compensates). The nugget check is the plain gradient in w, the
    unbounded coordinate the solver actually optimizes.
    """
    n = len(x_train)

    def negloglik_single_output(
        theta_i: Float[Array, "d*g"], w_i: Scalar, y_i: Float[Array, "n"]
    ) -> Scalar:
        Koo = kernel(theta_i, x_train, x_train) + nugget_from_w(
            w_i, g_min, g_max
        ) * jnp.eye(n)
        loglik, _, _ = loglikelihood(Koo, y_i)
        return -loglik

    # chunked gradient sweep over outputs; autoregressive coordinates are structural
    # zeros, not optimization variables, so their gradient is masked out
    grad_theta, grad_w = jax.lax.map(
        lambda inputs: jax.grad(negloglik_single_output, argnums=(0, 1))(*inputs),
        (theta, w, y_train.T),
        batch_size=chunk_size,
    )
    if mask is not None:
        grad_theta = grad_theta * mask[:, :-1]

    grad_groups = admm.to_groups(grad_theta, group_size)
    theta_groups = admm.to_groups(theta, group_size)
    norms = jnp.linalg.norm(theta_groups, axis=-1)
    live = norms > 0.0

    # box projection: at an active bound, only the component pushing outward counts
    direction = theta_groups / jnp.where(live, norms, 1.0)[:, None]
    stationarity = grad_groups + l1 * direction
    lower, upper = bounds
    stationarity = jnp.where(
        theta_groups <= lower, jnp.minimum(stationarity, 0.0), stationarity
    )
    stationarity = jnp.where(
        theta_groups >= upper, jnp.maximum(stationarity, 0.0), stationarity
    )
    live_kkt = jnp.linalg.norm(stationarity, axis=-1)
    dead_slack = l1 - jnp.linalg.norm(grad_groups, axis=-1)
    return dict(
        max_live_kkt=jnp.max(jnp.where(live, live_kkt, 0.0)),
        live_kkt=jnp.where(live, live_kkt, jnp.nan),
        dead_slack=jnp.where(live, jnp.nan, dead_slack),
        nugget_grad=jnp.max(jnp.abs(grad_w)),
    )


class GLASSOADMMState(NamedTuple):
    """Format-4 state, kept only so pickles written before the ADMM port still load.

    Nothing constructs one any more; admm_state_from_legacy converts it. Pickle looks the
    class up by module and name, so deleting it would make every old result unreadable,
    not just unusable as a warmstart.
    """

    x: Float[Array, "o d*g+1"]
    z: Float[Array, "o d*g+1"]
    u: Float[Array, "o d*g+1"]
    rho: Scalar
    l1: Scalar
    group_size: int
    bounds: Optional[Float[Array, "2 o d*g+1"]] = None
    g_min: Scalar = jnp.array(1e-4)
    g_max: Scalar = jnp.array(1.0)


def admm_state_from_legacy(legacy: GLASSOADMMState) -> ADMMState:
    """Format-4 (o, d*g+1) iterates -> the (... g) layout, nugget moved into aux.

    Lossless: the theta and w parametrizations are unchanged, only the layout is, and the
    w column satisfied x = z with u = 0 identically, which is what aux now expresses.
    """
    x, z, u = (
        admm.to_groups(v[:, :-1], legacy.group_size)
        for v in (legacy.x, legacy.z, legacy.u)
    )
    return ADMMState(x=x, z=z, u=u, rho=legacy.rho, aux=legacy.x[:, -1])


def admm_state_from_pickle(results: dict) -> ADMMState:
    """The state out of a result pickle, converting the format-4 layout if it is one."""
    state = results["admm_state"]
    if isinstance(state, GLASSOADMMState):
        return admm_state_from_legacy(state)
    return state


@eqx.filter_jit
def x_update_solve(
    x0: Float[Array, "o d*g+1"],
    target_theta: Float[Array, "o d*g"],
    rho: Scalar,
    masked_designs: Float[Array, "n_groups n d*g"],
    y_train: Float[Array, "n o"],
    group_size: int,
    mask: Optional[Float[Array, "o d*g+1"]] = None,
    g_min: Scalar = jnp.array(1e-4),
    g_max: Scalar = jnp.array(1.0),
    maxiter: int = 50,
    chunk_size: int = 8,
    history_length: int = 40,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Float[Array, "o d*g+1"]:
    """Solve the o independent smooth subproblems, chunk_size of them at a time.

    L-BFGS over an x whose negative part the likelihood cannot see (the relu in
    admm_x_update_loss). The caller projects the result onto the theta box, and the two
    together are what make a dead group stable: its target is z - u = -u, so the
    minimiser sits at a negative theta on a likelihood plateau and the projection clips
    it to exactly zero. A long history matters, measured 221 steps at 40 against 735 at
    the default 10 and 627 for dense BFGS.

    rtol/atol are a *step* criterion, not a gradient one, so they are far looser than a
    gradient tolerance of the same magnitude: at 1e-4 the cold x-update reaches the same
    augmented-Lagrangian value as 1.49e-8 (relative 4e-6) in half the steps.
    """
    solver = optx.LBFGS(rtol=rtol, atol=atol, history_length=history_length)
    x0 = x0 if mask is None else x0 * mask
    group_of_output = jnp.arange(len(x0)) // group_size

    def solve_one_output(args) -> Float[Array, "d*g+1"]:
        x0_i, target_i, y_i, group_i, mask_i = args
        solution = optx.minimise(
            admm_x_update_loss,
            solver,
            x0_i,
            args=(target_i, rho, masked_designs[group_i], y_i, g_min, g_max),
            max_steps=maxiter,
            throw=False,
        )
        return solution.value * mask_i

    # lax.map with a batch size is a chunked vmap: it trades the o*n*n kernels a full
    # vmap would hold live against keeping the device fed
    return jax.lax.map(
        solve_one_output,
        (
            x0,
            target_theta,
            y_train.T,
            group_of_output,
            jnp.ones_like(x0) if mask is None else mask,
        ),
        batch_size=chunk_size,
    )


@eqx.filter_jit
def x_update_solve_bounded(
    x0: Float[Array, "o d*g+1"],
    target_theta: Float[Array, "o d*g"],
    rho: Scalar,
    masked_designs: Float[Array, "n_groups n d*g"],
    y_train: Float[Array, "n o"],
    group_size: int,
    bounds: tuple[Float[Array, "o d*g+1"], Float[Array, "o d*g+1"]],
    mask: Optional[Float[Array, "o d*g+1"]] = None,
    g_min: Scalar = jnp.array(1e-4),
    g_max: Scalar = jnp.array(1.0),
    maxiter: int = 5,
    chunk_size: int = 8,
    history_length: int = 40,
    tol: float = 1e-2,
    max_linesearch: int = 5,
) -> Float[Array, "o d*g+1"]:
    """The same subproblems as x_update_solve, under a real box instead of a projection.

    theta >= 0 is the solver's own feasible set here, so a dead group -- whose target
    z - u is negative -- returns exactly zero rather than the unconstrained minimiser the
    caller then has to clip, and the mirror well the even objective puts at -theta is
    outside the feasible set entirely.

    Neither budget converts one-for-one from the optimistix path. tol is the infinity norm
    of the *projected gradient* (scipy's pgtol), not a step criterion, so the same
    magnitude is far tighter; maxiter counts whole line searches rather than evaluations,
    which is why its default is 5 and not 50. max_linesearch is the straggler's leash and
    is close to free: a lax.map chunk costs the max over its members, so capping the line
    search bounds the worst output rather than the mean -- measured at inner_maxiter=12,
    cutting 30 to 5 took one x-update from 28.24s to 9.45s for the same objective to eight
    digits and the same projected gradient.
    """
    if lbfgsb_minimise is None:
        raise ImportError("the bounded x-update needs vlse.optim: `uv add jaxvlse`")
    x0 = x0 if mask is None else x0 * mask
    lower, upper = bounds
    group_of_output = jnp.arange(len(x0)) // group_size

    def solve_one_output(args) -> Float[Array, "d*g+1"]:
        x0_i, target_i, y_i, group_i, mask_i, lower_i, upper_i = args
        # vlse calls f(x, *args), optimistix calls f(x, args) -- one extra level of
        # nesting is what lets both drive the same loss
        solution = lbfgsb_minimise(
            admm_x_update_loss,
            x0_i,
            (lower_i, upper_i),
            args=((target_i, rho, masked_designs[group_i], y_i, g_min, g_max),),
            tol=tol,
            max_iterations=maxiter,
            history_length=history_length,
            max_linesearch=max_linesearch,
        )
        return solution.x * mask_i

    return jax.lax.map(
        solve_one_output,
        (
            x0,
            target_theta,
            y_train.T,
            group_of_output,
            jnp.ones_like(x0) if mask is None else mask,
            lower,
            upper,
        ),
        batch_size=chunk_size,
    )


class GroupLassoGaussianProcess(NamedTuple):
    theta: Float[Array, "o d*g"]
    g: Float[Array, "o"]
    b: Float[Array, "o"]
    nu: Float[Array, "o"]

    x_train: Float[Array, "n d*g"]
    y_train: Float[Array, "n o"]

    @eqx.filter_jit
    def predict(
        self, xs: Float[Array, "m d*g"], covariance: bool = False
    ) -> Float[Array, "o m"] | tuple[Float[Array, "o m"], Float[Array, "o m m"]]:
        def predict_single_output(
            theta: Float[Array, "d*g"],
            g: Scalar,
            b: Scalar,
            nu: Scalar,
            y_train: Float[Array, "n"],
        ) -> Float[Array, "m"] | tuple[Float[Array, "m"], Float[Array, "m m"]]:
            Kox = nu * kernel(theta, self.x_train, xs)
            Koo = nu * (
                kernel(theta, self.x_train, self.x_train)
                + g * jnp.eye(len(self.y_train))
            )
            Kxx = nu * kernel(theta, xs, xs) if covariance else None
            return gp_posterior(Kox, Koo, y_train, b, Kxx)

        # scan over outputs instead of vmap: only one n*n kernel is live at a time
        _, posterior = jax.lax.scan(
            lambda _, args: (_, predict_single_output(*args)),
            None,
            (self.theta, self.g, self.b, self.nu, self.y_train.T),
        )
        return posterior

    @classmethod
    @eqx.filter_jit
    def unpack_parameters(
        cls,
        admm_theta: Float[Array, "o d*g"],
        admm_w: Float[Array, "o"],
        x_train: Float[Array, "n d*g"],
        y_train: Float[Array, "n o"],
        g_min: Scalar = jnp.array(1e-4),
        g_max: Scalar = jnp.array(1.0),
    ) -> tuple[Self, Scalar]:
        # infer the rest; theta is reported as its magnitude, which is the equivalent
        # point in the old nonnegative orthant
        theta = jnp.abs(admm_theta)
        g = nugget_from_w(admm_w, g_min, g_max)

        # use scan instead of vmap over theta to avoid OOM
        def unpack_single_output(
            theta: Float[Array, "d*g"],
            g: Scalar,
            y_train: Float[Array, "n"],
        ) -> tuple[Scalar, Scalar, Scalar]:
            Koo = kernel(theta, x_train, x_train) + g * jnp.eye(len(y_train))
            return loglikelihood(Koo, y_train)

        _, (llk, b, nu) = jax.lax.scan(
            lambda _, args: (_, unpack_single_output(*args)),
            None,
            (theta, g, y_train.T),
        )
        self = cls(theta, g, b, nu, x_train, y_train)
        return self, llk.sum()

    @classmethod
    def fit(
        cls,
        x_train: Float[Array, "n d*g"],
        y_train: Float[Array, "n o"],
        l1_penalty: Scalar,
        group_size: int,
        autoregressive: bool = True,
        *,
        warmstart: Optional["Self | ADMMState"] = None,
        auto_bounds: Optional[
            tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d"]]
        ] = None,
        # the ceiling used to be 1.0, which 32% of real outputs sat exactly on: an
        # output noisier than its signal cannot say so, and the saturated ridge costs
        # iterations (103 against 52 on the toy fit). Measured identical at 10/100/1000
        g_range: tuple[float, float] = G_RANGE,
        g_init: float = 0.1,
        max_iterations: int = 300,
        tol: Scalar = jnp.array(1e-3),
        adapt_rho: bool = True,
        adapt_rho_iters: Optional[int] = None,
        # plain ADMM, deliberately: Boyd's 1.5-1.8 band is wrong for this problem. See
        # admm.z_and_u_update for why over-relaxation and the theta box are incompatible.
        alpha: float = 1.0,
        # "optimistix" is unconstrained L-BFGS plus a projection, "lbfgsb" is vlse's
        # bounded solver. Measured at one knot from a single warmstart, lbfgsb at
        # inner_maxiter=5 max_linesearch=5 took 67.4s and 16 ADMM iterations against
        # optimistix's 488.7s and 35, at equal-or-better loglik
        solver: str = "optimistix",
        inner_maxiter: int = 50,
        # first rung of the ramp. lbfgsb's budget counts whole line searches, so it runs
        # an order of magnitude smaller than optimistix's step count and a shared start of
        # 20 would sit above the cap from iteration 0 -- leaving the ramp inert and, worse,
        # reporting every x-update as exact so the convergence break could fire immediately
        inner_maxiter_init: Optional[int] = None,
        inner_rtol: float = 1e-4,
        inner_atol: float = 1e-4,
        # lbfgsb only: projected-gradient tolerance and the line-search cap
        inner_pgtol: float = 1e-2,
        inner_max_linesearch: int = 5,
        chunk_size: int = 8,
        history_length: int = 40,
        # tqdm goes to stderr and only shows the latest iteration; a periodic stdout row
        # is what makes a residual that stalls distinguishable from one still falling
        log_every: int = 0,
    ) -> tuple[Self, Scalar, ADMMState, dict]:
        _, d_times_g = x_train.shape
        _, o = y_train.shape
        assert d_times_g % group_size == 0
        assert solver in ("optimistix", "lbfgsb"), solver
        if inner_maxiter_init is None:
            inner_maxiter_init = 1 if solver == "lbfgsb" else 20

        # data-driven (hetgpy) lengthscale bounds; they only depend on x_train, so a
        # lambda sweep computes them once and passes them in
        lower, upper = auto_bounds or hetgpy_auto_bounds(x_train)
        # theta >= 0 is not a statistical constraint -- the objective is even, so -theta
        # and theta are the same model -- but it is what lets a dead group sit at exactly
        # zero instead of limit-cycling between the two wells. See x_update_solve.
        bounds = theta_box(lower, o, d_times_g, group_size)
        # the same box in the layout the inner solver wants, one row per output, with the
        # nugget column left unbounded because w saturates rather than being clipped
        unbounded = jnp.full((o, 1), jnp.inf)
        solver_bounds = (
            jnp.concat([admm.to_outputs(bounds[0], group_size), -unbounded], axis=-1),
            jnp.concat([admm.to_outputs(bounds[1], group_size), unbounded], axis=-1),
        )
        theta_init = admm.to_groups(
            jnp.broadcast_to(
                jnp.sqrt(2.0 / (0.9 * upper + 0.1 * lower)), (o, d_times_g)
            ),
            group_size,
        )
        # the nugget is an aux variable, not part of the consensus, so it needs no bound
        # at all: it saturates into g_range instead of being clipped. It used to be a
        # clipped column of x, and outputs whose noise exceeded g_range[1] then held the
        # primal residual permanently above tol -- no lambda could ever converge.
        g_min, g_max = jnp.array(g_range[0]), jnp.array(g_range[1])
        w_init = jnp.full((o,), w_from_nugget(jnp.array(g_init), g_min, g_max))
        state = ADMMState.initialize(theta_init, aux=w_init)

        # warmstart the whole ADMM state (dual variable and rho included) from the
        # neighbouring lambda; a fitted model alone only carries the parameters
        if isinstance(warmstart, ADMMState):
            state = warmstart
        elif warmstart is not None:
            # a fitted model reports g in the linear space the state optimizes as w
            theta = admm.to_groups(warmstart.theta, group_size)
            w = w_from_nugget(warmstart.g, g_min, g_max)
            state = state._replace(x=theta, z=theta, aux=w)

        # each output must not see its own marker group: one masked design per group,
        # indexed by group rather than replicated per output
        n_groups = d_times_g // group_size
        if autoregressive:
            group_columns = jnp.arange(d_times_g)[None, :] // group_size
            keep = (jnp.arange(n_groups)[:, None] != group_columns).astype(
                x_train.dtype
            )
            masked_designs = x_train[None, :, :] * keep[:, None, :]
            mask = autoregressive_mask(o, d_times_g, group_size)
        else:
            masked_designs = jnp.broadcast_to(x_train, (n_groups, *x_train.shape))
            mask = None

        def x_update(state: ADMMState, iteration: int) -> tuple[ADMMState, bool]:
            # inexact early on: a tight x-update is wasted while z-u is still moving, and
            # reporting exactness only at the cap stops a cheap iteration faking convergence
            maxiter = min(inner_maxiter, inner_maxiter_init * 2**iteration)
            theta = admm.to_outputs(state.x, group_size)
            x0 = jnp.concat([theta, state.aux[..., None]], axis=-1)
            target = admm.to_outputs(state.z - state.u, group_size)
            if solver == "lbfgsb":
                solution = x_update_solve_bounded(
                    x0,
                    target,
                    state.rho,
                    masked_designs,
                    y_train,
                    group_size,
                    solver_bounds,
                    mask=mask,
                    g_min=g_min,
                    g_max=g_max,
                    maxiter=maxiter,
                    chunk_size=chunk_size,
                    history_length=history_length,
                    tol=inner_pgtol,
                    max_linesearch=inner_max_linesearch,
                )
            else:
                solution = x_update_solve(
                    x0,
                    target,
                    state.rho,
                    masked_designs,
                    y_train,
                    group_size,
                    mask=mask,
                    g_min=g_min,
                    g_max=g_max,
                    maxiter=maxiter,
                    chunk_size=chunk_size,
                    history_length=history_length,
                    rtol=inner_rtol,
                    atol=inner_atol,
                )
            # project onto the same box z lives in, so a group the prox is killing reaches
            # exactly zero rather than the unconstrained minimiser near -u
            theta = admm.to_groups(solution[:, :-1], group_size)
            state = state._replace(x=jnp.clip(theta, *bounds), aux=solution[:, -1])
            return state, maxiter == inner_maxiter

        state, info = admm.solve(
            x_update,
            state,
            l1_penalty,
            max_iterations=max_iterations,
            tol=tol,
            bounds=bounds,
            alpha=alpha,
            adapt_rho=adapt_rho,
            adapt_rho_iters=adapt_rho_iters,
            log_every=log_every,
        )

        # report z, not x: the prox output is the iterate carrying exact zeros now that
        # the smooth solver no longer pins coordinates against a lower bound
        theta = admm.to_outputs(state.z, group_size)
        theta = theta if mask is None else theta * mask[:, :-1]
        self, llk = cls.unpack_parameters(
            theta, state.aux, x_train, y_train, g_min=g_min, g_max=g_max
        )

        # optimality of the reported solution, independent of the ADMM stopping test:
        # converged=True has been measured grazing its tolerance at 0.999x, so the
        # certificate, not the stamp, is what downstream reporting should trust
        certificate = kkt_certificate(
            self.theta,
            state.aux,
            l1_penalty,
            x_train,
            y_train,
            group_size,
            bounds,
            mask=mask,
            g_min=g_min,
            g_max=g_max,
            chunk_size=chunk_size,
        )
        info["certificate"] = {k: np.asarray(v) for k, v in certificate.items()}
        return self, llk, state, info
