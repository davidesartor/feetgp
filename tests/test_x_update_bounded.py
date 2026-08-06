"""The bounded x-update against scipy's Fortran L-BFGS-B on the same subproblem.

vlse ships the solver and its own correctness battery; what belongs here is only the
part that is specific to this repo -- that `admm_x_update_loss` under the theta box
reaches the same minimiser scipy does.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import Bounds
from scipy.optimize import minimize as scipy_minimize

from feetgp.glassogp import admm_x_update_loss, nugget_from_w

minimise = pytest.importorskip("vlse.optim", reason="jaxvlse is not installed").minimise

jax.config.update("jax_enable_x64", True)

TOL = 1e-8


def scipy_reference(fun, x0, lower, upper, args, tol=TOL, history_length=10):
    """Same problem through scipy's Fortran L-BFGS-B, stopping on the same gtol."""
    value_and_grad = jax.jit(jax.value_and_grad(fun))

    def numpy_value_and_grad(x):
        f, grad = value_and_grad(jnp.asarray(x), args)
        return float(f), np.asarray(grad, dtype=np.float64)

    result = scipy_minimize(
        numpy_value_and_grad,
        np.asarray(x0, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=Bounds(
            np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
        ),
        options=dict(maxiter=1000, maxcor=history_length, ftol=0.0, gtol=tol),
    )
    return jnp.asarray(result.x), jnp.asarray(result.fun)


def test_gp_x_update_matches_scipy():
    rng = np.random.default_rng(0)
    n, d = 24, 12
    x_train = jnp.asarray(rng.uniform(size=(n, d)))
    y_train = jnp.asarray(rng.normal(size=n))
    target = jnp.asarray(rng.uniform(0.2, 1.5, size=d))
    args = (target, jnp.array(1.0), x_train, y_train, jnp.array(1e-4), jnp.array(100.0))

    x0 = jnp.concatenate([jnp.ones(d), jnp.zeros(1)])
    lower = jnp.concatenate([jnp.zeros(d), jnp.array([-jnp.inf])])
    upper = jnp.concatenate([jnp.full(d, 3.0), jnp.array([jnp.inf])])

    # vlse calls f(x, *args), so the packed tuple the loss expects needs one more level
    state = minimise(
        admm_x_update_loss,
        x0,
        (lower, upper),
        args=(args,),
        tol=TOL,
        max_iterations=500,
    )
    x_scipy, f_scipy = scipy_reference(
        admm_x_update_loss, x0, lower, upper, args=args, tol=TOL
    )

    assert state.f <= f_scipy + 1e-8 * (1.0 + abs(f_scipy))
    assert np.allclose(state.x[:-1], x_scipy[:-1], atol=1e-5)
    # w itself is not identified once the nugget saturates, the nugget is
    nugget = lambda w: nugget_from_w(w, args[-2], args[-1])
    assert nugget(state.x[-1]) == pytest.approx(nugget(x_scipy[-1]), rel=1e-6)
