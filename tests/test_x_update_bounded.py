import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import Bounds
from scipy.optimize import minimize as scipy_minimize

from conftest import x_update_objective

minimise = pytest.importorskip("vlse.optim", reason="jaxvlse is not installed").minimise

# the objective itself is float32, so scipy cannot be asked for more than that
TOL = 1e-5


def scipy_reference(fun, x0, lower, upper, tol=TOL, history_length=10):
    value_and_grad = jax.jit(jax.value_and_grad(fun))

    def numpy_value_and_grad(x):
        f, grad = value_and_grad(jnp.asarray(x, dtype=jnp.float32))
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


def test_gp_x_update_matches_scipy(profile):
    rng = np.random.default_rng(0)
    n, d = 24, 12
    x_train = jnp.asarray(rng.uniform(size=(n, d)))
    y_train = jnp.asarray(rng.normal(size=n))
    target = jnp.asarray(rng.uniform(0.2, 1.5, size=d))
    objective = x_update_objective(profile, x_train, y_train, target, jnp.array(1.0))

    x0 = jnp.concatenate([jnp.ones(d), jnp.zeros(1)])
    lower = jnp.concatenate([jnp.zeros(d), jnp.array([jnp.log(1e-4)])])
    upper = jnp.concatenate([jnp.full(d, 3.0), jnp.array([jnp.log(100.0)])])

    state = minimise(objective, x0, (lower, upper), tol=TOL, max_iterations=500)
    x_scipy, f_scipy = scipy_reference(objective, x0, lower, upper)

    # the two solvers agree on the value long before they agree on every coordinate
    assert state.f <= f_scipy + 1e-3 * (1.0 + abs(f_scipy))
    assert np.allclose(state.x[:-1], x_scipy[:-1], rtol=1e-2, atol=1e-2)
    assert jnp.exp(state.x[-1]) == pytest.approx(float(jnp.exp(x_scipy[-1])), rel=1e-2)
