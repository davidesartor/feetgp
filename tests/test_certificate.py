import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange

from feetgp.glasso_admm import kkt_certificate
from feetgp.gp import GaussianProcess, hetgpy_auto_bounds

from conftest import flat_design, group_norms, negative_loglikelihood_grad


def test_certificate_vanishes_at_a_stationary_point():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(3, 2, 4)))
    l1_penalty = jnp.array(0.7)

    # the subgradient of a live group is exactly minus its unit vector times lambda
    norms = jnp.sqrt(jnp.sum(x**2, axis=(0, 1)))
    grad = -l1_penalty * x / norms
    assert np.allclose(kkt_certificate(x, grad, l1_penalty), 0.0, atol=1e-6)


def test_certificate_reports_the_penalty_ball_slack_of_a_dead_group():
    l1_penalty = jnp.array(2.0)
    x = jnp.zeros((3, 2))
    grad = jnp.zeros((3, 2)).at[0, 0].set(0.5).at[0, 1].set(5.0)

    # dead groups are optimal while the gradient stays inside the ball of radius lambda
    certificate = kkt_certificate(x, grad, l1_penalty)
    assert np.allclose(certificate, [0.5 - 2.0, 5.0 - 2.0])
    assert certificate[0] < 0.0 < certificate[1]


def test_certificate_projects_out_gradients_blocked_by_the_bounds():
    lower, upper = jnp.zeros((2, 1)), jnp.full((2, 1), 3.0)
    l1_penalty = jnp.array(0.0)
    at_lower = jnp.array([[0.0], [1.0]])
    at_upper = jnp.array([[3.0], [1.0]])
    outward, inward = jnp.array([[1.0], [0.0]]), jnp.array([[-1.0], [0.0]])

    # a gradient pushing a variable through a bound it already sits on is not a violation
    assert np.allclose(
        kkt_certificate(at_lower, outward, l1_penalty, lower, upper), 0.0
    )
    assert np.allclose(kkt_certificate(at_upper, inward, l1_penalty, lower, upper), 0.0)

    # the same gradient pointing back into the feasible set still counts
    assert np.allclose(kkt_certificate(at_lower, inward, l1_penalty, lower, upper), 1.0)
    assert np.allclose(
        kkt_certificate(at_upper, outward, l1_penalty, lower, upper), 1.0
    )


def fit_certificate_at(x_train, y_train, theta, l1_penalty):
    """Recompute the fit's certificate at an arbitrary theta, bounds and all."""
    design = flat_design(x_train)
    _, _, g = x_train.shape
    _, lower, upper = hetgpy_auto_bounds(design)
    lower = jnp.zeros_like(lower)

    value_and_grad = negative_loglikelihood_grad("rbf", design, y_train[:, 0])
    grad = jax.vmap(value_and_grad, in_axes=(0, None))(
        rearrange(theta, "o d g -> o (d g)"), jnp.log(jnp.array(0.1))
    )[1]
    grad = rearrange(grad, "o (d g) -> o d g", g=g)
    bounds = (rearrange(v, "(d g) -> d g", g=g) for v in (lower, upper))
    return kkt_certificate(theta, grad, l1_penalty, *bounds)


def test_certificate_degrades_away_from_the_fit(toy_data):
    x_train, y_train = toy_data
    model, _, _, certificate = GaussianProcess(profile="rbf").fit(
        x_train, y_train, l1_penalty=jnp.array(0.0), max_iterations=60
    )
    assert np.isfinite(certificate).all()

    # the same measure, taken half way to the origin, has to look worse
    perturbed = fit_certificate_at(x_train, y_train, 0.5 * model.theta, jnp.array(0.0))
    reference = fit_certificate_at(x_train, y_train, model.theta, jnp.array(0.0))
    assert np.max(perturbed) > np.max(reference)


def test_certificate_is_satisfied_once_every_group_is_dead(toy_data):
    x_train, y_train = toy_data
    l1_penalty = 1e4
    model, _, _, certificate = GaussianProcess(profile="rbf").fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(l1_penalty),
        max_iterations=60,
    )
    assert np.allclose(model.theta, 0.0)
    assert np.allclose(group_norms(model.theta), 0.0)

    # a dead group scores its gradient's distance from the surface of the ball
    assert (certificate < 0.0).all()
    assert (certificate >= -l1_penalty).all()
