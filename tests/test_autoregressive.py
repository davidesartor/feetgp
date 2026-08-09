import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange

pytest.skip(
    "autoregressive classes are gone, the dataset flattens the targets instead",
    allow_module_level=True,
)

from feetgp.gp import AutoregressiveGaussianProcess, GaussianProcess  # noqa: E402
from feetgp.linear import AutoregressiveLinear, Linear  # noqa: E402


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    x_train = jnp.asarray(rng.uniform(size=(24, 4, 3)))
    x_test = jnp.asarray(rng.uniform(size=(5, 4, 3)))
    return x_train, x_test


def flatten(x):
    return rearrange(x, "... d g -> ... (d g)")


def test_linear_predict_matches_flat_model_with_reshaped_params(toy_data):
    x_train, x_test = toy_data
    rng = np.random.default_rng(1)
    _, d, g = x_train.shape

    theta = jnp.asarray(rng.normal(size=(d, g, d, g)))
    bias = jnp.asarray(rng.normal(size=(d, g)))
    autoregressive = AutoregressiveLinear(theta=theta, bias=bias)
    flat = Linear(
        theta=rearrange(theta, "do go di gi -> (do go) di gi"),
        bias=flatten(bias),
    )

    prediction = autoregressive.predict(x_test)
    assert prediction.shape == (len(x_test), d, g)
    assert np.allclose(flatten(prediction), flat.predict(x_test))


@pytest.mark.parametrize("profile", ["rbf", "matern52"])
def test_gp_predict_matches_flat_model_with_reshaped_params(profile, toy_data):
    x_train, x_test = toy_data
    rng = np.random.default_rng(2)
    _, d, g = x_train.shape

    theta = jnp.asarray(rng.uniform(0.5, 2.0, size=(d, g, d, g)))
    nugget = jnp.full((d, g), 0.1)
    b = jnp.asarray(rng.normal(size=(d, g)))
    nu = jnp.full((d, g), 1.3)
    autoregressive = AutoregressiveGaussianProcess(
        profile=profile,
        theta=theta, nugget=nugget, b=b, nu=nu, x_train=x_train
    )
    flat = GaussianProcess(
        profile=profile,
        theta=rearrange(theta, "do go di gi -> (do go) di gi"),
        nugget=flatten(nugget),
        b=flatten(b),
        nu=flatten(nu),
        x_train=x_train,
        y_train=flatten(x_train),
    )

    mean, cov = autoregressive.predict(x_test)
    expected_mean, expected_cov = flat.predict(x_test)
    assert mean.shape == (d, g, len(x_test))
    assert cov.shape == (d, g, len(x_test), len(x_test))
    assert np.allclose(rearrange(mean, "do go m -> (do go) m"), expected_mean)
    assert np.allclose(rearrange(cov, "do go a b -> (do go) a b"), expected_cov)


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_linear_predict_accepts_arbitrary_batch_dims(toy_data, batch):
    x_train, _ = toy_data
    rng = np.random.default_rng(3)
    _, d, g = x_train.shape
    o = d * g

    x = jnp.asarray(rng.uniform(size=(*batch, d, g)))
    theta = jnp.asarray(rng.normal(size=(d, g, d, g)))
    bias = jnp.asarray(rng.normal(size=(d, g)))
    autoregressive = AutoregressiveLinear(theta=theta, bias=bias)
    flat = Linear(
        theta=rearrange(theta, "do go di gi -> (do go) di gi"), bias=flatten(bias)
    )

    prediction = autoregressive.predict(x)
    flat_prediction = flat.predict(x)
    assert prediction.shape == (*batch, d, g)
    assert flat_prediction.shape == (*batch, o)

    # every batch element matches the same point predicted on its own
    flat_x = rearrange(x, "... d g -> (...) d g")
    for point, expected, flat_expected in zip(
        flat_x,
        rearrange(prediction, "... d g -> (...) d g"),
        rearrange(flat_prediction, "... o -> (...) o"),
    ):
        assert np.allclose(autoregressive.predict(point), expected, atol=1e-6)
        assert np.allclose(flat.predict(point), flat_expected, atol=1e-6)


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
@pytest.mark.parametrize("profile", ["rbf", "matern52"])
def test_gp_predict_accepts_arbitrary_batch_dims(profile, toy_data, batch):
    x_train, x_test = toy_data
    rng = np.random.default_rng(4)
    _, d, g = x_train.shape
    o, m = d * g, len(x_test)

    x = jnp.asarray(rng.uniform(size=(*batch, m, d, g)))
    theta = jnp.asarray(rng.uniform(0.5, 2.0, size=(d, g, d, g)))
    nugget, nu = jnp.full((d, g), 0.1), jnp.full((d, g), 1.3)
    b = jnp.asarray(rng.normal(size=(d, g)))
    autoregressive = AutoregressiveGaussianProcess(
        profile=profile,
        theta=theta, nugget=nugget, b=b, nu=nu, x_train=x_train
    )
    flat = GaussianProcess(
        profile=profile,
        theta=rearrange(theta, "do go di gi -> (do go) di gi"),
        nugget=flatten(nugget),
        b=flatten(b),
        nu=flatten(nu),
        x_train=x_train,
        y_train=flatten(x_train),
    )

    mean, cov = autoregressive.predict(x)
    flat_mean, flat_cov = flat.predict(x)
    assert mean.shape == (*batch, d, g, m)
    assert cov.shape == (*batch, d, g, m, m)
    assert flat_mean.shape == (*batch, o, m)
    assert flat_cov.shape == (*batch, o, m, m)

    # every batch element matches the same query set predicted on its own
    for query, expected, flat_expected in zip(
        rearrange(x, "... m d g -> (...) m d g"),
        rearrange(mean, "... do go m -> (...) do go m"),
        rearrange(flat_mean, "... o m -> (...) o m"),
    ):
        assert np.allclose(autoregressive.predict(query)[0], expected)
        assert np.allclose(flat.predict(query)[0], flat_expected)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_fit_matches_flat_fit_on_flattened_targets(toy_data, fit_intercept):
    x_train, x_test = toy_data
    l1_penalty = jnp.array(0.5)
    flat, flat_loss, flat_state, flat_certificate = Linear.fit(
        x_train, flatten(x_train), l1_penalty, fit_intercept=fit_intercept
    )
    model, loss, state, certificate = AutoregressiveLinear.fit(
        x_train, l1_penalty, fit_intercept=fit_intercept
    )

    assert np.allclose(
        rearrange(model.theta, "do go di gi -> (do go) di gi"), flat.theta
    )
    assert np.allclose(flatten(model.bias), flat.bias)
    assert np.allclose(loss, flat_loss)
    assert np.allclose(certificate, flat_certificate)
    assert state.iteration == flat_state.iteration
    assert np.allclose(flatten(model.predict(x_test)), flat.predict(x_test))


@pytest.mark.parametrize("profile", ["rbf", "matern52"])
def test_gp_fit_matches_flat_fit_on_flattened_targets(profile, toy_data):
    x_train, x_test = toy_data
    l1_penalty = jnp.array(1.0)
    flat, flat_llk, flat_state, flat_certificate = GaussianProcess.fit(
        x_train, flatten(x_train), l1_penalty, profile=profile, max_iterations=10
    )
    model, llk, state, certificate = AutoregressiveGaussianProcess.fit(
        x_train, l1_penalty, profile=profile, max_iterations=10
    )

    # the two fits differ only in reduction order, which float32 l-bfgs-b amplifies
    assert np.allclose(
        rearrange(model.theta, "do go di gi -> (do go) di gi"),
        flat.theta,
        rtol=1e-2,
        atol=1e-2,
    )
    for field in ("nugget", "b", "nu"):
        assert np.allclose(
            flatten(getattr(model, field)), getattr(flat, field), rtol=1e-3, atol=1e-4
        ), field
    assert np.allclose(llk, flat_llk, rtol=1e-4)
    assert np.allclose(certificate, flat_certificate, rtol=1e-2, atol=1e-2)
    assert state.iteration == flat_state.iteration

    mean, _ = model.predict(x_test)
    expected_mean, _ = flat.predict(x_test)
    assert np.allclose(
        rearrange(mean, "do go m -> (do go) m"), expected_mean, rtol=1e-2, atol=1e-2
    )
