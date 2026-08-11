import jax.numpy as jnp
import numpy as np
import pytest

from feetgp.gp import GaussianProcess

from conftest import group_norms


@pytest.fixture
def signal_data():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(40, 2, 3))
    signal = np.sin(4.0 * x[:, 0, 0]) + x[:, 0, 1]
    y = np.stack([signal, -signal], axis=1) + 0.05 * rng.normal(size=(40, 2))
    return jnp.asarray(x), jnp.asarray(y)


def fit(x_train, y_train, l1, warmstart=None):
    return GaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(l1),
        profile="rbf",
        warmstart=warmstart,
        max_iterations=200,
    )


def test_a_falsely_dead_group_is_worse_than_the_dense_fit(signal_data):
    x_train, y_train = signal_data
    l1 = 0.3

    dense_model, dense_llk, dense_state, _ = fit(x_train, y_train, l1)
    dense_norms = group_norms(dense_model.theta)
    assert dense_norms[0] > 1e-3, dense_norms

    # kill the first group by hand and let the solver restart from that iterate
    dead_state = dense_state._replace(x=dense_state.x.at[..., 0].set(0.0))
    sparse_model, sparse_llk, _, _ = fit(x_train, y_train, l1, warmstart=dead_state)

    def objective(model, llk):
        return float(llk + l1 * group_norms(model.theta).sum())

    assert objective(dense_model, dense_llk) <= objective(sparse_model, sparse_llk)


def test_restart_can_revive_a_group_zeroed_in_the_warmstart(signal_data):
    x_train, y_train = signal_data
    _, _, state, _ = fit(x_train, y_train, 0.0)

    # restart drops z and u, so a live group is not trapped at zero by a stale dual
    dead_state = state._replace(x=state.x.at[..., 0].set(0.0))
    model, _, _, _ = fit(x_train, y_train, 0.0, warmstart=dead_state)
    assert group_norms(model.theta)[0] > 1e-3
