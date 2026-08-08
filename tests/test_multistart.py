import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange

from feetgp.gp import GroupLassoGaussianProcess, penalized_objective

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def signal_data():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(40, 2, 3))
    signal = np.sin(4.0 * x[:, 0, 0]) + x[:, 0, 1]
    y = np.stack([signal, -signal], axis=1) + 0.05 * rng.normal(size=(40, 2))
    return jnp.asarray(x), jnp.asarray(y)


def group_norms(model):
    return np.linalg.norm(
        np.asarray(rearrange(model.theta, "o d g -> d (o g)")), axis=-1
    )


def test_dense_start_resurrects_a_falsely_dead_group(signal_data):
    x_train, y_train = signal_data
    l1 = 0.3

    dense_model, _, dense_state, _ = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(l1),
        autoregressive=False,
        max_iterations=200,
        chunk_size=2,
    )
    dense_norms = group_norms(dense_model)
    assert dense_norms[0] > 1e-3, dense_norms

    dead_state = dense_state._replace(
        x=dense_state.x.at[..., 0].set(0.0),
        z=dense_state.z.at[..., 0].set(0.0),
        u=dense_state.u.at[..., 0].set(0.0),
    )
    sparse_model, _, _, _ = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(l1),
        autoregressive=False,
        warmstart=dead_state,
        max_iterations=200,
        chunk_size=2,
    )
    sparse_norms = group_norms(sparse_model)
    assert sparse_norms[0] < 1e-8, sparse_norms

    def objective(model):
        return float(
            penalized_objective(model.theta, model.g, jnp.array(l1), x_train, y_train)
        )

    assert objective(dense_model) < objective(sparse_model)
