import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feetgp import admm
from feetgp.glassogp import GroupLassoGaussianProcess, penalized_objective

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def signal_data():
    """Group 0 (columns 0-2) carries the signal, group 1 is pure noise."""
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(40, 6))
    signal = np.sin(4.0 * x[:, 0]) + x[:, 1]
    y = np.stack([signal, -signal], axis=1) + 0.05 * rng.normal(size=(40, 2))
    return jnp.asarray(x), jnp.asarray(y)


def test_dense_start_resurrects_a_falsely_dead_group(signal_data):
    """Continuation bias is corrected by ranking starts on the true objective.

    Death is absorbing: a warmstart with the signal group forced dead keeps it dead,
    so the chained start alone can never win it back. The dense start can, and the
    penalized objective ranks it above the biased solution.
    """
    x_train, y_train = signal_data
    group_size, l1 = 3, 0.3

    dense_model, _, dense_state, _ = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(l1),
        group_size=group_size,
        autoregressive=False,
        max_iterations=200,
        chunk_size=2,
    )
    dense_norms = np.linalg.norm(
        np.asarray(admm.to_groups(dense_model.theta, group_size)), axis=-1
    )
    assert dense_norms[0] > 1e-3, dense_norms

    # forcing x = z = u = 0 on the signal group reproduces what a sparser-than-true
    # chained warmstart hands the next lambda
    dead_state = dense_state._replace(
        x=dense_state.x.at[0].set(0.0),
        z=dense_state.z.at[0].set(0.0),
        u=dense_state.u.at[0].set(0.0),
    )
    sparse_model, _, _, _ = GroupLassoGaussianProcess.fit(
        x_train,
        y_train,
        l1_penalty=jnp.array(l1),
        group_size=group_size,
        autoregressive=False,
        warmstart=dead_state,
        max_iterations=200,
        chunk_size=2,
    )
    sparse_norms = np.linalg.norm(
        np.asarray(admm.to_groups(sparse_model.theta, group_size)), axis=-1
    )
    assert sparse_norms[0] < 1e-8, sparse_norms

    def objective(model):
        return float(
            penalized_objective(
                model.theta, model.g, jnp.array(l1), x_train, y_train, group_size
            )
        )

    assert objective(dense_model) < objective(sparse_model)
