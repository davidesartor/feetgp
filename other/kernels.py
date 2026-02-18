from typing import NamedTuple, Protocol, Optional
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.numpy.linalg import norm


def pairwise_distance(
    x1: Float[Array, "n d"],
    x2: Optional[Float[Array, "m d"]] = None,
    order: int = 2,
) -> Float[Array, "n m"]:
    def dist(a, b):
        # IMPORTANT: lax.cond avoids propagating NaNs in the gradient
        # other control flow options (select, jnp.where, ...) do not work here!!!
        return jax.lax.cond(
            jnp.all(a == b), lambda: 0.0, lambda: norm(a - b, ord=order)
        )

    dist = jax.vmap(jax.vmap(dist, (None, 0)), (0, None))
    return dist(x1, x2 if x2 is not None else x1)


class Kernel(Protocol):
    def __call__(
        self, x1: Float[Array, "n d"], x2: Optional[Float[Array, "m d"]] = None
    ) -> Float[Array, "n m"]: ...


@dataclass
class SquaredExponential(Kernel):
    norm_order: int = 2

    def __call__(
        self,
        x1: Float[Array, "n d"],
        x2: Optional[Float[Array, "m d"]] = None,
    ) -> Float[Array, "n m"]:
        d = pairwise_distance(x1, x2, order=self.norm_order)
        return jnp.exp(-(d**2))


@dataclass
class Matern(Kernel):
    smoothness_order: int = 2
    norm_order: int = 2

    def __call__(
        self,
        x1: Float[Array, "n d"],
        x2: Optional[Float[Array, "m d"]] = None,
    ) -> Float[Array, "n m"]:
        d = pairwise_distance(x1, x2, order=self.norm_order)
        if self.smoothness_order == 0:
            K = jnp.exp(-d)
        elif self.smoothness_order == 1:
            K = (1 + jnp.sqrt(3) * d) * jnp.exp(-jnp.sqrt(3) * d)
        elif self.smoothness_order == 2:
            K = (1 + jnp.sqrt(5) * d + 5 / 3 * d**2) * jnp.exp(-jnp.sqrt(5) * d)
        else:
            raise ValueError(f"Unsupported smoothness order {self.smoothness_order}")
        return K