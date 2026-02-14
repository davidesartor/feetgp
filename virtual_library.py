from dataclasses import dataclass
from functools import cache
from jaxtyping import Array, Float, Key
import jax.numpy as jnp
import jax.random as jr


@dataclass
class Uniform:
    low: float
    high: float

    def sample(self, rng: Key, shape: tuple[int, ...]) -> Float[Array, "..."]:
        return jr.uniform(rng, shape, minval=self.low, maxval=self.high)


@dataclass
class Normal:
    mean: float
    std: float

    def sample(self, rng: Key, shape: tuple[int, ...]) -> Float[Array, "..."]:
        return jr.normal(rng, shape) * self.std + self.mean


@dataclass
class LogUniform:
    low: float
    high: float

    def sample(self, rng: Key, shape: tuple[int, ...]) -> Float[Array, "..."]:
        return jnp.exp(
            jr.uniform(rng, shape, minval=jnp.log(self.low), maxval=jnp.log(self.high))
        )


@dataclass
class LogNormal:
    mean: float
    std: float

    def sample(self, rng: Key, shape: tuple[int, ...]) -> Float[Array, "..."]:
        return jnp.exp(jr.normal(rng, shape) * self.std + self.mean)


class BoreHole:
    """
    Borehole function.
    Original implementation from
    https://www.sfu.ca/~ssurjano/borehole.html
    """

    inputs = {
        "rw": Normal(mean=0.10, std=0.0161812),
        "r": LogNormal(mean=7.71, std=1.0056),
        "Tu": Uniform(low=63070, high=115600),
        "Hu": Uniform(low=990, high=1110),
        "Tl": Uniform(low=63.1, high=116),
        "Hl": Uniform(low=700, high=820),
        "L": Uniform(low=1120, high=1680),
        "Kw": Uniform(low=9855, high=12045),
    }

    @staticmethod
    def __call__(x: Float[Array, "... 8"]) -> Float[Array, "..."]:
        rw, r, Tu, Hu, Tl, Hl, L, Kw = jnp.split(x, 8, axis=-1)
        frac1 = 2 * jnp.pi * Tu * (Hu - Hl)
        frac2a = 2 * L * Tu / (jnp.log(r / rw) * rw**2 * Kw)
        frac2b = Tu / Tl
        frac2 = jnp.log(r / rw) * (1 + frac2a + frac2b)
        y = frac1 / frac2
        return y.squeeze(-1)

    @classmethod
    def sample(
        cls, n: int, seed: int = 0, normalize: bool = False
    ) -> tuple[Float[Array, "n 8"], Float[Array, "n"]]:
        rngs = jr.split(jr.key(seed), len(cls.inputs))
        x = {
            k: dist.sample(rng, (n, 1))
            for (k, dist), rng in zip(cls.inputs.items(), rngs)
        }

        x = jnp.concatenate(list(x.values()), axis=-1)
        y = cls.__call__(x)

        if normalize:
            # log scale r parameter
            x = x.at[:, 1].set(jnp.log(x[:, 1]))
            x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)

            y_mean, y_std = cls.output_scale()
            y = (y - y_mean) / y_std
        return x, y

    @classmethod
    @cache
    def output_scale(cls) -> tuple[Float[Array, ""], Float[Array, ""]]:
        x, y = cls.sample(1000, normalize=False)
        return y.mean(axis=0), y.std(axis=0)


class Cantilever:
    """
    Cantilever Beam function.
    Original implementation from
    https://www.sfu.ca/~ssurjano/canti.html
    """

    inputs = {
        "R": Normal(mean=40000, std=2000),
        "E": Normal(mean=2.9e7, std=1.45e6),
        "X": Normal(mean=500, std=100),
        "Y": Normal(mean=1000, std=100),
    }

    @staticmethod
    def __call__(x: Float[Array, "... 4"]) -> Float[Array, "... 2"]:
        R, E, X, Y = jnp.split(x, 4, axis=-1)
        L = 100
        D_0 = 2.2535
        w = 4.0
        t = 2.0

        Sterm1 = 600 * Y / (w * (t**2))
        Sterm2 = 600 * X / ((w**2) * t)
        S = Sterm1 + Sterm2
        Dfact1 = 4 * (L**3) / (E * w * t)
        Dfact2 = jnp.sqrt((Y / (t**2)) ** 2 + (X / (w**2)) ** 2)
        D = Dfact1 * Dfact2
        y = jnp.concatenate([S, D], axis=-1)
        return y

    @classmethod
    def sample(
        cls, n: int, seed: int = 0, normalize: bool = False
    ) -> tuple[Float[Array, "n 4"], Float[Array, "n 2"]]:
        rngs = jr.split(jr.key(seed), len(cls.inputs))
        x = {
            k: dist.sample(rng, (n, 1))
            for (k, dist), rng in zip(cls.inputs.items(), rngs)
        }

        x = jnp.concatenate(list(x.values()), axis=-1)
        y = cls.__call__(x)

        if normalize:
            x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
            y_mean, y_std = cls.output_scale()
            y = (y - y_mean) / y_std
        return x, y

    @classmethod
    @cache
    def output_scale(cls) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        x, y = cls.sample(1000, normalize=False)
        return y.mean(axis=0), y.std(axis=0)


class ShortColumn:
    """
    Short Column function.
    Original implementation from
    https://www.sfu.ca/~ssurjano/shortcol.html
    """

    inputs = {
        "Y": LogNormal(mean=5.0, std=0.5),
        "M": Normal(mean=2000, std=400),
        "P": Normal(mean=500, std=100),
    }

    @staticmethod
    def __call__(x: Float[Array, "... 3"]) -> Float[Array, "..."]:
        Y, M, P = jnp.split(x, 3, axis=-1)
        b = 5
        h = 15
        term1 = -4 * M / (b * (h**2) * Y)
        term2 = -(P**2) / ((b**2) * (h**2) * (Y**2))
        y = 1 + term1 + term2
        return y.squeeze(-1)

    @classmethod
    def sample(
        cls, n: int, seed: int = 0, normalize: bool = False
    ) -> tuple[Float[Array, "n 3"], Float[Array, "n"]]:
        rngs = jr.split(jr.key(seed), len(cls.inputs))
        x = {
            k: dist.sample(rng, (n, 1))
            for (k, dist), rng in zip(cls.inputs.items(), rngs)
        }

        x = jnp.concatenate(list(x.values()), axis=-1)
        y = cls.__call__(x)

        if normalize:
            # log scale Y parameter
            x = x.at[:, 0].set(jnp.log(x[:, 0]))
            x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
            y_mean, y_std = cls.output_scale()
            y = (y - y_mean) / y_std
        return x, y

    @classmethod
    @cache
    def output_scale(cls) -> tuple[Float[Array, ""], Float[Array, ""]]:
        x, y = cls.sample(1000, normalize=False)
        return y.mean(axis=0), y.std(axis=0)


class SteelColumn:
    """
    Steel Column function.
    Original implementation from
    https://www.sfu.ca/~ssurjano/steelcol.html
    """

    inputs = {
        "Fs": LogNormal(mean=400, std=35),
        "P1": Normal(mean=500000, std=50000),
        "P2": Normal(mean=600000, std=90000),  # supposes to be Gumbel
        "P3": Normal(mean=600000, std=90000),  # supposed to be Gumbel
        "B": LogNormal(mean=300, std=3),
        "D": LogNormal(mean=20, std=2),
        "H": LogNormal(mean=300, std=5),
        "F0": Normal(mean=30, std=10),
        "E": Normal(mean=210000, std=4200),  # supposed to be Weibull
    }

    @staticmethod
    def __call__(x: Float[Array, "... 9"]) -> Float[Array, "..."]:
        Fs, P1, P2, P3, B, D, H, F0, E = jnp.split(x, 9, axis=-1)
        L = 7500
        P = P1 + P2 + P3
        Eb = (jnp.pi**2) * E * B * D * (H**2) / (2 * (L**2))
        term1 = 1 / (2 * B * D)
        term2 = F0 * Eb / (B * D * H * (Eb - P))
        y = Fs - P * (term1 + term2)
        return y.squeeze(-1)

    @classmethod
    def sample(
        cls, n: int, seed: int = 0, normalize: bool = False
    ) -> tuple[Float[Array, "n 3"], Float[Array, "n"]]:
        rngs = jr.split(jr.key(seed), len(cls.inputs))
        x = {
            k: dist.sample(rng, (n, 1))
            for (k, dist), rng in zip(cls.inputs.items(), rngs)
        }

        x = jnp.concatenate(list(x.values()), axis=-1)
        y = cls.__call__(x)

        if normalize:
            # log scale Fs parameter
            x = x.at[:, 0].set(jnp.log(x[:, 0]))
            # log scale B,D,H parameters
            x = x.at[:, 4:7].set(jnp.log(x[:, 4:7]))
            x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
            y_mean, y_std = cls.output_scale()
            y = (y - y_mean) / y_std
        return x, y

    @classmethod
    @cache
    def output_scale(cls) -> tuple[Float[Array, ""], Float[Array, ""]]:
        x, y = cls.sample(1000, normalize=False)
        return y.mean(axis=0), y.std(axis=0)


class SulfurModel:
    """
    Sulfur model function.
    Original implementation from
    https://www.sfu.ca/~ssurjano/sulfur.html
    """

    inputs = {
        "Tr": LogNormal(mean=0.76, std=1.2),
        "1-Ac": LogNormal(mean=0.39, std=1.1),
        "1-Rs": LogNormal(mean=0.85, std=1.1),
        "beta": LogNormal(mean=0.3, std=1.3),
        "Psi_e": LogNormal(mean=5.0, std=1.4),
        "f_Psi_e": LogNormal(mean=1.7, std=1.2),
        "Q": LogNormal(mean=71.0, std=1.15),
        "Y": LogNormal(mean=0.5, std=1.5),
        "L": LogNormal(mean=5.5, std=1.5),
    }

    @staticmethod
    def __call__(x: Float[Array, "... 9"]) -> Float[Array, "..."]:
        Tr, Ac, Rs, beta, Psi_e, f_Psi_e, Q, Y, L = jnp.split(x, 9, axis=-1)
        Ac, Rs = 1 - Ac, 1 - Rs
        S0 = 1366
        A = 5 * 1e14
        fact1 = (S0**2) * (1 - Ac) * (Tr**2) * (1 - Rs) ** 2 * beta * Psi_e * f_Psi_e
        fact2 = 3 * Q * Y * L / A
        y = -1 / 2 * fact1 * fact2
        return y.squeeze(-1)

    @classmethod
    def sample(
        cls, n: int, seed: int = 0, normalize: bool = False
    ) -> tuple[Float[Array, "n 3"], Float[Array, "n"]]:
        rngs = jr.split(jr.key(seed), len(cls.inputs))
        x = {
            k: dist.sample(rng, (n, 1))
            for (k, dist), rng in zip(cls.inputs.items(), rngs)
        }

        x = jnp.concatenate(list(x.values()), axis=-1)
        y = cls.__call__(x)

        if normalize:
            # log scale all parameters
            x = jnp.log(x)
            x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
            y_mean, y_std = cls.output_scale()
            y = (y - y_mean) / y_std
        return x, y

    @classmethod
    @cache
    def output_scale(cls) -> tuple[Float[Array, ""], Float[Array, ""]]:
        x, y = cls.sample(1000, normalize=False)
        return y.mean(axis=0), y.std(axis=0)


class OTLCircuit:
    """
    OTL Circuit function.
    Original implementation from
    https://www.sfu.ca/~ssurjano/otlcircuit.html
    """

    inputs = {
        "Rb1": Uniform(low=50, high=150),
        "Rb2": Uniform(low=25, high=70),
        "Rf": Uniform(low=0.5, high=3),
        "Rc1": Uniform(low=1.2, high=2.5),
        "Rc2": Uniform(low=0.25, high=1.2),
        "beta": Uniform(low=50, high=300),
    }

    @staticmethod
    def __call__(x: Float[Array, "... 6"]) -> Float[Array, "..."]:
        Rb1, Rb2, Rf, Rc1, Rc2, beta = jnp.split(x, 6, axis=-1)
        Vb1 = 12 * Rb2 / (Rb1 + Rb2)
        term1a = (Vb1 + 0.74) * beta * (Rc2 + 9)
        term1b = beta * (Rc2 + 9) + Rf
        term1 = term1a / term1b

        term2a = 11.35 * Rf
        term2b = beta * (Rc2 + 9) + Rf
        term2 = term2a / term2b

        term3a = 0.74 * Rf * beta * (Rc2 + 9)
        term3b = (beta * (Rc2 + 9) + Rf) * Rc1
        term3 = term3a / term3b

        y = term1 + term2 + term3
        return y.squeeze(-1)

    @classmethod
    def sample(
        cls, n: int, seed: int = 0, normalize: bool = False
    ) -> tuple[Float[Array, "n 3"], Float[Array, "n"]]:
        rngs = jr.split(jr.key(seed), len(cls.inputs))
        x = {
            k: dist.sample(rng, (n, 1))
            for (k, dist), rng in zip(cls.inputs.items(), rngs)
        }

        x = jnp.concatenate(list(x.values()), axis=-1)
        y = cls.__call__(x)

        if normalize:
            x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
            y_mean, y_std = cls.output_scale()
            y = (y - y_mean) / y_std
        return x, y

    @classmethod
    @cache
    def output_scale(cls) -> tuple[Float[Array, ""], Float[Array, ""]]:
        x, y = cls.sample(1000, normalize=False)
        return y.mean(axis=0), y.std(axis=0)


class WingWeight:
    """
    Wing weight function.
    Original implementation from
    https://www.sfu.ca/~ssurjano/wingweight.html
    """

    inputs = {
        "Sw": Uniform(low=150, high=200),
        "Wfw": Uniform(low=220, high=300),
        "A": Uniform(low=6, high=10),
        "LamCaps": Uniform(low=-10, high=10),
        "q": Uniform(low=16, high=45),
        "lam": Uniform(low=0.5, high=1),
        "tc": Uniform(low=0.08, high=0.18),
        "Nz": Uniform(low=2.5, high=6),
        "Wdg": Uniform(low=1700, high=2500),
        "Wp": Uniform(low=0.025, high=0.08),
    }

    @staticmethod
    def __call__(x: Float[Array, "... 10"]) -> Float[Array, "..."]:
        Sw, Wfw, A, LamCaps, q, lam, tc, Nz, Wdg, Wp = jnp.split(x, 10, axis=-1)
        fact1 = 0.036 * Sw**0.758 * Wfw**0.0035
        fact2 = (A / ((jnp.cos(LamCaps)) ** 2)) ** 0.6
        fact3 = q**0.006 * lam**0.04
        fact4 = (100 * tc / jnp.cos(LamCaps)) ** (-0.3)
        fact5 = (Nz * Wdg) ** 0.49
        term1 = Sw * Wp
        y = fact1 * fact2 * fact3 * fact4 * fact5 + term1
        return y.squeeze(-1)

    @classmethod
    def sample(
        cls, n: int, seed: int = 0, normalize: bool = False
    ) -> tuple[Float[Array, "n 3"], Float[Array, "n"]]:
        rngs = jr.split(jr.key(seed), len(cls.inputs))
        x = {
            k: dist.sample(rng, (n, 1))
            for (k, dist), rng in zip(cls.inputs.items(), rngs)
        }

        x = jnp.concatenate(list(x.values()), axis=-1)
        y = cls.__call__(x)

        if normalize:
            x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
            y_mean, y_std = cls.output_scale()
            y = (y - y_mean) / y_std
        return x, y

    @classmethod
    @cache
    def output_scale(cls) -> tuple[Float[Array, ""], Float[Array, ""]]:
        x, y = cls.sample(1000, normalize=False)
        return y.mean(axis=0), y.std(axis=0)
