"""Temporal sampling for efficient frequency component extraction."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .typing import Range


def omegas(freq_range: Range) -> jax.Array:
    # Convert to angular frequency.
    start, stop, num = freq_range
    if num == 1:
        return jnp.array([(start + stop) / 2])
    else:
        return jnp.linspace(start, stop, num)


def sampling_interval(freq_range: Range) -> float:
    w = omegas(freq_range)

    if len(w) == 1:
        return float(jnp.pi / (2 * w[0]))  # Quarter-period.
    else:
        w_avg = (w[0] + w[-1]) / 2
        dw = (w[-1] - w[0]) / (len(w) - 1)
        return _round_to_mult(
            2 * jnp.pi / (len(w) * dw),
            multiple=jnp.pi / (len(w) * w_avg),
            offset=0.5,
        )


def _round_to_mult(x, multiple, offset=0):
    return (round(x / multiple - offset) + offset) * multiple


def freq_projection(
    snapshots: ArrayLike,
    omegas: ArrayLike,
    t: ArrayLike,
) -> jax.Array:
    # Build ``P`` matrix.
    wt = dt * omegas[None, :] * (snapshots.arange()[:, None] + 0.5)
    P = jnp.concatenate([jnp.cos(wt), -jnp.sin(wt)], axis=1)

    # Project out frequency components.
    res = jnp.einsum("ij,j...->i...", jnp.linalg.inv(P), snapshots)
    return res[: len(omegas)] + 1j * res[len(omegas) :]


class FreqSpace(NamedTuple):
    start: float
    stop: float
    num: int

    @property
    def freqs(self) -> jax.Array:
        if self.num == 1:
            return jnp.array([(self.start + self.stop) / 2])
        else:
            return jnp.linspace(self.start, self.stop, self.num)

    # TODO: I do not think this belongs here.
    @property
    def phases(self) -> jax.Array:
        # return jnp.zeros((self.num,))
        return -1 * jnp.linspace(
            start=0,
            stop=self.num * jnp.pi,
            endpoint=False,
            num=self.num,
        )

    @property
    def sampling_interval(self) -> float:
        w_avg = (self.start + self.stop) / 2
        if self.num == 1:
            return float(jnp.pi / (2 * w_avg))  # Quarter-period.
        else:
            dw = (self.stop - self.start) / (self.num - 1)
            return float(
                self.round_to_mult(
                    2 * jnp.pi / (self.num * dw),
                    multiple=jnp.pi / (self.num * w_avg),
                    offset=0.5,
                )
            )

    @staticmethod
    def round_to_mult(x, multiple, offset=0):
        # return x
        return (round(x / multiple - offset) + offset) * multiple
