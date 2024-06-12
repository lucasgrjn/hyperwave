"""Temporal sampling for efficient frequency component extraction."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


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
